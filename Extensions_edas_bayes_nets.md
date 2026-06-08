# Extending `edas_bayes_nets` for full pateda integration

**Audience:** an implementing agent who will modify the `edas_bayes_nets`
(`bayes_nets`) library.
**Scope:** this document only *describes* the changes. It makes **no** changes
to `edas_bayes_nets`. All pateda-side wiring described here is already partly in
place (see "Current state") or is trivial glue that can follow once the
`bayes_nets` features below exist.

---

## 0. Background and current state

`pateda` has been migrated off `pgmpy` for its Bayesian-network EDAs. The
Bayesian-network learners now delegate to `bayes_nets`:

| pateda file | role | now uses |
|-------------|------|----------|
| `packages/pateda/src/pateda/learning/ebna.py` | `LearnEBNA` | `bayes_nets.BayesianNetwork.fit` / `set_structure` + `learn_parameters` |
| `packages/pateda/src/pateda/learning/boa.py`  | `LearnBOA`  | `bayes_nets.BayesianNetwork.fit` (K2 / BIC) |
| `packages/pateda/src/pateda/sampling/bayesian_network.py` | `SampleBayesianNetwork` | ancestral sampling over the learned CPDs |

### The integration contract (do not break)

Both learners return a `pateda.core.models.BayesianNetworkModel` with:

* `structure` — an `(n_vars, n_vars)` **numpy adjacency matrix**, parent → child
  (`structure[i, j] == 1` ⟺ edge `i → j`). Identical to `bayes_nets`'
  `BayesianNetwork.adjacency` and to `bn.to_adjacency_matrix()`.
* `parameters` — a **dict** `{var: {"parents": [...], "cpd": np.ndarray}}`.
  Root CPD is 1-D `(card[var],)`; non-root CPD is 2-D
  `(n_parent_configs, card[var])` with rows summing to 1. Parent-config index is
  **row-major in `parents` order with the first parent varying fastest**
  (`idx = Σ_j value(parent_j) · ∏_{l<j} card(parent_l)`), matching
  `bayes_nets.parameter_learning` and `bayes_nets.sampling`.

Every feature below must keep producing / consuming these two structures so that
`SampleBayesianNetwork`, the knowledge-extraction modules, the visualizers, and
the cluster runners keep working unchanged.

The same numpy contract is what `pateda`'s factorized machinery
(`FactorizedModel`, `SampleFDA`, MAP/k-MPC) already speaks, so the extensions
below are mostly *format adapters* plus a few genuinely new algorithms
(triangulation, junction tree, exact MPC).

---

## 1. REQUIRED FIX (blocking): zero-count guard in BIC/AIC scoring

**Symptom.** With `alpha = 0` (the default for `LearnEBNA`), `bayes_nets`'
BIC/AIC scoring returns `nan` whenever a state has zero count, because
`bayes_nets/scoring.py::_log_likelihood` evaluates

```python
ll = np.sum(counts * np.log(counts / total))   # counts may contain 0 → 0*log(0) = nan
```

with no `0·log(0) = 0` guard. Since `nan > best_score` is always `False`, greedy
hill-climbing **never adds an edge**, so `LearnEBNA` silently degrades to UMDA
and fails on problems that need dependencies (e.g. Deceptive-3). The old pateda
implementation guarded this with `if count > 0`.

**Fix.** In `bayes_nets/scoring.py::_log_likelihood`, compute the term only over
positive counts, e.g.

```python
nz = counts > 0
ll = float(np.sum(counts[nz] * np.log(counts[nz] / total)))
```

(both in the no-parent and the with-parents branches). This makes
`alpha = 0` behave as a proper maximum-likelihood score, matching the K2 path
which is already safe (it uses `gammaln`).

**Acceptance test.**
`BICScoringMethod(alpha=0.0).local_score(...)` must be finite for data with
zero-count states; `pateda`'s `tests/test_discrete_eda.py::TestEBNA::
test_ebna_on_deceptive` must pass reliably (it currently passes only
intermittently because EBNA defaults to `alpha=0.0`).

> Until this fix lands, the canonical workaround on the pateda side is to use a
> small positive `alpha` (e.g. `LearnEBNA(alpha=0.1)`, as in
> `examples/ebna_deceptive.py`). The pateda default is intentionally left at
> `alpha=0.0` so the defect remains visible.

---

## 2. (Task 2.1) Convert a BN into an FDA-style clique factorization

EDAs of the FDA / MN-FDA family sample from a **junction-tree factorization**:
a set of cliques with marginal/conditional probability tables. `bayes_nets`
should be able to turn a learned BN into exactly that representation, obtained
from a **triangulated** version of the (moralized) BN, with optional **pruning
to bound the maximum clique width**.

### 2.1.1 Target output format (must match pateda's `FactorizedModel`)

`pateda.core.models.FactorizedModel` uses:

* `structure` — a **cliques matrix**; each row is
  `[n_overlap, n_new, overlap_vars..., new_vars...]`
  (`n_overlap` = number of "already-sampled" separator variables for that clique,
  `n_new` = number of variables first introduced by that clique). This row layout
  is parsed by `pateda/sampling/fda.py` and `pateda/sampling/map_sampling.py::
  _extract_cliques_and_tables`.
* `parameters` — a **list of probability tables**, one per clique, ordered to
  match `structure`. Root cliques (`n_overlap == 0`) store a marginal over their
  `new_vars`; non-root cliques store `p(new_vars | overlap_vars)`. Variable
  combinations are encoded with `find_acc_card` / `index_convert_card`
  (`pateda/learning/utils/conversions.py`) — first variable fastest.

The reference builders already exist on the pateda side and define the exact
semantics to reproduce:
`pateda/learning/utils/markov_network.py::{find_maximal_cliques_greedy,
order_cliques_for_sampling, convert_cliques_to_factorized_structure}`.

### 2.1.2 Algorithm to implement in `bayes_nets`

Add a module `bayes_nets/factorization.py` exposing:

```python
def moralize(adjacency) -> np.ndarray:
    """Return the undirected moral graph: drop edge directions and connect
    every pair of parents that share a common child."""

def triangulate(moral_adjacency, cardinality, method="min-fill",
                max_clique_width=None):
    """Return (triangulated_adjacency, elimination_order, cliques).
    Use a minimum-fill-in or minimum-degree elimination heuristic.
    `cliques` is a list of variable-index arrays (the elimination cliques).
    If `max_clique_width` is given, see 2.1.3."""

def junction_tree(cliques):
    """Build a clique (junction) tree by maximum-spanning-tree on the
    separator-cardinality-weighted clique graph; return tree edges + separators."""

class CliqueFactorization:
    structure: np.ndarray          # FactorizedModel cliques matrix
    tables:    list[np.ndarray]    # one table per clique (FactorizedModel order)
    cliques:   list[np.ndarray]    # raw maximal cliques
    separators: list[np.ndarray]
```

and a one-call entry point on the BN itself:

```python
class BayesianNetwork:
    def to_factorization(self, data=None, alpha=1.0,
                         max_clique_width=None) -> CliqueFactorization:
        """Moralize → triangulate → junction tree → clique tables.

        Clique marginals/conditionals are estimated either
        (a) from `data` directly (counting over each clique's variables), or
        (b) by local inference from the BN's CPDs when `data` is None.
        Returns a CliqueFactorization whose `.structure`/`.tables` are directly
        usable to build a pateda FactorizedModel and to run SampleFDA / MAP."""
```

### 2.1.3 Clique-width pruning

When the triangulated graph contains a clique whose joint table would exceed a
budget (either a user `max_clique_width`, or the
"table size ≤ n_samples" rule pateda already uses via
`learning/utils/table_size.py::joint_table_size`), **prune** before building
tables:

* Drop the lowest-mutual-information edges inside the oversized clique (or the
  weakest BN edges feeding it) and re-triangulate, **or**
* Split the clique along its separator and renormalise the resulting
  conditional tables.

Mirror the existing pateda behaviour in
`learning/mnfda.py::_apply_table_size_limit` (splits oversized cliques so each
joint table fits in the sample size). The pruning policy should be a parameter
(`width_control="mi" | "split"`), default `"split"`.

### 2.1.4 pateda-side glue (small, after the above exists)

* Add `pateda/learning/utils/bn_to_fda.py` with
  `bn_factorization_to_model(cf) -> FactorizedModel` (a 3-line adapter wrapping
  `CliqueFactorization.structure/.tables`).
* This immediately lets `SampleFDA`, `SampleInsertMAP`, and `SampleInsertKMAP`
  consume a BN-derived factorization with **no further changes** (they already
  parse the `[n_overlap, n_new, ...]` row layout).

### 2.1.5 Acceptance tests

* For a BN that is already a tree, `to_factorization()` must reproduce the
  Chow-Liu clique set (pairs) and `SampleFDA` over it must match
  `SampleBayesianNetwork` in distribution (KL within tolerance).
* For a BN with v-structures, the moral graph must contain the parent-parent
  fill edge; every clique table must sum to 1 over its `new_vars` per
  `overlap_vars` configuration.
* With `max_clique_width=k`, no clique's joint table exceeds `∏ card` for `k`
  variables.

---

## 3. (Task 2.2) Visualization and problem-information extraction

The goal is that a BN learned by `bayes_nets` plugs into pateda's existing
knowledge-extraction and visualization pipeline **without bespoke code**.

### 3.1 What pateda expects

* `pateda/knowledge_extraction/model_visualizations.py` consumes
  `run_structures['all_big_matrices']` — a per-generation list of **adjacency
  "big matrices"** — for dendrogram (`view_dendrogram_structure`) and glyph
  (`view_glyph_structure`) comparison. These are plain `(n_vars, n_vars)`
  matrices, exactly `bn.adjacency`.
* `pateda/knowledge_extraction/dependency_analysis.py::learn_bayesian_network`
  returns `{'adjacency_matrix': ..., 'edges': ..., 'score': ...}` using its own
  duplicated numpy structure search. `analyze_variable_dependencies` and the
  edge/structure-score helpers build on that.

### 3.2 Extensions in `bayes_nets`

1. **Structure export helpers** on `BayesianNetwork`:
   ```python
   def big_matrix(self) -> np.ndarray            # alias of adjacency, for clarity
   def edge_list(self) -> list[tuple[int,int]]   # [(parent, child), ...]
   def to_run_structure(self, generation, run=0) -> dict
       # {'adjacency': ..., 'generation': ..., 'run': ...}, ready to append into
       # run_structures['all_big_matrices'] for the pateda dendrogram/glyph views
   ```
2. **Problem-information extraction** consistent with
   `dependency_analysis.py`:
   ```python
   def structure_score(self, data, score="bic") -> float
   def variable_dependencies(self, data) -> dict   # per-variable parents,
                                                    # Markov blanket, degree,
                                                    # pairwise MI matrix
   def markov_blanket(self, var) -> list[int]      # parents ∪ children ∪
                                                    # children's other parents
   ```
   These should return the **same dict keys** that
   `analyze_variable_dependencies` already produces (`'adjacency_matrix'`,
   `'edges'`, `'score'`, `'mi_matrix'`) so pateda can delegate
   `dependency_analysis.learn_bayesian_network` to `bayes_nets` and delete its
   duplicate search.
3. **Visualization parity.** `bayes_nets/visualization.py` already provides
   `plot_bayesian_network` and `plot_marginals`. Extend it to honour pateda's
   figure conventions (from the repo `CLAUDE.md`): **no titles** by default
   (captions live in LaTeX), `.eps`/`.pdf`-friendly output, large fonts, and an
   optional `node_order`/`pos` argument so a structure can be drawn with the same
   layout across generations (needed for visually comparing structure evolution).

### 3.3 pateda-side glue

* `dependency_analysis.learn_bayesian_network(...)` becomes a thin wrapper that
  builds a `bayes_nets.BayesianNetwork`, calls `fit`, and returns
  `bn.variable_dependencies(data)` — removing the duplicated search while
  keeping the return signature.
* `model_visualizations` needs no change: it already takes adjacency matrices,
  which `bn.big_matrix()` / `bn.to_run_structure()` supply.

### 3.4 Acceptance tests

* `bn.to_run_structure(...)` appended across generations feeds
  `view_dendrogram_structure` / `view_glyph_structure` without error.
* `bn.variable_dependencies(data)` matches the keys and (within rounding) the
  values of `analyze_variable_dependencies` for the same data.

---

## 4. (Task 2.3) Most-probable-configuration (MPC) sampling integration

pateda samples "most probable configurations" through
`pateda/sampling/map_sampling.py` (`SampleInsertMAP`, `SampleTemplateMAP`,
`SampleHybridMAP`) and `pateda/sampling/kmap_sampling.py`, backed by
`pateda/inference/map_inference.py::MAPInference` and
`pateda/inference/kmpc.py`. Today:

* `MAPInference` exact mode is the **only remaining `pgmpy` dependency in
  pateda** (junction-tree / belief propagation). It already has numpy fallbacks
  (`bp`, `decimation`, `greedy`).
* `pateda/inference/kmpc.py::KMPCBayesianNetwork._extract_factorized` expects a
  **pgmpy-style** `structure` (`.nodes()`, `.predecessors()`, `cpd.get_values()`)
  and silently **falls back to "all variables independent"** for the numpy
  adjacency BN models pateda actually produces — i.e. it is currently a no-op for
  real BN models. This must be fixed by consuming the integration contract from
  §0.

### 4.1 Two supported paths (per the task statement)

`bayes_nets` should support computing the (k) most probable configuration(s)
either:

**(a) directly from the BN** — exact MPC by variable elimination / max-product
over the BN's CPDs (Viterbi-style), giving the single MPC and, via Nilsson's
scheme, the k-MPC; or

**(b) via the clique factorization from §2** — convert the BN to a
`CliqueFactorization`, then run MPC on the factorization (this is what pateda's
`MAPInference` / `_KMPCPartition` already do for cliques + tables).

### 4.2 Extensions in `bayes_nets`

Add `bayes_nets/inference.py`:

```python
class MaxProductInference:
    """Exact MPC on a BN via bucket elimination / max-product."""
    def __init__(self, bn: BayesianNetwork): ...
    def most_probable_config(self, evidence=None) -> tuple[np.ndarray, float]
    def k_most_probable_configs(self, k, evidence=None) -> tuple[np.ndarray, np.ndarray]
    def marginals(self, evidence=None) -> list[np.ndarray]   # exact node marginals

class BayesianNetwork:
    def most_probable_config(self, evidence=None): ...        # path (a)
    def k_most_probable_configs(self, k, evidence=None): ...  # path (a)
```

The factorization path (b) is obtained by `bn.to_factorization(...)` (§2) plus
the partition engine already present in pateda
(`kmpc.py::_KMPCPartition`), so `bayes_nets` need only guarantee that
`CliqueFactorization.structure/.tables` plug into `MAPInference(cliques, tables,
cardinalities)`.

### 4.3 pateda-side glue (after the above exists)

1. **Fix `KMPCBayesianNetwork._extract_factorized`** to read the numpy contract
   (`model.structure` adjacency + `model.parameters` cpds dict) instead of the
   pgmpy API. Concretely: topological order from the adjacency, and for each
   variable build a clique `[parents..., var]` whose table is the CPD reshaped to
   `(card[parent_0], …, card[var])`. (This alone makes k-MPC work for BN models;
   it can also simply call `bn.to_factorization()`.)
2. **Drop `pgmpy` from `MAPInference`.** Replace `_build_pgmpy_model` /
   `_compute_map_exact` (BeliefPropagation) with either the numpy
   `bp`/`decimation` methods already implemented, or — preferred — route exact
   inference through `bayes_nets.MaxProductInference` /
   `bayes_nets`'s junction-tree marginals. After this, remove the `pgmpy` import
   block at the top of `map_inference.py`; pateda then has **zero** `pgmpy`
   references and `pgmpy` can be removed from `requirements.txt` / `setup.py`
   (already removed from `packages/pateda/pyproject.toml`).
3. `SampleInsertMAP._compute_map` / `_compute_k_map` need no change: they call
   `MAPInference` / `compute_kmpc`, which will now be `pgmpy`-free.

### 4.4 Acceptance tests

* On a small BN (≤ 12 binary vars) `bn.most_probable_config()` must equal the
  brute-force argmax over the full joint.
* `bn.k_most_probable_configs(k)` must return strictly decreasing
  log-probabilities and match brute-force top-k.
* `pateda/inference/kmpc.py::KMPCBayesianNetwork` must, for a chain BN, return
  the chain's true MPC instead of the current independent-fallback result.
* `pateda` test suite passes with `pgmpy` uninstalled.

---

## 5. Suggested implementation order

1. **§1 zero-count guard** — one-line correctness fix; unblocks EBNA defaults.
2. **§2 factorization** (`moralize`/`triangulate`/`junction_tree`/
   `to_factorization` + clique-width pruning) — foundation reused by §4(b).
3. **§4 inference** (`MaxProductInference`, k-MPC) + pateda fix of
   `KMPCBayesianNetwork` and removal of the `pgmpy` exact path.
4. **§3 visualization / extraction parity** — mostly adapters once the structure
   export helpers exist.

Each step keeps the §0 contract intact, so pateda continues to run after every
individual step.

---

## 6. Files referenced (for the implementer)

**In `edas_bayes_nets` (to be modified by the other agent):**
`bayes_nets/scoring.py` (§1), new `bayes_nets/factorization.py` (§2), new
`bayes_nets/inference.py` (§4), `bayes_nets/visualization.py` and
`bayes_nets/bayesian_network.py` (§2–§4 methods, §3 helpers).

**In `pateda` (already migrated / thin glue only):**
`packages/pateda/src/pateda/learning/{ebna,boa}.py` (done),
`packages/pateda/src/pateda/sampling/bayesian_network.py` (done),
`packages/pateda/src/pateda/core/models.py` (done),
`.../inference/map_inference.py` and `.../inference/kmpc.py` (§4.3),
`.../knowledge_extraction/{dependency_analysis,model_visualizations}.py` (§3.3),
`.../learning/utils/{markov_network,conversions,table_size}.py` (reference
semantics for §2), `.../sampling/{fda,map_sampling,kmap_sampling}.py`
(consumers of the §2 factorization).
