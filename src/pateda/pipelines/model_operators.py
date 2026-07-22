"""
Model modifiers (MODMOD) and model conversors (MODCONV) for EDA pipelines

An EDA pipeline chains consistent components into a working EDA.  A recurring
obstacle is that a *learning method* (LM) produces a model of one type while a
*sampling method* (SM) expects another, so only a few (LM, SM) pairs are
directly compatible.  Two families of operators lift this restriction and let
the pipeline grammar (:mod:`pateda.pipelines.grammar`) combine many more LMs and
SMs:

- **MODMOD** (model modifier): given a probabilistic model (graph + tables),
  produce a *new* model (graph + tables), usually simpler.  Implemented here:

  * :func:`prune_factorized` -- prune a factorized / junction model by reducing
    every clique to at most ``K`` variables, marginalizing the dropped ones out
    of the clique factors;
  * :func:`tree_to_forest` -- cut the weakest tree edges, turning a tree into a
    forest of independent components;
  * :func:`tree_to_malign` -- keep only *malign* edges (the interaction changes
    the conditional mode), the model-level analogue of Tree-EDA-M.

- **MODCONV** (model conversor): change the *type* of a model so that a
  different sampler can consume it.  Implemented here:

  * :func:`bn_to_factorized` -- turn a Bayesian-network model into an equivalent
    factorized model (one conditional clique per variable, in ancestral order),
    which the factorized samplers (``SampleFDA``, ``SampleGibbs``) can sample.

The operators are exposed to a pipeline through :class:`ModifiedLearning`, a
:class:`~pateda.core.components.LearningMethod` decorator that runs a base LM and
applies a chain of MODMOD/MODCONV operators to its output.  The binary
"``LM MODCONV SM``" combination of the task is therefore realized as
``ModifiedLearning(LM, [conversor])`` paired with ``SM``.

Every operator degrades gracefully: if it cannot transform a given model it
returns it unchanged, so a pipeline never breaks because of a modifier.

References
----------
- de Sa, A. et al. (2017). "RECIPE: A Grammar-based Framework for Automatically
  Evolving Classification Pipelines." EuroGP 2017.
- Marinescu, R. et al. (2021). "Searching for Machine Learning Pipelines Using a
  Context-Free Grammar." AAAI 2021.
"""

from typing import Any, Callable, List, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import Model, FactorizedModel, BayesianNetworkModel
from pateda.learning.utils.conversions import find_acc_card, index_convert_card


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def model_type(model: Any) -> str:
    """Return a short tag for the kind of model: ``"bn"``, ``"factorized"``,
    ``"intfda"``, ``"regmarkov"``, ``"markovnet"`` or ``"other"``."""
    cls = type(model).__name__
    if cls == "BayesianNetworkModel":
        return "bn"
    if cls == "MarkovNetworkModel":
        return "markovnet"
    if cls == "FactorizedModel":
        return "factorized"
    if isinstance(model, Model) and isinstance(getattr(model, "parameters", None), dict):
        mt = model.metadata.get("model_type", "")
        if mt == "IntFDA":
            return "intfda"
        if mt == "RegularizedMarkov":
            return "regmarkov"
    return "other"


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _marginal_over_overlap(table: np.ndarray) -> np.ndarray:
    """Average a conditional table ``P(new | overlap)`` (shape ``(O, Nn)``) over
    its overlap configurations to obtain a marginal ``P(new)`` (shape ``(Nn,)``),
    renormalized.  Uniform average is used as the overlap distribution."""
    marg = np.asarray(table, dtype=float).mean(axis=0)
    s = marg.sum()
    return marg / s if s > 0 else np.full_like(marg, 1.0 / marg.size)


def _reduce_overlap(
    table: np.ndarray, overlap_cards: List[int], keep: List[bool]
) -> np.ndarray:
    """Marginalize the *dropped* overlap variables out of a conditional table.

    ``table`` has shape ``(prod(overlap_cards), n_new)``; ``keep[i]`` says whether
    overlap variable ``i`` is retained.  Returns the reduced conditional table of
    shape ``(prod(kept_cards), n_new)`` with rows renormalized."""
    table = np.asarray(table, dtype=float)
    O, n_new = table.shape
    m = len(overlap_cards)
    acc = find_acc_card(m, np.asarray(overlap_cards, dtype=int))
    kept_idx = [i for i in range(m) if keep[i]]
    kept_cards = [overlap_cards[i] for i in kept_idx]
    kacc = find_acc_card(len(kept_idx), np.asarray(kept_cards, dtype=int)) if kept_idx \
        else np.array([], dtype=int)
    n_kept = int(np.prod(kept_cards)) if kept_idx else 1

    acc_t = np.zeros((n_kept, n_new))
    cnt = np.zeros(n_kept)
    for o in range(O):
        vals = index_convert_card(o, m, acc)
        kvals = [vals[i] for i in kept_idx]
        ko = int(np.dot(kvals, kacc)) if kept_idx else 0
        acc_t[ko] += table[o]
        cnt[ko] += 1
    cnt[cnt == 0] = 1
    reduced = acc_t / cnt[:, None]
    rs = reduced.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return reduced / rs


def _clique_members(row: np.ndarray):
    """Return (n_overlap, n_new, overlap_vars, new_vars) of a clique row."""
    n_ov = int(row[0])
    n_new = int(row[1])
    overlap = [int(v) for v in row[2:2 + n_ov]]
    new = [int(v) for v in row[2 + n_ov:2 + n_ov + n_new]]
    return n_ov, n_new, overlap, new


# ---------------------------------------------------------------------------
# MODCONV: Bayesian network  ->  factorized model
# ---------------------------------------------------------------------------

def bn_to_factorized(model: Any, **_: Any) -> Model:
    """
    MODCONV -- convert a Bayesian-network model to an equivalent factorized model.

    A Bayesian network factorizes the joint as ``prod_i P(X_i | Pa_i)``, which is
    exactly a factorized model with one conditional clique
    ``[|Pa_i|, 1, Pa_i..., i]`` per variable and the CPD as its table.  Emitting
    the cliques in ancestral (topological) order lets the factorized samplers
    (``SampleFDA``, ``SampleGibbs``) draw from it -- so a BN learner (EBNA, BOA,
    ...) becomes usable with the factorized samplers.

    Returns the input unchanged if it is not a Bayesian-network model.
    """
    if model_type(model) != "bn":
        return model

    adj = np.asarray(model.structure)
    n_vars = adj.shape[0]
    cpds = model.parameters                    # {var: {"parents": [...], "cpd": array}}

    # Topological order: parents before children (Kahn's algorithm on the DAG).
    in_deg = adj.sum(axis=0).astype(int)
    order, queue = [], [v for v in range(n_vars) if in_deg[v] == 0]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w in np.where(adj[v] > 0)[0]:
            in_deg[w] -= 1
            if in_deg[w] == 0:
                queue.append(int(w))
    if len(order) != n_vars:                   # cyclic (shouldn't happen) -> bail
        return model

    cliques, tables = [], []
    for var in order:
        parents = list(cpds[var]["parents"])
        cpd = np.asarray(cpds[var]["cpd"], dtype=float)
        if len(parents) == 0:
            cliques.append([0, 1, var, 0])
            tables.append(cpd.ravel())
        else:
            cliques.append([len(parents), 1] + parents + [var])
            tables.append(cpd)                 # already (n_parent_configs, card)

    width = max(len(c) for c in cliques)
    structure = np.zeros((len(cliques), width), dtype=int)
    for r, c in enumerate(cliques):
        structure[r, :len(c)] = c

    meta = dict(getattr(model, "metadata", {}))
    meta.update({"model_type": "Factorized (from BN)", "converted_from": "bn"})
    return FactorizedModel(structure=structure, parameters=tables, metadata=meta)


# ---------------------------------------------------------------------------
# MODMOD: prune a factorized / junction model to clique width K
# ---------------------------------------------------------------------------

def prune_factorized(model: Any, K: int = 2, **_: Any) -> Model:
    """
    MODMOD -- reduce every clique of a factorized model to at most ``K`` variables.

    For a clique with more than ``K`` variables the excess *overlap* (parent)
    variables --- the ones furthest from the new variable --- are marginalized
    out of the clique factor, lowering the junction width / maximal-clique size
    to ``K``.  The result is a simpler, cheaper-to-sample factorized model.

    Returns the input unchanged if it is not a factorized model.
    """
    if model_type(model) not in ("factorized", "markovnet"):
        return model
    K = max(1, int(K))
    structure = np.asarray(model.structure)
    tables = model.parameters
    cardinality = model.metadata.get("cardinality")

    new_rows, new_tables = [], []
    for c in range(structure.shape[0]):
        n_ov, n_new, overlap, new = _clique_members(structure[c])
        table = tables[c]
        total = n_ov + n_new
        if total <= K or n_ov == 0:
            new_rows.append(list(structure[c])); new_tables.append(table); continue

        # Keep the new variables and the K - n_new overlap variables closest to
        # them (the last overlap entries), drop and marginalize out the rest.
        n_keep_ov = max(0, K - n_new)
        keep = [i >= (n_ov - n_keep_ov) for i in range(n_ov)]
        if cardinality is not None:
            ov_cards = [int(cardinality[v]) for v in overlap]
        else:
            # Infer overlap cardinalities from the table shape when possible.
            ov_cards = _infer_overlap_cards(table.shape[0], n_ov)
        reduced = _reduce_overlap(np.asarray(table), ov_cards, keep) if n_keep_ov > 0 \
            else _marginal_over_overlap(np.asarray(table))
        kept_overlap = [overlap[i] for i in range(n_ov) if keep[i]]
        row = [len(kept_overlap), n_new] + kept_overlap + new
        new_rows.append(row)
        new_tables.append(reduced)

    width = max(len(r) for r in new_rows)
    out = np.zeros((len(new_rows), width), dtype=int)
    for r, row in enumerate(new_rows):
        out[r, :len(row)] = row
    meta = dict(model.metadata); meta["pruned_width_K"] = K
    return FactorizedModel(structure=out, parameters=new_tables, metadata=meta)


def _infer_overlap_cards(n_overlap_configs: int, n_ov: int) -> List[int]:
    """Best-effort per-variable cardinality when only the product is known:
    assume an equal cardinality ``round(prod ** (1/n_ov))`` per variable."""
    c = max(2, int(round(n_overlap_configs ** (1.0 / max(1, n_ov)))))
    return [c] * n_ov


# ---------------------------------------------------------------------------
# MODMOD: tree -> forest  and  tree -> malign tree
# ---------------------------------------------------------------------------

def _to_root_clique(child: int, table: np.ndarray, row_len: int):
    """Build a root (marginal) clique row + marginal table for ``child``."""
    marg = _marginal_over_overlap(np.asarray(table, dtype=float)) if np.asarray(table).ndim == 2 \
        else np.asarray(table, dtype=float)
    return [0, 1, child, 0], marg


def _cut_edges(model: Any, should_cut: Callable[[int, np.ndarray], bool], tag: str) -> Model:
    """Turn every conditional clique for which ``should_cut(clique_index, table)``
    is true into a root (marginal) clique -- disconnecting that edge."""
    structure = np.asarray(model.structure)
    tables = model.parameters
    new_rows, new_tables = [], []
    for c in range(structure.shape[0]):
        n_ov, n_new, overlap, new = _clique_members(structure[c])
        table = tables[c]
        if n_ov == 1 and n_new == 1 and should_cut(c, np.asarray(table)):
            row, marg = _to_root_clique(new[0], table, structure.shape[1])
            new_rows.append(row); new_tables.append(marg)
        else:
            new_rows.append(list(structure[c])); new_tables.append(table)
    width = max(len(r) for r in new_rows) if new_rows else structure.shape[1]
    out = np.zeros((len(new_rows), width), dtype=int)
    for r, row in enumerate(new_rows):
        out[r, :len(row)] = row
    meta = dict(model.metadata); meta[tag] = True
    return FactorizedModel(structure=out, parameters=new_tables, metadata=meta)


def tree_to_forest(model: Any, cut_fraction: float = 0.3, rng=None, **_: Any) -> Model:
    """
    MODMOD -- cut the weakest edges of a tree/forest model to make it sparser.

    A ``cut_fraction`` of the conditional (parent -> child) cliques are turned
    into root cliques with a marginal table, disconnecting them.  The weakest
    edges are chosen by the mutual-information matrix stored in the model
    metadata when available, otherwise at random.  Returns a factorized model.
    """
    if model_type(model) != "factorized":
        return model
    rng = rng or np.random.default_rng()
    structure = np.asarray(model.structure)
    cond = [c for c in range(structure.shape[0]) if int(structure[c, 0]) == 1]
    if not cond:
        return model
    n_cut = int(round(cut_fraction * len(cond)))
    if n_cut <= 0:
        return model

    mi = model.metadata.get("mi_matrix")
    if mi is not None:
        mi = np.asarray(mi)
        strength = []
        for c in cond:
            p, ch = int(structure[c, 2]), int(structure[c, 3])
            strength.append(mi[p, ch])
        cut_set = set(np.asarray(cond)[np.argsort(strength)[:n_cut]].tolist())
    else:
        cut_set = set(rng.choice(cond, size=n_cut, replace=False).tolist())

    return _cut_edges(model, lambda c, t: c in cut_set, "tree_to_forest")


def tree_to_malign(model: Any, **_: Any) -> Model:
    """
    MODMOD -- keep only *malign* edges of a tree model (model-level Tree-EDA-M).

    An edge ``parent -> child`` is *benign* when the child's most probable value
    is the same for every parent value (the interaction does not change the main
    effect); such edges are cut to root cliques.  *Malign* edges --- where the
    conditional mode of the child depends on the parent --- are kept.  Returns a
    factorized model (usually a forest).
    """
    if model_type(model) != "factorized":
        return model

    def benign(c, table):
        # table: P(child | parent), shape (card_parent, card_child).
        if table.ndim != 2 or table.shape[0] < 2:
            return False
        modes = np.argmax(table, axis=1)
        return bool(np.all(modes == modes[0]))     # constant mode -> benign -> cut

    return _cut_edges(model, benign, "tree_to_malign")


# ---------------------------------------------------------------------------
# Registry + ModifiedLearning decorator
# ---------------------------------------------------------------------------

# Each operator: name -> (function, input_type, output_type, kind)
MODEL_OPERATORS = {
    "bn_to_factorized": (bn_to_factorized, "bn", "factorized", "MODCONV"),
    "prune_factorized": (prune_factorized, "factorized", "factorized", "MODMOD"),
    "tree_to_forest": (tree_to_forest, "factorized", "factorized", "MODMOD"),
    "tree_to_malign": (tree_to_malign, "factorized", "factorized", "MODMOD"),
}


class ModifiedLearning(LearningMethod):
    """
    A :class:`~pateda.core.components.LearningMethod` that applies MODMOD/MODCONV
    operators to the model produced by a base learning method.

    This realizes the ``LM MODCONV SM`` combination: wrap a learner with the
    operator(s) that convert / modify its model so that a chosen sampler can use
    it.  Operators are applied left to right; each is robust (returns the model
    unchanged if inapplicable), so the wrapper never breaks a pipeline.
    """

    def __init__(self, base_learner: LearningMethod,
                 operators: List, operator_params: Optional[dict] = None):
        """
        Args:
            base_learner: The learning method whose model is post-processed.
            operators: A list of operator names (keys of :data:`MODEL_OPERATORS`)
                or callables ``model -> model``.
            operator_params: Optional ``{operator_name: {kwarg: value}}`` extra
                arguments (e.g. ``{"prune_factorized": {"K": 2}}``).
        """
        self.base_learner = base_learner
        self.operators = operators
        self.operator_params = operator_params or {}

    def learn(self, generation, n_vars, cardinality, population, fitness, **params):
        model = self.base_learner.learn(generation, n_vars, cardinality,
                                        population, fitness, **params)
        # Make cardinality available to operators that need it (e.g. pruning).
        try:
            if isinstance(getattr(model, "metadata", None), dict):
                model.metadata.setdefault("cardinality", np.asarray(cardinality))
        except Exception:
            pass
        for op in self.operators:
            try:
                if callable(op):
                    model = op(model)
                else:
                    fn = MODEL_OPERATORS[op][0]
                    model = fn(model, **self.operator_params.get(op, {}))
            except Exception:
                pass                                # robust: skip a failed operator
        return model
