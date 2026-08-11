"""Phase 2 prototype: bounded-treewidth HBOA-Light.

Idea #2 of docs/BN_bounded_clique_learning.md, made concrete.  A k-tree scaffold
(built from the same pairwise-MI statistic Phase 1 uses) restricts every
variable's decision-graph parent search to the k-clique it attaches to, so every
family lies inside a scaffold clique of size <= k+1.  Because a k-tree is already
triangulated, the learned BN's moral graph is a subgraph of the k-tree and its
treewidth is <= k **by construction** -- the guarantee MI restriction alone
could not give (Phase 1 reached treewidth 25-32).  The same code path serves all
five variants (dt / dg / dg_ndg), since bayes_nets' decision-graph learners all
restrict candidates through (permutation, interaction_matrix).

Exports:
  build_mi_ktree(mi, k)            -> (ktree_adj, order, candidate_parents)
  bounded_hboa_learn(data, card, method, k, **variant_kwargs) -> (adjacency, cpds, ktree)
  LearnBoundedHBOALight            -> a pateda LearningMethod (for end-to-end EDA use)
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def build_mi_ktree(mi, k):
    """MI-guided k-tree: introduce variables high-MI-degree first, attach each
    new node to the existing k-clique with which it shares the most MI.

    Returns (ktree adjacency, introduction order, {var: attach-clique parents}).
    Guarantee: a DAG whose parents come from candidate_parents has treewidth<=k.
    """
    from itertools import combinations
    mi = np.asarray(mi, dtype=float).copy()
    np.fill_diagonal(mi, 0.0)
    n = mi.shape[0]
    k = max(1, min(int(k), n - 1))
    order = list(np.argsort(mi.sum(axis=1))[::-1].astype(int))   # hubs first

    ktree = np.zeros((n, n), dtype=int)
    candidate_parents = {int(v): [] for v in range(n)}
    n_init = min(k + 1, n)
    init = [int(order[j]) for j in range(n_init)]
    for a in range(n_init):
        for b in range(a):
            ktree[init[a], init[b]] = ktree[init[b], init[a]] = 1
        candidate_parents[init[a]] = init[:a]

    k_cliques = ([tuple(sorted(c)) for c in combinations(init, k)]
                 if n_init >= k and k > 0 else [])

    for j in range(n_init, n):
        u = int(order[j])
        if k_cliques:
            # attach to the k-clique sharing the most MI with u
            best_c, best_w = k_cliques[0], -np.inf
            for c in k_cliques:
                w = float(mi[u, list(c)].sum())
                if w > best_w:
                    best_w, best_c = w, c
            parents = list(best_c)
        else:
            parents = init[:k]
        candidate_parents[u] = parents
        for x in parents:
            ktree[u, x] = ktree[x, u] = 1
        for x in parents:
            nc = tuple(sorted(set(parents) - {x} | {u}))
            if len(nc) == k:
                k_cliques.append(nc)

    return ktree, np.asarray(order, dtype=int), candidate_parents


def treewidth_via_order(adj, order):
    """Treewidth of the MORAL graph witnessed by eliminating in reverse
    introduction order.

    For a BN whose families lie inside k-tree cliques, eliminating the moral
    graph in the scaffold's reverse order gives max induced clique <= k+1, so
    this returns the *guaranteed* treewidth (unlike the min-fill heuristic,
    which can overestimate by a little on dense-but-bounded graphs)."""
    adj = np.asarray(adj)
    n = adj.shape[0]
    # Moral graph: skeleton + edges between co-parents of every node.
    nb = {v: set(np.where((adj[v] | adj[:, v]) != 0)[0]) for v in range(n)}
    for v in range(n):
        pa = list(np.where(adj[:, v] != 0)[0])
        for a in pa:
            nb[a].update(p for p in pa if p != a)
    pos = {int(v): i for i, v in enumerate(np.asarray(order, dtype=int))}
    maxc = 1
    for v in sorted(range(n), key=lambda x: -pos[x]):     # last introduced first
        earlier = [u for u in nb[v] if pos[u] < pos[v]]
        maxc = max(maxc, len(earlier) + 1)
        for a in earlier:                                  # fill remaining clique
            nb[a].update(x for x in earlier if x != a)
        for u in nb:
            nb[u].discard(v)
    return maxc - 1


def bounded_hboa_learn(data, card, method, k, *, alpha=1.0, max_parents=6,
                       sample_weights=None, **variant_kwargs):
    """Learn a treewidth-<=k HBOA-Light structure for one selected population.

    `method` in {"dt","dg","dg_ndg"}; `variant_kwargs` carries the per-variant
    knobs (local_structure, fast_local_scoring, max_leaves, split_score).
    """
    from bayes_nets.bayesian_network import BayesianNetwork
    from bayes_nets.polytree_learning import _pairwise_mutual_information
    data = np.asarray(data, dtype=int)
    card = np.asarray(card, dtype=int)
    n = data.shape[1]

    mi = np.asarray(_pairwise_mutual_information(data, card, sample_weights),
                    dtype=float)
    ktree, order, _ = build_mi_ktree(mi, k)

    bn = BayesianNetwork(n_vars=n, cardinality=card)
    fit_kwargs = dict(method=method, max_parents=min(int(max_parents), int(k)),
                      alpha=alpha, limit_table_size=False,
                      sample_weights=sample_weights,
                      interaction_matrix=ktree, permutation=order)
    fit_kwargs.update(variant_kwargs)
    bn.fit(data, **fit_kwargs)
    return bn.to_adjacency_matrix(), bn.cpds, ktree, order


# Per-variant decision-graph kwargs (mirrors the HBOA_Light_A* wrappers).
VARIANT_KW = {
    "A1_dt":   dict(method="dt",     local_structure="dt"),
    "A2_dg":   dict(method="dg",     local_structure="dg"),
    "A3_fast": dict(method="dg",     local_structure="dg", fast_local_scoring=True),
    "A4_mdl":  dict(method="dg",     local_structure="dg", max_leaves=32,
                    split_score="mdl"),
    "A5_ndg":  dict(method="dg_ndg", local_structure="dg"),
}

# The full report BN-EDA set -> bayes_nets method for the bounded scaffold.
# All are DAG learners that honour (permutation, interaction_matrix), so the
# k-tree scaffold gives every one a guaranteed treewidth <= k.  LFDA shares the
# BIC greedy search with EBNA_BIC (they differ only by warm-start, which does
# not apply to a single learn); PC only *removes* edges from the k-tree
# skeleton, so its result stays a subgraph of the scaffold.
BN_EDA_KW = {
    "EBNA_BIC":   dict(method="bic"),
    "EBNA_K2":    dict(method="k2_pen"),
    "EBNA_PC":    dict(method="stable_pc"),
    "LFDA":       dict(method="bic"),
    "BOA":        dict(method="k2"),
    "SARTRE":     dict(method="sartre"),
    "A1_dt":      dict(method="dt",     local_structure="dt"),
    "A2_dg":      dict(method="dg",     local_structure="dg"),
    "A3_fast":    dict(method="dg",     local_structure="dg", fast_local_scoring=True),
    "A4_mdl":     dict(method="dg",     local_structure="dg", max_leaves=32,
                       split_score="mdl"),
    "A5_ndg":     dict(method="dg_ndg", local_structure="dg"),
}

# Backwards-compatible alias: the learn function was always method-generic.
bounded_bn_learn = bounded_hboa_learn


try:
    from pateda.core.components import LearningMethod
    from pateda.core.models import BayesianNetworkModel

    class LearnBoundedBN(LearningMethod):
        """A pateda LearningMethod that learns a treewidth-<=k BN each
        generation via the MI k-tree scaffold, for any bayes_nets ``method``.

        Usable to build bounded EDAs for the end-to-end MN-FDA comparison:
        pair with :class:`SampleLocalStructureBN` (decision-graph methods) or
        :class:`SampleBayesianNetwork` (tabular methods).
        """

        def __init__(self, method="dg", k=8, max_parents=6, alpha=1.0,
                     local_structure=None, **variant_kwargs):
            self.method = method
            self.k = int(k)
            self.max_parents = int(max_parents)
            self.alpha = float(alpha)
            self.local_structure = local_structure
            self.variant_kwargs = variant_kwargs

        def learn(self, generation, n_vars, cardinality, population, fitness,
                  **params):
            from pateda.learning.utils.weights import normalize_probabilities
            data = np.asarray(population, dtype=int)
            card = np.asarray(cardinality, dtype=int)
            sw = normalize_probabilities(params.get("p"), data.shape[0])
            kw = dict(self.variant_kwargs)
            if self.local_structure is not None:
                kw["local_structure"] = self.local_structure
            adj, cpds, _, _ = bounded_bn_learn(
                data, card, method=self.method, k=self.k, alpha=self.alpha,
                max_parents=self.max_parents, sample_weights=sw, **kw)
            return BayesianNetworkModel(
                structure=adj, parameters=cpds,
                metadata={"generation": generation, "model_type": "BoundedBN",
                          "method": self.method, "treewidth_bound": self.k})
except Exception:   # pragma: no cover - pateda import optional for pure microbench
    LearnBoundedBN = None


# ---------------------------------------------------------------------------
# Most-probable configuration (MPC / MAP) for a bounded-treewidth BN
# ---------------------------------------------------------------------------
def _bn_family_factors(n_vars, model, cardinality):
    """Represent a BayesianNetworkModel as (cliques, tables): one factor per
    variable over its family {parents, var}, the dense CPD reshaped to a C-order
    joint potential.  The product of these factors is the BN joint, so
    max-product over them yields the exact MPC."""
    card = np.asarray(cardinality)
    cpds = model.parameters
    cliques, tables = [], []
    for v in range(n_vars):
        info = cpds[v]
        pa = [int(p) for p in info["parents"]]
        cpd = np.asarray(info["cpd"], dtype=float)
        if not pa:
            cliques.append([v])
            tables.append(cpd.ravel())                       # marginal p(v)
        else:
            m = len(pa)
            k = int(card[v])
            pcard = [int(card[p]) for p in pa]
            # CPD (config, v) with config mixed-radix first-parent-fastest ->
            # tensor over [pa0..pa{m-1}, v] in C-order (validated build).
            T = cpd.reshape((*reversed(pcard), k))
            T = np.transpose(T, list(range(m))[::-1] + [m])
            cliques.append(pa + [v])
            tables.append(T)
    return cliques, tables


def _bn_ancestral_argmax(n_vars, model, cardinality):
    """Greedy fallback: assign each variable (topological order) the argmax of
    its CPD given the already-assigned parents.  Used only if exact max-product
    ever reports intractable (never for a treewidth-bounded model)."""
    from pateda.sampling.bayesian_network import SampleBayesianNetwork
    card = np.asarray(cardinality)
    order = SampleBayesianNetwork(n_samples=1)._topological_sort(model.structure)
    cpds = model.parameters
    x = -np.ones(n_vars, dtype=int)
    for v in order:
        info = cpds[int(v)]
        pa = [int(p) for p in info["parents"]]
        cpd = np.asarray(info["cpd"], dtype=float)
        if not pa:
            x[v] = int(np.argmax(cpd.ravel()))
        else:
            pcard = [int(card[p]) for p in pa]
            cfg, mult = 0, 1
            for j, p in enumerate(pa):
                cfg += int(x[p]) * mult
                mult *= pcard[j]
            x[v] = int(np.argmax(cpd[cfg, :]))
    return x


def mpc_from_bn(n_vars, model, cardinality, order_method="min_degree"):
    """Exact most-probable configuration of a (bounded-treewidth) BN, by
    junction-tree max-product over the CPD family factors.  Falls back to the
    greedy ancestral argmax only if the model is reported intractable."""
    from pateda.inference.max_product_mpc import (
        max_product_mpc_cliques, MPCIntractable)
    cliques, tables = _bn_family_factors(n_vars, model, np.asarray(cardinality))
    for order in (order_method, "min_fill"):
        try:
            x, _ = max_product_mpc_cliques(cliques, tables,
                                           np.asarray(cardinality), order_method=order)
            return x
        except MPCIntractable:
            continue
    return _bn_ancestral_argmax(n_vars, model, np.asarray(cardinality))


try:
    from pateda.core.components import SamplingMethod

    class SampleBNWithMPC(SamplingMethod):
        """Bounded-treewidth BN sampling with the most-probable configuration
        inserted -- the BN analogue of :class:`SampleFDAWithMPC` (MN-FDA-P).

        The first individual is the exact MPC of the learned BN (junction-tree
        max-product; tractable because the treewidth is bounded by construction),
        the remaining ``n_samples - 1`` are drawn by the wrapped base sampler
        (``SampleLocalStructureBN`` for decision-graph EDAs,
        ``SampleBayesianNetwork`` for the tabular ones)."""

        def __init__(self, base_sampler, n_samples, mpc_order="min_degree"):
            self.base = base_sampler
            self.n_samples = n_samples
            self.mpc_order = mpc_order

        def sample(self, n_vars, model, cardinality, aux_pop=None,
                   aux_fitness=None, rng=None, **params):
            if rng is None:
                rng = np.random.default_rng()
            n = params.get("n_samples", self.n_samples)
            mpc = mpc_from_bn(n_vars, model, cardinality, self.mpc_order)
            if n <= 1:
                return mpc.reshape(1, n_vars)
            rest = self.base.sample(n_vars, model, cardinality, aux_pop=aux_pop,
                                    aux_fitness=aux_fitness, rng=rng,
                                    n_samples=n - 1)
            return np.vstack([mpc.reshape(1, n_vars), rest.astype(int)])
except Exception:   # pragma: no cover
    SampleBNWithMPC = None


if __name__ == "__main__":
    # Self-test: prove the treewidth guarantee on a captured LABS population.
    import time, ioh
    import compare_bn_variants_pbo as C
    from bayes_nets.factorization import moralize, triangulate
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8]
    prob = ioh.get_problem(18, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda("EBNA_K2", "BZ", prob, 200, 3, 0.5, 7)
    cap = {}
    ol = eda.components.learning.learn

    def lw(g, nv, cd, pop, fit, **kw):
        m = ol(g, nv, cd, pop, fit, **kw)
        cap.update(pop=pop.copy(), card=np.asarray(cd)); return m
    eda.components.learning.learn = lw
    eda.run(verbose=False)
    data, card = cap["pop"], cap["card"]

    def tw_of(adj):
        _, _, cl = triangulate(moralize(adj), card, method="min-fill",
                               max_clique_width=None)
        return max((len(c) for c in cl), default=1)

    print(f"self-test on LABS f18 n={dim}, {data.shape[0]} rows")
    for lbl, kw in VARIANT_KW.items():
        for k in ks:
            t0 = time.perf_counter()
            adj, _, ktree, order = bounded_hboa_learn(data, card, k=k, alpha=1.0,
                                                      max_parents=6, **kw)
            dt = time.perf_counter() - t0
            tw = treewidth_via_order(adj, order)         # guaranteed witness
            twf = tw_of(adj)                             # min-fill heuristic
            ok = "OK" if tw <= k else "VIOLATION"
            print(f"  {lbl:8s} k={k:<2} learn={dt:6.2f}s  treewidth={tw:2d} "
                  f"(<= {k}? {ok})  [min-fill={twf}]  edges={int(adj.sum())}  "
                  f"fam={int(adj.sum(0).max())+1}")
