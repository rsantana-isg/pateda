"""Does an MI-based candidate-parent restriction cut learning time AND bound the
maximum clique of the learned BN?  For each score-based BN learner we capture a
realistic selected population (LABS f18, n=100) and re-learn the structure

  * unconstrained (interaction_matrix = None, the current pateda behaviour), and
  * restricted to each variable's top-K mutual-information neighbours,

reporting learning time, the scenario-1 treewidth (moralize+triangulate max
clique), the scenario-2 family clique (max in-degree + 1), edge count and the
BIC score of the learned structure on the selected data.

Usage: python3 scripts/bench_bn_learning_constraints.py [dim] [topk_list]
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_learning"))
POP = 200
# (label, bayes_nets method) for the report's score-based learners.
METHODS = [("EBNA_BIC", "bic"), ("EBNA_K2", "k2_pen"),
           ("BOA", "k2"), ("SARTRE", "sartre"), ("EBNA_PC", "stable_pc")]


def _capture_population(dim, seed=7):
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    prob = ioh.get_problem(18, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda("EBNA_K2", "BZ", prob, POP, 4, 0.5, seed)
    cap = {}
    ol = eda.components.learning.learn

    def lw(g, nv, card, pop, fit, **kw):
        m = ol(g, nv, card, pop, fit, **kw)
        cap["pop"] = pop.copy(); cap["card"] = np.asarray(card); return m
    eda.components.learning.learn = lw
    eda.run(verbose=False)
    return cap["pop"], cap["card"]


def _mi_interaction_matrix(data, card, topk):
    """Symmetric 0/1 matrix keeping each variable's top-K MI neighbours."""
    import numpy as np
    from bayes_nets.polytree_learning import _pairwise_mutual_information
    mi = np.asarray(_pairwise_mutual_information(data, card, None), dtype=float)
    n = mi.shape[0]
    np.fill_diagonal(mi, -np.inf)
    im = np.zeros((n, n), dtype=int)
    for v in range(n):
        nbrs = np.argsort(mi[v])[::-1][:topk]
        im[v, nbrs] = 1
    im = ((im + im.T) > 0).astype(int)   # symmetrise
    np.fill_diagonal(im, 0)
    return im, mi


def _treewidth(adj, card):
    import numpy as np
    from bayes_nets.factorization import moralize, triangulate
    moral = moralize(adj)
    _, _, cliques = triangulate(moral, np.asarray(card), method="min-fill",
                               max_clique_width=None)
    return max((len(c) for c in cliques), default=1), len(cliques)


def _bic(adj, data, card):
    import numpy as np
    from scipy.special import gammaln
    n = adj.shape[0]; N = data.shape[0]
    total = 0.0
    for v in range(n):
        pa = list(np.where(adj[:, v] != 0)[0])
        k = int(card[v])
        if pa:
            pcard = [int(card[p]) for p in pa]
            mult = np.cumprod([1] + pcard[:-1])
            cfg = (data[:, pa] * mult).sum(axis=1).astype(int)
            ncfg = int(np.prod(pcard))
        else:
            cfg = np.zeros(N, dtype=int); ncfg = 1
        ll = 0.0
        for c in range(ncfg):
            idx = cfg == c
            if not idx.any():
                continue
            counts = np.bincount(data[idx, v], minlength=k).astype(float)
            tot = counts.sum()
            nz = counts > 0
            ll += float((counts[nz] * np.log(counts[nz] / tot)).sum())
        params = ncfg * (k - 1)
        total += ll - 0.5 * params * np.log(N)
    return total


def _run(args):
    import numpy as np
    label, method, data, card, topk = args
    from bayes_nets.bayesian_network import BayesianNetwork
    im = None
    if topk is not None:
        im, _ = _mi_interaction_matrix(data, card, topk)
    t0 = time.perf_counter()
    bn = BayesianNetwork(n_vars=data.shape[1], cardinality=card)
    bn.fit(data, method=method, max_parents=6, alpha=1.0,
           interaction_matrix=im, sample_weights=None)
    t_learn = time.perf_counter() - t0
    adj = bn.to_adjacency_matrix()
    tw, ncl = _treewidth(adj, card)
    indeg = int(adj.sum(0).max())
    return {"method": label, "topk": (topk if topk is not None else "none"),
            "t_learn_s": t_learn, "treewidth": tw, "family_clique": indeg + 1,
            "edges": int(adj.sum()), "bic": _bic(adj, data, card)}


def main():
    import numpy as np, pandas as pd
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    topks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [None, 20, 12, 8]
    os.makedirs(OUT, exist_ok=True)
    print(f"capturing LABS f18 n={dim} selected population (pop={POP}) ...", flush=True)
    data, card = _capture_population(dim)
    print(f"  captured {data.shape}", flush=True)
    tasks = [(lbl, mth, data, card, tk) for lbl, mth in METHODS
             for tk in ([None] + [t for t in topks if t is not None])]
    rows = []
    with ProcessPoolExecutor(max_workers=min(14, len(tasks)),
                             max_tasks_per_child=1) as ex:
        for r in ex.map(_run, tasks):
            rows.append(r)
            print(f"  {r['method']:9s} topk={str(r['topk']):>4}  "
                  f"learn={r['t_learn_s']:7.2f}s  tw(sc1)={r['treewidth']:2d}  "
                  f"family_clique(sc2)={r['family_clique']}  edges={r['edges']:3d}  "
                  f"BIC={r['bic']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"bn_learning_dim{dim}.csv"), index=False)
    print(f"\nCSV: {os.path.join(OUT, f'bn_learning_dim{dim}.csv')}")


if __name__ == "__main__":
    main()
