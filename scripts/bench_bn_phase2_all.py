"""Phase 2 for ALL report BN-EDAs: MI k-tree scaffold -> guaranteed treewidth<=k.

Generalises bench_hboa_phase2.py from the HBOA-Light family to every Bayesian-
network EDA in the pbo_bn_variants report:

  EBNA_BIC (bic), EBNA_K2 (k2_pen), EBNA_PC (stable_pc), LFDA (bic),
  BOA (k2), SARTRE (sartre), A1_dt (dt), A2_dg/A3_fast/A4_mdl (dg), A5_ndg (dg_ndg).

All are DAG learners that honour (permutation, interaction_matrix); restricting
each variable's parents to its k-tree attach-clique makes every family lie inside
a scaffold clique of size <= k+1, so the moral graph is a subgraph of the k-tree
and the treewidth is <= k by construction (witnessed by the scaffold order).

For each (EDA, k) on the same captured LABS f18 population (seed 7, matching the
Phase 0/1 rows), records learning time, guaranteed treewidth, min-fill treewidth,
family clique, edges and dense BIC.  Verifies zero treewidth-bound violations.

Usage: python3 scripts/bench_bn_phase2_all.py [dims] [ks] [timeout_s]
"""
import os
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_learning"))
POP = 200


def _capture(dim, seed=7):
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    prob = ioh.get_problem(18, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda("EBNA_K2", "BZ", prob, POP, 3, 0.5, seed)
    cap = {}
    ol = eda.components.learning.learn

    def lw(g, nv, card, pop, fit, **kw):
        m = ol(g, nv, card, pop, fit, **kw)
        cap["pop"] = pop.copy(); cap["card"] = np.asarray(card); return m
    eda.components.learning.learn = lw
    eda.run(verbose=False)
    return cap["pop"], cap["card"]


def _run(args):
    label, kw, k, data, card, timeout_s = args
    import numpy as np
    from hboa_bounded_tw import bounded_bn_learn, treewidth_via_order
    from bench_bn_learning_constraints import _treewidth, _bic

    def _h(s, f):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(timeout_s))
    try:
        t0 = time.perf_counter()
        adj, _, ktree, order = bounded_bn_learn(
            data, card, k=k, alpha=1.0, max_parents=6, **kw)
        t_learn = time.perf_counter() - t0
        signal.alarm(0)
    except TimeoutError:
        return {"eda": label, "k": k, "t_learn_s": float("inf"), "tw_guar": -1,
                "tw_minfill": -1, "family_clique": -1, "edges": -1,
                "bic": float("nan"), "status": f"timeout>{int(timeout_s)}s"}
    tw_g = treewidth_via_order(adj, order)
    tw_mf, _ = _treewidth(adj, card)
    return {"eda": label, "k": k, "t_learn_s": t_learn, "tw_guar": tw_g,
            "tw_minfill": tw_mf, "family_clique": int(adj.sum(0).max()) + 1,
            "edges": int(adj.sum()), "bic": _bic(adj, data, card),
            "status": "ok" if tw_g <= k else "VIOLATION"}


def main():
    import numpy as np, pandas as pd
    from hboa_bounded_tw import BN_EDA_KW
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 100, 256]
    ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 12]
    timeout_s = float(sys.argv[3]) if len(sys.argv) > 3 else 300
    os.makedirs(OUT, exist_ok=True)

    all_rows = []
    for dim in dims:
        print(f"\n=== n={dim}: capture + bounded learn (all 11 BN-EDAs) x k={ks} ===",
              flush=True)
        data, card = _capture(dim)
        tasks = [(lbl, kw, k, data, card, timeout_s)
                 for lbl, kw in BN_EDA_KW.items() for k in ks]
        with ProcessPoolExecutor(max_workers=14, max_tasks_per_child=1) as ex:
            for r in ex.map(_run, tasks):
                r["dim"] = dim
                all_rows.append(r)
                tl = "  timeout" if r["t_learn_s"] == float("inf") else f"{r['t_learn_s']:6.2f}s"
                print(f"  {r['eda']:9s} k={r['k']:<2} learn={tl}  "
                      f"tw_guar={r['tw_guar']:2d} (<=k? {r['status']})  "
                      f"[min-fill={r['tw_minfill']:2d}]  fam={r['family_clique']:2d}  "
                      f"edges={r['edges']:4d}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "bn_phase2_all.csv"), index=False)
    viol = df[df["status"] == "VIOLATION"]
    print(f"\nrows: {len(df)}   treewidth-bound violations: {len(viol)}")
    if len(viol):
        print(viol[["eda", "dim", "k", "tw_guar"]].to_string(index=False))
    print(f"CSV: {os.path.join(OUT, 'bn_phase2_all.csv')}")


if __name__ == "__main__":
    main()
