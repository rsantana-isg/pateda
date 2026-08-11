"""Phase 2 benchmark: bounded-treewidth HBOA-Light vs. the Phase 0/1 baselines.

For each variant (A1-A5) and treewidth bound k, learn a scaffold-restricted
structure on the SAME captured LABS f18 population Phase 1 used (seed 7, so the
rows match hboa_mi_restriction.csv), and record: learning time, the *guaranteed*
treewidth (witnessed by the scaffold elimination order), the min-fill treewidth
(for reference), family clique, edge count, and dense-tabular BIC.

The point: unlike Phase 1 (treewidth 15-52, unbounded), every row here must show
treewidth <= k, at a learning cost that is small because each variable searches
over <= k candidate parents.

Usage: python3 scripts/bench_hboa_phase2.py [dims] [ks] [timeout_s]
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
    from hboa_bounded_tw import bounded_hboa_learn, treewidth_via_order
    from bench_bn_learning_constraints import _treewidth, _bic

    def _h(s, f):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(timeout_s))
    try:
        t0 = time.perf_counter()
        adj, _, ktree, order = bounded_hboa_learn(
            data, card, k=k, alpha=1.0, max_parents=6, **kw)
        t_learn = time.perf_counter() - t0
        signal.alarm(0)
    except TimeoutError:
        return {"variant": label, "k": k, "t_learn_s": float("inf"),
                "tw_guar": -1, "tw_minfill": -1, "family_clique": -1,
                "edges": -1, "bic": float("nan"), "status": f"timeout>{int(timeout_s)}s"}
    tw_g = treewidth_via_order(adj, order)
    tw_mf, _ = _treewidth(adj, card)
    return {"variant": label, "k": k, "t_learn_s": t_learn,
            "tw_guar": tw_g, "tw_minfill": tw_mf,
            "family_clique": int(adj.sum(0).max()) + 1, "edges": int(adj.sum()),
            "bic": _bic(adj, data, card),
            "status": "ok" if tw_g <= k else "VIOLATION"}


def main():
    import numpy as np, pandas as pd
    from hboa_bounded_tw import VARIANT_KW
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 100, 256]
    ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 12]
    timeout_s = float(sys.argv[3]) if len(sys.argv) > 3 else 300
    os.makedirs(OUT, exist_ok=True)

    all_rows = []
    for dim in dims:
        print(f"\n=== n={dim}: capture + bounded learn A1-A5 x k={ks} ===", flush=True)
        data, card = _capture(dim)
        tasks = [(lbl, kw, k, data, card, timeout_s)
                 for lbl, kw in VARIANT_KW.items() for k in ks]
        with ProcessPoolExecutor(max_workers=min(14, len(tasks)),
                                 max_tasks_per_child=1) as ex:
            for r in ex.map(_run, tasks):
                r["dim"] = dim
                all_rows.append(r)
                tl = "  timeout" if r["t_learn_s"] == float("inf") else f"{r['t_learn_s']:6.2f}s"
                print(f"  {r['variant']:8s} k={r['k']:<2} learn={tl}  "
                      f"tw_guar={r['tw_guar']:2d} (<=k? {r['status']})  "
                      f"[min-fill={r['tw_minfill']}]  fam={r['family_clique']:2d}  "
                      f"edges={r['edges']:4d}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "hboa_phase2.csv"), index=False)
    viol = df[df["status"] == "VIOLATION"]
    print(f"\nrows: {len(df)}   treewidth-bound violations: {len(viol)}")
    print(f"CSV: {os.path.join(OUT, 'hboa_phase2.csv')}")


if __name__ == "__main__":
    main()
