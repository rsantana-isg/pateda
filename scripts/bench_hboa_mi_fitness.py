"""Phase 0/1, Part B: end-to-end EDA fitness -- does MI candidate restriction
preserve optimization quality for HBOA-Light A1-A5 while cutting runtime?

For each variant we run the full EDA (BZ selection, niching, as in the report)
twice per (function, seed): baseline (candidate_parents forced to None) vs.
MI-restricted (candidate_parents="mi:K"), and record the final best fitness and
the total wall time.  The baseline vs. MI fitness gap is the gate for Phase 2:
if MI restriction is fitness-neutral (within seed noise) at a large speed-up, the
same lever is safe to build the bounded-treewidth scaffold on.

Usage: python3 scripts/bench_hboa_mi_fitness.py [dim] [ngen] [fids] [seeds] [mi_k]
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_learning"))
POP = 200
VARIANTS = ["A1_dt", "A2_mi", "A3_fast", "A4_mdl", "A5_ndg"]


def _run(args):
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    alg, mode, fid, dim, ngen, seed = args
    prob = ioh.get_problem(fid, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda(alg, "BZ", prob, POP, ngen, 0.5, seed)
    # Force a uniform baseline (None) or MI restriction, overriding each
    # variant's default (A2 ships with mi:10) so the contrast is clean.
    eda.components.learning.candidate_parents = (None if mode == "base" else mode)
    t0 = time.time()
    eda.run(verbose=False)
    dt = time.time() - t0
    return {"variant": alg, "mode": mode, "fid": fid, "dim": dim, "seed": seed,
            "best": float(prob.state.current_best.y), "time_s": dt}


def main():
    import numpy as np, pandas as pd
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    ngen = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    fids = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 18]
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2]
    mi_k = sys.argv[5] if len(sys.argv) > 5 else "mi:12"
    os.makedirs(OUT, exist_ok=True)

    tasks = [(alg, mode, fid, dim, ngen, seed)
             for alg in VARIANTS for mode in ("base", mi_k)
             for fid in fids for seed in seeds]
    print(f"end-to-end fitness: {len(tasks)} runs  (n={dim}, ngen={ngen}, "
          f"pop={POP}, fids={fids}, seeds={seeds}, mi={mi_k})", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=14, max_tasks_per_child=1) as ex:
        for r in ex.map(_run, tasks):
            rows.append(r)
            print(f"  {r['variant']:8s} {r['mode']:6s} f{r['fid']:<2} "
                  f"s{r['seed']}  best={r['best']:.4f}  time={r['time_s']:7.1f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"hboa_mi_fitness_dim{dim}.csv"), index=False)

    print("\n=== baseline vs MI: mean best (higher=better) and mean time ===")
    for alg in VARIANTS:
        b = df[(df["variant"] == alg) & (df["mode"] == "base")]
        m = df[(df["variant"] == alg) & (df["mode"] == mi_k)]
        if b.empty or m.empty:
            continue
        spd = b["time_s"].mean() / max(m["time_s"].mean(), 1e-9)
        print(f"  {alg:8s}: best base={b['best'].mean():.4f}  "
              f"MI={m['best'].mean():.4f}  (Δ={m['best'].mean()-b['best'].mean():+.4f}) | "
              f"time base={b['time_s'].mean():6.1f}s  MI={m['time_s'].mean():6.1f}s  "
              f"(x{spd:.1f} faster)")
    print(f"\nCSV: {os.path.join(OUT, f'hboa_mi_fitness_dim{dim}.csv')}")


if __name__ == "__main__":
    main()
