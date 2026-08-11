"""
Profile the component time distribution of the 11 BN-variant EDAs (the
pbo_bn_variants study), by wrapping each EDA component with a timer.

Components timed: structure+parameter learning, sampling, fitness evaluation,
selection, replacement, and 'other' (seeding, statistics, weighting, loop
overhead).  Runs the algorithms in parallel; reports per-component % of total.

Usage: python3 scripts/profile_bn_variants.py [dims] [fid] [ngen]
"""

import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_profile"))
POP = 200


def _profile(args):
    alg, dim, fid, ngen = args
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    prob = ioh.get_problem(fid, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda(alg, "BZ", prob, POP, ngen, 0.5, 1)
    T = defaultdict(float)

    def wrap(obj, name, key):
        orig = getattr(obj, name)
        def wrapped(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            T[key] += time.perf_counter() - t0
            return r
        setattr(obj, name, wrapped)

    wrap(eda.components.learning, "learn", "learn")
    wrap(eda.components.sampling, "sample", "sample")
    wrap(eda, "evaluate_fitness", "evaluate")
    wrap(eda.components.selection, "select", "select")
    if eda.components.replacement is not None:
        wrap(eda.components.replacement, "replace", "replace")

    t0 = time.perf_counter()
    eda.run(verbose=False)
    total = time.perf_counter() - t0
    T["total"] = total
    T["other"] = total - sum(v for k, v in T.items() if k != "total")
    T["sampler"] = type(eda.components.sampling).__name__
    return alg, dim, dict(T)


def main():
    import pandas as pd
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 100]
    fid = int(sys.argv[2]) if len(sys.argv) > 2 else 18   # LABS: structured model
    ngen = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(OUT, exist_ok=True)
    import compare_bn_variants_pbo as C
    tasks = [(a, d, fid, ngen) for a in C.ALGORITHM_NAMES for d in dims]
    print(f"profiling {len(tasks)} (alg,dim): f{fid}, pop={POP}, ngen={ngen}, BZ")

    rows = []
    with ProcessPoolExecutor(max_workers=14, max_tasks_per_child=1) as ex:
        for alg, dim, T in ex.map(_profile, tasks):
            rows.append({"alg": alg, "dim": dim, **T})
            print(f"  {alg:11s} n={dim:<4} done ({T['total']:.1f}s, "
                  f"{T['sampler']})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "bn_profile.csv"), index=False)
    comps = ["learn", "sample", "evaluate", "select", "replace", "other"]
    for dim in dims:
        print(f"\n=== n={dim}: component time distribution (% of total; "
              f"per-generation ms) ===")
        print(f"{'alg':11s} {'total/gen':>9} | " +
              " ".join(f"{c:>8}" for c in comps) + "  sampler")
        for _, r in df[df.dim == dim].iterrows():
            tot = r["total"]
            pct = " ".join(f"{100*r.get(c,0)/tot:7.1f}%" for c in comps)
            print(f"{r['alg']:11s} {1000*tot/max(1,r.get('ngen',8)):8.1f}m | {pct}  "
                  f"{r['sampler']}")
    # aggregate sample vs learn share by sampler family
    print("\n=== sampling share (median %% of total) by sampler ===")
    df["sample_pct"] = 100 * df["sample"] / df["total"]
    df["learn_pct"] = 100 * df["learn"] / df["total"]
    for smp, sub in df.groupby("sampler"):
        print(f"  {smp:24s}: learn={sub.learn_pct.median():.1f}%  "
              f"sample={sub.sample_pct.median():.1f}%  (n={len(sub)})")
    print(f"\nCSV: {os.path.join(OUT,'bn_profile.csv')}")


if __name__ == "__main__":
    main()
