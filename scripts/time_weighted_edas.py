"""
Timing feasibility probe for the weighted-probability PBO EDAs.

Before launching the full grid (``compare_weighted_edas_pbo.py`` /
``launch_weighted_pbo_experiments.py``), this script measures the wall-clock
cost of every algorithm so the model-building EDAs (EBNA, BOA, LFDA, HBOA,
SARTRE, BINOTEARS, PCBN, ...) can be checked for feasibility at ``n >= 100``.

For each algorithm it runs ONE seed of ``PROBE_GEN`` generations on one
representative structured PBO function (NK landscapes, f25) at several
dimensions, then extrapolates the cost of a full ``N_GEN``-generation run
(learning cost per generation is roughly constant, so
``full ~= t_probe / (PROBE_GEN + 1) * (N_GEN + 1)``).

Timing is done under the heaviest selection method (``RTS``: Boltzmann
weighting + restricted-tournament replacement); FP and BZ are never slower.

Usage:
    python scripts/time_weighted_edas.py [dims] [gens] [fid] [algs]

    dims  comma-separated dims       (default: 64,100,196)
    gens  probe generations          (default: 5)
    fid   PBO function id            (default: 25, NK landscapes)
    algs  comma-separated algs/all   (default: all)
"""

import os
import sys
import time
import csv

import numpy as np
import ioh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compare_weighted_edas_pbo import (   # noqa: E402
    ALGORITHM_NAMES, ALGORITHM_BUILDERS, apply_selection_method,
    make_fitness, INSTANCE, POP_SIZE, N_GEN, SEL_RATIO,
)

PROBE_SEL = "RTS"       # heaviest selection method
OUT_CSV = os.path.join(
    os.path.dirname(SCRIPT_DIR), "results", "pbo_weighted_timing.csv")


def time_one(alg, dim, fid, gens, seed=1):
    problem = ioh.get_problem(fid, instance=INSTANCE, dimension=dim,
                              problem_class=ioh.ProblemClass.PBO)
    builder = ALGORITHM_BUILDERS[alg]
    eda = builder(problem.meta_data.n_variables, 2, make_fitness(problem),
                  POP_SIZE, gens, SEL_RATIO, seed)
    apply_selection_method(eda, PROBE_SEL, POP_SIZE)
    t0 = time.time()
    eda.run(verbose=False)
    elapsed = time.time() - t0
    evals = int(problem.state.evaluations)
    return elapsed, evals


def main():
    dims = [64, 100, 196]
    gens = 5
    fid = 25
    algs = ALGORITHM_NAMES

    if len(sys.argv) > 1:
        dims = [int(d) for d in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        gens = int(sys.argv[2])
    if len(sys.argv) > 3:
        fid = int(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4].lower() != "all":
        algs = [a for a in sys.argv[4].split(",") if a in ALGORITHM_BUILDERS]

    scale = (N_GEN + 1) / (gens + 1)     # extrapolate probe -> full run

    print(f"Timing probe (selection method = {PROBE_SEL}, heaviest)")
    print(f"  probe gens={gens}  pop={POP_SIZE}  fid=f{fid}  dims={dims}")
    print(f"  full run = {N_GEN} gens  ->  extrapolation factor x{scale:.1f}\n")
    header = f"{'algorithm':<12}" + "".join(f"  n={d:<4} probe/full(s)" for d in dims)
    print(header)
    print("-" * len(header))

    rows = []
    for alg in algs:
        cells = []
        parts = [f"{alg:<12}"]
        for dim in dims:
            try:
                elapsed, evals = time_one(alg, dim, fid, gens)
                full = elapsed * scale
                parts.append(f"  {elapsed:6.1f}/{full:7.1f}")
                cells.append((dim, elapsed, full, evals))
            except Exception as exc:
                parts.append(f"  ERR:{str(exc)[:10]:<11}")
                cells.append((dim, float('nan'), float('nan'), -1))
        line = "".join(parts)
        print(line, flush=True)
        for dim, elapsed, full, evals in cells:
            rows.append({"algorithm": alg, "selection": PROBE_SEL, "dim": dim,
                         "probe_gens": gens, "probe_seconds": round(elapsed, 3),
                         "full_gens": N_GEN, "est_full_seconds": round(full, 2),
                         "probe_evals": evals})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Feasibility summary at the largest experiment dimension actually used.
    print("\nFeasibility flags (est. full single-run time):")
    exp_dim = max([d for d in dims if d <= 100], default=max(dims))
    worst = sorted([r for r in rows if r["dim"] == exp_dim],
                   key=lambda r: (r["est_full_seconds"] or -1), reverse=True)
    for r in worst:
        t = r["est_full_seconds"]
        flag = ("OK" if t == t and t < 60 else
                "moderate" if t == t and t < 300 else
                "SLOW" if t == t else "ERROR")
        print(f"  n={exp_dim:<4} {r['algorithm']:<12} "
              f"{t:>8.1f}s  [{flag}]")
    print(f"\nCSV written to {OUT_CSV}")


if __name__ == "__main__":
    main()
