"""
Generate the sbatch grid for the BN-EDA comparison study: one cluster job per
(seed, problem, algorithm), each handled by ``scripts/run_bn_eda.py`` via
``slurm/slurm_bn_eda.sh``.

Grid & feasibility filter
-------------------------
    5 seeds  x  33 problems  x  up to 19 BN learning algorithms.

All EDAs share every component except the BN learning algorithm (truncation
T=0.5, Boltzmann-weighted learning, 100 generations, pop_size = 10 * n).

Rather than hand-pick a fixed subset of algorithms, this launcher includes
**every** algorithm and filters each ``(algorithm, problem)`` pair by a
data-driven feasibility estimate taken from the offline BN benchmark
(``eda_cluster_results.csv``).  For a given pair we extrapolate the measured
one-shot learning time (at the benchmark's 800 training samples) to the EDA's
selected-set size (5 * n) over ``N_GEN`` generations:

    est_hours = N_GEN * max_time(alg, problem) * (5 * n / 800) / 3600

using the **worst** measured time over temperatures/splits (an EDA repeats the
learning every generation, so a single slow learning stalls the whole run).  A
pair is launched only when ``est_hours <= MAX_EST_HOURS``.  Pairs with no
benchmark time (the handful of largest offline jobs that never finished) are
treated as infeasible and skipped.

This automatically keeps every fast method on all sizes and every expensive /
borderline method (aic, k2_mb, pc, stable_pc, k2_plus, k2_refine, dmbbn,
stable_hc, dt, ...) on exactly the problem sizes where it is feasible -- e.g.
``dt`` on the small MaxClique instances it wins offline, but not on n=256.

Usage
-----
    python3 slurm/launch_bn_eda.py                    # print all sbatch lines
    python3 slurm/launch_bn_eda.py --report           # per-algorithm job counts
    python3 slurm/launch_bn_eda.py | head -400 | bash # <= 400 jobs in flight
    python3 slurm/launch_bn_eda.py | grep Ising | bash

Tune the grid with the constants below or the environment variables
``BN_EDA_MAX_EST_HOURS`` and ``BN_EDA_SEEDS``.  Each job is idempotent, so the
grid is safe to re-launch.
"""

import csv
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bn_edas_analysis/
_CSV = os.path.join(_ROOT, "eda_cluster_results.csv")
SBATCH_SCRIPT = "slurm/slurm_bn_eda.sh"

# ---- knobs -----------------------------------------------------------------
SEEDS = [int(s) for s in os.environ.get("BN_EDA_SEEDS", "1,2,3,4,5").split(",")]
N_GEN = 100
POP_MULT = 10                       # pop_size = POP_MULT * n; selected = 5 * n
BENCH_SAMPLES = 800                 # training-set size used in the offline benchmark
# feasibility threshold in hours (leave margin under the 48h SLURM wall-time)
MAX_EST_HOURS = float(os.environ.get("BN_EDA_MAX_EST_HOURS", "40"))

# The full pool of algorithms (all methods evaluated in the offline benchmark).
ALGORITHMS = [
    "univ_bn",
    "k2", "k2_mi", "k2_mb", "k2_refine", "k2_ensemble", "k2_plus",
    "fi_k2", "rfe_k2",
    "bic", "aic", "stable_hc",
    "pc", "stable_pc",
    "dt", "dmbbn", "sartre", "binotears", "bounded_tw",
]

# Some learners only support binary variables and error out on non-binary
# problems.  Braid is the only non-binary family here (cardinality 4, the four
# Fibonacci-anyon generators), and `binotears` (BinaryNotearsLearner) is the only
# binary-only method -- verified by running every algorithm on Braid.  Such
# (algorithm, problem) pairs are excluded from the grid.
BINARY_ONLY_ALGORITHMS = {"binotears"}
NON_BINARY_FAMILIES = {"Braid"}


def _incompatible(alg, problem):
    family = problem.rsplit("_", 1)[0]
    return alg in BINARY_ONLY_ALGORITHMS and family in NON_BINARY_FAMILIES


def _float(x):
    try:
        v = float(x)
        return v if v == v else None      # drop NaN
    except (TypeError, ValueError):
        return None


def load_benchmark():
    """max measured learning time and n per (algorithm, problem)."""
    max_time = defaultdict(float)
    n_of = {}
    seen = set()
    with open(_CSV) as fh:
        for row in csv.DictReader(fh):
            t = _float(row["time"])
            key = (row["algorithm"], row["problem"])
            n_of[row["problem"]] = int(row["n"])
            seen.add(key)
            if t is not None:
                max_time[key] = max(max_time[key], t)
    return max_time, n_of, seen


def est_hours(alg, problem, max_time, n_of, seen):
    """Estimated full-run wall time in hours; None if not estimable (skip)."""
    key = (alg, problem)
    if key not in seen or key not in max_time:
        return None
    n = n_of[problem]
    scale = (POP_MULT // 2 * n) / BENCH_SAMPLES      # selected = 5 * n
    return N_GEN * max_time[key] * scale / 3600.0


def feasible_pairs():
    max_time, n_of, seen = load_benchmark()
    problems = sorted(n_of)
    pairs, skipped = [], []
    for problem in problems:
        for alg in ALGORITHMS:
            if _incompatible(alg, problem):
                skipped.append((problem, alg, "binary-only"))
                continue
            h = est_hours(alg, problem, max_time, n_of, seen)
            if h is not None and h <= MAX_EST_HOURS:
                pairs.append((problem, alg, h))
            else:
                skipped.append((problem, alg, h))
    return pairs, skipped, problems


def main():
    pairs, skipped, problems = feasible_pairs()
    report = "--report" in sys.argv

    per_alg = defaultdict(int)
    for problem, alg, _h in pairs:
        per_alg[alg] += 1

    n_jobs = len(pairs) * len(SEEDS)
    print(f"# {len(problems)} problems x {len(ALGORITHMS)} algorithms: "
          f"{len(pairs)} feasible (alg,problem) pairs, {len(skipped)} skipped "
          f"(est > {MAX_EST_HOURS}h) -> {n_jobs} jobs "
          f"({len(SEEDS)} seeds)", file=sys.stderr)
    print(f"# feasible pairs per algorithm (of {len(problems)} problems):",
          file=sys.stderr)
    for alg in ALGORITHMS:
        print(f"#   {alg:12s} {per_alg[alg]:3d}", file=sys.stderr)

    if report:
        print("\nskipped (algorithm, problem, est_hours):", file=sys.stderr)
        for problem, alg, h in skipped:
            if isinstance(h, str):
                hs = h
            elif h is None:
                hs = "no-benchmark"
            else:
                hs = f"{h:.1f}h"
            print(f"#   {alg:12s} {problem:18s} {hs}", file=sys.stderr)
        return

    try:
        for seed in SEEDS:
            for problem, alg, _h in pairs:
                print(f"sbatch {SBATCH_SCRIPT} {seed} {problem} {alg}")
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
