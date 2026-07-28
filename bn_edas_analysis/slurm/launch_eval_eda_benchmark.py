"""
Generate sbatch commands for the extended EDA benchmark: one cluster job per
(problem, train_set, algorithm, seed, temperature) combination, each handled by
``scripts/gen_eval_eda_benchmark.py``.

Grid
----
    10 seeds  x  3 temperatures {0.1, 1.0, 10}  x  3 train sets {0,1,2}
    x  19 BN learning algorithms  x  all problems in data/eda_datasets/

The 19 algorithms are every method implemented in ``bayes_nets`` (including the
Univ_BN baseline and the objective-guided FI_k2 / RFE_k2 orderings) **except**
the six excluded by the experiment design — rcd, gs, tabu, dg, iterdsla, rpcd —
and the exact ``levelwise``/``exact`` DP (hard-capped at n<=20, infeasible for
every dataset here).

Usage
-----
    # print the full grid of sbatch lines
    python3 slurm/launch_eval_eda_benchmark.py

    # keep <= 400 jobs in flight at a time (project convention)
    python3 slurm/launch_eval_eda_benchmark.py | head -400 | bash
    python3 slurm/launch_eval_eda_benchmark.py | sed -n '401,800p' | bash

    # or restrict the grid on the fly
    python3 slurm/launch_eval_eda_benchmark.py | grep Braid_36 | bash

Each job is idempotent (the runner skips a combination whose result file
already exists), so the grid is safe to re-launch.  Results are collected in
``results/eda_eval_cluster/`` and can be aggregated afterwards.
"""

import glob
import os
import sys

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
SEEDS = list(range(1, 2))              # 10 seeds: 1..10
TEMPERATURES = ["0.1", "1.0", "10"]     # strings -> preserved in result filenames
TRAIN_SETS = [0, 1, 2]

# All implemented methods except {rcd, gs, tabu, dg, iterdsla, rpcd} and the
# exact/exponential levelwise DP (n<=20 guard, infeasible on these datasets).
ALGORITHMS = [
    "univ_bn",                                             # independent baseline
    "k2", "k2_mi", "k2_mb", "k2_refine", "k2_ensemble", "k2_plus",
    "fi_k2", "rfe_k2",                                     # objective-guided K2
    "bic", "aic", "stable_hc",
    "pc", "stable_pc",
    "dt", "dmbbn", "sartre", "binotears", "bounded_tw",
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT, "data", "eda_datasets")
SBATCH_SCRIPT = "slurm/slurm_eval_eda_benchmark.sh"


def problems():
    """All dataset names (sorted) discovered under data/eda_datasets/."""
    names = [
        os.path.basename(f).replace("_structure.dat", "")
        for f in sorted(glob.glob(os.path.join(_DATA_DIR, "*_structure.dat")))
    ]
    return names


if __name__ == "__main__":
    probs = problems()
    total = (len(probs) * len(ALGORITHMS) * len(TRAIN_SETS)
             * len(TEMPERATURES) * len(SEEDS))
    # A short banner on stderr keeps stdout a clean stream of sbatch lines.
    print(f"# {len(probs)} problems x {len(ALGORITHMS)} algorithms x "
          f"{len(TRAIN_SETS)} train sets x {len(TEMPERATURES)} temperatures x "
          f"{len(SEEDS)} seeds = {total} jobs", file=sys.stderr)

    try:
        for problem in probs:
            for algo in ALGORITHMS:
                for train in TRAIN_SETS:
                    for T in TEMPERATURES:
                        for seed in SEEDS:
                            print(f"sbatch {SBATCH_SCRIPT} "
                                  f"{problem} {train} {algo} {seed} {T}")
    except BrokenPipeError:                 # e.g. when piping through `head`
        pass
