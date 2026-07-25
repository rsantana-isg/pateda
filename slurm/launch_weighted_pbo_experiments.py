"""
Generate sbatch commands for the weighted-probability PBO experiments (one job
per algorithm x selection-method x PBO function x dimension; all seeds run
inside the same job so they share one IOH data folder).

Full default grid:
    22 algorithms x 3 selection-methods x 25 functions x 3 dimensions = 4950 jobs.

Keep <= 400 jobs running simultaneously, e.g. slice with head/tail or launch
one (dimension, selection-method) block at a time:

    python3 slurm/launch_weighted_pbo_experiments.py | head -400 | bash
    python3 slurm/launch_weighted_pbo_experiments.py | sed -n '401,800p' | bash

Dimension 625 is intentionally excluded (prohibitively expensive for the
model-building EDAs).  After all jobs finish, analyze jointly with:

    python3 scripts/analyze_weighted_pbo_results.py results/pbo_weighted_data_cluster
"""

import os
import sys

# Reuse the single source of truth for algorithm and selection-method names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
from compare_weighted_edas_pbo import (   # noqa: E402
    ALGORITHM_NAMES, SELECTION_ORDER, POP_SIZE, N_GEN, SEL_RATIO,
)

# Fixed parameters (must match compare_weighted_edas_pbo.py for comparability).
base_seed = 1
n_runs = 5
pop_size = POP_SIZE
n_gen = N_GEN
sel_ratio = SEL_RATIO

# PBO functions (1..25) and standard dimensions (perfect squares, required by
# f23 N-Queens).  625 omitted on purpose.
fids = list(range(1, 26))
dims = [16, 64, 100]
dims = [625]

algorithms = list(ALGORITHM_NAMES)
selections = list(SELECTION_ORDER)

if __name__ == "__main__":
    try:
        for dim in dims:
            for sel in selections:
                for fid in fids:
                    for alg in algorithms:
                        cmd = (f"sbatch slurm/slurm_weighted_pbo.sh {base_seed} "
                               f"{n_runs} {alg} {sel} {fid} {dim} {pop_size} "
                               f"{n_gen} {sel_ratio}")
                        print(cmd)
    except BrokenPipeError:      # e.g. when piping through `head`
        pass
