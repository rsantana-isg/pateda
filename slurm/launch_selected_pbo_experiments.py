"""
Generate sbatch commands for the SELECTED weighted-probability PBO experiments
(one job per algorithm x selection-method x PBO function x dimension; all seeds
run inside the same job so they share one IOH data folder).

Based on ``launch_weighted_pbo_experiments.py`` (weighted probabilities, three
selection schemes) but restricted to the affinity / MN-FDA-S / MN-FDA-sparse /
MN-FDA-S-sparse / MN-FDA-P family with their hyper-parameter sweeps.

Full default grid:
    17 algorithms x 3 selection-methods x 25 functions x 4 dimensions = 5100 jobs.

Keep <= 400 jobs running simultaneously, e.g. slice with head/tail or launch one
(dimension, selection-method) block at a time:

    python3 slurm/launch_selected_pbo_experiments.py | head -400 | bash
    python3 slurm/launch_selected_pbo_experiments.py | sed -n '401,800p' | bash

The largest dimension (625) is included; the prohibitive large-dim jobs that do
not finish are simply skipped by the analysis, which uses whichever runs
completed.  After the jobs finish, analyse with:

    python3 scripts/analyze_selected_pbo_results.py results/pbo_selected_data_cluster \\
        results/pbo_analysis
"""

import os
import sys

# Reuse the single source of truth for algorithm and selection-method names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
from compare_selected_edas_pbo import (   # noqa: E402
    MAIN_STUDY_ALGORITHMS, SELECTION_ORDER, POP_SIZE, N_GEN, SEL_RATIO,
    DIMENSIONS,
)

# Fixed parameters (must match compare_selected_edas_pbo.py for comparability).
base_seed = 1
n_runs = 5
pop_size = POP_SIZE
n_gen = N_GEN
sel_ratio = SEL_RATIO

fids = list(range(1, 26))
dims = list(DIMENSIONS)               # [16, 64, 100, 625]
algorithms = list(MAIN_STUDY_ALGORITHMS)   # the 17 core configs (no MN-FDA-F)
selections = list(SELECTION_ORDER)

if __name__ == "__main__":
    try:
        for dim in dims:
            for sel in selections:
                for fid in fids:
                    for alg in algorithms:
                        cmd = (f"sbatch slurm/slurm_selected_pbo.sh {base_seed} "
                               f"{n_runs} {alg} {sel} {fid} {dim} {pop_size} "
                               f"{n_gen} {sel_ratio}")
                        print(cmd)
    except BrokenPipeError:            # e.g. when piping through `head`
        pass
