"""
Generate sbatch commands for the MN-FDA-P / MN-FDA-F sweep of the selected-PBO
study (a focused variant of ``launch_selected_pbo_experiments.py``).

It launches only the six configurations:

    MN-FDA-P (max_clique = 3, 4, 5)   -> rerun (the OOM at n=625 is fixed: MN-FDA-P
                                         now learns a bounded-treewidth
                                         running-intersection forest, so the exact
                                         MPC can no longer exhaust memory),
    MN-FDA-F (max_clique = 3, 4, 5)   -> new (the forest model without the MPC
                                         insertion; every solution sampled by PLS).

under the three weighted-probability selection schemes (FP, BZ, RTS), for the
PBO functions 1..25 and dimensions {16, 64, 100, 625}:

    6 algorithms x 3 selection-methods x 25 functions x 4 dimensions = 1800 jobs.

Keep <= 400 jobs running simultaneously, e.g. slice with head/tail:

    python3 slurm/launch_selected_pbo_PF_experiments.py | head -400 | bash
    python3 slurm/launch_selected_pbo_PF_experiments.py | sed -n '401,800p' | bash

RERUNNING MN-FDA-P
------------------
Both the SLURM wrapper (``slurm_selected_pbo.sh``, skips if the ``.dat`` output
file already exists) and ``run_selected_pbo_eda.py`` (skips if the IOH folder
already exists) are idempotent.  A job that was OOM-killed left a *stale*
``.dat`` file and/or a partial IOH folder, so it would be skipped.  To force the
MN-FDA-P jobs to re-run, first delete their stale artefacts, e.g.:

    rm -f results_spbo_MNFDAP*_*.dat
    rm -rf results/pbo_selected_data_cluster/MNFDAP*__*

After the jobs finish, analyse (per selection method) with:

    python3 scripts/analyze_selected_pbo_results.py results/pbo_selected_data_cluster \\
        results/pbo_analysis
"""

import os
import sys

# Reuse the single source of truth for builders and selection-method names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
from compare_selected_edas_pbo import (   # noqa: E402
    PF_ALGORITHMS, SELECTION_ORDER, POP_SIZE, N_GEN, SEL_RATIO, DIMENSIONS,
)

# Fixed parameters (must match compare_selected_edas_pbo.py for comparability).
base_seed = 1
n_runs = 5
pop_size = POP_SIZE
n_gen = N_GEN
sel_ratio = SEL_RATIO

fids = list(range(1, 26))
dims = list(DIMENSIONS)               # [16, 64, 100, 625]
algorithms = list(PF_ALGORITHMS)      # MNFDAP{3,4,5} + MNFDAF{3,4,5}
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
