"""
Generate sbatch commands to run the recently-developed BN variants on the PBO
suite under the three selection schemes (FP, BZ, RTS), one job per
(algorithm, selection-method, function, dimension); all seeds run inside the
same job so they share one IOH data folder.

Algorithms (11, all with common max_parents=6):
    EBNA_BIC, EBNA_K2, EBNA_PC, LFDA, BOA, SARTRE,
    A1_dt, A2_mi, A3_fast, A4_mdl, A5_ndg

Full default grid:
    11 algorithms x 3 selection-methods x 25 functions x 4 dimensions = 3300 jobs.

Keep <= 400 jobs running simultaneously, e.g. slice with head/tail or launch one
(dimension, selection-method) block at a time:

    python3 slurm/launch_bn_variants_pbo_experiments.py | head -400 | bash
    python3 slurm/launch_bn_variants_pbo_experiments.py | sed -n '401,800p' | bash

Dimensions {16, 64, 100, 625} are all included; the prohibitive large-dim jobs
(e.g. A3_fast/A4_mdl and the exact BN learners at n=625) that do not finish are
simply skipped by the analysis, which uses whichever runs completed.

After the jobs finish, analyse and build the report exactly as for the previous
weighted-PBO study (the folder naming is identical, {ALG}__{SEL}):

    python3 scripts/analyze_weighted_pbo_results.py results/pbo_bnvariants_data_cluster \\
        results/pbo_bnvariants_analysis
    python3 scripts/make_pbo_latex_report.py results/pbo_bnvariants_analysis
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
from compare_bn_variants_pbo import (   # noqa: E402
    ALGORITHM_NAMES, SELECTION_ORDER, POP_SIZE, N_GEN, SEL_RATIO, DIMENSIONS,
)

# Fixed parameters (match compare_bn_variants_pbo.py for comparable results).
base_seed = 1
n_runs = 5
pop_size = POP_SIZE          # 200
n_gen = N_GEN                # 50
sel_ratio = SEL_RATIO        # 0.5

fids = list(range(1, 26))
dims = list(DIMENSIONS)      # [16, 64, 100, 625]
algorithms = list(ALGORITHM_NAMES)
selections = list(SELECTION_ORDER)

if __name__ == "__main__":
    try:
        for dim in dims:
            for sel in selections:
                for fid in fids:
                    for alg in algorithms:
                        cmd = (f"sbatch slurm/slurm_bn_variants_pbo.sh {base_seed} "
                               f"{n_runs} {alg} {sel} {fid} {dim} {pop_size} "
                               f"{n_gen} {sel_ratio}")
                        print(cmd)
    except BrokenPipeError:      # e.g. when piping through `head`
        pass
