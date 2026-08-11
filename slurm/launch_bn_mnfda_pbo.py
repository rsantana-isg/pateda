"""
Generate sbatch commands for the BN-EDAs-vs-MN-FDA comparison on the PBO suite
at n in {16, 64, 625}, one job per (algorithm, selection, function, dimension).

Algorithm set (from compare_bn_mnfda_pbo.ALGORITHM_NAMES) -- the unbounded BN
variants are removed; every learned model has bounded treewidth, and all
algorithms share the SAME treewidth budget k for each k in BOUNDED_KS = [4, 8]:
  * bounded Phase-2 variants        (BND{k}_*, treewidth <= k by construction)
  * bounded variants + MPC          (BND{k}_*_MPC: MN-FDA-P-style exact
                                      most-probable configuration inserted each
                                      generation, computed by junction-tree
                                      max-product on the bounded model)
  * MN-FDA                          (MNFDA_k{k}, MNFDAP_k{k}; unified RI-forest
                                      learner with max_clique_size = k+1, i.e.
                                      the same treewidth k as the bounded BNs)

Per-dimension budget (n = 625 is expensive, so it mirrors
launch_pbo_dim625_experiments.py: bigger population, one seed per job):

    n = 16, 64 : pop = 200,  n_gen = 50,  n_runs = 5  (5 seeds in one job)
    n = 625    : pop = 1000, n_gen = 100, n_runs = 1  (5 seeds -> 5 jobs)

The point of the study is n = 625: the unbounded BN learners (and, for the
decision-graph ones, already n = 100/256) do not finish, whereas the BND{k}_*
variants -- and MN-FDA -- do.  Unfinished jobs are simply skipped by the
analysis, which uses whichever runs completed.

Default grid size (SELECTIONS = FP,BZ,RTS; BOUNDED_KS = [4,8] -> 22 bounded +
22 bounded+MPC + 4 MN-FDA = 48 algorithms):
    48 alg x 3 sel x 25 fn x [ (16:1 job)+(64:1 job)+(625:5 jobs) ] = 48*3*25*7
    = 25200 jobs.  Trim by editing SELECTIONS / BOUNDED_KS / dims, and keep
    <= 400 jobs in flight:

    python3 slurm/launch_bn_mnfda_pbo.py | head -400 | bash
    python3 slurm/launch_bn_mnfda_pbo.py | sed -n '401,800p' | bash

After the runs, analyse with the shared weighted-PBO pipeline (folder naming is
the identical {ALG}__{SEL}):

    python3 scripts/analyze_weighted_pbo_results.py \\
        results/pbo_bn_mnfda_data_cluster results/pbo_bn_mnfda_analysis
    python3 scripts/make_pbo_latex_report.py results/pbo_bn_mnfda_analysis
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
from compare_bn_mnfda_pbo import ALGORITHM_NAMES, SELECTION_ORDER   # noqa: E402

# Per-dimension run configuration: (pop_size, n_gen, n_runs, seeds).
DIM_CONFIG = {
    16:  dict(pop=200,  n_gen=50,  n_runs=5, seeds=[1]),          # seeds 1..5, 1 job
    64:  dict(pop=200,  n_gen=50,  n_runs=5, seeds=[1]),          # seeds 1..5, 1 job
    625: dict(pop=1000, n_gen=100, n_runs=1, seeds=[1, 2, 3, 4, 5]),  # 1 seed/job
}

SEL_RATIO = 0.5
FIDS = list(range(1, 26))
SELECTIONS = list(SELECTION_ORDER)        # ["FP", "BZ", "RTS"]; trim to ["BZ"] to shrink
ALGORITHMS = list(ALGORITHM_NAMES)


if __name__ == "__main__":
    try:
        for dim, cfg in DIM_CONFIG.items():
            for sel in SELECTIONS:
                for fid in FIDS:
                    for alg in ALGORITHMS:
                        for seed in cfg["seeds"]:
                            cmd = (f"sbatch slurm/slurm_bn_mnfda.sh {seed} "
                                   f"{cfg['n_runs']} {alg} {sel} {fid} {dim} "
                                   f"{cfg['pop']} {cfg['n_gen']} {SEL_RATIO}")
                            print(cmd)
    except BrokenPipeError:      # e.g. when piping through `head`
        pass
