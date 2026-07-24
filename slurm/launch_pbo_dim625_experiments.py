"""
Generate sbatch commands for the PBO suite at the highest standard
benchmark dimension, n = 625 (the largest dimension used in the PBO
reference setup of Doerr et al., GECCO'19; all 25 functions support it,
and f23 N-Queens requires a perfect square).

Differences with launch_pbo_experiments.py (the n in {16,64,100} grid):

  * One seed per job (n_runs = 1): runs at n = 625 are expensive
    (measured on f1: UMDA ~1 min, MIMIC ~10 min, BMDA ~20 min,
    TreeEDA ~1.5 h per run with pop=1000, n_gen=100), so replicates are
    parallelized instead of run sequentially.  Each job writes its own
    IOH folder results/pbo_data_cluster/{ALG}_f{FID}_dim625_s{SEED}/ and
    the analyzer merges them by algorithm name.
  * Larger population (1000) for the 625-variable models.
  * EBNA / BOA / MTED are commented out: Bayesian-network structure
    learning and mixtures of trees over 625 variables are prohibitive at
    this population size.  Re-enable them deliberately if needed.

Default grid: 10 algorithms x 25 functions x 5 seeds = 1250 jobs.
Keep <= 400 jobs running simultaneously:

    python3 slurm/launch_pbo_dim625_experiments.py | head -400 | bash
    python3 slurm/launch_pbo_dim625_experiments.py | sed -n '401,800p' | bash
    ...

Results land in the same tree as the other dimensions, so a single
joint analysis covers everything:

    python3 scripts/analyze_pbo_results.py results/pbo_data_cluster
"""

# Fixed parameters for the n = 625 experiments
dim = 625
n_runs = 1                    # one seed per job (see docstring)
seeds = range(1, 6)           # 5 replicates as separate jobs
pop_size = 1000
n_gen = 100                   # budget per run = 101,000 evaluations
sel_ratio = 0.5

fids = list(range(1, 26))

# Discrete pateda EDAs feasible at 625 variables
algorithms = [
    'UMDA', 'BMDA', 'TreeEDA', 'MIMIC', 'PBIL',
    'AffEDA', 'MKEDA',
    'MNFDA', 'FDA', 'BSC',
    'EBNA',    # BN structure learning at n=625: prohibitive
    'BOA',     # BN structure learning at n=625: prohibitive
    'MTED',    # mixture of trees at n=625: several hours per run
]

if __name__ == '__main__':
    try:
        for seed in seeds:
            for fid in fids:
                for alg in algorithms:
                    cmd = (f"sbatch slurm/slurm_pbo.sh {seed} {n_runs} "
                           f"{alg} {fid} {dim} {pop_size} {n_gen} {sel_ratio}")
                    print(cmd)
    except BrokenPipeError:    # e.g. when piping through `head`
        pass
