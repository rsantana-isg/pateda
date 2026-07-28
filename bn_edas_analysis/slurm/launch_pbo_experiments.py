"""
Generate sbatch commands for the PBO suite experiments (one job per
combination of algorithm, PBO function and dimension; all seeds run
inside the same job so they share one IOH data folder).

Full default grid: 13 algorithms x 25 functions x 3 dimensions = 975 jobs.
Keep <= 400 jobs running simultaneously, e.g. launch dimension by
dimension, or pipe through head/tail:

    python3 slurm/launch_pbo_experiments.py | head -400 | bash
    python3 slurm/launch_pbo_experiments.py | sed -n '401,800p' | bash

After all jobs finish, analyze jointly with:

    python3 scripts/analyze_pbo_results.py results/pbo_data_cluster
"""

# Fixed parameters (must match the local defaults of
# scripts/compare_discrete_edas_pbo.py for comparable results)
base_seed = 1
n_runs = 5
pop_size = 200
n_gen = 50
sel_ratio = 0.5

# PBO functions (1..25) and standard dimensions (all perfect squares,
# required by f23 N-Queens).  Add 625 explicitly if needed.
fids = list(range(1, 26))
dims = [16, 64, 100]

# Discrete pateda EDAs (see ALGORITHMS in scripts/compare_discrete_edas_pbo.py)
algorithms = [
    'UMDA', 'BMDA', 'TreeEDA', 'MIMIC', 'PBIL',
    'EBNA', 'BOA', 'AffEDA', 'MKEDA', 'MTED',
    'MNFDA', 'FDA', 'BSC',
]

if __name__ == '__main__':
    try:
        for dim in dims:
            for fid in fids:
                for alg in algorithms:
                    cmd = (f"sbatch slurm/slurm_pbo.sh {base_seed} {n_runs} "
                           f"{alg} {fid} {dim} {pop_size} {n_gen} {sel_ratio}")
                    print(cmd)
    except BrokenPipeError:    # e.g. when piping through `head`
        pass
