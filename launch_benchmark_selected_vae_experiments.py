import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.5

if __name__ == '__main__':
    # Real-world problem instances to test
    # SAT instances
    sat_instances = ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05']
    
    # Ising instances
    ising_instances = ['SG_100_1', 'SG_100_2', 'SG_100_3', 'SG_100_4']
    
    # UBQP instances
    ubqp_instances = ['bqp100']
    
    # Combine all problem instances
    problem_configs = []
    for instance in sat_instances:
        problem_configs.append(('SAT', instance))
    for instance in ising_instances:
        problem_configs.append(('Ising', instance))
    for instance in ubqp_instances:
        problem_configs.append(('UBQP', instance))
    
    # VAE variants supported by discrete_EDA_RW.py
    vae_variants = ['VAE', 'VAE-Extended']
    
    # Alpha (mutation parameter) - testing both baseline (no mutation) and mutation-enabled
    # 0.0 = no mutation, 0.95 = frequency balance mutation for better exploration
    alpha_values = [0.0, 0.95]
    
    # Seeds to test
    seeds = np.arange(1, 21)  # 20 different seeds (matching launch_selected_vae_experiments.py)
    
    # Problem size configuration (n=100 for all real-world instances)
    n = 100
    p_size = n * 5  # Population size = 500

    # Generate all combinations
    for seed in seeds:
        for problem_type, instance_name in problem_configs:
            for variant in vae_variants:
                for alpha in alpha_values:
                    # Build command
                    # Format: discrete_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <alg> [alpha] [truncation]
                    cmd = (f"sbatch slurm_benchmark_vae.sh examples/discrete_EDA_RW.py "
                          f"{seed} {problem_type} {instance_name} {p_size} {n_gen} "
                          f"{variant} {alpha} {trunc}")
                    print(cmd)
