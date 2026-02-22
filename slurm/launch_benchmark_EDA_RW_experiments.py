import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
truncation_values = [0.1,0.5]
truncation_ratio = 0.1
n = 100
p_size = n * 5  # Following the N=5n convention
alpha_values = [0,0.95]  # Standard threshold for mutation

if __name__ == '__main__':
    # Problem types and their specific instances
    problems = {
        'SAT': ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05'],
        'Ising': ['SG_100_1', 'SG_100_2', 'SG_100_3', 'SG_100_4'],
        'UBQP': ['bqp100']
    }

    # Traditional algorithms supported by discrete_EDA_RW.py
    traditional_algos = [
        'UMDA', 'TreeEDA', 'EBNA', 'MN-FDA', 'MN-FDAG',
        'MK-EDA1', 'MK-EDA2', 'MK-EDA3']

    # Seeds to test
    seeds = np.arange(11, 21)

    # Generate all combinations
    for seed in seeds:
        for problem_type, instance_names in problems.items():
            for instance_name in instance_names:
                for alg in traditional_algos:
                    for alpha in alpha_values:
                        for truncation_ration in truncation_values:
                            # Build command using the new slurm_benchmark_EDA_RW.sh script
                            # Arguments: script, seed, problem, instance, pop_size, n_gen, alg, alpha, truncation
                            cmd = (f"sbatch slurm_benchmark_EDA_RW.sh examples/discrete_EDA_RW.py "
                                   f"{seed} {problem_type} {instance_name} {p_size} {n_gen} "
                                   f"{alg} {alpha} {truncation_ratio}")
                    
                            print(cmd)                    
