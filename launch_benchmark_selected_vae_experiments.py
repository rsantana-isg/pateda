import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.1

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
    
    # VAE variants to explore (matching launch_selected_vae_experiments.py)
    # TODO: Implement C-VAE support in discrete_EDA_RW.py to enable full variant exploration
    #variants = ['VAE', 'E-VAE', 'C-VAE']
    variants = ['E-VAE', 'C-VAE']
    
    # VAE-specific parameters to explore (matching launch_selected_vae_experiments.py)
    activation_enc_options = ['elu', 'relu', 'tanh']
    activation_dec_options = ['elu', 'relu', 'tanh']
    beta_start_options = [0.0]
    beta_end_options = [1.0]

    # Latent dimension (0 means use default)
    # latent_dim_options = [0, 8]
    latent_dim_options = [0]
    
    epochs_options = [50]
    
    #mi_layer_options = [0, 1]
    mi_layer_options = [0]
    
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
            for variant in variants:
                for alpha in alpha_values:
                    for activation_enc in activation_enc_options:
                        for activation_dec in activation_dec_options:
                            for latent_dim in latent_dim_options:
                                for beta_start in beta_start_options:
                                    for beta_end in beta_end_options:
                                        for epochs in epochs_options:
                                            for mi_layer in mi_layer_options:
                                                                                     
                                                alg_variant = variant
                                            
                                                # Build command with all parameters
                                                # Format: discrete_VAE_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <alg>
                                                #         [alpha] [truncation] [activation_enc] [activation_dec] [beta_start] [beta_end] [epochs] [mi_layer]
                                                cmd = (f"sbatch slurm_benchmark_vae.sh examples/discrete_VAE_EDA_RW.py "
                                                       f"{seed} {problem_type} {instance_name} {p_size} {n_gen} "
                                                       f"{trunc} {alg_variant} {activation_enc} {activation_dec} "
                                                       f"{beta_start} {beta_end} {latent_dim} {epochs} {mi_layer} {alpha}")
                                                print(cmd)
