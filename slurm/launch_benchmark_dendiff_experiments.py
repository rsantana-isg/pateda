import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.1
n_sampling_steps = 20

problem = 'UBQP'

if __name__ == '__main__':
    # Objective functions to test
    #obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']

    if problem=='SAT':
        instance_names = ['uf100-01','uf100-02','uf100-03','uf100-04','uf100-05']
    elif  problem == 'Ising':
        instance_names = ['SG_100_1','SG_100_2','SG_100_3','SG_100_4']
    elif  problem == 'UBQP':
        instance_names = ['bqp100']


    # Dendiff variantss
    #variants = ['dendiff_gumbel', 'dendiff_corruption', 'dendiff_ste', 'dendiff_hard_concrete', 'dendiff_deterministic']
    variants = ['dendiff_gumbel']
    
    # Activation functions
    #activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']
    activations = ['elu', 'relu', 'tanh']

    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']

    # Fitness guidance options
    fitness_guided_options = [0, 1]

    num_alpha_samples_list = {0, 0.95}

    # Seeds for multiple runs
    seeds = np.arange(1, 11)  # 30 independent run
    
    n_timesteps = 400
    temperature = 1.0
    beta_start = 0.01
    beta_end = 1


    # Alpha (mutation)
    alpha_values = [0, 0.95]

    
    # Seeds to test
    seeds = np.arange(1, 21)  # 30 different seeds

    
    n = 100
    # Generate all combinations
    for seed in seeds:
        for instance_name in instance_names:                       
            p_size = n * 5
            for variant in variants:
                # Set variant-dependent parameters
                if variant == 'dendiff_gumbel':
                    sampling_strategy = 'gumbel'                 
                elif variant == 'dendiff_corruption':
                    sampling_strategy = 'corruption'                 
                elif variant == 'dendiff_hard_concrete':
                    sampling_strategy = 'hard_concrete'                      
                elif variant == 'dendiff_deterministic':
                    sampling_strategy = 'deterministic'
                for activation in activations:
                    for loss in loss_functions:                      
                             for fitness_guided in fitness_guided_options:                                       
                                            for alpha in alpha_values:
                                            # Build command
                                                cmd = (
                                                    f"sbatch slurm_benchmark_dendiff.sh examples/discrete_Dendiff_EDA_RW.py "
                                                    f"{seed} {problem} {instance_name} {p_size} {n_gen} {trunc} "
                                                    f"{variant} {sampling_strategy} {activation} {loss} "
                                                    f"{n_timesteps} {n_sampling_steps} {fitness_guided} "
                                                    f"{temperature} {beta_start} {beta_end} {alpha}"
                                                )
                                                print(cmd)
                                               
