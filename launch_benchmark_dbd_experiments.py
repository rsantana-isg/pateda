import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.1

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
   
    dbd_variants = ['dbd_cs', 'dbd_cd', 'dbd_cs_t', 'dbd_cd_t']
  

    # Activation functions (all in the specified set)
    #activations = ['leaky_relu', 'relu', 'tanh']
    activations = ['elu', 'relu', 'tanh']
    

    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    

    # Number of alpha samples for blending
    num_alpha_samples_list = [100]

    # Number of denoising steps
    n_steps_list = [20]

    # Markov chain orders (for transformation variants)
    k_values = [1]


    # Alpha smoothing parameter (fixed)
    alpha_smooth = 0.1

    # Fitness guidance flag    
    fitness_guided_values = [0,1]

    # Markov initialization flag
    use_markov_init_values = [0]

    # Alpha (mutation)
    alpha_values = [0.95]
    #alpha_values = [0.8, 0.95]

    
    # Seeds to test
    seeds = np.arange(11, 21)  # 30 different seeds

    
    n = 100
    # Generate all combinations
    for seed in seeds:
        for instance_name in instance_names:                       
            p_size = n * 5
            for variant in dbd_variants:
                for activation in activations:
                    for loss in loss_functions:
                        for num_alpha_samples in num_alpha_samples_list:
                            for n_steps in n_steps_list:
                                # For transformation variants, test different k values
                                if '_t' in variant:
                                    k_list = k_values
                                else:
                                    k_list = [0]  # k is not used for non-transformation variants

                                for k in k_list:
                                    for fitness_guided in fitness_guided_values:
                                        for use_markov_init in use_markov_init_values:
                                            for alpha in alpha_values:
                                            # Build command
                                                  cmd = (f"sbatch slurm_benchmark_dbd.sh examples/discrete_DbD_EDA_RW.py "
                                                  f"{seed} {problem} {instance_name} {p_size} {n_gen} {trunc} "
                                                  f"{variant} {activation} {loss} {num_alpha_samples} "
                                                  f"{n_steps} {k} {alpha_smooth} {fitness_guided} "
                                                  f"{use_markov_init} {alpha}")
                                                  print(cmd)
