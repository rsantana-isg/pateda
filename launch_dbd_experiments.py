import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.5

if __name__ == '__main__':
    # Objective functions to test
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']

    # DbD variants to test
    #dbd_variants = ['dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 'dbd_us',
    #                'dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t', 'dbd_us_t']
    #dbd_variants = ['dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 
    #                'dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t']
    dbd_variants = ['dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t']


    # Activation functions (all in the specified set)
    activations = ['leaky_relu', 'relu', 'tanh']

    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']

    # Number of alpha samples for blending
    num_alpha_samples_list = [20]

    # Number of denoising steps
    n_steps_list = [20]

    # Markov chain orders (for transformation variants)
    k_values = [2]

    # Alpha smoothing parameter (fixed)
    alpha_smooth = 0.1

    # Fitness guidance flag
    fitness_guided_values = [0, 1]

    # Markov initialization flag
    use_markov_init_values = [0]

    # Seeds to test
    seeds = np.arange(1, 11)  # 30 different seeds

    # Generate all combinations
    for seed in seeds:
        for obj_func in obj_functions:
            # Set problem size based on objective function
            if obj_func == 'HIFF':
                n = 64
            else:
                n = 30
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
                                            # Build command
                                            cmd = (f"sbatch slurm_dbd.sh examples/discrete_DbD_EDA.py "
                                                  f"{seed} {obj_func} {n} {p_size} {n_gen} {trunc} "
                                                  f"{variant} {activation} {loss} {num_alpha_samples} "
                                                  f"{n_steps} {k} {alpha_smooth} {fitness_guided} "
                                                  f"{use_markov_init}")
                                            print(cmd)
