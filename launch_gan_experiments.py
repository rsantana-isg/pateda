import sys
import os
import numpy as np

if __name__ == '__main__':
    # Fixed parameters
    n_gen = 250
    trunc = 0.5
    
    # Objective functions to test
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    
    # GAN variants
    variants = ['WGAN-GP', 'Cond-Fit-GAN', 'Aux-GAN', 'Repulsion-GAN']
    
    # Activation functions for Generator
    activation_g_options = ['relu', 'tanh', 'leaky_relu']
    
    # Activation functions for Discriminator
    activation_d_options = ['leaky_relu', 'relu']
    
    # Activation function for Encoder (only used for Hybrid-GAN-VAE)
    activation_e = 'relu'
    
    # Dropout rates
    #dropout_options = [0.3, 0.5]
    dropout_options = [0.5]
    
    # Temperature
    temperature_options = [0.5, 1.0]
    
    # Use surrogate (only for Aux-GAN)
    use_surrogate_options = [0, 1]
    
    # Seeds
    for seed in np.arange(6, 11):
        for obj_func in obj_functions:
            # Set n based on objective function
            if obj_func == 'HIFF':
                n = 64
            else:
                n = 30
            p_size = n * 5
            
            # Generate all combinations
            for variant in variants:
                for activation_g in activation_g_options:
                    for activation_d in activation_d_options:
                        for dropout in dropout_options:
                            for temperature in temperature_options:
                                # Handle use_surrogate specifically for Aux-GAN
                                if variant == 'Aux-GAN':
                                    surrogate_values = use_surrogate_options
                                else:
                                    surrogate_values = [0]  # Only test 0 for non-Aux-GAN
                                
                                for use_surrogate in surrogate_values:
                                    # Build command with positional arguments
                                    cmd = f"sbatch slurm_gan.sh examples/discrete_GAN_EDA.py {seed} {obj_func} {n} {p_size} {n_gen} {trunc} {variant} {activation_g} {activation_d} {activation_e} {dropout} {temperature} {use_surrogate}"
                                    
                                    print(cmd)

