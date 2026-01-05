import sys
import os
import numpy as np

if __name__ == '__main__':
    # Fixed parameters
    n_gen = 250
    trunc = 0.5
    
    # Objective functions to test
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    
    # VAE variants - all 12 variants (original + enhanced)
    variants = ['VAE', 'E-VAE', 'C-VAE', 'Desc-VAE', 'Reg-VAE', 'Mom-VAE', 
                'BA-VAE', 'AA-VAE', 'FW-VAE', 'GS-VAE', 'HS-VAE', 'TC-VAE']
    
    # Activation functions for Encoder
    activation_enc_options = ['relu', 'tanh', 'leaky_relu']
    
    # Activation functions for Decoder
    activation_dec_options = ['relu', 'tanh']
    
    # Beta annealing parameters
    #beta_start_options = [0.0, 0.1]
    beta_start_options = [0.0]
    beta_end_options = [1.0]
    
    # Latent dimension (0 means use default)
    # latent_dim_options = [0, 8]
    latent_dim_options = [0]

    # Epochs
    #epochs_options = [30, 50]
    epochs_options = [50]
    
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
                for activation_enc in activation_enc_options:
                    for activation_dec in activation_dec_options:
                        for beta_start in beta_start_options:
                            for beta_end in beta_end_options:
                                for latent_dim in latent_dim_options:
                                    for epochs in epochs_options:
                                        # Build command with positional arguments
                                        cmd = f"sbatch slurm_vae.sh examples/discrete_VAE_EDA.py {seed} {obj_func} {n} {p_size} {n_gen} {trunc} {variant} {activation_enc} {activation_dec} {beta_start} {beta_end} {latent_dim} {epochs}"
                                        
                                        print(cmd)

