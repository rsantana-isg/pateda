import sys
import os
import numpy as np

if __name__ == '__main__':
    # Fixed parameters
    n_gen = 250
    trunc = 0.5
    
    # Objective functions to test
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    
    # Backdrive variants
    #variants = ['backdrive', 'backdrive_adaptive', 'backdrive_descriptors']
    variants = ['backdrive', 'backdrive_descriptors']
    
    # Initialization methods
    #init_methods = ['random', 'perturb_best', 'perturb_selected']
    init_methods = ['random']
    
    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    
    # Activation functions
    activation_functions = ['leaky_relu', 'relu', 'tanh']
    
    # Boolean flags (weight-transfer, early-stopping, surrogate-filtering)
    weight_transfer_options = [1, 0]  # 1=True, 0=False
    #early_stopping_options = [1, 0]
    early_stopping_options = [1]  # 1=True
    surrogate_filtering_options = [1, 0]  # 1=True, 0=False
    
    # Seeds
    for seed in np.arange(2, 11):
        for obj_func in obj_functions:
            # Set n based on objective function
            if obj_func == 'HIFF':
                n = 64
            else:
                n = 30
            p_size = n * 5
            
            # Generate all combinations
            for variant in variants:
                for init_method in init_methods:
                    for loss_function in loss_functions:
                        for activation in activation_functions:
                            for weight_transfer in weight_transfer_options:
                                for early_stopping in early_stopping_options:
                                    for surrogate_filtering in surrogate_filtering_options:
                                        # Build command with positional arguments
                                        cmd = f"sbatch slurm_backdrive.sh examples/discrete_Backdrive_EDA.py {seed} {obj_func} {n} {p_size} {n_gen} {trunc} {variant} {init_method} {loss_function} {activation} {weight_transfer} {early_stopping} {surrogate_filtering}"
                                        
                                        print(cmd)

