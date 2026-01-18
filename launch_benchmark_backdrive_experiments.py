import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.1

problem = 'UBQP'

if __name__ == '__main__':
    # Real-world problem instances
    if problem=='SAT':
        instance_names = ['uf100-01','uf100-02','uf100-03','uf100-04','uf100-05']
    elif  problem == 'Ising':
        instance_names = ['SG_100_1','SG_100_2','SG_100_3','SG_100_4']
    elif  problem == 'UBQP':
        instance_names = ['bqp100']
    
    # Backdrive variants
    variants = ['backdrive', 'backdrive_descriptors']
    
    # Initialization methods
    init_methods = ['random']
    
    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    
    # Activation functions
    activation_functions = ['elu', 'relu', 'tanh']
    
    # Boolean flags (weight-transfer, early-stopping, surrogate-filtering)
    weight_transfer_options = [0]  # 1=True, 0=False
    
    early_stopping_options = [0]
    
    surrogate_filtering_options = [0]  # 1=True, 0=False
    
    # Alpha (mutation)
    alpha_values = [0.95]
    
    # Seeds
    seeds = np.arange(11, 21)
    
    n = 100
    # Generate all combinations
    for seed in seeds:
        for instance_name in instance_names:
            p_size = n * 5
            
            # Generate all combinations
            for variant in variants:
                for init_method in init_methods:
                    for loss_function in loss_functions:
                        for activation in activation_functions:
                            for weight_transfer in weight_transfer_options:
                                for early_stopping in early_stopping_options:
                                    for surrogate_filtering in surrogate_filtering_options:
                                        for alpha in alpha_values:
                                            # Build command with positional arguments
                                            cmd = f"sbatch slurm_benchmark_backdrive.sh examples/discrete_Backdrive_EDA_RW.py {seed} {problem} {instance_name} {p_size} {n_gen} {trunc} {variant} {init_method} {loss_function} {activation} {weight_transfer} {early_stopping} {surrogate_filtering} {alpha}"
                                            
                                            print(cmd)
