"""
Launch Script for Discrete Dendiff EDA Experiments
===================================================

This script generates all combinations of Dendiff EDA experiments with different
parameters for cluster execution using SLURM.

Based on: lanzar_discrete_EDA.py
For program: examples/discrete_Dendiff_EDA.py

Fixed Parameters:
- Generations: 250
- Truncation Percent: 0.5
- n: 64 for HIFF, 30 for others
- p_size: n*5
- n_sampling_steps: 20

Variable Parameters:
- variant: dendiff_gumbel, dendiff_corruption
- activation: leaky_relu, relu, tanh, sigmoid
- loss: mse, weighted_mse, ranking, huber
- fitness_guided: 0, 1

Dependent Parameters (set based on variant):
- sampling_strategy: matches variant (gumbel or corruption)
- n_timesteps: 100 for gumbel, 50 for corruption
- temperature: 1.0 for gumbel, 0.5 for corruption
- beta_start: 0.0001 for gumbel, 0.01 for corruption
- beta_end: 0.3 for gumbel, 0.5 for corruption

Usage:
    python launch_dendiff_experiments.py
"""

import sys
import os
import numpy as np

# Fixed parameters
n_gen = 250
trunc = 0.5
n_sampling_steps = 20

if __name__ == '__main__':
    # Objective functions to test
    # obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    obj_functions = ['OneMax']

    # Dendiff variants
    variants = ['dendiff_gumbel', 'dendiff_corruption']

    # Activation functions
    activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']

    # Loss functions
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']

    # Fitness guidance options
    fitness_guided_options = [0, 1]

    # Seeds for multiple runs
    seeds = np.arange(1, 31)  # 30 independent runs

    # Generate all combinations
    for seed in seeds:
        for obj_func in obj_functions:
            # Set problem size based on objective function
            if obj_func == 'HIFF':
                n = 64
            else:
                n = 30
            p_size = n * 5

            for variant in variants:
                # Set variant-dependent parameters
                if variant == 'dendiff_gumbel':
                    sampling_strategy = 'gumbel'
                    n_timesteps = 100
                    temperature = 1.0
                    beta_start = 0.0001
                    beta_end = 0.3
                else:  # dendiff_corruption
                    sampling_strategy = 'corruption'
                    n_timesteps = 50
                    temperature = 0.5
                    beta_start = 0.01
                    beta_end = 0.5

                for activation in activations:
                    for loss in loss_functions:
                        for fitness_guided in fitness_guided_options:
                            # Build SLURM command
                            # Format: sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py <16 parameters>
                            cmd = (
                                f"sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py "
                                f"{seed} {obj_func} {n} {p_size} {n_gen} {trunc} "
                                f"{variant} {sampling_strategy} {activation} {loss} "
                                f"{n_timesteps} {n_sampling_steps} {fitness_guided} "
                                f"{temperature} {beta_start} {beta_end}"
                            )
                            print(cmd)
