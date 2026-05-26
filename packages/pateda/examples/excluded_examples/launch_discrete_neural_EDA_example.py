"""
Example Launch Script for Cluster Execution
============================================

This example demonstrates how to generate commands for launching
discrete_EDA.py across multiple seeds, algorithms, and problems
in a cluster environment using a job scheduler like SLURM.

The script generates commands that would be executed with sbatch
to run independent EDA experiments in parallel.

Usage:
    python launch_discrete_neural_EDA_example.py

Note: This is an example script. In a real cluster environment,
you would need to:
1. Create a SLURM batch script (e.g., run_nn_EDA.sh)
2. Uncomment the subprocess.run() call to actually submit jobs
3. Adjust resource requirements based on your cluster

==============================================================================
"""

import numpy as np
# import subprocess  # Uncomment to actually submit jobs


def generate_commands():
    """
    Generate discrete_EDA.py commands for cluster execution
    
    This mimics the launch script from the problem statement,
    generating commands for all combinations of:
    - Seeds (for statistical reproducibility)
    - Algorithms (both neural and traditional EDAs)
    - Problems (with appropriate population sizes and generations)
    """
    # Problem configurations
    obj_functions = ['OneMax-20', 'Deceptive3-30', 'KDeceptive3-30']
    p_sizes = [80, 100, 120]
    n_gens = [20, 30, 30]
    
    # All supported algorithms
    NN_EDAs = ['VAE', 'GAN', 'Backdrive', 'DAE', 'RBM', 'DbD', 
               'UMDA', 'TreeEDA', 'EBNA', 'MOA']
    
    commands = []
    
    # Generate commands for multiple seeds
    for seed in np.arange(0, 20):
        for alg in NN_EDAs:
            for i in range(len(obj_functions)):
                # Format: discrete_EDA.py <seed> <obj_func> <pop_size> <n_gen> <alg>
                cmd = (f"python discrete_EDA.py {seed} {obj_functions[i]} "
                       f"{p_sizes[i]} {n_gens[i]} {alg}")
                
                # In a real cluster environment, you would use sbatch:
                # sbatch_cmd = f"sbatch run_nn_EDA.sh {cmd}"
                
                commands.append(cmd)
    
    return commands


def main():
    """Generate and display/execute commands"""
    print("=" * 80)
    print("DISCRETE EDA CLUSTER LAUNCH SCRIPT")
    print("=" * 80)
    print()
    
    commands = generate_commands()
    
    print(f"Generated {len(commands)} commands")
    print()
    print("Example commands (first 10):")
    print("-" * 80)
    for cmd in commands[:10]:
        print(cmd)
        # To actually submit jobs in a cluster environment:
        # subprocess.run(["sbatch", "run_nn_EDA.sh"] + cmd.split()[2:], check=True)
    
    print("...")
    print()
    print("Configuration Summary:")
    print("-" * 80)
    print(f"Seeds:      20 (0-19)")
    print(f"Algorithms: 10 (VAE, GAN, Backdrive, DAE, RBM, DbD, UMDA, TreeEDA, EBNA, MOA)")
    print(f"Problems:   3 (OneMax-20, Deceptive3-30, KDeceptive3-30)")
    print(f"Total runs: {len(commands)}")
    print()
    print("To execute in a cluster:")
    print("1. Create a SLURM batch script (e.g., run_nn_EDA.sh)")
    print("2. Uncomment subprocess.run() in this script")
    print("3. Run: python launch_discrete_neural_EDA_example.py")
    print()
    print("Example SLURM script (run_nn_EDA.sh):")
    print("-" * 80)
    print("#!/bin/bash")
    print("#SBATCH --job-name=discrete_eda")
    print("#SBATCH --output=logs/eda_%j.out")
    print("#SBATCH --error=logs/eda_%j.err")
    print("#SBATCH --time=01:00:00")
    print("#SBATCH --mem=4G")
    print("#SBATCH --cpus-per-task=1")
    print()
    print("python $@")
    print("=" * 80)


if __name__ == '__main__':
    main()
