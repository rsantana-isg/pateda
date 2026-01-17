#!/bin/bash

# Job name
#SBATCH --job-name=discrete_benchmark_dendiff

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/M_%A_%a.out
#SBATCH --error=outputs/M_%A_%a.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
# #SBATCH --cpus-per-task=20
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
# SBATCH --mem-per-cpu=4G

# Select one partition
# GPU
# SBATCH --partition=CPU
## SBATCH --gpus=1


# If you are using arrays, specify the number of tasks in the array
## SBATCH --array=1-1


# Parameters:
#   $1  = script path (examples/discrete_Dendiff_EDA_RW.py)
#   $2  = seed
#   $3  = problem
#   s4  = instance_name
#   $5  = pop_size
#   $6  = n_gen
#   $7  = trunc
#   $8  = variant
#   $9  = sampling_strategy
#   $10 = activation
#   $11 = loss
#   $12 = n_timesteps
#   $13 = n_sampling_steps
#   $14 = fitness_guided
#   $15 = temperature
#   $16 = beta_start
#   $17 = beta_end
#   $18 = alpha


echo   "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18}> results_benchmark_dendiff_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${16}_${17}_${18}_${2}.dat"
        bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18}> results_benchmark_dendiff_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${16}_${17}_${18}_${2}.dat


# Example usage:
# sbatch slurm_benchmark_dendiff.sh examples/discrete_Dendiff_EDA_RW.py \
#     111 SAT uf100-01 500 0.1 dendiff_gumbel gumbel relu mse \
#     400 20 0 1.0 0.0001 0.3 0.95

