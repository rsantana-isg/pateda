#!/bin/bash

# Job name
#SBATCH --job-name=discrete_dbd

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
# SBATCH --mem-per-cpu=8G

# Select one partition
# GPU
# SBATCH --partition=GPU
## SBATCH --gpus=1


# If you are using arrays, specify the number of tasks in the array
## SBATCH --array=1-1



# Parameters:
# $1  = script path (examples/discrete_DbD_EDA.py)
# $2  = seed
# $3  = obj_func
# $4  = n
# $5  = pop_size
# $6  = n_gen
# $7  = trunc
# $8  = variant
# $9  = activation
# $10 = loss
# $11 = num_alpha_samples
# $12 = n_steps
# $13 = k
# $14 = alpha_smooth
# $15 = fitness_guided
# $16 = use_markov_init

echo   "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} > results_dbd_${3}_${8}_${9}_${10}_${11}_${12}_${13}_${15}_${16}_${2}.dat"
        bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} > results_dbd_${3}_${8}_${9}_${10}_${11}_${12}_${13}_${15}_${16}_${2}.dat


# Example usage:
# sbatch slurm_dbd.sh examples/discrete_DbD_EDA.py 0 OneMax 20 100 250 0.5 dbd relu mse 20 20 0 0.1 0 0
