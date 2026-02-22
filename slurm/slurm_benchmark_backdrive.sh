#!/bin/bash

# Job name
#SBATCH --job-name=discrete_benchmark_backdrive

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
# $1  = script path (examples/discrete_Backdrive_EDA_RW.py)
# $2  = seed
# $3  = problem
# $4  = instance_name
# $5  = pop_size
# $6  = n_gen
# $7  = trunc
# $8  = variant
# $9  = init
# $10 = loss
# $11 = activation
# $12 = weight_transfer
# $13 = early_stopping
# $14 = surrogate_filtering
# $15 = alpha

echo   "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} > results_benchmark_backdrive_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${2}.dat"
        bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} > results_benchmark_backdrive_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${2}.dat


# Example usage:
# sbatch slurm_benchmark_backdrive.sh examples/discrete_Backdrive_EDA_RW.py 111 SAT uf100-01 500 250 0.1 backdrive random mse relu 0 0 0 0.95

