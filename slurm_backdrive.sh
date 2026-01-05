#!/bin/bash

# Job name
#SBATCH --job-name=discrete_backdrive_eda

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



# Extract positional parameters for filename
# $1: script, $2: seed, $3: obj_func, $4: n, $5: pop_size, $6: n_gen, $7: trunc
# Additional parameters are optional flags
echo "bnd -exec python3 $@ > results_discrete_backdrive_${3}_${4}_${5}_${6}_${7}_${2}.dat"
bnd -exec python3 "$@" > "results_discrete_backdrive_${3}_${4}_${5}_${6}_${7}_${2}.dat"


# sbatch slurm_backdrive.sh examples/discrete_Backdrive_EDA.py 111 OneMax 30 150 250 0.5 --variant backdrive --init random --loss mse --activation relu
