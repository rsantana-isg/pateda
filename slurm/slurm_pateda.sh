#!/bin/bash

# Job name
#SBATCH --job-name=discrete_eda

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



echo   "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 > results_discrete_eda_v2_$3_$4_$5_$6_$7_$8_$2.dat"        
        bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 > results_discrete_eda_v2_$3_$4_$5_$6_$7_$8_$2.dat


# sbatch examples/slurm_pateda.sh discrete_eda.py 111 OneMax 20 80 20 VAE 0.95
