#!/bin/bash

# SLURM Batch Script for Discrete Dendiff EDA Experiments
# =========================================================
#
# This script runs discrete Dendiff EDA experiments on SLURM cluster.
# Based on: slurm_pateda.sh
# For program: examples/discrete_Dendiff_EDA.py
#
# Parameters:
#   $1  = script path (examples/discrete_Dendiff_EDA.py)
#   $2  = seed
#   $3  = obj_func
#   $4  = n (number of variables)
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
#
# Usage:
#   sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
#       <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> \
#       <variant> <sampling_strategy> <activation> <loss> \
#       <n_timesteps> <n_sampling_steps> <fitness_guided> \
#       <temperature> <beta_start> <beta_end>
#
# Example:
#   sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
#       0 OneMax 20 80 20 0.5 dendiff_gumbel gumbel relu mse \
#       100 20 0 1.0 0.0001 0.3

# Job name
#SBATCH --job-name=dendiff_eda

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/dendiff_%A_%a.out
#SBATCH --error=outputs/dendiff_%A_%a.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use (uncomment if needed)
# #SBATCH --cpus-per-task=20
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
# SBATCH --mem-per-cpu=8G

# Select one partition
# GPU (uncomment if GPU is available and beneficial)
# SBATCH --partition=GPU
## SBATCH --gpus=1

# If you are using arrays, specify the number of tasks in the array
## SBATCH --array=1-1

# Create output filename with key parameters for easy identification
# Format: results_dendiff_<obj_func>_<n>_<variant>_<activation>_<loss>_<fitness_guided>_seed<seed>.dat
output_file="results_dendiff_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${16}_${17}_${18}_${2}.dat"

# Print command for logging
echo "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} > $output_file"

# Execute command
      bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} > $output_file

# Example usage:
# sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
#     0 OneMax 30 150 250 0.5 dendiff_gumbel gumbel relu mse \
#     100 20 0 1.0 0.0001 0.3 0.95
