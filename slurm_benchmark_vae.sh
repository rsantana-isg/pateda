#!/bin/bash

# Job name
#SBATCH --job-name=discrete_benchmark_vae

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
# Uncomment and adjust if needed for larger problem instances
#SBATCH --mem-per-cpu=4G

# Select one partition
# Uncomment appropriate partition for your cluster
#SBATCH --partition=CPU
## SBATCH --partition=GPU
## SBATCH --gpus=1


# If you are using arrays, specify the number of tasks in the array
## SBATCH --array=1-1



# Parameters:
# $1  = script path (examples/discrete_EDA_RW.py)
# $2  = seed
# $3  = problem_type (SAT, Ising, UBQP)
# $4  = instance_name (e.g., uf100-01, SG_100_1, bqp100)
# $5  = pop_size
# $6  = n_gen
# $7  = algorithm (VAE, VAE-Extended)
# $8  = alpha (mutation parameter)
# $9  = truncation
# $10 = activation_enc (encoder activation function)
# $11 = activation_dec (decoder activation function)
# $12 = beta_start (beta annealing start)
# $13 = beta_end (beta annealing end)
# $14 = beta_end (beta annealing end)
# $15 = epochs (training epochs)
# $16 = mi_layer (MI layer flag: 0 or 1)

echo   "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} > results_benchmark_vae_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${16}_${2}.dat"
        bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} > results_benchmark_vae_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${10}_${11}_${12}_${13}_${14}_${15}_${16}_${2}.dat


# Example usage:
# sbatch slurm_benchmark_vae.sh examples/discrete_EDA_RW.py 1 SAT uf100-01 500 250 VAE 0.0 0.5 relu relu 0.0 1.0 0 50 0
# sbatch slurm_benchmark_vae.sh examples/discrete_EDA_RW.py 1 Ising SG_100_1 500 250 VAE-Extended 0.95 0.5 elu tanh 0.0 1.0 0 50 1


