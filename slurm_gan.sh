#!/bin/bash

# Job name
#SBATCH --job-name=discrete_gan_eda

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/GAN_%A_%a.out
#SBATCH --error=outputs/GAN_%A_%a.err

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



# Arguments:
# $1: script, $2: seed, $3: obj_func, $4: n, $5: pop_size, $6: n_gen, $7: trunc
# $8: variant, $9: activation_g, $10: activation_d, $11: activation_e
# $12: dropout, $13: temperature, $14: use_surrogate

echo "bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} > results_discrete_gan_$3_$4_$5_$6_$7_$2_$8_$9_${10}_${11}_${12}_${13}_${14}.dat"
bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} > results_discrete_gan_$3_$4_$5_$6_$7_$2_$8_$9_${10}_${11}_${12}_${13}_${14}.dat


# sbatch slurm_gan.sh examples/discrete_GAN_EDA.py 111 OneMax 30 150 250 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0
