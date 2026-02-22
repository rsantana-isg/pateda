#!/bin/bash

# Job name
#SBATCH --job-name=traditional_benchmark_eda

# Define output and error files
#SBATCH --output=outputs/TRAD_%A_%a.out
#SBATCH --error=outputs/TRAD_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

# Parameters:
# $1 = script path (examples/discrete_EDA_RW.py)
# $2 = seed
# $3 = problem_type (SAT, Ising, UBQP)
# $4 = instance_name
# $5 = pop_size
# $6 = n_gen
# $7 = algorithm (UMDA, TreeEDA, etc.)
# $8 = alpha (mutation threshold)
# $9 = truncation (selection ratio)

# Construct output filename for systematic results collection
OUTPUT_FILE="results_benchmark_EDA_RW_${3}_${4}_${5}_${6}_${7}_${8}_${9}_${2}.dat"

echo "Executing: bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 > $OUTPUT_FILE"

# Execute the experiment
                 bnd -exec python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 > "$OUTPUT_FILE"

# Example usage:
# sbatch slurm_benchmark_EDA_RW.sh examples/discrete_EDA_RW.py 1 SAT uf100-01 500 250 UMDA 0.95 0.5
