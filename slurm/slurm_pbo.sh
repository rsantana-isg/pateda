#!/bin/bash

# Job name
#SBATCH --job-name=pbo_eda
# Define output and error files
#SBATCH --output=outputs/PBO_%A_%a.out
#SBATCH --error=outputs/PBO_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

# Parameters (seed first, matching scripts/run_pbo_eda.py):
# $1 = seed (base seed; runs seed..seed+n_runs-1)
# $2 = n_runs
# $3 = algorithm (UMDA, TreeEDA, EBNA, ...)
# $4 = fid (PBO function id, 1..25)
# $5 = dim (16, 64, 100, ...)
# $6 = pop_size
# $7 = n_gen
# $8 = sel_ratio (truncation selection ratio)

# Output filename encodes all parameters (self-describing results)
OUTPUT_FILE="results_pbo_${3}_f${4}_dim${5}_${6}_${7}_${8}_${1}.dat"

# Skip if already done (idempotent, safe to re-launch).
# The IOH data folder results/pbo_data_cluster/${3}_f${4}_dim${5}_s${1}/
# is additionally checked inside run_pbo_eda.py itself.
if [ -f "$OUTPUT_FILE" ]; then
    exit 0
fi

echo "Executing: bnd -exec python3 scripts/run_pbo_eda.py $1 $2 $3 $4 $5 $6 $7 $8 > $OUTPUT_FILE"

# Execute the experiment
bnd -exec python3 scripts/run_pbo_eda.py $1 $2 $3 $4 $5 $6 $7 $8 > "$OUTPUT_FILE"

# Example usage:
# sbatch slurm/slurm_pbo.sh 1 5 UMDA 19 100 200 50 0.5
