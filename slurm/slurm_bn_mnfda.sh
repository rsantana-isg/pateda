#!/bin/bash

# Job name
#SBATCH --job-name=bnmnfda_pbo
# Define output and error files
#SBATCH --output=outputs/BNMNFDA_%A_%a.out
#SBATCH --error=outputs/BNMNFDA_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

# Parameters (seed first, matching scripts/run_bn_mnfda_pbo.py):
# $1 = seed (base seed; runs seed..seed+n_runs-1)
# $2 = n_runs
# $3 = algorithm  (BND8_* bounded, BND8_*_MPC bounded+MPC, MNFDA, MNFDAP)
# $4 = sel        (selection method: FP, BZ, RTS)
# $5 = fid        (PBO function id, 1..25)
# $6 = dim        (16, 64, 625)
# $7 = pop_size
# $8 = n_gen
# $9 = sel_ratio  (truncation selection ratio)

# Output filename encodes all parameters (self-describing results)
OUTPUT_FILE="results_bnmnfda_${3}_${4}_f${5}_dim${6}_${7}_${8}_${9}_${1}.dat"

# Skip if already done (idempotent, safe to re-launch).  The IOH data folder
# results/pbo_bn_mnfda_data_cluster/${3}__${4}_f${5}_dim${6}_s${1}/ is
# additionally checked inside run_bn_mnfda_pbo.py itself.
if [ -f "$OUTPUT_FILE" ]; then
    exit 0
fi

echo "Executing: bnd -exec python3 scripts/run_bn_mnfda_pbo.py $1 $2 $3 $4 $5 $6 $7 $8 $9 > $OUTPUT_FILE"

# Execute the experiment
bnd -exec python3 scripts/run_bn_mnfda_pbo.py $1 $2 $3 $4 $5 $6 $7 $8 $9 > "$OUTPUT_FILE"

# Example usage:
# sbatch slurm/slurm_bn_mnfda.sh 1 5 BND8_A3_fast     BZ 18 64  200  50  0.5
# sbatch slurm/slurm_bn_mnfda.sh 1 5 BND8_A3_fast_MPC BZ 18 64  200  50  0.5
# sbatch slurm/slurm_bn_mnfda.sh 1 1 MNFDAP           BZ 18 625 1000 100 0.5
