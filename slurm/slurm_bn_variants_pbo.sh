#!/bin/bash

# Job name
#SBATCH --job-name=bnvar_pbo
# Define output and error files
#SBATCH --output=outputs/BNVAR_%A_%a.out
#SBATCH --error=outputs/BNVAR_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

# Parameters (seed first, matching scripts/run_bn_variant_pbo.py):
# $1 = seed (base seed; runs seed..seed+n_runs-1)
# $2 = n_runs
# $3 = algorithm (EBNA_BIC, EBNA_K2, EBNA_PC, LFDA, BOA, SARTRE, A1_dt..A5_ndg)
# $4 = sel      (selection method: FP, BZ, RTS)
# $5 = fid      (PBO function id, 1..25)
# $6 = dim      (16, 64, 100, 625)
# $7 = pop_size
# $8 = n_gen
# $9 = sel_ratio (truncation selection ratio)

# Output filename encodes all parameters (self-describing results)
OUTPUT_FILE="results_bnvar_${3}_${4}_f${5}_dim${6}_${7}_${8}_${9}_${1}.dat"

# Skip if already done (idempotent, safe to re-launch).  The IOH data folder
# results/pbo_bnvariants_data_cluster/${3}__${4}_f${5}_dim${6}_s${1}/ is
# additionally checked inside run_bn_variant_pbo.py itself.
if [ -f "$OUTPUT_FILE" ]; then
    exit 0
fi

echo "Executing: bnd -exec python3 scripts/run_bn_variant_pbo.py $1 $2 $3 $4 $5 $6 $7 $8 $9 > $OUTPUT_FILE"

# Execute the experiment
bnd -exec python3 scripts/run_bn_variant_pbo.py $1 $2 $3 $4 $5 $6 $7 $8 $9 > "$OUTPUT_FILE"

# Example usage:
# sbatch slurm/slurm_bn_variants_pbo.sh 1 5 A5_ndg RTS 19 100 200 50 0.5
