#!/bin/bash

# Job name
#SBATCH --job-name=eda_eval
# Output and error files (one per array/job)
#SBATCH --output=outputs/EDAEVAL_%A_%a.out
#SBATCH --error=outputs/EDAEVAL_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=CPU

# Parameters (positional, matching scripts/gen_eval_eda_benchmark.py):
# $1 = problem      (dataset name in data/eda_datasets/, e.g. Braid_36)
# $2 = train_set    (0, 1 or 2: which split is the training pool)
# $3 = algorithm    (BN learning method, e.g. k2, fi_k2, dmbbn, ...)
# $4 = seed         (integer RNG seed)
# $5 = temperature  (Boltzmann T, e.g. 0.1, 1.0, 10)

# Directory where the runner writes its self-named result file.
OUT_DIR="results/eda_eval_cluster"
mkdir -p "$OUT_DIR" outputs

# Result filename encodes all parameters (self-describing).
OUTPUT_FILE="${OUT_DIR}/eval_${1}_tr${2}_${3}_T${5}_s${4}.dat"

# Always run and overwrite the result (no skip-if-exists check).
#
# IMPORTANT: the runner writes $OUTPUT_FILE itself (atomically).  Do NOT
# redirect this command's stdout into $OUTPUT_FILE — that would fill the
# result file with the container banner and log text.  OUT_DIR is passed so
# the runner writes into the intended directory regardless of the working
# directory; stdout/stderr go to the SLURM logs (see --output/--error above).

echo "Executing: bnd -exec python3 scripts/gen_eval_eda_benchmark.py $1 $2 $3 $4 $5 $OUT_DIR"

# Execute one benchmark combination.
bnd -exec python3 scripts/gen_eval_eda_benchmark.py "$1" "$2" "$3" "$4" "$5" "$OUT_DIR"

# Example usage:
# sbatch slurm/slurm_eval_eda_benchmark.sh Braid_36 0 fi_k2 7 1.0
