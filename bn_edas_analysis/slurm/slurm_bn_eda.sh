#!/bin/bash

# Job name
#SBATCH --job-name=bn_eda
# Output and error files (one per job)
#SBATCH --output=outputs/BNEDA_%A_%a.out
#SBATCH --error=outputs/BNEDA_%A_%a.err

# Resource requirements
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=47:59:00
#SBATCH --partition=CPU

# One BN-EDA run: one BN learning algorithm on one problem for one seed.
#
# Parameters (positional, matching scripts/run_bn_eda.py -- SEED FIRST):
#   $1 = seed         (integer RNG seed)
#   $2 = problem      ("<Family>_<n>" as in eda_cluster_results.csv, e.g. Ising_100)
#   $3 = algorithm    (bayes_nets BN learning method, e.g. bic, sartre, univ_bn)
#
# The remaining EDA settings are fixed inside run_bn_eda.py:
#   truncation selection T=0.5, Boltzmann-weighted learning, 100 generations,
#   pop_size = 10 * n, max_parents = 5.

OUT_DIR="results/bn_eda_cluster"
mkdir -p "$OUT_DIR" outputs

# Self-describing result file (encodes all parameters).
OUTPUT_FILE="${OUT_DIR}/bneda_${2}_${3}_s${1}.json"

# Idempotent: skip a run whose result already exists (safe to re-launch).
if [ -f "$OUTPUT_FILE" ]; then
    echo "SKIP (exists): $OUTPUT_FILE"
    exit 0
fi

# IMPORTANT: run_bn_eda.py writes $OUTPUT_FILE itself.  Do NOT redirect this
# command's stdout into the result file -- stdout/stderr go to the SLURM logs.
echo "Executing: bnd -exec python3 scripts/run_bn_eda.py $1 $2 $3 $OUT_DIR"

# `bnd -exec` provides the container/environment with pateda + bayes_nets.
bnd -exec python3 scripts/run_bn_eda.py "$1" "$2" "$3" "$OUT_DIR"

# Example usage:
# sbatch slurm/slurm_bn_eda.sh 1 Ising_100 bic
