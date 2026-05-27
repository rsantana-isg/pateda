#!/bin/bash
#
# SLURM template for one execute_discrete_edas_RW.py run.
#
# Positional arguments mirror the python script (no flags):
#   $1 = n_vars
#   $2 = pop_size
#   $3 = n_gen
#   $4 = selection_ratio
#   $5 = random_seed
#   $6 = alg                (UMDA, TreeEDA, ...)
#   $7 = problem            (SAT | Ising | UBQP)
#   $8 = instance           (full path to the .cnf / .txt instance file)
#
# Output:
#   A self-describing .dat file written to ``results/`` whose name encodes
#   every parameter, making it easy to harvest results in batch.  Files are
#   skipped if they already exist so the script is idempotent.

#SBATCH --job-name=eda_rw
#SBATCH --output=outputs/EDA_RW_%A_%a.out
#SBATCH --error=outputs/EDA_RW_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU

set -euo pipefail

if [ "$#" -ne 8 ]; then
    echo "Usage: sbatch slurm_compare_EDAs_RW.sh n_vars pop_size n_gen selection_ratio seed alg problem instance" >&2
    exit 1
fi

N_VARS="$1"
POP_SIZE="$2"
N_GEN="$3"
SEL_RATIO="$4"
SEED="$5"
ALG="$6"
PROBLEM="$7"
INSTANCE="$8"

# Resolve the repository root so the script works regardless of where SLURM
# spawned us from.  The launcher (launch_compare_EDAs_RW.sh) lives in
# slurm/ alongside this file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p results outputs

# Build a self-describing output filename.  Strip the extension off the
# instance path so the name stays compact.
INSTANCE_BASE="$(basename "${INSTANCE}")"
INSTANCE_TAG="${INSTANCE_BASE%.*}"

OUTPUT_FILE="results/results_EDA_RW_${PROBLEM}_${ALG}_${N_VARS}_${INSTANCE_TAG}_pop${POP_SIZE}_gen${N_GEN}_sel${SEL_RATIO}_seed${SEED}.dat"

# Skip if already done (idempotent re-launches).
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping: ${OUTPUT_FILE} already exists."
    exit 0
fi

# Use the interpreter where pateda is installed (editable install lives
# under python3.11 on this project).  Override with PATEDA_PYTHON if needed.
PYTHON_BIN="${PATEDA_PYTHON:-python3.11}"

echo "Running: ${PYTHON_BIN} scripts/execute_discrete_edas_RW.py ${N_VARS} ${POP_SIZE} ${N_GEN} ${SEL_RATIO} ${SEED} ${ALG} ${PROBLEM} ${INSTANCE}"
"${PYTHON_BIN}" scripts/execute_discrete_edas_RW.py \
    "${N_VARS}" "${POP_SIZE}" "${N_GEN}" "${SEL_RATIO}" "${SEED}" \
    "${ALG}" "${PROBLEM}" "${INSTANCE}" \
    > "${OUTPUT_FILE}"
