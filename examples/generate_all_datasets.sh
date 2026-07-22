#!/bin/bash
#
# generate_all_datasets.sh
# ========================
# Run run_eda_search.py on every problem/size in the task description and
# produce the structure + samples files for each one.  Runs several problems
# in parallel (one single-threaded Python process per job).
#
# Fixed experimental parameters (as requested):
#   population size        POP  = 1000
#   number of generations  GEN  = 30
#   number of repetitions  REPS = 5
#   dataset size (SAMP)    SAMP = 3000
#
# Usage:
#   ./generate_all_datasets.sh [EDA] [OUTPUT_DIR] [BASE_SEED] [JOBS]
#
#   EDA         EDA algorithm to use (default: UMDA).  Supported: UMDA, TreeEDA,
#               EBNA, MOA, MN-FDA, MN-FDAG, MK-EDA1, MK-EDA2, MK-EDA3,
#               MT-EDA2, MT-EDA3.
#   OUTPUT_DIR  Directory for the .dat files (default: ./eda_datasets).
#   BASE_SEED   Base random seed (default: 1).
#   JOBS        Number of problems to run concurrently (default: 10).
#
# Each Python process is pinned to a single BLAS/OpenMP thread so that JOBS
# concurrent jobs occupy ~JOBS cores without oversubscription.
#
# The run is idempotent: a (problem, size) whose samples file already exists in
# OUTPUT_DIR is skipped, so the script can be safely re-run after interruption.

set -u

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
export POP=1000
export GEN=30
export REPS=5
export SAMP=3000

export EDA="${1:-UMDA}"
OUTPUT_DIR="${2:-eda_datasets}"
export BASE_SEED="${3:-1}"
JOBS="${4:-10}"

# Python 3.11 is required (pateda is installed for 3.11; the default python3 is 3.8).
export PYTHON="${PYTHON:-python3.11}"

# Keep each process single-threaded so JOBS processes use ~JOBS cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUNNER="${SCRIPT_DIR}/run_eda_search.py"

# Problem -> list of sizes (from the task description).
PROBLEMS=(
    "OneMax        36 64 100 256"
    "Trap          36 64 100 256"
    "Deceptive3    39 66 102 258"
    "Checkerboard  36 64 100 256"
    "EqualProducts 36 64 100 256"
    "Ising         36 64 100 256"
    "UBQP          50 100"
    "MaxClique     30 60 125"
    "Braid         36 64 100 256"
)

# --------------------------------------------------------------------------- #
# Per-job worker (invoked in parallel by xargs)
# --------------------------------------------------------------------------- #
run_one() {
    local func="$1" n="$2"
    local samples="${func}_${n}_${EDA}_samples.dat"
    if [ -f "${samples}" ]; then
        echo ">> SKIP  ${func} n=${n} (already done)"
        return 0
    fi
    local log="logs/${func}_${n}.log"
    echo ">> RUN   ${func} n=${n}"
    if "${PYTHON}" "${RUNNER}" "${func}" "${n}" "${EDA}" \
            "${POP}" "${GEN}" "${REPS}" "${SAMP}" "${BASE_SEED}" > "${log}" 2>&1; then
        echo ">> OK    ${func} n=${n}"
    else
        echo "!! FAIL  ${func} n=${n} (see ${log})"
    fi
}
export -f run_one

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

echo "======================================================================"
echo "Generating EDA search datasets (parallel)"
echo "  EDA           : ${EDA}"
echo "  Population     : ${POP}"
echo "  Generations    : ${GEN}"
echo "  Repetitions    : ${REPS}"
echo "  Dataset (SAMP) : ${SAMP}"
echo "  Base seed      : ${BASE_SEED}"
echo "  Parallel jobs  : ${JOBS}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "======================================================================"

# run_eda_search.py writes the .dat files in the current directory, so run from
# OUTPUT_DIR.
cd "${OUTPUT_DIR}" || exit 1
mkdir -p logs

# Build the flat list of "func n" jobs.
JOB_LIST=()
for entry in "${PROBLEMS[@]}"; do
    read -r func sizes <<< "${entry}"
    for n in ${sizes}; do
        JOB_LIST+=("${func} ${n}")
    done
done

# Dispatch JOBS at a time.  -L1 passes each line's words as args to run_one.
printf '%s\n' "${JOB_LIST[@]}" \
    | xargs -P "${JOBS}" -L1 bash -c 'run_one "$@"' _

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo "======================================================================"
n_done=0
n_missing=0
for entry in "${PROBLEMS[@]}"; do
    read -r func sizes <<< "${entry}"
    for n in ${sizes}; do
        if [ -f "${func}_${n}_${EDA}_samples.dat" ]; then
            n_done=$((n_done + 1))
        else
            echo "MISSING: ${func} n=${n}"
            n_missing=$((n_missing + 1))
        fi
    done
done
echo "Datasets present: ${n_done}   missing: ${n_missing}"
echo "Files written to: ${OUTPUT_DIR}"
echo "======================================================================"

[ "${n_missing}" -eq 0 ]
