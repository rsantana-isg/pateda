#!/bin/bash
#
# Launcher for the real-world discrete EDA benchmark.
#
# Prints one ``sbatch slurm_compare_EDAs_RW.sh ...`` line per
# ``(algorithm, problem, instance, seed)`` combination required by the
# experimental matrix described in docs/EXPERIMENT_PLAN.md (paraphrased):
#
#   * SAT    : ``uf$n-$j.cnf``  for n in {20, 50, 75, 100}
#                                and j in {01..09, 010}        (10 instances/n)
#   * Ising  : ``SG_$n_$j.txt`` for n in {16, 36, 64, 100, 256, 400}
#                                and j in {1, 2, 3, 4}          ( 4 instances/n)
#   * UBQP   : ``bqp$n.txt``    for n in {50, 100, 250, 500}    ( 1 instance/n)
#
# Each combination is repeated for seeds in ``np.arange(1, 21)`` (i.e. 1..20).
#
# The script is *non-destructive*: it only prints commands.  Pipe its output
# to ``bash`` (or ``sh``) to actually submit, e.g.::
#
#     bash slurm/launch_compare_EDAs_RW.sh | head -5    # inspect first 5
#     bash slurm/launch_compare_EDAs_RW.sh | bash       # submit them all
#
# To submit only a subset (e.g. one problem family), grep first::
#
#     bash slurm/launch_compare_EDAs_RW.sh | grep " SAT " | bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths so the launcher works no matter where it is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/slurm_compare_EDAs_RW.sh"

SAT_DIR="${REPO_ROOT}/packages/pateda/src/pateda/functions/SAT_instances"
ISING_DIR="${REPO_ROOT}/packages/pateda/src/pateda/functions/Ising_Instances"
UBQP_DIR="${REPO_ROOT}/packages/pateda/src/pateda/functions/UBQP_Instances"

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

# Every plug-and-play discrete EDA that lives in scripts/execute_discrete_edas_RW.py
ALGORITHMS=(
    UMDA BMDA TreeEDA TreeEDA-r MIMIC PBIL EBNA BOA AffEDA MKEDA MTED
    MNFDA MNFDAR MNFDAG MNFDAGR MOA FDA BSC
)

# Seeds 1..20 (matches the np.arange(1, 21) convention).
SEEDS=$(seq 1 20)

# Problem sizes.
SAT_SIZES=(20 50 75 100)
SAT_INDICES=(01 02 03 04 05 06 07 08 09 010)   # match on-disk filenames
ISING_SIZES=(16 36 64 100 256 400)
ISING_INDICES=(1 2 3 4)
UBQP_SIZES=(50 100 250 500)

# Selection ratio is fixed at 0.5; pop_size follows the N = 5 * n_vars
# convention used elsewhere in the project; n_gen is fixed at 250.
SEL_RATIO=0.5
N_GEN=250
POP_FACTOR=5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

emit_jobs () {
    # $1 = problem (SAT | Ising | UBQP)
    # $2 = n_vars
    # $3 = full path to instance
    local problem="$1"
    local n_vars="$2"
    local instance="$3"

    local pop_size=$(( n_vars * POP_FACTOR ))

    for alg in "${ALGORITHMS[@]}"; do
        for seed in ${SEEDS}; do
            echo "sbatch ${SBATCH_SCRIPT} ${n_vars} ${pop_size} ${N_GEN} ${SEL_RATIO} ${seed} ${alg} ${problem} ${instance}"
        done
    done
}

# ---------------------------------------------------------------------------
# SAT
# ---------------------------------------------------------------------------
for n in "${SAT_SIZES[@]}"; do
    for j in "${SAT_INDICES[@]}"; do
        instance="${SAT_DIR}/uf${n}-${j}.cnf"
        if [ ! -f "${instance}" ]; then
            echo "# missing: ${instance}" >&2
            continue
        fi
        emit_jobs SAT "${n}" "${instance}"
    done
done

# ---------------------------------------------------------------------------
# Ising
# ---------------------------------------------------------------------------
for n in "${ISING_SIZES[@]}"; do
    for j in "${ISING_INDICES[@]}"; do
        instance="${ISING_DIR}/SG_${n}_${j}.txt"
        if [ ! -f "${instance}" ]; then
            echo "# missing: ${instance}" >&2
            continue
        fi
        emit_jobs Ising "${n}" "${instance}"
    done
done

# ---------------------------------------------------------------------------
# UBQP
# ---------------------------------------------------------------------------
for n in "${UBQP_SIZES[@]}"; do
    instance="${UBQP_DIR}/bqp${n}.txt"
    if [ ! -f "${instance}" ]; then
        echo "# missing: ${instance}" >&2
        continue
    fi
    emit_jobs UBQP "${n}" "${instance}"
done
