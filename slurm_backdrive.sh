#!/bin/bash

# Job name
#SBATCH --job-name=discrete_backdrive_eda

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/M_%A_%a.out
#SBATCH --error=outputs/M_%A_%a.err

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



# Extract positional parameters for filename
# $1: script, $2: seed, $3: obj_func, $4: n, $5: pop_size, $6: n_gen, $7: trunc
# Additional parameters are optional flags that should be included in filename

# Save all original arguments for execution
ORIG_ARGS=("$@")

# Save first 7 positional parameters for later use
SCRIPT="$1"
SEED="$2"
OBJ_FUNC="$3"
N="$4"
POP_SIZE="$5"
N_GEN="$6"
TRUNC="$7"

# Parse optional flags from command line arguments
VARIANT=""
INIT=""
LOSS=""
ACTIVATION=""
WEIGHT_TRANSFER=""
EARLY_STOPPING=""
SURROGATE_FILTERING=""

shift 7  # Skip the first 7 positional parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --init)
            INIT="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --activation)
            ACTIVATION="$2"
            shift 2
            ;;
        --weight-transfer)
            WEIGHT_TRANSFER="wt"
            shift
            ;;
        --early-stopping)
            EARLY_STOPPING="es"
            shift
            ;;
        --surrogate-filtering)
            SURROGATE_FILTERING="sf"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Build flag string for filename
FLAG_STRING=""
[[ -n "$VARIANT" ]] && FLAG_STRING="${FLAG_STRING}_${VARIANT}"
[[ -n "$INIT" ]] && FLAG_STRING="${FLAG_STRING}_${INIT}"
[[ -n "$LOSS" ]] && FLAG_STRING="${FLAG_STRING}_${LOSS}"
[[ -n "$ACTIVATION" ]] && FLAG_STRING="${FLAG_STRING}_${ACTIVATION}"
[[ -n "$WEIGHT_TRANSFER" ]] && FLAG_STRING="${FLAG_STRING}_${WEIGHT_TRANSFER}"
[[ -n "$EARLY_STOPPING" ]] && FLAG_STRING="${FLAG_STRING}_${EARLY_STOPPING}"
[[ -n "$SURROGATE_FILTERING" ]] && FLAG_STRING="${FLAG_STRING}_${SURROGATE_FILTERING}"

# Create output filename with positional parameters and flags
OUTPUT_FILE="results_discrete_backdrive_${OBJ_FUNC}_${N}_${POP_SIZE}_${N_GEN}_${TRUNC}_${SEED}${FLAG_STRING}.dat"

echo "bnd -exec python3 \"\$@\" > ${OUTPUT_FILE}"
bnd -exec python3 "${ORIG_ARGS[@]}" > "${OUTPUT_FILE}"


# sbatch slurm_backdrive.sh examples/discrete_Backdrive_EDA.py 111 OneMax 30 150 250 0.5 --variant backdrive --init random --loss mse --activation relu
