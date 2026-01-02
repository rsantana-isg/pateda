# Discrete EDA - Cluster Execution Tools

This directory contains tools for running discrete EDA experiments in parallel on cluster environments.

## Files

### discrete_EDA.py

Main program for running individual EDA experiments with specified parameters.

**Usage:**
```bash
python discrete_EDA.py <seed> <obj_func> <pop_size> <n_gen> <alg>
```

**Arguments:**
- `seed`: Random seed (integer) for reproducibility
- `obj_func`: Objective function name (e.g., 'OneMax-20', 'Deceptive3-30', 'KDeceptive3-30')
- `pop_size`: Population size (integer)
- `n_gen`: Number of generations (integer)
- `alg`: Algorithm name

**Supported Algorithms:**
- Neural EDAs: VAE, GAN, Backdrive, DAE, RBM, DbD
- Traditional EDAs: UMDA, TreeEDA, EBNA, MOA

**Examples:**
```bash
# Run UMDA on OneMax-20 with seed 0
python discrete_EDA.py 0 OneMax-20 80 20 UMDA

# Run VAE on Deceptive3-30 with seed 5
python discrete_EDA.py 5 Deceptive3-30 100 30 VAE

# Run EBNA on KDeceptive3-30 with seed 10
python discrete_EDA.py 10 KDeceptive3-30 120 30 EBNA
```

### launch_discrete_neural_EDA_example.py

Example script demonstrating how to generate commands for cluster execution across multiple seeds, algorithms, and problems.

**Usage:**
```bash
python launch_discrete_neural_EDA_example.py
```

This script generates commands for 600 independent runs (20 seeds × 10 algorithms × 3 problems) suitable for parallel execution on a cluster using job schedulers like SLURM.

## Cluster Execution

### Example SLURM Batch Script (run_nn_EDA.sh)

```bash
#!/bin/bash
#SBATCH --job-name=discrete_eda
#SBATCH --output=logs/eda_%j.out
#SBATCH --error=logs/eda_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

python $@
```

### Submitting Jobs

```bash
# Submit a single job
sbatch run_nn_EDA.sh discrete_EDA.py 0 OneMax-20 80 20 VAE

# Submit multiple jobs (modify launch_discrete_neural_EDA_example.py)
python launch_discrete_neural_EDA_example.py
```

## Algorithm Selection Guide

### Neural EDAs
- **VAE**: Good balance of quality and speed, recommended for most problems
- **GAN**: Can suffer from mode collapse, use with caution
- **Backdrive**: Network inversion approach, good for smooth fitness landscapes
- **DAE**: Simple and effective, fast training
- **RBM**: Classical energy-based model, well-established
- **DbD**: New diffusion-based method, promising alternative

### Traditional EDAs
- **UMDA**: Good for separable problems
- **TreeEDA**: Good balance, efficient learning
- **EBNA**: Better for epistatic problems with dependencies
- **MOA**: Good for problems with local dependencies

## Problem Configurations

| Problem | Variables | Population | Generations | Optimal Fitness |
|---------|-----------|------------|-------------|-----------------|
| OneMax-20 | 20 | 80 | 20 | 20.0 |
| Deceptive3-30 | 30 | 100 | 30 | 10.0 |
| KDeceptive3-30 | 30 | 120 | 30 | 30.0 |

## Output Format

Each run produces:
- Best fitness found
- Success indicator (within 1% of optimal)
- Elapsed time
- Best solution found

Example output:
```
================================================================================
RESULTS
================================================================================
Best Fitness:     20.0000
Optimal Fitness:  20.0000
Gap:              0.0000
Success:          Yes
Elapsed Time:     0.13 seconds
Best Solution:    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
================================================================================
```

## Notes

- Different seeds produce statistically independent runs for robust performance evaluation
- Neural EDAs typically require more computational resources than traditional EDAs
- Adjust population sizes and generation counts based on problem difficulty
- Monitor memory usage for large-scale experiments
