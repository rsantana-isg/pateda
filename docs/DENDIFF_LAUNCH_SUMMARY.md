# Dendiff Launch Scripts - Implementation Summary

## Overview

This document summarizes the implementation of launch scripts for comprehensive Dendiff EDA experiments on SLURM clusters.

---

## Files Created

### 1. `launch_dendiff_experiments.py`
**Purpose**: Generate all combinations of Dendiff EDA experiments for SLURM execution

**Based on**: `lanzar_discrete_EDA.py`

**Key Features**:
- Systematically generates all parameter combinations
- Automatically sets variant-dependent parameters
- Supports multiple objective functions and seeds
- Outputs SLURM sbatch commands

**Fixed Parameters**:
```python
n_gen = 250                  # Number of generations
trunc = 0.5                  # Selection ratio (50%)
n_sampling_steps = 20        # Denoising steps during sampling

# Problem-dependent sizes
n = 64 if obj_func == 'HIFF' else 30
p_size = n * 5               # Population size
```

**Variable Parameters**:
```python
variants = ['dendiff_gumbel', 'dendiff_corruption']
activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']
loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
fitness_guided_options = [0, 1]
seeds = np.arange(1, 31)     # 30 independent runs
```

**Variant-Dependent Parameters** (automatically set):

| Variant | sampling_strategy | n_timesteps | temperature | beta_start | beta_end |
|---------|------------------|-------------|-------------|------------|----------|
| dendiff_gumbel | gumbel | 100 | 1.0 | 0.0001 | 0.3 |
| dendiff_corruption | corruption | 50 | 0.5 | 0.01 | 0.5 |

**Total Combinations per Objective Function**:
- Per seed: 2 variants × 4 activations × 4 losses × 2 fitness_guided = **64 combinations**
- For 30 seeds: 64 × 30 = **1,920 experiments**

---

### 2. `slurm_dendiff.sh`
**Purpose**: SLURM batch script for running individual Dendiff EDA experiments

**Based on**: `slurm_pateda.sh`

**Key Features**:
- Handles 17 parameters (script path + 16 Dendiff parameters)
- Descriptive output filenames
- SLURM configuration for cluster execution
- Error and output logging

**Parameters** (in order):
1. Script path (`examples/discrete_Dendiff_EDA.py`)
2. seed
3. obj_func
4. n (number of variables)
5. pop_size
6. n_gen
7. trunc
8. variant
9. sampling_strategy
10. activation
11. loss
12. n_timesteps
13. n_sampling_steps
14. fitness_guided
15. temperature
16. beta_start
17. beta_end

**Output Filename Format**:
```
results_dendiff_<obj_func>_n<n>_<variant>_<activation>_<loss>_fg<fitness_guided>_seed<seed>.dat
```

**Examples**:
```
results_dendiff_OneMax_n30_dendiff_gumbel_relu_mse_fg0_seed1.dat
results_dendiff_HIFF_n64_dendiff_corruption_tanh_weighted_mse_fg1_seed15.dat
```

**SLURM Configuration**:
```bash
#SBATCH --job-name=dendiff_eda
#SBATCH --output=outputs/dendiff_%A_%a.out
#SBATCH --error=outputs/dendiff_%A_%a.err
#SBATCH --ntasks=1
# SBATCH --mem-per-cpu=8G
```

---

### 3. `DENDIFF_EXPERIMENTS_README.md`
**Purpose**: Comprehensive documentation for using the launch scripts

**Contents**:
- File descriptions and usage
- Step-by-step instructions
- Customization options
- Result parsing and analysis
- Troubleshooting guide
- Experiment design recommendations
- Performance estimates

**Key Sections**:
1. **Usage**: Step-by-step guide from generation to result analysis
2. **Customization**: How to modify parameters and reduce experiments
3. **Experiment Design**: Baseline, full factorial, and ablation study recommendations
4. **Analysis**: Commands for parsing results and extracting metrics
5. **Troubleshooting**: Common issues and solutions

---

## Usage Workflow

### Step 1: Generate Experiment Commands

```bash
python launch_dendiff_experiments.py > run_dendiff_experiments.sh
```

This creates a shell script with all SLURM sbatch commands.

### Step 2: Review Generated Commands

```bash
# Check total number of experiments
wc -l run_dendiff_experiments.sh

# View first few commands
head -10 run_dendiff_experiments.sh

# View specific examples
grep "dendiff_gumbel.*relu.*mse.*fg0" run_dendiff_experiments.sh | head -5
```

### Step 3: Submit Jobs

```bash
# Submit all jobs
bash run_dendiff_experiments.sh

# Or submit in batches
head -n 100 run_dendiff_experiments.sh | bash  # First 100 jobs
```

### Step 4: Monitor Execution

```bash
# Check job queue
squeue -u $USER

# Count running jobs
squeue -u $USER | wc -l

# Check specific job status
squeue -j <job_id>
```

### Step 5: Collect Results

```bash
# List all completed results
ls results_dendiff_*.dat

# Count completed experiments
ls results_dendiff_*.dat | wc -l

# Extract best fitness for all runs
for f in results_dendiff_*.dat; do
    echo -n "$f: "
    grep "Best Fitness:" $f | tail -1 | awk '{print $3}'
done > all_results.txt
```

---

## Example Commands Generated

### Example 1: Dendiff-Gumbel with ReLU and MSE (Baseline)
```bash
sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
    1 OneMax 30 150 250 0.5 \
    dendiff_gumbel gumbel relu mse \
    100 20 0 1.0 0.0001 0.3
```

### Example 2: Dendiff-Corruption with Tanh and Weighted MSE (Enhanced)
```bash
sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
    1 OneMax 30 150 250 0.5 \
    dendiff_corruption corruption tanh weighted_mse \
    50 20 0 0.5 0.01 0.5
```

### Example 3: Dendiff-Gumbel with Fitness Guidance (Advanced)
```bash
sbatch slurm_dendiff.sh examples/discrete_Dendiff_EDA.py \
    1 HIFF 64 320 250 0.5 \
    dendiff_gumbel gumbel elu ranking \
    100 20 1 1.0 0.0001 0.3
```

---

## Experiment Scale and Performance

### Default Configuration (1 Objective Function)

**Parameter Space**:
- 2 variants
- 4 activation functions
- 4 loss functions
- 2 fitness guidance options
- 30 seeds

**Total Experiments**: 2 × 4 × 4 × 2 × 30 = **1,920 experiments**

**Time Estimates**:
- Per experiment: ~30-60 seconds
- Sequential: ~16-32 hours
- Parallel (10 nodes): ~1.6-3.2 hours
- Parallel (100 nodes): ~10-20 minutes

**Storage**:
- Per result file: ~10-50 KB
- Total (1,920 files): ~20-100 MB

### Multiple Objective Functions

For 6 objective functions (OneMax, KDeceptive3, Deceptive3, HIFF, KDeceptive5, FC5):
- **Total experiments**: 1,920 × 6 = **11,520**
- **Sequential time**: ~96-192 hours (~4-8 days)
- **Parallel time** (100 nodes): ~1-2 hours

---

## Customization Examples

### Reduced Experiment Set (Quick Testing)

Modify `launch_dendiff_experiments.py`:

```python
# Test with fewer parameters
variants = ['dendiff_gumbel']           # Only 1 variant
activations = ['relu']                  # Only 1 activation
loss_functions = ['mse', 'weighted_mse'] # Only 2 losses
fitness_guided_options = [0, 1]         # Both options
seeds = np.arange(1, 6)                 # Only 5 seeds

# Result: 1 × 1 × 2 × 2 × 5 = 20 experiments
```

### Ablation Study (Activation Functions)

```python
# Compare all activation functions with baseline settings
variants = ['dendiff_gumbel']
activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']  # All 4
loss_functions = ['mse']                # Fixed
fitness_guided_options = [0]            # Fixed
seeds = np.arange(1, 31)                # Full seeds

# Result: 1 × 4 × 1 × 1 × 30 = 120 experiments
```

### Ablation Study (Loss Functions)

```python
# Compare all loss functions with baseline settings
variants = ['dendiff_gumbel', 'dendiff_corruption']  # Both variants
activations = ['relu']                               # Fixed
loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']  # All 4
fitness_guided_options = [0]                         # Fixed
seeds = np.arange(1, 31)                             # Full seeds

# Result: 2 × 1 × 4 × 1 × 30 = 240 experiments
```

### Fitness Guidance Study

```python
# Compare impact of fitness guidance
variants = ['dendiff_gumbel', 'dendiff_corruption']
activations = ['relu', 'elu']                       # 2 activations
loss_functions = ['mse', 'weighted_mse']            # 2 losses
fitness_guided_options = [0, 1]                     # Both options
seeds = np.arange(1, 31)                            # Full seeds

# Result: 2 × 2 × 2 × 2 × 30 = 480 experiments
```

---

## Result Analysis

### Extract Success Rates

```bash
# Count successful runs
grep "Success:.*Yes" results_dendiff_*.dat | wc -l

# Success rate by variant
for variant in dendiff_gumbel dendiff_corruption; do
    total=$(ls results_dendiff_*_${variant}_*.dat | wc -l)
    success=$(grep -l "Success:.*Yes" results_dendiff_*_${variant}_*.dat | wc -l)
    echo "$variant: $success/$total ($(echo "scale=2; $success*100/$total" | bc)%)"
done
```

### Extract Convergence Statistics

```bash
# Average best fitness by configuration
for variant in dendiff_gumbel dendiff_corruption; do
    for act in leaky_relu relu tanh sigmoid; do
        for loss in mse weighted_mse ranking huber; do
            for fg in 0 1; do
                pattern="results_dendiff_*_${variant}_${act}_${loss}_fg${fg}_*.dat"
                if ls $pattern 2>/dev/null | grep -q .; then
                    avg=$(grep "Best Fitness:" $pattern | awk '{sum+=$3; count++} END {print sum/count}')
                    echo "$variant,$act,$loss,$fg,$avg"
                fi
            done
        done
    done
done > summary_stats.csv
```

### Compare Variants

```bash
# Compare gumbel vs corruption
echo "Gumbel average:"
grep "Best Fitness:" results_dendiff_*_dendiff_gumbel_*.dat | \
    awk '{sum+=$3; count++} END {print sum/count}'

echo "Corruption average:"
grep "Best Fitness:" results_dendiff_*_dendiff_corruption_*.dat | \
    awk '{sum+=$3; count++} END {print sum/count}'
```

---

## Integration with Existing Workflow

These scripts complement the existing PATEDA experiment infrastructure:

**Existing**:
- `lanzar_discrete_EDA.py` → General EDA experiments
- `slurm_pateda.sh` → SLURM script for discrete_EDA.py

**New (Dendiff-specific)**:
- `launch_dendiff_experiments.py` → Dendiff EDA experiments
- `slurm_dendiff.sh` → SLURM script for discrete_Dendiff_EDA.py

**Relationship**:
```
lanzar_discrete_EDA.py  ──┐
                          ├──→ discrete_EDA.py (all EDAs including Dendiff)
slurm_pateda.sh        ──┘

launch_dendiff_experiments.py ──┐
                                ├──→ discrete_Dendiff_EDA.py (Dendiff only, comprehensive parameters)
slurm_dendiff.sh              ──┘
```

**Use Cases**:
- `discrete_EDA.py` + `lanzar_discrete_EDA.py`: Compare Dendiff with other EDAs (VAE, GAN, DbD, etc.)
- `discrete_Dendiff_EDA.py` + `launch_dendiff_experiments.py`: Deep dive into Dendiff variants and parameters

---

## Quality Assurance

### Verification Tests

```bash
# 1. Check syntax
python -m py_compile launch_dendiff_experiments.py
bash -n slurm_dendiff.sh

# 2. Generate test commands
python launch_dendiff_experiments.py | head -10

# 3. Count total experiments
python launch_dendiff_experiments.py | wc -l

# 4. Verify parameter values
python launch_dendiff_experiments.py | grep "dendiff_gumbel.*100.*1.0.*0.0001.*0.3" | head -1
python launch_dendiff_experiments.py | grep "dendiff_corruption.*50.*0.5.*0.01.*0.5" | head -1
```

### Expected Output Counts

For default configuration (1 objective function):
```bash
# Total commands
python launch_dendiff_experiments.py | wc -l
# Expected: 1920 (2 × 4 × 4 × 2 × 30)

# Gumbel variant commands
python launch_dendiff_experiments.py | grep dendiff_gumbel | wc -l
# Expected: 960 (half of total)

# Corruption variant commands
python launch_dendiff_experiments.py | grep dendiff_corruption | wc -l
# Expected: 960 (half of total)

# Commands with fitness guidance enabled
python launch_dendiff_experiments.py | grep " 1 1.0\| 1 0.5" | wc -l
# Expected: 960 (half of total)
```

---

## Summary

The Dendiff launch scripts provide:

✅ **Systematic experiment generation** for all parameter combinations
✅ **Variant-specific parameter handling** (gumbel vs corruption)
✅ **SLURM cluster integration** for parallel execution
✅ **Descriptive output filenames** for easy result identification
✅ **Comprehensive documentation** for usage and customization
✅ **Flexible configuration** for different experiment designs
✅ **Integration** with existing PATEDA workflow

This enables comprehensive evaluation of Dendiff EDA variants across different:
- Objective functions
- Activation functions
- Loss functions
- Fitness guidance strategies
- Multiple independent runs (seeds)

The scripts make it easy to run thousands of experiments systematically and analyze results to identify best configurations for different problem types.

---

**Created**: 2026-01-07
**Version**: 1.0
**Status**: Production Ready
