# Dendiff Experiments Launch Scripts

This directory contains scripts for launching comprehensive Dendiff EDA experiments on SLURM clusters.

## Files

### 1. `launch_dendiff_experiments.py`

Python script that generates all combinations of Dendiff EDA experiments.

**Fixed Parameters:**
- Generations: 250
- Truncation Percent: 0.5
- n_sampling_steps: 20
- Problem size:
  - n = 64 for HIFF
  - n = 30 for all other problems
  - pop_size = n × 5

**Variable Parameters:**
- **variants**: `dendiff_gumbel`, `dendiff_corruption`
- **activations**: `leaky_relu`, `relu`, `tanh`, `sigmoid`
- **loss_functions**: `mse`, `weighted_mse`, `ranking`, `huber`
- **fitness_guided**: `0` (no), `1` (yes)
- **seeds**: 1-30 (30 independent runs)

**Variant-Dependent Parameters:**

For `dendiff_gumbel`:
- sampling_strategy: `gumbel`
- n_timesteps: `100`
- temperature: `1.0`
- beta_start: `0.0001`
- beta_end: `0.3`

For `dendiff_corruption`:
- sampling_strategy: `corruption`
- n_timesteps: `50`
- temperature: `0.5`
- beta_start: `0.01`
- beta_end: `0.5`

### 2. `slurm_dendiff.sh`

SLURM batch script for running individual Dendiff EDA experiments.

**Parameters (17 total):**
1. Script path (`examples/discrete_Dendiff_EDA.py`)
2. Seed
3. Objective function
4. Number of variables (n)
5. Population size
6. Number of generations
7. Truncation ratio
8. Variant
9. Sampling strategy
10. Activation function
11. Loss function
12. Number of timesteps
13. Number of sampling steps
14. Fitness guided (0/1)
15. Temperature
16. Beta start
17. Beta end

**Output Files:**
Results are saved with descriptive filenames:
```
results_dendiff_<obj_func>_n<n>_<variant>_<activation>_<loss>_fg<fitness_guided>_seed<seed>.dat
```

Example:
```
results_dendiff_OneMax_n30_dendiff_gumbel_relu_mse_fg0_seed1.dat
```

## Usage

### Step 1: Configure Objective Functions

Edit `launch_dendiff_experiments.py` to select objective functions:

```python
# Test all functions
obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']

# Or test a subset
obj_functions = ['OneMax', 'HIFF']
```

### Step 2: Generate SLURM Commands

Run the launch script to generate all experiment combinations:

```bash
python launch_dendiff_experiments.py > run_dendiff_experiments.sh
```

This creates a shell script with all SLURM batch job submissions.

### Step 3: Review Generated Commands

Check the generated script:

```bash
head run_dendiff_experiments.sh
wc -l run_dendiff_experiments.sh  # Count total experiments
```

### Step 4: Submit Jobs to SLURM

```bash
# Submit all jobs
bash run_dendiff_experiments.sh

# Or submit interactively (review each before submitting)
bash -x run_dendiff_experiments.sh

# Or submit in batches
head -n 100 run_dendiff_experiments.sh | bash  # First 100 jobs
```

### Step 5: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel job if needed
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Step 6: Check Results

Output files are in the current directory:

```bash
# List all results
ls results_dendiff_*.dat

# Check specific result
cat results_dendiff_OneMax_n30_dendiff_gumbel_relu_mse_fg0_seed1.dat

# Count completed experiments
ls results_dendiff_*.dat | wc -l
```

## Experiment Combinations

**Total experiments per seed:**
- 2 variants × 4 activations × 4 loss functions × 2 fitness_guided options = 64 combinations
- For each objective function

**Total experiments for 30 seeds:**
- 64 combinations × 30 seeds = 1,920 experiments per objective function

**Example for 6 objective functions:**
- 1,920 × 6 = 11,520 total experiments

## Customization

### Reduce Number of Experiments

#### Option 1: Fewer Seeds
```python
seeds = np.arange(1, 11)  # Only 10 seeds instead of 30
```

#### Option 2: Subset of Parameters
```python
# Only test specific activations
activations = ['relu', 'elu']

# Only test specific loss functions
loss_functions = ['mse', 'weighted_mse']

# Only test without fitness guidance
fitness_guided_options = [0]
```

#### Option 3: Selected Combinations
Modify the loop to include only specific combinations:

```python
# Example: Only test baseline and best configurations
for variant in variants:
    for activation in ['relu']:  # Only ReLU
        for loss in ['mse', 'weighted_mse']:  # Only these losses
            for fitness_guided in [0, 1]:  # Both options
                # Generate command...
```

### Modify Fixed Parameters

Edit these constants in `launch_dendiff_experiments.py`:

```python
n_gen = 250              # Increase for longer runs
trunc = 0.5              # Adjust selection pressure
n_sampling_steps = 20    # Increase for better quality
```

### Add New Objective Functions

```python
obj_functions = [
    'OneMax',
    'HIFF',
    'Polytree3',
    'FC3',
    # Add more...
]
```

For functions requiring specific n values, add conditions:

```python
if obj_func == 'HIFF':
    n = 64  # Must be power of 2
elif obj_func == 'Polytree5':
    n = 30  # Must be multiple of 5
elif obj_func == 'FC3':
    n = 30  # Must be multiple of 5
else:
    n = 30  # Default
```

## Output File Analysis

Each output file contains:
- Configuration summary
- Generation-by-generation best fitness
- Final results (best fitness, gap, success, time, solution)

Example output structure:
```
================================================================================
DISCRETE DENDIFF EDA - Configuration
================================================================================
Seed:               1
Problem:            OneMax
Variables:          30
Optimal Fitness:    30.0
Population Size:    150
Generations:        250
Variant:            dendiff_gumbel
Activation:         relu
Loss Function:      mse
...

Generation 0: Best Fitness = 18.0000
Generation 1: Best Fitness = 20.0000
...
Generation 250: Best Fitness = 30.0000

================================================================================
RESULTS
================================================================================
Best Fitness:     30.0000
Optimal Fitness:  30.0000
Gap:              0.0000
Success:          Yes
Elapsed Time:     45.23 seconds
Best Solution:    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]...
================================================================================
```

## Parsing Results

### Extract Best Fitness

```bash
# For a single file
grep "Best Fitness:" results_dendiff_OneMax_n30_dendiff_gumbel_relu_mse_fg0_seed1.dat

# For all files
for f in results_dendiff_*.dat; do
    echo -n "$f: "
    grep "Best Fitness:" $f | tail -1 | awk '{print $3}'
done
```

### Extract Success Rate

```bash
# Count successful runs
grep "Success:.*Yes" results_dendiff_*.dat | wc -l

# Total runs
ls results_dendiff_*.dat | wc -l
```

### Extract Convergence

```bash
# Get all generation-by-generation progress for a file
grep "Generation.*:" results_dendiff_OneMax_n30_dendiff_gumbel_relu_mse_fg0_seed1.dat
```

## Troubleshooting

### Jobs Not Starting

Check SLURM status:
```bash
squeue -u $USER
sinfo  # Check cluster status
```

### Jobs Failing

Check error files:
```bash
cat outputs/dendiff_<job_id>.err
```

Common issues:
- Missing dependencies (numpy, torch)
- Insufficient memory (increase `--mem-per-cpu`)
- GPU not available (remove GPU requirement or request CPU)

### Missing Output Files

Check if job completed:
```bash
squeue -j <job_id>
cat outputs/dendiff_<job_id>.out
```

### Insufficient Memory

Increase memory in `slurm_dendiff.sh`:
```bash
#SBATCH --mem-per-cpu=16G  # Increase from 8G
```

## Experiment Design Recommendations

### Baseline Comparison

First run baseline experiments:
```python
# Only standard settings
variants = ['dendiff_gumbel', 'dendiff_corruption']
activations = ['relu']
loss_functions = ['mse']
fitness_guided_options = [0]
seeds = np.arange(1, 31)
```

This gives 2 × 1 × 1 × 1 × 30 = 60 experiments per objective function.

### Full Factorial Design

Then run full factorial for comprehensive comparison:
```python
variants = ['dendiff_gumbel', 'dendiff_corruption']
activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']
loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
fitness_guided_options = [0, 1]
seeds = np.arange(1, 31)
```

This gives 2 × 4 × 4 × 2 × 30 = 1,920 experiments per objective function.

### Ablation Studies

Test specific factors:

**Activation Function Study:**
```python
variants = ['dendiff_gumbel']
activations = ['leaky_relu', 'relu', 'tanh', 'sigmoid']
loss_functions = ['mse']
fitness_guided_options = [0]
```

**Loss Function Study:**
```python
variants = ['dendiff_gumbel']
activations = ['relu']
loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
fitness_guided_options = [0]
```

**Fitness Guidance Study:**
```python
variants = ['dendiff_gumbel', 'dendiff_corruption']
activations = ['relu']
loss_functions = ['mse', 'weighted_mse']
fitness_guided_options = [0, 1]
```

## Performance Estimates

**Time per experiment:** ~30-60 seconds (varies by problem size and parameters)

**Total time for 1,920 experiments:**
- Sequential: ~16-32 hours
- Parallel (10 nodes): ~1.6-3.2 hours
- Parallel (100 nodes): ~10-20 minutes

**Disk space:**
- Each result file: ~10-50 KB
- 1,920 experiments: ~20-100 MB total

**Memory per job:**
- Typical: 2-4 GB
- Safe allocation: 8 GB

## Contact and Support

For issues or questions:
1. Check error logs in `outputs/dendiff_*.err`
2. Verify SLURM configuration with cluster administrators
3. Consult `DISCRETE_DENDIFF_EDA_GUIDE.md` for parameter recommendations
4. Review `examples/discrete_Dendiff_EDA.py` for detailed usage

---

**Created:** 2026-01-07
**For:** PATEDA Discrete Dendiff EDA Experiments
**Version:** 1.0
