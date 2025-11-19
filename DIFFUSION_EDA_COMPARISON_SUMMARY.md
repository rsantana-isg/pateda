# Diffusion-Based EDAs Comparison: Implementation Summary

## Task Completion Status

✅ **Step 0**: Updated to the most recent version of pateda from GitHub
✅ **Step 1**: Analyzed pateda structure and identified diffusion-based EDAs
✅ **Step 2**: Identified five continuous benchmark functions for n=20
✅ **Step 3**: Set problem dimension to n=20 in all implementations
✅ **Step 4**: Created comprehensive experimental script with evolution tracking
⏳ **Step 5**: Ready to run experiments (PyTorch installation in progress)
⏳ **Step 6**: Comparison framework implemented in script

## Identified Diffusion-Based EDAs

### 1. DbD-CS (Denoising-by-Deblending: Current to Selected)
- **Location**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`
- **Method**: Alpha-deblending from current population to selected population
- **Neural Network**: MLP [64, 64]

### 2. DbD-CD (Denoising-by-Deblending: Current to Distance-matched)
- **Location**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`
- **Method**: Alpha-deblending to distance-matched solutions
- **Neural Network**: MLP [64, 64]

### 3. DbD-UC (Denoising-by-Deblending: Univariate to Current)
- **Location**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`
- **Method**: Alpha-deblending from univariate Gaussian to current population
- **Neural Network**: MLP [64, 64]

### 4. DbD-US (Denoising-by-Deblending: Univariate to Selected)
- **Location**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`
- **Method**: Alpha-deblending from univariate Gaussian to selected population
- **Neural Network**: MLP [64, 64]

### 5. DenDiff (Denoising Diffusion EDA)
- **Location**: `pateda/learning/dendiff.py`, `pateda/sampling/dendiff.py`
- **Method**: Full DDPM with 1000 timesteps, DDIM fast sampling (50 steps)
- **Neural Network**: MLP [128, 64] with time embedding (dim=32)

## Benchmark Functions (All at n=20)

| Function | Domain | Optimum | Characteristics |
|----------|--------|---------|----------------|
| Sphere | [-5.12, 5.12]²⁰ | 0.0 | Unimodal, separable, convex |
| Rosenbrock | [-5, 5]²⁰ | 0.0 | Unimodal, non-separable, valley |
| Rastrigin | [-5.12, 5.12]²⁰ | 0.0 | Highly multimodal, separable |
| Ackley | [-5, 5]²⁰ | 0.0 | Multimodal, non-separable |
| Griewank | [-5, 5]²⁰ | 0.0 | Multimodal, non-separable |

## Experimental Design

- **EDAs**: 5 (DbD-CS, DbD-CD, DbD-UC, DbD-US, DenDiff)
- **Functions**: 5 (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank)
- **Runs per combination**: 5 (seeds: 42-46)
- **Generations**: 50
- **Population size**: 200
- **Selection ratio**: 0.3 (60 individuals)
- **Total experiments**: 125

### Algorithm Parameters

**DbD EDAs**:
- Hidden layers: [64, 64]
- Training epochs: 30
- Batch size: 32
- Learning rate: 1e-3
- Alpha samples: 10
- Sampling iterations: 10
- Restart trigger: 5 generations without improvement

**DenDiff EDA**:
- Hidden layers: [128, 64]
- Time embedding: 32 dimensions
- Training epochs: 30
- Diffusion timesteps: 1000
- DDIM sampling steps: 50
- Beta schedule: Linear

## Performance Metrics Collected

### 1. Solution Quality
- Best fitness achieved
- Fitness history over generations
- Final error from known optimum
- Mean and standard deviation across runs

### 2. Computational Efficiency
- Learning time per generation
- Sampling time per generation
- Total time per generation
- Function evaluations

### 3. Statistical Analysis
- Best fitness comparison table (mean ± std)
- Average time breakdown per EDA
- Best EDA identification per function
- Overall ranking across all functions

## Files Created

### Main Experimental Script
**File**: `experiments/diffusion_eda_comparison.py`

**Features**:
- Complete implementation of all 5 EDAs
- All 5 benchmark functions with vectorized evaluation
- Automatic experiment execution and data collection
- JSON results export with timestamp
- Comprehensive statistical analysis
- Four analysis tables automatically generated

### Documentation
**File**: `experiments/README_DIFFUSION_COMPARISON.md`

**Contents**:
- Detailed EDA descriptions
- Benchmark function specifications
- Experimental protocol
- Performance metrics explanations
- Usage instructions
- Expected runtime estimates

## How to Run the Experiments

### 1. Ensure PyTorch is Installed
```bash
# Check if torch is available
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

### 2. Run Full Comparison
```bash
cd /home/user/pateda
python experiments/diffusion_eda_comparison.py
```

### 3. Quick Test (Reduced)
Edit the `main()` function in the script:
```python
results = run_comparison_experiments(
    n_runs=3,          # Instead of 5
    n_generations=30,  # Instead of 50
    save_results=True
)
```

## Expected Output

### Console Output
```
================================================================================
DIFFUSION-BASED EDAs COMPARISON EXPERIMENT
================================================================================
Dimension: n = 20
Generations: 50
Independent runs: 5
Benchmark functions: 5
EDAs: 5 (DbD-CS, DbD-CD, DbD-UC, DbD-US, DenDiff)
================================================================================

Testing DbD-CS:
--------------------------------------------------------------------------------
  Sphere: ..... Avg: 1.23e-05 (5/125)
  Rosenbrock: ..... Avg: 18.456 (10/125)
  ...

[Analysis Tables]
Table 1: Best Fitness Achieved (mean ± std)
Table 2: Average Time per Generation (seconds)
Table 3: Best EDA per Function
Table 4: Overall Ranking
```

### Results File
**Location**: `experiments/results/diffusion_eda_comparison_YYYYMMDD_HHMMSS.json`

**Format**:
```json
[
  {
    "eda_name": "DbD-CS",
    "function_name": "Sphere",
    "run": 1,
    "best_fitness": 1.23e-05,
    "best_solution": [...],
    "fitness_history": [...],
    "learning_times": [...],
    "sampling_times": [...],
    "total_time": 123.45,
    "avg_learning_time": 2.34,
    "avg_sampling_time": 0.12,
    "avg_generation_time": 2.47,
    "function_evaluations": 10200,
    "final_error": 1.23e-05
  },
  ...
]
```

## Expected Runtime

- **Per experiment**: ~2-3 minutes
- **Full comparison (125 experiments)**: ~4-6 hours
- **Quick test (75 experiments)**: ~2.5-3.5 hours

Runtime depends on:
- Neural network training (30 epochs)
- Population size (200 individuals)
- Number of generations (50)
- Diffusion sampling steps

## Key Research Questions Addressed

1. **Which diffusion-based EDA performs best across different problem landscapes?**
   - Comparison across unimodal, multimodal, separable, and non-separable functions

2. **How do the four DbD variants compare?**
   - CS vs CD: random pairing vs distance-based pairing
   - UC vs US: univariate baseline with current vs selected population

3. **How does DenDiff compare to DbD methods?**
   - Full DDPM vs simplified alpha-deblending
   - Trade-off between model complexity and performance

4. **What is the computational overhead?**
   - Learning time vs sampling time
   - Performance per unit of computation time

## Implementation Notes

### Restart Mechanism (DbD EDAs)
All DbD variants include automatic restart that triggers when:
- No improvement for 5 consecutive generations, OR
- Diversity drops below 1e-6

On restart:
- Population reinitialized
- Best 2 solutions preserved
- Counter reset

### Fast Sampling (DenDiff)
DenDiff uses DDIM for efficient sampling:
- Reduces from 1000 to 50 steps (20x speedup)
- Deterministic sampling (eta=0.0)
- Maintains sample quality

### Data Normalization
All EDAs normalize data to [0, 1] range:
- Improves neural network stability
- Prevents numerical issues
- Samples denormalized to original bounds

## Git Repository Status

**Branch**: `claude/crossover-learning-sampling-01Wuco6bUq6UtGwxuLCcUk4T`

**Recent Commit**:
```
30f28b3 - Add comprehensive comparison framework for diffusion-based EDAs
```

**Changes Pushed**:
- ✅ Experimental script (`experiments/diffusion_eda_comparison.py`)
- ✅ Documentation (`experiments/README_DIFFUSION_COMPARISON.md`)

## Next Steps

1. **Wait for PyTorch Installation**: Currently installing in background
   - Main package: torch-2.9.1 (~900 MB)
   - CUDA libraries: ~1.5 GB total

2. **Run Experiments**: Once PyTorch is installed
   ```bash
   python experiments/diffusion_eda_comparison.py
   ```

3. **Analyze Results**: Review the generated tables and JSON file
   - Identify best-performing EDA overall
   - Compare computational efficiency
   - Examine convergence behavior

4. **Optional Extensions**:
   - Test at higher dimensions (n=50, n=100)
   - Add more benchmark functions
   - Tune hyperparameters
   - Create visualizations of fitness histories

## References

### DbD (Denoising-by-Deblending)
- Paper: "Learning search distributions in estimation of distribution algorithms with minimalist diffusion models"
- Implementation: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`
- Example: `pateda/examples/dbd_eda_example.py`

### DenDiff (Denoising Diffusion)
- Based on: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Implementation: `pateda/learning/dendiff.py`, `pateda/sampling/dendiff.py`
- Example: `pateda/examples/dendiff_eda_example.py`

## Contact

For questions about this comparison framework, refer to:
- Main README: `README_PATEDA.md`
- Detailed docs: `experiments/README_DIFFUSION_COMPARISON.md`
- PATEDA design: `PATEDA_DESIGN.md`

---

**Status**: Framework complete, PyTorch installation in progress, ready to run experiments
**Date**: 2025-11-19
**Commit**: 30f28b3
