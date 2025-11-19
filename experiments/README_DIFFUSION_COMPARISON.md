# Diffusion-Based EDAs Comparison Framework

## Overview

This framework provides a comprehensive comparison of diffusion-based Estimation of Distribution Algorithms (EDAs) for continuous optimization problems at dimension n=20.

## Diffusion-Based EDAs Included

### 1. DbD-CS (Denoising-by-Deblending: Current to Selected)
- **Method**: Alpha-deblending diffusion from current population to selected population
- **Key Feature**: Direct distribution shift from full population to elite solutions
- **Implementation**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`

### 2. DbD-CD (Denoising-by-Deblending: Current to Distance-matched)
- **Method**: Alpha-deblending from current population to distance-matched selected solutions
- **Key Feature**: Pairs source and target based on proximity in solution space
- **Implementation**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`

### 3. DbD-UC (Denoising-by-Deblending: Univariate to Current)
- **Method**: Alpha-deblending from univariate Gaussian to current population
- **Key Feature**: Starts from simpler distribution assumption
- **Implementation**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`

### 4. DbD-US (Denoising-by-Deblending: Univariate to Selected)
- **Method**: Alpha-deblending from univariate Gaussian to selected population
- **Key Feature**: Combines univariate baseline with elite targeting
- **Implementation**: `pateda/learning/dbd.py`, `pateda/sampling/dbd.py`

### 5. DenDiff (Denoising Diffusion EDA)
- **Method**: Denoising Diffusion Probabilistic Model (DDPM) based on Ho et al. (2020)
- **Key Feature**: Full diffusion process with 1000 timesteps, uses DDIM for fast sampling
- **Implementation**: `pateda/learning/dendiff.py`, `pateda/sampling/dendiff.py`

## Benchmark Functions (n=20)

All functions are tested at dimension n=20:

### 1. Sphere Function
```
f(x) = sum(x_i^2)
```
- **Domain**: [-5.12, 5.12]^20
- **Optimum**: f(0,...,0) = 0
- **Characteristics**: Unimodal, separable, convex

### 2. Rosenbrock Function
```
f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
```
- **Domain**: [-5, 5]^20
- **Optimum**: f(1,...,1) = 0
- **Characteristics**: Unimodal, non-separable, valley-shaped

### 3. Rastrigin Function
```
f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
```
- **Domain**: [-5.12, 5.12]^20
- **Optimum**: f(0,...,0) = 0
- **Characteristics**: Highly multimodal, separable

### 4. Ackley Function
```
f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e
```
- **Domain**: [-5, 5]^20
- **Optimum**: f(0,...,0) = 0
- **Characteristics**: Multimodal, non-separable

### 5. Griewank Function
```
f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1)))
```
- **Domain**: [-5, 5]^20
- **Optimum**: f(0,...,0) = 0
- **Characteristics**: Multimodal, non-separable

## Experimental Protocol

### Algorithm Parameters

#### DbD EDAs (all variants)
- **Population size**: 200
- **Selection ratio**: 0.3 (60 individuals selected)
- **Neural network**: MLP with hidden layers [64, 64]
- **Training epochs**: 30
- **Batch size**: 32
- **Learning rate**: 1e-3
- **Alpha samples**: 10 per training pair
- **Sampling iterations**: 10
- **Restart trigger**: 5 generations without improvement
- **Diversity threshold**: 1e-6
- **Keep best**: 2 solutions on restart

#### DenDiff EDA
- **Population size**: 200
- **Selection ratio**: 0.3 (60 individuals selected)
- **Neural network**: MLP with hidden layers [128, 64]
- **Time embedding dimension**: 32
- **Diffusion timesteps**: 1000
- **Beta schedule**: Linear
- **Training epochs**: 30
- **Batch size**: 32
- **Learning rate**: 1e-3
- **DDIM sampling steps**: 50 (fast sampling)
- **DDIM eta**: 0.0 (deterministic)

### Experimental Design
- **Independent runs**: 5 per algorithm-function pair
- **Generations**: 50
- **Random seeds**: 42, 43, 44, 45, 46
- **Total experiments**: 5 EDAs × 5 functions × 5 runs = 125 experiments

### Performance Metrics

1. **Solution Quality**
   - Best fitness achieved
   - Final error from known optimum
   - Mean and standard deviation across runs

2. **Computational Efficiency**
   - Average learning time per generation
   - Average sampling time per generation
   - Total time per generation
   - Function evaluations

3. **Convergence Behavior**
   - Fitness history over generations
   - Improvement rate
   - Convergence speed

## Output Files

### Results File Format
Results are saved as JSON with structure:
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

### Results Location
- **Directory**: `experiments/results/`
- **Filename pattern**: `diffusion_eda_comparison_YYYYMMDD_HHMMSS.json`

## Analysis Tables

The framework automatically generates:

### Table 1: Best Fitness (mean ± std)
Comparison of final best fitness across all EDA-function combinations

### Table 2: Average Time per Generation
Breakdown of computational time:
- Learning time
- Sampling time
- Total time per generation

### Table 3: Best EDA per Function
Identifies the best-performing EDA for each benchmark function

### Table 4: Overall Ranking
Ranks EDAs based on average performance across all functions

## Running the Comparison

### Prerequisites
```bash
pip install torch numpy
```

### Execute Comparison
```bash
python experiments/diffusion_eda_comparison.py
```

### Quick Test (fewer runs)
Modify the main() function:
```python
results = run_comparison_experiments(
    n_runs=3,          # Reduced from 5
    n_generations=30,  # Reduced from 50
    save_results=True
)
```

## Expected Runtime

With the specified parameters:
- **Per experiment**: ~2-3 minutes
- **Total runtime**: ~4-6 hours for 125 experiments
- **Factors affecting time**:
  - Neural network training (30 epochs)
  - Population evaluations (200 individuals × 50 generations)
  - Diffusion sampling steps

## Implementation Notes

### Restart Mechanism (DbD EDAs only)
All DbD variants include an automatic restart mechanism that triggers when:
1. No improvement for 5 consecutive generations, OR
2. Diversity drops below threshold (1e-6)

On restart:
- Population is reinitialized randomly
- Best 2 solutions are preserved
- Restart counter is reset

### Fast Sampling (DenDiff only)
DenDiff uses DDIM (Denoising Diffusion Implicit Models) for efficient sampling:
- Reduces sampling from 1000 to 50 steps
- Maintains sample quality
- Significantly speeds up generation time

### Data Normalization
All EDAs normalize training data to [0, 1] range:
- Improves neural network training stability
- Prevents numerical issues
- Samples are denormalized back to original bounds

## Key Research Questions

1. **Which diffusion-based EDA performs best across different problem landscapes?**
   - Unimodal vs multimodal
   - Separable vs non-separable
   - Smooth vs rugged

2. **How do the four DbD variants compare?**
   - Impact of distribution pairing strategy
   - Effect of univariate baseline vs full population

3. **How does DenDiff compare to DbD methods?**
   - Trade-off between model complexity and performance
   - Computational cost vs solution quality

4. **What is the computational overhead of diffusion models?**
   - Learning time
   - Sampling time
   - Scalability to higher dimensions

## References

### DbD (Denoising-by-Deblending)
- Paper: "Learning search distributions in estimation of distribution algorithms with minimalist diffusion models"
- Method: Alpha-deblending diffusion models
- Key innovation: Simplified diffusion process using linear interpolation

### DenDiff (Denoising Diffusion)
- Based on: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Method: Full DDPM with MLP architecture
- Key innovation: Time-conditioned denoising for distribution learning

## Future Extensions

1. **Higher Dimensions**: Test at n=50, n=100
2. **More Functions**: Include CEC benchmark suites
3. **Constraint Handling**: Add constrained optimization problems
4. **Hybrid Methods**: Combine with local search
5. **Multi-objective**: Extend to multi-objective optimization
6. **Real-world Problems**: Apply to engineering design problems

## Contact & Contributions

For questions or contributions to this comparison framework, please refer to the main PATEDA repository documentation.
