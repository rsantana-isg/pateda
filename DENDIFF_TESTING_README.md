# Dendiff Distribution Approximation Testing

This document describes the comprehensive testing framework for evaluating the `learn_dendiff` and `sample_dendiff` functions in the PATEDA library.

## Overview

The testing framework evaluates how accurately the denoising diffusion model (dendiff) can approximate various probability distributions. Two main scripts have been created:

1. **`pateda/tests/test_dendiff_distributions.py`** - PyTest-based test suite
2. **`benchmark_dendiff_distributions.py`** - Standalone benchmark script with detailed reporting

## Installation

Before running the tests, ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- numpy >= 1.21.0
- scipy >= 1.7.0
- torch >= 2.0.0
- pytest >= 7.0.0

## Test Distributions

The framework tests dendiff on the following probability distributions:

### 1. Univariate Gaussian Distribution
- **Description**: Independent Gaussian distributions for each variable
- **Parameters**: Different means and standard deviations for each dimension
- **Difficulty**: Easy (baseline test)

### 2. Multivariate Gaussian Distribution
- **Description**: Correlated Gaussian with predetermined eigenvectors
- **Parameters**: Random covariance matrix with controlled eigenvalues
- **Difficulty**: Medium (tests ability to capture correlations)

### 3. Multivariate Cauchy Distribution
- **Description**: Independent Cauchy distributions (heavy-tailed)
- **Parameters**: Different locations and scales
- **Difficulty**: Hard (tests robustness to outliers and heavy tails)

### 4. Gaussian Mixture Model
- **Description**: Mixture of 3 Gaussian components
- **Parameters**: Random means, covariances, and mixture weights
- **Difficulty**: Hard (tests multi-modal distribution learning)

### 5. Uniform Distribution
- **Description**: Independent uniform distributions
- **Parameters**: Different ranges for each dimension
- **Difficulty**: Medium (tests ability to learn sharp boundaries)

### 6. Empirical Fitness Distributions (NEW!)

The framework now includes **empirical fitness distributions** that simulate realistic EDA scenarios. These distributions are created by:

1. Generating N random solutions (e.g., 1000)
2. Evaluating them using an objective function
3. Selecting the top S solutions with best fitness (e.g., top 500)
4. Using these selected solutions as the training distribution

This approach models the actual selection step in EDAs and tests dendiff in realistic optimization contexts.

#### Objective Functions Available:

1. **Sphere Function**
   - Type: Unimodal, smooth, separable
   - Bounds: [-5.12, 5.12]
   - Global minimum: f(0,...,0) = 0
   - Difficulty: Easy

2. **Ellipsoid Function**
   - Type: Unimodal, axis-parallel, ill-conditioned
   - Bounds: [-5.12, 5.12]
   - Global minimum: f(0,...,0) = 0
   - Difficulty: Easy-Medium

3. **Rastrigin Function**
   - Type: Highly multimodal, many local optima
   - Bounds: [-5.12, 5.12]
   - Global minimum: f(0,...,0) = 0
   - Difficulty: Hard

4. **Rosenbrock Function**
   - Type: Valley-shaped, non-separable
   - Bounds: [-2.048, 2.048]
   - Global minimum: f(1,...,1) = 0
   - Difficulty: Hard

5. **Ackley Function**
   - Type: Multimodal with nearly flat outer region
   - Bounds: [-32.768, 32.768]
   - Global minimum: f(0,...,0) = 0
   - Difficulty: Hard

6. **Griewank Function**
   - Type: Many local optima with variable interactions
   - Bounds: [-600, 600]
   - Global minimum: f(0,...,0) = 0
   - Difficulty: Hard

7. **Schwefel Function**
   - Type: Deceptive with many local optima
   - Bounds: [-500, 500]
   - Global minimum: f(420.97,...,420.97) ≈ 0
   - Difficulty: Very Hard

## Evaluation Metrics

The framework computes several metrics to compare original and sampled distributions:

### Distance Metrics

1. **Jensen-Shannon (JS) Divergence**
   - Range: [0, 1] (base 2)
   - Symmetric version of KL divergence
   - Interpretation:
     - < 0.1: Excellent approximation
     - 0.1-0.3: Good approximation
     - 0.3-0.5: Moderate approximation
     - > 0.5: Poor approximation

2. **Kullback-Leibler (KL) Divergence**
   - Range: [0, ∞)
   - Measures information loss
   - Interpretation:
     - < 0.1: Excellent approximation
     - 0.1-0.5: Good approximation
     - 0.5-1.0: Moderate approximation
     - > 1.0: Poor approximation

3. **Wasserstein Distance**
   - Optimal transport distance
   - Lower is better
   - Scale-dependent

### Statistical Metrics

1. **Mean Difference**: L2 norm of difference between means
2. **Standard Deviation Difference**: L2 norm of difference between standard deviations
3. **Covariance Difference**: Frobenius norm of difference between covariance matrices
4. **Relative Metrics**: Normalized versions of the above

## Running the Tests

### Option 1: PyTest Test Suite

Run all tests:
```bash
cd /home/user/pateda
pytest pateda/tests/test_dendiff_distributions.py -v -s
```

Run specific test class:
```bash
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffUnivariateGaussian -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffMultivariateGaussian -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffCauchy -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffMixture -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffUniform -v -s
```

Run parameter variation tests:
```bash
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffParameterVariations -v -s
```

### Option 2: Standalone Benchmark Script

Run comprehensive benchmark (includes both standard and fitness-based distributions):
```bash
cd /home/user/pateda
python benchmark_dendiff_distributions.py
```

This will:
1. **Part 1: Standard Distributions**
   - Test 5 probability distribution types (Gaussian, Cauchy, mixtures, uniform)
   - Report detailed metrics for each
   - Run parameter variation experiments

2. **Part 2: Empirical Fitness Distributions** (NEW!)
   - Test 5 objective functions (sphere, ellipsoid, rastrigin, rosenbrock, ackley)
   - Simulate realistic EDA selection scenarios
   - Compare sampled distributions with independent fitness samples
   - Report fitness statistics and distribution quality metrics
   - Run parameter variation experiments (selection ratio, timesteps, etc.)

3. **Generate comprehensive summary report**

Expected runtime: 15-30 minutes depending on hardware (longer due to fitness benchmarks)

## Test Structure

### Test Workflow

For each distribution:

1. **Generate Original Data**
   ```python
   original_samples, metadata = generator(n_samples=500, n_vars=10, seed=42)
   ```

2. **Create Fitness Vector**
   ```python
   fitness = np.random.randn(n_samples)  # Random fitness for testing
   ```

3. **Learn Dendiff Model**
   ```python
   model = learn_dendiff(original_samples, fitness, params=params)
   ```

4. **Sample from Learned Model**
   ```python
   sampled_data = sample_dendiff(model, n_samples=500, bounds=bounds, params=params)
   ```

5. **Compare Distributions**
   - Compute JS divergence, KL divergence, Wasserstein distance
   - Compute statistical differences (mean, std, covariance)
   - Report results

### Fitness-Based Test Workflow (NEW!)

For empirical fitness distributions:

1. **Generate MAT (Training Distribution)**
   ```python
   # Generate n_initial random solutions (e.g., 1000)
   initial_population = np.random.uniform(bounds[0], bounds[1], (n_initial, n_vars))

   # Evaluate fitness
   fitness = objective_function(initial_population)

   # Select top n_selected solutions (e.g., top 500)
   MAT = select_best(initial_population, fitness, n_selected)
   MAT_fitness = select_best(fitness, n_selected)
   ```

2. **Learn Dendiff Model**
   ```python
   model = learn_dendiff(MAT, MAT_fitness, params=params)
   ```

3. **Sample from Learned Model**
   ```python
   Sampled_MAT = sample_dendiff(model, n_samples=n_samples, bounds=bounds)
   Sampled_MAT_fitness = objective_function(Sampled_MAT)
   ```

4. **Generate MAT_ANOTHER (Independent Reference)**
   ```python
   # Generate another independent sample from same empirical fitness distribution
   # This provides a reference for comparison
   MAT_ANOTHER, MAT_ANOTHER_fitness = generate_empirical_fitness_distribution(
       objective_name, n_initial, n_selected, n_vars, seed=different_seed
   )
   ```

5. **Compare Distributions**
   - Compare Sampled_MAT with MAT_ANOTHER (both should represent fitness distribution)
   - Compute distribution metrics (JS divergence, KL divergence, Wasserstein)
   - Compare fitness statistics (mean, std, best fitness values)
   - Evaluate how well dendiff captured the empirical fitness landscape

6. **Additional Fitness Metrics**
   - Best fitness comparison: MAT vs Sampled vs MAT_ANOTHER
   - Mean fitness difference: measures average quality preservation
   - Fitness distribution shape: how well the fitness landscape is preserved

### Key Parameters

Default parameters used in tests:
```python
params = {
    'epochs': 100,              # Number of training epochs
    'n_timesteps': 500,         # Number of diffusion timesteps
    'hidden_dims': [128, 64],   # Network architecture
    'batch_size': None,         # Auto-computed based on data
    'learning_rate': 1e-3,      # Adam learning rate
    'beta_schedule': 'linear',  # Noise schedule type
    'time_emb_dim': 32,         # Time embedding dimension
}
```

## Parameter Tuning Recommendations

### For Better Quality

1. **Increase Sample Size**
   - More data → better model learning
   - Recommended: n_samples >= 500

2. **Increase Training Epochs**
   - More training → better convergence
   - Recommended: epochs >= 100
   - For complex distributions: epochs >= 150

3. **Increase Diffusion Timesteps**
   - More steps → smoother denoising process
   - Recommended: n_timesteps >= 500
   - For complex distributions: n_timesteps >= 800

4. **Larger Network Architecture**
   - More capacity → better approximation
   - Recommended: hidden_dims = [128, 64]
   - For complex distributions: hidden_dims = [256, 128] or [256, 128, 64]

5. **Alternative Beta Schedule**
   - Try cosine schedule for better stability
   - Set: beta_schedule='cosine'

### Trade-offs

- **More timesteps**: Better quality but slower sampling
- **Larger networks**: Better capacity but slower training
- **More epochs**: Better fit but risk of overfitting with small samples

## Expected Results

### Typical Performance by Distribution Type

| Distribution | Expected JS Div | Expected KL Div | Difficulty |
|--------------|----------------|-----------------|------------|
| Univariate Gaussian | < 0.3 | < 0.5 | Easy |
| Multivariate Gaussian | < 0.4 | < 0.8 | Medium |
| Cauchy | < 1.0 | < 2.0 | Hard |
| Gaussian Mixture | < 0.6 | < 1.0 | Hard |
| Uniform | < 0.8 | < 1.5 | Medium |

Note: Actual results may vary based on:
- Random seed
- Sample size
- Model parameters
- Hardware/computation

## Understanding the Results

### Good Performance Indicators

✓ JS divergence < 0.4
✓ Mean difference relative < 0.5
✓ Std difference relative < 0.5
✓ Model successfully captures the general shape of the distribution

### Poor Performance Indicators

✗ JS divergence > 0.7
✗ Mean difference relative > 1.0
✗ Sampled distribution looks very different from original
✗ Training loss not converging

### Common Issues

1. **High divergence with all distributions**
   - Solution: Increase epochs, use larger network, more timesteps

2. **Good performance on Gaussian, poor on others**
   - Expected: Dendiff works best with smooth, continuous distributions
   - Solution: Use specialized parameters for complex distributions

3. **Poor correlation capture in multivariate**
   - Solution: Increase network capacity, more training epochs

4. **Mode collapse in mixtures**
   - Solution: Increase timesteps, use cosine schedule

## Example Usage

```python
import numpy as np
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff

# Generate test data (e.g., multivariate Gaussian)
mean = np.zeros(10)
cov = np.eye(10)
original_samples = np.random.multivariate_normal(mean, cov, 500)
fitness = np.random.randn(500)

# Learn model
params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [128, 64]
}
model = learn_dendiff(original_samples, fitness, params=params)

# Sample from model
bounds = np.array([
    original_samples.min(axis=0) - 3,
    original_samples.max(axis=0) + 3
])
sampled_data = sample_dendiff(model, n_samples=500, bounds=bounds)

# Compare distributions
from scipy.spatial.distance import jensenshannon
# ... compute metrics ...
```

## Example Usage: Fitness-Based Distributions (NEW!)

```python
import numpy as np
from benchmark_dendiff_distributions import (
    generate_empirical_fitness_distribution,
    OBJECTIVE_FUNCTIONS,
    evaluate_dendiff_on_fitness_distribution
)
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff

# Step 1: Generate empirical fitness distribution
objective_name = 'sphere'
n_initial = 1000  # Generate 1000 random solutions
n_selected = 500  # Select top 500 based on fitness
n_vars = 10

MAT, MAT_fitness, metadata = generate_empirical_fitness_distribution(
    objective_name, n_initial, n_selected, n_vars, seed=42
)

print(f"Best fitness in MAT: {metadata['best_fitness']}")
print(f"Mean fitness in MAT: {metadata['mean_fitness']}")

# Step 2: Learn dendiff model
params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [128, 64]
}
model = learn_dendiff(MAT, MAT_fitness, params=params)

# Step 3: Sample from learned model
bounds_info = OBJECTIVE_FUNCTIONS[objective_name]
bounds = np.array([
    [bounds_info['bounds'][0]] * n_vars,
    [bounds_info['bounds'][1]] * n_vars
])

Sampled_MAT = sample_dendiff(model, n_samples=500, bounds=bounds)

# Step 4: Evaluate fitness of sampled solutions
obj_function = bounds_info['function']
Sampled_MAT_fitness = obj_function(Sampled_MAT)

print(f"Best fitness in Sampled_MAT: {np.min(Sampled_MAT_fitness)}")
print(f"Mean fitness in Sampled_MAT: {np.mean(Sampled_MAT_fitness)}")

# Step 5: Generate MAT_ANOTHER for comparison
MAT_ANOTHER, MAT_ANOTHER_fitness, _ = generate_empirical_fitness_distribution(
    objective_name, n_initial, n_selected, n_vars, seed=1042
)

# Step 6: Compare distributions
from scipy.spatial.distance import jensenshannon
# Compute JS divergence between Sampled_MAT and MAT_ANOTHER
# This tells us how well dendiff captured the empirical fitness distribution

# Or use the comprehensive evaluation function
results = evaluate_dendiff_on_fitness_distribution(
    objective_name='sphere',
    n_initial=1000,
    n_selected=500,
    n_vars=10,
    params=params,
    seed=42
)

print(f"JS Divergence: {results['js_divergence']}")
print(f"Fitness Mean Difference: {results['fitness_mean_diff']}")
```

## Expected Results for Fitness-Based Distributions

### Performance by Objective Function Type

| Objective Type | Expected JS Div | Expected Fitness Diff | Difficulty |
|----------------|----------------|----------------------|------------|
| Sphere (unimodal) | < 0.3 | < 5% of mean fitness | Easy |
| Ellipsoid (unimodal) | < 0.4 | < 8% of mean fitness | Easy-Medium |
| Rastrigin (multimodal) | < 0.6 | < 15% of mean fitness | Hard |
| Rosenbrock (valley) | < 0.6 | < 15% of mean fitness | Hard |
| Ackley (multimodal) | < 0.7 | < 20% of mean fitness | Hard |

### Quality Indicators for Fitness Distributions

**Excellent Performance (unimodal functions):**
- ✓ JS divergence < 0.3
- ✓ Sampled best fitness similar to or better than MAT best
- ✓ Fitness mean difference < 5% of MAT mean fitness
- ✓ Similar fitness distribution shape to MAT_ANOTHER

**Good Performance (multimodal functions):**
- ✓ JS divergence < 0.6
- ✓ Sampled solutions maintain reasonable fitness quality
- ✓ Fitness mean difference < 15% of MAT mean fitness
- ✓ Model captures general fitness landscape structure

**Warning Signs:**
- ✗ JS divergence > 0.8
- ✗ Sampled best fitness much worse than MAT best (>50% worse)
- ✗ Fitness mean difference > 25% of MAT mean fitness
- ✗ Sampled solutions cluster away from good fitness regions

### Impact of Selection Ratio

The selection ratio (n_selected / n_initial) affects distribution shape:

- **High selection (ratio=0.7)**: Broader distribution, easier to learn
  - Expected JS divergence: Lower
  - More diverse solutions in MAT

- **Medium selection (ratio=0.5)**: Balanced distribution
  - Expected JS divergence: Moderate
  - Good mix of quality and diversity

- **Low selection (ratio=0.3)**: Narrow distribution, focused on best solutions
  - Expected JS divergence: May be higher (harder to learn sharp distribution)
  - More challenging but potentially better for optimization

### Fitness Preservation

A good dendiff model should preserve fitness characteristics:

1. **Mean Fitness**: Sampled mean should be close to MAT mean
2. **Best Fitness**: Sampled best should be similar to or better than MAT best
3. **Fitness Std**: Distribution of fitness values should be similar
4. **Fitness Landscape**: Model should sample from promising regions

If sampled solutions have significantly worse fitness than MAT, the model may not have learned the fitness landscape well. Consider:
- Increasing epochs or timesteps
- Using larger network architecture
- Adjusting selection ratio

## Files Created

1. **`pateda/tests/test_dendiff_distributions.py`**
   - Comprehensive pytest test suite
   - Tests all distribution types
   - Tests parameter variations
   - Includes assertions for quality thresholds

2. **`benchmark_dendiff_distributions.py`** (UPDATED!)
   - Standalone benchmark script
   - Tests both standard and empirical fitness distributions
   - Includes 7 objective functions for fitness-based testing
   - Generates detailed reports with fitness statistics
   - Includes interpretation guides for both distribution types
   - Tests parameter variations (timesteps, architectures, selection ratios)

3. **`DENDIFF_TESTING_README.md`** (this file)
   - Documentation and usage guide

## References

The dendiff implementation is based on:

1. **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020)
2. **Improved Denoising Diffusion Probabilistic Models** (Nichol & Dhariwal, ICML 2021)
3. **DiffImpute: Tabular Data Imputation with Denoising Diffusion Probabilistic Model** (Nazzal et al., 2024)

## Contact

For questions or issues with the testing framework, please refer to the PATEDA project documentation or create an issue in the project repository.
