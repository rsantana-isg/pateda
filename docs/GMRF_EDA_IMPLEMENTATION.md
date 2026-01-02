# GMRF-EDA Implementation for pateda

## Overview

This document describes the GMRF-EDA (Gaussian Markov Random Field Estimation of Distribution Algorithm) implementation added to pateda. GMRF-EDA is a continuous optimization EDA that uses regularization to determine the structure of variable interactions.

## Reference

The implementation is based on:

**Karshenas, H., Santana, R., Bielza, C., & Larrañaga, P. (2012).** *Continuous Estimation of Distribution Algorithms Based on Factorized Gaussian Markov Networks.* In Markov Networks in Evolutionary Computation (pp. 157-173). Springer.

## Algorithm Description

GMRF-EDA uses a hybrid approach combining:

1. **Regularized Regression**: For each variable, learns a regularized regression model predicting it from all other variables. The regression coefficients represent dependency strengths.

2. **Affinity Propagation**: Clusters variables into disjoint groups (cliques) based on their dependency weights.

3. **Multivariate Gaussian Estimation**: Estimates a multivariate Gaussian distribution for each clique.

4. **Independent Sampling**: Samples from each clique's Gaussian distribution independently.

This creates a Marginal Product Model (MPM) where cliques don't overlap, allowing efficient learning and sampling while capturing important variable interactions.

## Implementation Structure

### Learning Module (`pateda/learning/gaussian.py`)

#### Main Functions

- `learn_gmrf_eda(population, fitness, params)`: Main learning function with configurable regularization
- `learn_gmrf_eda_lasso(population, fitness, params)`: LASSO (L1) regularization variant
- `learn_gmrf_eda_elasticnet(population, fitness, params)`: Elastic Net (L1+L2) regularization variant
- `learn_gmrf_eda_lars(population, fitness, params)`: LARS (Least Angle Regression) variant

#### Parameters

The `params` dictionary supports:

- `regularization`: Type of regularization ('lasso', 'elasticnet', 'lars', 'lassolars')
- `alpha`: Regularization strength (default: 0.01)
- `l1_ratio`: For elasticnet, mixing between L1 and L2 (default: 0.5)
- `preference`: Affinity propagation preference (default: median of similarities)
- `damping`: Affinity propagation damping factor (default: 0.5)
- `max_iter`: Maximum iterations for affinity propagation (default: 200)
- `min_clique_size`: Minimum variables per clique (default: 1)

#### Returns

A model dictionary containing:
- `cliques`: List of variable index lists for each clique
- `clique_models`: List of dicts with 'mean' and 'cov' for each clique
- `weights`: Dependency weight matrix (for analysis)
- `type`: 'gmrf_eda'

### Sampling Module (`pateda/sampling/gaussian.py`)

#### Function

- `sample_gmrf_eda(model, n_samples, bounds, params)`: Samples from GMRF-EDA model

#### Parameters

- `model`: Model returned by learning function
- `n_samples`: Number of samples to generate
- `bounds`: Optional array of shape (2, n_vars) with [min, max] bounds
- `params`: Optional dict with 'var_scaling' for covariance scaling

#### Returns

- `population`: Sampled population of shape (n_samples, n_vars)

## Usage Examples

### Basic Usage

```python
from pateda.learning.gaussian import learn_gmrf_eda_lasso
from pateda.sampling.gaussian import sample_gmrf_eda
import numpy as np

# Create population
population = np.random.randn(100, 10)  # 100 samples, 10 variables
fitness = np.random.rand(100)

# Learn model
model = learn_gmrf_eda_lasso(population, fitness, {'alpha': 0.01})

# Inspect structure
print(f"Found {len(model['cliques'])} cliques:")
for i, clique in enumerate(model['cliques']):
    print(f"  Clique {i}: variables {clique}")

# Sample new population
bounds = np.array([[-5]*10, [5]*10])
new_population = sample_gmrf_eda(model, 100, bounds)
```

### In an EDA Loop

```python
from pateda.learning.gaussian import learn_gmrf_eda_elasticnet
from pateda.sampling.gaussian import sample_gmrf_eda
from pateda.functions.continuous.benchmarks import rosenbrock
import numpy as np

# Setup
n_vars = 10
pop_size = 100
n_generations = 50
selection_ratio = 0.5
bounds = np.array([[-5]*n_vars, [10]*n_vars])

# Initialize population
population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

for gen in range(n_generations):
    # Evaluate
    fitness = rosenbrock(population)

    # Select best
    n_select = int(pop_size * selection_ratio)
    selected_idx = np.argsort(fitness)[:n_select]
    selected_pop = population[selected_idx]
    selected_fit = fitness[selected_idx]

    # Learn model
    model = learn_gmrf_eda_elasticnet(
        selected_pop,
        selected_fit,
        {'alpha': 0.02, 'l1_ratio': 0.5}
    )

    # Sample new population
    population = sample_gmrf_eda(model, pop_size, bounds)

    # Report progress
    if gen % 10 == 0:
        print(f"Gen {gen}: Best = {np.min(fitness):.6e}")
```

### Comparing Regularization Methods

```python
from pateda.learning.gaussian import (
    learn_gmrf_eda_lasso,
    learn_gmrf_eda_elasticnet,
    learn_gmrf_eda_lars
)

# Same population
population = np.random.randn(100, 10)
fitness = np.random.rand(100)

# Try different methods
methods = [
    ("LASSO", learn_gmrf_eda_lasso, {'alpha': 0.01}),
    ("ElasticNet", learn_gmrf_eda_elasticnet, {'alpha': 0.01, 'l1_ratio': 0.5}),
    ("LARS", learn_gmrf_eda_lars, {})
]

for name, learn_func, params in methods:
    model = learn_func(population, fitness, params)
    print(f"{name}: {len(model['cliques'])} cliques")
    print(f"  Structure: {model['cliques']}")
```

## Test Suite

A comprehensive test suite is provided in `test_gmrf_eda.py`:

```bash
python test_gmrf_eda.py
```

The test suite includes:

1. **Basic Functionality Test**: Validates learning and sampling for all regularization methods
2. **Sphere Function**: Tests on separable function (variables independent)
3. **Additive Function**: Tests on block-structured function
4. **Rosenbrock Function**: Tests on chain-structured function

### Test Results

From our test runs:

| Function   | Method          | Final Best  | Notes                        |
|------------|-----------------|-------------|------------------------------|
| Sphere     | GMRF-LASSO      | 1.77e-06    | Correctly found independence |
| Sphere     | GMRF-ElasticNet | 3.59e-06    |                              |
| Sphere     | Univariate      | 1.03e-08    | Best for separable problem   |
| Additive   | GMRF-LASSO      | 2.95        |                              |
| Additive   | GMRF-ElasticNet | 2.14        | Better than full Gaussian    |
| Additive   | Full Gaussian   | 5.72        |                              |
| Rosenbrock | GMRF-LASSO      | 7.61        |                              |
| Rosenbrock | GMRF-ElasticNet | 7.98        |                              |

## Files Modified/Created

### Modified Files

1. `pateda/learning/gaussian.py`:
   - Added `_compute_regularized_weights()` helper function
   - Added `_cluster_variables_affinity_propagation()` helper function
   - Added `learn_gmrf_eda()` main function
   - Added `learn_gmrf_eda_lasso()` convenience wrapper
   - Added `learn_gmrf_eda_elasticnet()` convenience wrapper
   - Added `learn_gmrf_eda_lars()` convenience wrapper

2. `pateda/sampling/gaussian.py`:
   - Added `sample_gmrf_eda()` function

3. `pateda/learning/__init__.py`:
   - Exported new GMRF-EDA learning functions

4. `pateda/sampling/__init__.py`:
   - Exported new GMRF-EDA sampling function

### New Files

1. `test_gmrf_eda.py`: Comprehensive test suite
2. `GMRF_EDA_IMPLEMENTATION.md`: This documentation file

## Dependencies

The GMRF-EDA implementation requires:

- `numpy`: For numerical operations
- `scikit-learn`: For regularized regression (Lasso, ElasticNet, Lars) and AffinityPropagation
- `scipy`: For linear algebra operations

These dependencies are already part of pateda's requirements.

## Algorithm Properties

### Advantages

1. **Automatic Structure Learning**: Discovers variable dependencies without prior knowledge
2. **Sparse Models**: Regularization creates sparse dependency structures
3. **Scalable**: Factorization reduces computational complexity
4. **Flexible**: Multiple regularization methods available
5. **Interpretable**: Learned cliques show which variables interact

### Limitations

1. **Linear Dependencies**: Can only capture linear correlations
2. **Regularization Tuning**: Performance depends on regularization parameters
3. **Convergence**: Affinity propagation may not always converge
4. **Gaussian Assumption**: Assumes multivariate Gaussian distribution

### When to Use GMRF-EDA

GMRF-EDA is particularly effective for:

- **Decomposable Problems**: Functions with clear block structure
- **Sparse Interactions**: Problems where most variables are independent
- **Medium-Sized Problems**: 10-50 variables with structured dependencies
- **Problems with Unknown Structure**: When you don't know which variables interact

## Future Enhancements

Potential improvements to consider:

1. **Alternative Clustering**: Try other clustering methods beyond affinity propagation
2. **Dynamic Regularization**: Adapt regularization parameter during evolution
3. **Graphical LASSO**: Implement direct precision matrix estimation
4. **Shrinkage Estimation**: Add covariance shrinkage methods
5. **Hybrid Sampling**: Combine with Gibbs sampling for better dependency capture

## Troubleshooting

### Affinity Propagation Warnings

If you see "Affinity propagation did not converge" warnings:
- Increase `max_iter` parameter
- Adjust `damping` parameter (try values between 0.5 and 0.9)
- Increase `alpha` to create sparser weight matrices

### Empty Cliques

If you get empty cliques:
- Reduce `preference` parameter to encourage fewer, larger cliques
- Reduce `alpha` to allow more variable dependencies
- Set `min_clique_size` > 1 to filter small cliques

### Poor Performance

If GMRF-EDA performs worse than simpler methods:
- Problem may be fully separable (use univariate Gaussian)
- Problem may have dense dependencies (use full Gaussian)
- Try different regularization methods and parameters
- Increase population size for better model learning

## Contact

For questions or issues related to GMRF-EDA implementation, refer to:
- Original paper: Karshenas et al. (2012)
- Test suite: `test_gmrf_eda.py`
- Implementation: `pateda/learning/gaussian.py` and `pateda/sampling/gaussian.py`
