# Mixture of Trees EDA (MT-EDA) Enhancements

## Overview

This document describes the enhancements made to the Mixture of Trees Estimation of Distribution Algorithm (MT-EDA) implementation in PATEDA. The enhancements are based on research presented in several papers, particularly:

1. **"The mixture of trees Factorized Distribution Algorithm"** - Original MT-FDA formulation
2. **"RepMutMTFDA.pdf"** - Priors and adaptive learning for MT-FDA
3. **"Learning a small mixture of trees" (NIPS 2009)** - Efficient mixture of trees learning

## Key Enhancements

### 1. Exact Likelihood Computation

**File:** `learning/mixture_trees.py`

**Method:** `exact_likelihood()`

The original implementation used a placeholder approximation for computing tree likelihood. The enhanced implementation computes the exact probability using the tree factorization formula:

```
T(x) = P(X_root) * prod_{v != root} P(X_v | X_{pa(v)})
```

where `pa(v)` is the parent of variable `v` in the tree.

**Implementation Details:**
- Iterates through tree structure (root and parent-child relationships)
- For root nodes: uses marginal probability `P(X_v = x_v)`
- For non-root nodes: uses conditional probability `P(X_v | X_{pa(v)})`
- Handles both 1D and 2D probability tables
- Computes log-likelihood for numerical stability, then exponentiates

**Usage:**
```python
learner = LearnMixtureTrees(n_components=3)
model = learner.learn(...)

# Compute likelihood for samples
likelihood = learner.exact_likelihood(component, population, cardinality)
```

### 2. Priors for Mutation-like Effect (Section 4)

**Based on:** RepMutMTFDA.pdf, Section 4

**Theory:**
Priors prevent premature convergence by adding probability mass to unseen configurations, acting as a mutation-like mechanism. Two methods are implemented:

#### Fixed Prior (r*)
```
r' = r* = 2^(-(k-1)) * r
r = I_tau * M / n
```
where:
- `I_tau`: truncation ratio
- `M`: population size
- `n`: number of variables

#### Adaptive Prior (r^k)
```
r^k = P_bar(D^c) * M / (lambda^k * n)
P_bar(D^c) = 1 - P_bar(D)
```
where `P_bar(D)` is the average probability of data points under the mixture model.

**Implementation Details:**
- `_compute_priors()`: Computes component-wise priors
- `_compute_data_probability()`: Computes average data probability
- Priors stored in model parameters for use during sampling

**Usage:**
```python
learner = LearnMixtureTrees(
    n_components=3,
    use_priors=True,
    truncation_ratio=0.5,  # I_tau
    min_prior=1e-6
)
```

### 3. Adaptive Learning (Section 5)

**Based on:** RepMutMTFDA.pdf, Section 5

**Theory:**
Adaptive learning prevents overfitting by monitoring the probability of the data under the learned model. Learning stops when:

```
P_bar(D) >= mu
```

where:
- `P_bar(D) = (1/N) * sum_i Q(x_i)` is the average probability of data
- `mu` is the stopping threshold (typically 0.9)

**Implementation Details:**
- `_should_stop_adaptive()`: Checks if learning should stop
- `_compute_data_probability()`: Computes P_bar(D)
- Metadata includes `adaptive_stopped` and `p_bar_d` for diagnostics

**Usage:**
```python
learner = LearnMixtureTrees(
    n_components=3,
    use_adaptive=True,
    adaptive_mu=0.9  # Stopping threshold
)
```

### 4. Fitness-Weighted Components

**New Feature**

Components are weighted based on the fitness of samples that have high likelihood under each component. This gives higher weight to components that capture high-fitness regions of the search space.

**Algorithm:**
1. Compute responsibilities (soft assignment to components)
2. For each component, compute weighted average fitness
3. Weight = responsibility_sum * average_fitness
4. Normalize weights to sum to 1

**Implementation Details:**
- `_learn_mixture_weights_fitness()`: Implements fitness-weighted learning
- Uses exact likelihood for responsibility computation
- Normalizes fitness to [0, 1] for numerical stability

**Usage:**
```python
learner = LearnMixtureTrees(
    n_components=3,
    weight_learning="fitness_proportional"
)
```

## Configuration Options

### LearnMixtureTrees Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 3 | Number of tree components |
| `component_learning` | str | "tree" | Method for learning trees ("tree", "random_tree", "greedy") |
| `alpha` | float | 0.0 | Smoothing parameter |
| `weight_learning` | str | "uniform" | Weight learning method ("uniform", "em", "fitness_proportional") |
| `em_iterations` | int | 10 | Number of EM iterations |
| `use_priors` | bool | False | Enable priors for mutation |
| `use_adaptive` | bool | False | Enable adaptive learning |
| `truncation_ratio` | float | 0.5 | Truncation ratio (I_tau) for prior |
| `adaptive_mu` | float | 0.9 | Threshold for adaptive stopping |
| `min_prior` | float | 1e-6 | Minimum prior value |

## Mathematical Background

### Mixture of Trees Model

The mixture model combines multiple tree distributions:

```
Q(x) = sum_{k=1}^{m} lambda_k * T^k(x)
```

where:
- `lambda_k` are mixture weights (sum to 1)
- `T^k(x)` is the k-th tree component

### Tree Distribution

Each tree component factorizes as:

```
T(x) = prod_{v} T_{v|pa(v)}(x_v | x_{pa(v)})
```

For root nodes, `T_{v|pa(v)}(x_v) = P(x_v)` (marginal distribution).

### EM Algorithm for Mixture Learning

**E-step:** Calculate responsibilities
```
gamma_{jk} = (lambda_k * T^k(x_j)) / sum_l (lambda_l * T^l(x_j))
```

**M-step:** Update weights
```
lambda_k = (1/N) * sum_j gamma_{jk}
```

## Testing

A comprehensive test script is provided in `examples/test_mteda.py`:

```bash
# Run all tests
python examples/test_mteda.py --verbose

# Run specific tests
python examples/test_mteda.py --test likelihood
python examples/test_mteda.py --test priors
python examples/test_mteda.py --test fitness
python examples/test_mteda.py --test benchmark
```

### Test Coverage

1. **Exact Likelihood Test**: Verifies likelihood computation produces valid probabilities
2. **Priors Test**: Verifies prior computation and storage in model
3. **Fitness-Weighted Test**: Verifies fitness-weighted learning produces correct weights
4. **Benchmark Test**: Tests MT-EDA variants on OneMax, Deceptive3, KDeceptive3

## References

1. Santana, R., Ochoa, A., & Soto, M.R. (2001). "The Mixture of Trees Factorized Distribution Algorithm." GECCO 2001, pp. 543-550.

2. Meila, M., & Jordan, M.I. (2000). "Learning with Mixtures of Trees." Journal of Machine Learning Research, 1:1-48.

3. Anandkumar, A., Hsu, D., Javanmard, A., & Kakade, S. (2012). "Learning a small mixture of trees." NIPS 2009.

4. Chow, C., & Liu, C. (1968). "Approximating discrete probability distributions with dependence trees." IEEE Transactions on Information Theory, 14(3):462-467.

5. MATEDA-2.0 User Guide, Sections 4.2 and 4.4.

## Changelog

### Version 2.0 (Current)
- Added `exact_likelihood()` for correct probability computation
- Implemented priors for mutation-like effect (Section 4)
- Implemented adaptive learning (Section 5)
- Added fitness-weighted component learning
- Updated EM learning to use exact likelihood
- Created comprehensive test script
- Added new parameters: `use_priors`, `use_adaptive`, `truncation_ratio`, `adaptive_mu`
- Extended model metadata with diagnostic information
