# Permutation EDA Implementation Summary

## Overview

This document summarizes the implementation of permutation-based Estimation of Distribution Algorithms (EDAs) ported from the MATEDA perm_mateda toolbox to Python.

## What Was Implemented

### 1. Generalized Mallows Model with Kendall Distance ✓

**Files:**
- `pateda/learning/mallows.py` - `LearnGeneralizedMallowsKendall` class
- `pateda/sampling/mallows.py` - `SampleGeneralizedMallowsKendall` class

**Key Features:**
- Position-dependent spread parameters (theta vector of length n-1)
- Each position j has its own theta_j parameter
- Allows modeling different levels of uncertainty at different positions
- Uses Kendall distance (bubble sort distance / inversion distance)
- V-vector (Lehmer code) factorization for efficient computation

**Learning Algorithm:**
- Finds consensus permutation using Borda count or set median
- Estimates theta_j for each position independently using MLE
- Computes position-specific psi normalization constants
- Generates v-vector probability matrix

**Sampling Algorithm:**
- Samples v-vectors from position-specific probability distributions
- Converts v-vectors to permutations via Lehmer code
- Composes with consensus permutation

**MATLAB Reference:** `GMallows_kendall_learning.m`, `GMallows_kendall_sampling.m`

### 2. Generalized Mallows Model with Cayley Distance ✓

**Files:**
- `pateda/learning/mallows.py` - `LearnGeneralizedMallowsCayley` class
- `pateda/sampling/mallows.py` - `SampleGeneralizedMallowsCayley` class

**Key Features:**
- Position-dependent spread parameters (theta vector of length n-1)
- Uses Cayley distance (minimum number of swaps, not necessarily adjacent)
- Cycle decomposition for efficient computation
- X-vector factorization

**Learning Algorithm:**
- Finds consensus permutation using Borda count or set median
- Estimates theta_j for each position using analytical solution
- Formula: theta_j = -log((1/E[X_j] - 1) / (n-j))
- Computes position-specific psi normalization constants
- Generates x-vector probability matrix (n-1 x 2)

**Sampling Algorithm:**
- Samples x-vectors from position-specific Bernoulli distributions
- Converts x-vectors to permutations via random swaps
- Composes with consensus permutation

**MATLAB Reference:** `GMallows_cayley_learning.m`, `GMallows_cayley_sampling.m`

### 3. Testing ✓

**Test Files:**
- `tests/test_generalized_mallows.py` - Comprehensive pytest-based tests
- `tests/test_gmallows_simple.py` - Simple standalone tests

**Test Coverage:**
- Learning correctness (model structure, dimensions, probability constraints)
- Sampling correctness (valid permutations, proper dimensions)
- Position-dependent theta values
- Consensus preservation
- Integration tests comparing with regular Mallows

## Already Existing in Python (From Previous Work)

### Distance Functions ✓
- Kendall distance (`kendall_distance`)
- Cayley distance (`cayley_distance`)
- Ulam distance (`ulam_distance`)
- Hamming distance (`hamming_distance`)

### Consensus Methods ✓
- Borda count (`find_consensus_borda`)
- Set median (`find_consensus_median`)

### Regular Mallows Models ✓
- Mallows with Kendall distance (learning + sampling)
- Mallows with Cayley distance (learning + sampling)

### Problem Functions ✓
- Traveling Salesman Problem (TSP)
- Quadratic Assignment Problem (QAP)
- Linear Ordering Problem (LOP)

## Not Yet Implemented

### 1. Mallows with Ulam Distance (Complex)

**Reason for Not Implementing:**
The Mallows model with Ulam distance requires Young tableaux generation and Ferrer shapes computation, which is significantly more complex than the other models. The MATLAB implementation requires pre-computed data files (`FerrerShapes_ncounts_n.mat`, `FerrerShapes_Lengths_n.mat`) for each problem size.

**Required Components:**
- Young tableaux generation (`GenYoungTableaux.m`)
- Hook number calculations (`n_hook_number.m`)
- Ferrer shapes computation (`ComputeFerrerShapes.m`)
- Permutation generation from LIS (`GeneratePermuAtLIS.m`)
- Pre-computed partition tables

**Estimated Complexity:** High - requires implementing combinatorial structures and pre-computation

**MATLAB Files:**
- `Mallows_Ulam_learning.m`
- `Mallows_Ulam_sampling.m`
- `GenYoungTableaux.m`
- `ComputeFerrerShapes.m`
- Multiple auxiliary functions in `/Mallows/Ulam/`

### 2. Additional Integration Tests

While basic tests are provided, more comprehensive integration tests using actual optimization problems (TSP, QAP, LOP) could be added.

## Technical Details

### Model Structure

**Generalized Mallows Model Dictionary:**
```python
{
    "model_type": "generalized_mallows_kendall" or "generalized_mallows_cayley",
    "consensus": np.ndarray,  # Consensus permutation (length n)
    "theta": np.ndarray,  # Position-dependent spread parameters (length n-1)
    "psis": np.ndarray,  # Normalization constants (length n-1)
    "v_probs": np.ndarray,  # (Kendall) Probability matrix (n-1 x n)
    "x_probs": np.ndarray,  # (Cayley) Probability matrix (n-1 x 2)
}
```

### Key Differences: Regular vs Generalized Mallows

| Aspect | Regular Mallows | Generalized Mallows |
|--------|----------------|---------------------|
| Theta  | Single value θ | Vector θ = (θ₁, ..., θₙ₋₁) |
| Flexibility | Same spread everywhere | Position-dependent spread |
| Parameters | 1 + consensus | (n-1) + consensus |
| Expressiveness | Lower | Higher |
| Computational Cost | Lower | Slightly higher |

### References

1. **MATEDA Toolbox Paper:**
   E. Irurozki, J. Ceberio, J. Santamaria, R. Santana, A. Mendiburu: "perm_mateda: A matlab toolbox of estimation of distribution algorithms for permutation-based combinatorial optimization problems." ACM TOMS, 2016.

2. **Generalized Mallows:**
   M.A. Fligner, J.S. Verducci: "Distance based ranking models." JRSS, 1986.

3. **Applications:**
   J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: "A Distance-based Ranking Model Estimation of Distribution Algorithm for the Flowshop Scheduling Problem." IEEE TEVC, 2014.

## Usage Examples

### Example 1: Using Generalized Mallows with Kendall Distance

```python
import numpy as np
from pateda.learning.mallows import LearnGeneralizedMallowsKendall
from pateda.sampling.mallows import SampleGeneralizedMallowsKendall

# Create population
n_vars = 10
pop_size = 100
population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
fitness = np.random.rand(pop_size)

# Learn model
learner = LearnGeneralizedMallowsKendall()
model = learner(
    generation=0,
    n_vars=n_vars,
    cardinality=np.arange(n_vars),
    selected_pop=population,
    selected_fitness=fitness,
    initial_theta=0.1,
    upper_theta=10.0,
    max_iter=100,
    consensus_method="borda"
)

# Sample from model
sampler = SampleGeneralizedMallowsKendall()
new_pop = sampler(
    n_vars=n_vars,
    model=model,
    cardinality=np.arange(n_vars),
    population=population,
    fitness=fitness,
    sample_size=50
)
```

### Example 2: Using Generalized Mallows with Cayley Distance

```python
from pateda.learning.mallows import LearnGeneralizedMallowsCayley
from pateda.sampling.mallows import SampleGeneralizedMallowsCayley

# Learn model
learner = LearnGeneralizedMallowsCayley()
model = learner(
    generation=0,
    n_vars=n_vars,
    cardinality=np.arange(n_vars),
    selected_pop=population,
    selected_fitness=fitness,
    consensus_method="borda"
)

# Sample from model
sampler = SampleGeneralizedMallowsCayley()
new_pop = sampler(
    n_vars=n_vars,
    model=model,
    cardinality=np.arange(n_vars),
    population=population,
    fitness=fitness,
    sample_size=50
)
```

## File Structure

```
pateda/
├── learning/
│   └── mallows.py           # Mallows and GM learning algorithms
├── sampling/
│   └── mallows.py           # Mallows and GM sampling algorithms
├── permutation/
│   ├── __init__.py          # Exports distance functions
│   ├── distances.py         # Kendall, Cayley, Ulam distances
│   ├── consensus.py         # Borda and median consensus
│   └── README.md            # Documentation
├── functions/permutation/
│   ├── tsp.py               # Traveling Salesman Problem
│   ├── qap.py               # Quadratic Assignment Problem
│   └── lop.py               # Linear Ordering Problem
└── tests/
    ├── test_mallows_cayley.py
    ├── test_generalized_mallows.py
    └── test_gmallows_simple.py
```

## Commit Information

**Branch:** `claude/crossover-learning-sampling-01Wuco6bUq6UtGwxuLCcUk4T`

**Commit Message:**
```
Add Generalized Mallows models for Kendall and Cayley distances

Implements the Generalized Mallows model with position-dependent spread
parameters (theta vector) for both Kendall and Cayley distances.
```

**Files Modified:**
- `pateda/learning/mallows.py` (+357 lines)
- `pateda/sampling/mallows.py` (+186 lines)
- `tests/test_generalized_mallows.py` (new, +414 lines)
- `tests/test_gmallows_simple.py` (new, +167 lines)

**Total:** 1,124 lines added

## Recommendations for Future Work

1. **Mallows with Ulam Distance:**
   - Implement Young tableaux generation algorithms
   - Create pre-computation scripts for Ferrer shapes
   - Port the complex sampling algorithm

2. **Performance Optimization:**
   - Vectorize v-vector and x-vector computations
   - Use numba JIT compilation for bottlenecks
   - Implement parallel sampling

3. **Additional Models:**
   - Plackett-Luce model
   - Hamming distance-based models
   - Mixture models

4. **Benchmarking:**
   - Compare Generalized Mallows vs Regular Mallows on real problems
   - Performance comparisons with MATEDA implementations
   - Scalability studies for large permutations

5. **Documentation:**
   - Add more usage examples
   - Create tutorial notebooks
   - Document when to use GM vs regular Mallows

## Conclusion

Successfully implemented Generalized Mallows models for both Kendall and Cayley distances, bringing advanced permutation-based EDA capabilities to the pateda library. The implementations follow the modular structure of pateda and are compatible with the existing EDA framework.

The code has been tested, committed, and pushed to the repository.
