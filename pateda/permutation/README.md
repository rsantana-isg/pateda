# Permutation-Based EDAs

This module implements permutation-based probability models for Estimation of Distribution Algorithms (EDAs). These models are designed to solve combinatorial optimization problems over permutation spaces, such as the Traveling Salesman Problem (TSP), Quadratic Assignment Problem (QAP), and others.

## Overview

Permutation-based EDAs learn and sample from probability distributions over the space of permutations. Unlike discrete EDAs that work with binary or categorical variables, permutation EDAs must respect the constraint that each element appears exactly once in a valid permutation.

This implementation is based on the methods described in the MATEDA toolbox and includes several state-of-the-art permutation models.

## Available Models

### 1. Mallows Model with Kendall Distance

The Mallows model is a location-spread model for permutations, parameterized by:
- A **consensus permutation** σ₀ (location)
- A **spread parameter** θ ≥ 0

The probability of a permutation σ is proportional to exp(-θ · d(σ, σ₀)), where d is a permutation distance metric.

**Learning**: `pateda.learning.mallows.LearnMallowsKendall`
- Finds consensus using Borda count or median method
- Estimates θ via maximum likelihood using v-vector statistics
- Computes normalization constants (Psi values)

**Sampling**: `pateda.sampling.mallows.SampleMallowsKendall`
- Samples v-vectors from learned probability matrix
- Converts v-vectors to permutations via Lehmer code
- Composes with consensus permutation

**Parameters**:
- `initial_theta`: Initial guess for θ (default: 0.1)
- `upper_theta`: Upper bound for θ search (default: 10.0)
- `max_iter`: Maximum iterations for optimization (default: 100)
- `consensus_method`: 'borda' or 'median' (default: 'borda')

### 2. Edge Histogram Model (EHM)

The Edge Histogram Model learns the probability of transitions between consecutive positions in a permutation. It maintains a matrix of edge frequencies.

**Learning**: `pateda.learning.histogram.LearnEHM`
- Counts edge transitions: P(item_j follows item_i)
- Supports symmetric and asymmetric variants
- Applies smoothing via beta_ratio parameter

**Sampling**: `pateda.sampling.histogram.SampleEHM`
- Builds permutations sequentially
- Samples next item based on edge probabilities
- Tracks visited items to ensure validity

**Parameters**:
- `symmetric`: Use symmetric EHM (default: True)
- `beta_ratio`: Smoothing parameter (default: 0.01)

### 3. Node Histogram Model (NHM)

The Node Histogram Model learns the probability of items appearing at specific positions, assuming independence between positions.

**Learning**: `pateda.learning.histogram.LearnNHM`
- Counts item occurrences at each position
- Applies smoothing via beta_ratio parameter

**Sampling**: `pateda.sampling.histogram.SampleNHM`
- Samples positions independently
- Repairs conflicts to ensure valid permutations
- Uses greedy assignment for duplicate items

**Parameters**:
- `beta_ratio`: Smoothing parameter (default: 0.01)

## Distance Metrics

The module provides three standard permutation distance metrics:

### Kendall Distance (`kendall_distance`)
- Counts the minimum number of adjacent transpositions needed to transform one permutation into another
- Also known as bubble sort distance or inversion distance
- Computed efficiently via v-vectors (Lehmer code)
- Complexity: O(n)

### Cayley Distance (`cayley_distance`)
- Counts the minimum number of swaps (not necessarily adjacent) needed
- Equivalent to n minus the number of cycles in the permutation composition
- Complexity: O(n)

### Ulam Distance (`ulam_distance`)
- Based on the length of the longest increasing subsequence (LIS)
- Distance = n - LIS(σ₁⁻¹ ∘ σ₂)
- Useful for permutations with large contiguous blocks
- Complexity: O(n log n)

## Consensus Methods

Consensus finding is crucial for Mallows models. Two methods are provided:

### Borda Count (`find_consensus_borda`)
- Position-based scoring: item at position i gets score (n-i)
- Consensus ranks items by total score
- Fast: O(n × pop_size)
- Generally produces good consensus approximations

### Median Method (`find_consensus_median`)
- Finds permutation minimizing total distance to population
- Tries each population member as candidate
- More accurate but slower: O(pop_size² × n)
- Guaranteed to find exact median for Kendall distance

## Usage Examples

### Example 1: Mallows EDA for TSP

```python
import numpy as np
from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.selection import Truncation
from pateda.learning.mallows import LearnMallowsKendall
from pateda.sampling.mallows import SampleMallowsKendall
from pateda.replacement import Elitist
from pateda.stop_conditions import MaxGenerations
from pateda.functions.permutation import create_random_tsp

# Create TSP instance
n_cities = 20
tsp = create_random_tsp(n_cities, seed=42)

# Configure EDA components
components = EDAComponents(
    seeding=RandomInit(),
    selection=Truncation(),
    learning=LearnMallowsKendall(),
    sampling=SampleMallowsKendall(),
    replacement=Elitist(),
    stop_condition=MaxGenerations(50),
)

# Set parameters
components.learning_params = {
    "initial_theta": 0.1,
    "upper_theta": 10.0,
    "max_iter": 100,
    "consensus_method": "borda",
}
components.sampling_params = {"sample_size": 100}
components.selection_params = {"ratio": 0.5}
components.replacement_params = {"elite_size": 10}

# Initialize and run
eda = EDA(
    pop_size=100,
    n_vars=n_cities,
    fitness_func=tsp,
    cardinality=np.arange(n_cities),
    components=components,
)

stats, cache = eda.run(verbose=True)
print(f"Best fitness: {stats.best_fitness_overall:.2f}")
```

### Example 2: Edge Histogram Model for TSP

```python
from pateda.learning.histogram import LearnEHM
from pateda.sampling.histogram import SampleEHM

# Configure EDA with EHM
components = EDAComponents(
    seeding=RandomInit(),
    selection=Truncation(),
    learning=LearnEHM(),
    sampling=SampleEHM(),
    replacement=Elitist(),
    stop_condition=MaxGenerations(40),
)

# Set EHM parameters
components.learning_params = {
    "symmetric": True,
    "beta_ratio": 0.01,
}
components.sampling_params = {"sample_size": 80}
components.selection_params = {"ratio": 0.5}
components.replacement_params = {"elite_size": 8}

# Run as before...
```

### Example 3: Using Distance Metrics

```python
from pateda.permutation import kendall_distance, cayley_distance, ulam_distance

perm1 = np.array([0, 1, 2, 3, 4])
perm2 = np.array([1, 0, 3, 2, 4])

print(f"Kendall distance: {kendall_distance(perm1, perm2)}")  # 2
print(f"Cayley distance: {cayley_distance(perm1, perm2)}")     # 2
print(f"Ulam distance: {ulam_distance(perm1, perm2)}")         # 2
```

### Example 4: Finding Consensus

```python
from pateda.permutation import find_consensus_borda, find_consensus_median

population = np.array([
    [0, 1, 2, 3, 4],
    [0, 2, 1, 3, 4],
    [1, 0, 2, 3, 4],
])

consensus_borda = find_consensus_borda(population)
consensus_median = find_consensus_median(population, kendall_distance)

print(f"Borda consensus: {consensus_borda}")
print(f"Median consensus: {consensus_median}")
```

## Complete Working Examples

See the `examples/` directory for complete, runnable examples:
- `examples/mallows_tsp_example.py` - Mallows model with Kendall distance for TSP
- `examples/ehm_tsp_example.py` - Edge Histogram Model for TSP

## Permutation Indexing

The module supports both 0-indexed (Python standard) and 1-indexed (MATLAB-style) permutations:
- Internally, all operations use 0-indexing
- Input permutations are automatically detected and converted
- Output permutations match the input indexing style

## Implementation Notes

### V-Vectors (Lehmer Code)
The v-vector representation is crucial for Mallows models:
- v[i] = number of elements to the right of position i that are smaller than element at i
- Provides a canonical factorization of permutations
- Enables efficient distance computation and sampling

### Normalization Constants (Psi Values)
For Mallows models, the Psi values are normalization constants:
- Ψⱼ(θ) = Σᵣ₌₀ⁿ⁻ʲ exp(-r·θ)
- Required for computing v-vector probabilities
- Computed recursively for efficiency

### Edge Frequency Smoothing
Both EHM and NHM use beta_ratio for Laplace-style smoothing:
- Prevents zero probabilities for unseen edges/positions
- Small values (0.01-0.1) typically work well
- Higher values increase exploration

## Performance Considerations

- **Kendall distance**: O(n) via v-vectors - very efficient
- **Cayley distance**: O(n) via cycle counting - efficient
- **Ulam distance**: O(n log n) via LIS - more expensive but still practical
- **Mallows learning**: O(pop_size × n × max_iter) - dominated by optimization
- **Mallows sampling**: O(sample_size × n²) - v-vector to permutation conversion
- **EHM sampling**: O(sample_size × n²) - sequential construction
- **NHM sampling**: O(sample_size × n²) - conflict resolution

## References

1. Ceberio, J., Irurozki, E., Mendiburu, A., & Lozano, J. A. (2012). A review of distances for the Mallows and generalized Mallows estimation of distribution algorithms. *Computational Optimization and Applications*, 51(2), 869-896.

2. Ceberio, J., Mendiburu, A., & Lozano, J. A. (2015). The Plackett-Luce ranking model on permutation-based optimization problems. *IEEE Congress on Evolutionary Computation (CEC)*, 494-501.

3. Tsutsui, S., Pelikan, M., & Ghosh, A. (2006). Performance of aggregation pheromone system on unimodal and multimodal problems. *Congress on Evolutionary Computation (CEC)*, 880-887.

4. Santana, R. (2005). Estimation of Distribution Algorithms with Kikuchi Approximations. *Evolutionary Computation*, 13(1), 67-97.

5. MATEDA 3.0 documentation - Model-based algorithms for permutations (included in `paper/Mateda3_Model_Based.pdf`)

## Module Structure

```
pateda/permutation/
├── __init__.py          # Main exports
├── distances.py         # Kendall, Cayley, Ulam distances
├── consensus.py         # Borda and median consensus methods
└── README.md           # This file

pateda/learning/
├── mallows.py          # Mallows model learning
└── histogram.py        # EHM and NHM learning

pateda/sampling/
├── mallows.py          # Mallows model sampling
└── histogram.py        # EHM and NHM sampling

pateda/functions/permutation/
└── tsp.py              # TSP benchmark problem
```

## Future Extensions

Potential additions to the permutation module:
- Generalized Mallows model (position-dependent θ)
- Cayley and Ulam distance variants for Mallows
- Plackett-Luce model
- Additional benchmark problems (QAP, LOP, PFSP)
- Hybrid local search operators
- Visualization tools for permutation distributions
