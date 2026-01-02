# MAP-Based Sampling Implementation for PATEDA

## Overview

This implementation adds complete support for Maximum A Posteriori (MAP) based sampling methods to the PATEDA framework, specifically for Markov network-based EDAs (MN-FDA, MN-FDAG, and MOA).

Based on: **Santana, R. (2013). "Message Passing Methods for Estimation of Distribution Algorithms Based on Markov Networks"**

## What Was Implemented

### 1. MAP Inference Module (`pateda/inference/`)

**File:** `pateda/inference/map_inference.py`

#### Classes:
- **`MAPInference`**: Main inference engine for computing MAP configurations
  - Supports multiple inference methods:
    - `"exact"`: Junction tree exact inference (via pgmpy)
    - `"bp"`: Belief propagation (max-product, loopy BP)
    - `"decimation"`: Iterative variable fixing based on marginals
  - Methods:
    - `compute_map()`: Returns single MAP configuration
    - `compute_k_map(k)`: Returns k most probable configurations

- **`MAPResult`**: Dataclass holding MAP inference results
  - Configuration, probability, log-probability, method used, metadata

- **`KMAPResult`**: Dataclass holding k-MAP results
  - Multiple configurations with probabilities

#### Convenience Functions:
```python
compute_map(cliques, tables, cardinalities, method="exact")
compute_k_map(cliques, tables, cardinalities, k=5, method="exact")
compute_map_decimation(cliques, tables, cardinalities, max_iterations=100)
```

#### Key Features:
- Works with junction trees and junction graphs
- Handles arbitrary variable cardinalities
- Falls back gracefully when pgmpy unavailable
- Efficient k-MAP using beam search with priority queue
- Temperature control for decimation-based inference

---

### 2. MAP-Based Sampling Methods (`pateda/sampling/map_sampling.py`)

Three sampling strategies from Santana (2013):

#### **SampleInsertMAP** (Strategy S1)
Directly inserts MAP configuration into population.

**Algorithm:**
1. Sample M individuals using Probabilistic Logic Sampling (PLS)
2. Compute MAP configuration using message passing
3. Replace worst individual(s) with MAP configuration(s)

**Parameters:**
- `n_samples`: Population size
- `map_method`: MAP inference method ("exact", "bp", "decimation")
- `n_map_inserts`: Number of MAP configurations to insert (default 1)
- `k_map`: Use k-MAP to insert k best configurations (default 1)
- `replace_worst`: Replace worst individuals (True) or random (False)

**Example:**
```python
from pateda.sampling.map_sampling import SampleInsertMAP

sampler = SampleInsertMAP(
    n_samples=100,
    map_method="bp",
    n_map_inserts=1,
    replace_worst=True
)
```

#### **SampleTemplateMAP** (Strategy S2)
Uses MAP as template for crossover-like variation.

**Algorithm:**
1. Compute MAP configuration
2. For each new individual:
   - Randomly select variables to inherit from MAP template
   - Sample remaining variables from learned model
3. Creates variation around high-quality MAP solution

**Parameters:**
- `n_samples`: Population size
- `map_method`: MAP inference method
- `template_prob`: Probability of inheriting variable from MAP (default 0.5)
- `min_template_vars`: Minimum variables to keep from template

**Example:**
```python
from pateda.sampling.map_sampling import SampleTemplateMAP

sampler = SampleTemplateMAP(
    n_samples=100,
    map_method="bp",
    template_prob=0.6,  # 60% of variables from template
    min_template_vars=5
)
```

#### **SampleHybridMAP** (Strategy S3)
Combines Insert-MAP and Template-MAP.

**Algorithm:**
1. Use Template-MAP to generate population
2. Ensure pure MAP configuration is included (first individual)
3. Balances exploration (template variation) with exploitation (MAP inclusion)

**Parameters:**
- `n_samples`: Population size
- `map_method`: MAP inference method
- `template_prob`: Template inheritance probability
- `n_map_inserts`: Number of pure MAP configurations

**Example:**
```python
from pateda.sampling.map_sampling import SampleHybridMAP

sampler = SampleHybridMAP(
    n_samples=100,
    map_method="bp",
    template_prob=0.5,
    n_map_inserts=1
)
```

---

### 3. Integration with MN-FDA and MOA

The implementation seamlessly integrates with existing Markov network EDAs through PATEDA's modular architecture:

#### **MN-FDA Integration:**

MN-FDA can return either:
- `FactorizedModel` (for PLS sampling) when `return_factorized=True`
- `MarkovNetworkModel` (for Gibbs/MAP sampling) when `return_factorized=False`

**Example:**
```python
from pateda.core.eda import EDA
from pateda.learning.mnfda import LearnMNFDA
from pateda.sampling.map_sampling import SampleInsertMAP

eda = EDA(
    n_vars=30,
    cardinality=np.array([2] * 30),
    fitness_function=onemax,
    pop_size=100,
    n_generations=50,
    learning=LearnMNFDA(
        max_clique_size=3,
        return_factorized=False  # For MAP sampling
    ),
    sampling=SampleInsertMAP(n_samples=100, map_method="bp"),
    selection=SelectTruncation(ratio=0.5)
)

result = eda.run()
```

#### **MOA Integration:**

MOA already returns `MarkovNetworkModel`, so MAP sampling works directly:

**Example:**
```python
from pateda.learning.moa import LearnMOA

eda = EDA(
    n_vars=30,
    cardinality=np.array([2] * 30),
    fitness_function=onemax,
    pop_size=100,
    n_generations=50,
    learning=LearnMOA(k_neighbors=3),
    sampling=SampleInsertMAP(n_samples=100, map_method="bp"),
    selection=SelectTruncation(ratio=0.5)
)
```

---

### 4. Comprehensive Tests (`test_map_sampling.py`)

Test suite covering all functionality:

#### Test Classes:
1. **TestMAPInference**: MAP computation correctness
   - Simple exact MAP
   - k-MAP computation
   - Decimation-based MAP

2. **TestMAPSampling**: Basic sampling functionality
   - Insert-MAP produces valid populations
   - Template-MAP creates variations
   - Hybrid MAP combines strategies

3. **TestMAPWithMNFDA**: Integration with MN-FDA
   - Insert-MAP on OneMax
   - Template-MAP on Trap-5
   - Hybrid MAP on various problems

4. **TestMAPWithMOA**: Integration with MOA
   - Insert-MAP with local Markov structures
   - Template-MAP with neighborhood-based models

5. **TestMAPPerformanceComparison**: Benchmarking
   - Comparison of all sampling methods (Insert-MAP, Template-MAP, Hybrid, Gibbs, PLS)
   - Comparison of MAP inference methods (BP vs Decimation)
   - Statistical analysis across multiple runs

6. **TestMAPHighCardinality**: Higher cardinality variables
   - Ternary variables (k=3)
   - Demonstrates MAP advantage on higher cardinalities

#### Running Tests:
```bash
# Run all tests
python test_map_sampling.py

# Run specific test class (requires pytest)
pytest test_map_sampling.py::TestMAPInference -v
```

---

### 5. Example Usage (`pateda/examples/example_map_sampling.py`)

Comprehensive examples demonstrating all features:

#### Examples Included:
1. **MN-FDA + Insert-MAP on OneMax**
2. **MN-FDA + Template-MAP on Trap-5**
3. **MN-FDA + Hybrid MAP**
4. **MOA + Insert-MAP**
5. **Higher Cardinality (Ternary Variables)**
6. **Comparing MAP Inference Methods**
7. **Full Comparison of All Sampling Strategies**

#### Running Examples:
```bash
python pateda/examples/example_map_sampling.py
```

---

## Key Findings from Santana (2013)

Based on the original paper and our implementation:

1. **Insert-MAP (S1) generally outperforms other strategies**
   - Directly ensures the most probable configuration is in the population
   - Particularly effective on deceptive problems

2. **Performance advantage increases with variable cardinality**
   - MAP methods excel when variables have more than 2 values
   - Traditional sampling struggles with large cardinality spaces

3. **Exact and approximate inference show similar performance**
   - Belief propagation (BP) performs nearly as well as exact methods
   - Decimation-based BP also effective
   - BP is much faster on large networks

4. **MAP methods effective on deceptive problems**
   - Trap functions, Four Peaks, hierarchical problems
   - Helps escape local optima by directly sampling global modes

5. **Hybrid strategies balance exploration and exploitation**
   - Template-MAP provides variation around MAP
   - Hybrid combines benefits of both approaches

---

## Implementation Details

### Design Decisions:

1. **Modular Architecture**
   - MAP inference is separate from sampling methods
   - Sampling methods work with any `MarkovNetworkModel` or `FactorizedModel`
   - Easy to add new inference or sampling algorithms

2. **Graceful Degradation**
   - Falls back to greedy methods if pgmpy unavailable
   - Works without optional dependencies
   - Always produces valid results

3. **Efficient k-MAP**
   - Beam search with priority queue
   - Avoids exponential enumeration
   - Configurable beam width

4. **PLS Integration**
   - Insert-MAP uses PLS for base population
   - Maintains efficiency of forward sampling
   - Compatible with factorized model ordering

5. **Template Variation**
   - Template-MAP uses conditional sampling from model
   - Respects learned dependencies
   - Configurable template inheritance probability

### Performance Characteristics:

- **MAP Inference Time**: O(n * k * iterations) for BP, where k is max clique size
- **k-MAP Time**: O(k * beam_width * n) for beam search
- **Insert-MAP Sampling**: O(M * n + MAP_time) where M is population size
- **Template-MAP Sampling**: O(MAP_time + M * n * average_clique_size)
- **Space**: O(n^2) for adjacency matrix, O(n * max_clique_size) for cliques

### Dependencies:

Required:
- numpy
- scipy
- networkx

Optional (for enhanced functionality):
- pgmpy (for exact junction tree inference)

---

## Usage Recommendations

### When to Use Insert-MAP:
- Problems with clear global optimum
- Deceptive landscapes
- When fast convergence is critical
- Higher cardinality variables (k > 2)

### When to Use Template-MAP:
- Need more exploration
- Complex multi-modal landscapes
- When diversity is important
- Avoiding premature convergence

### When to Use Hybrid MAP:
- Balance between exploitation and exploration
- Uncertain problem characteristics
- Default choice for most problems

### MAP Inference Method Selection:
- **BP ("bp")**: Default, works well in most cases, fast
- **Decimation**: When BP struggles to converge, more robust
- **Exact**: When network is small (< 20 variables) and tree-like

---

## Files Modified/Created

### New Files:
```
pateda/inference/__init__.py                   # Inference module exports
pateda/inference/map_inference.py              # MAP inference implementation (850+ lines)
pateda/sampling/map_sampling.py                # MAP sampling methods (700+ lines)
pateda/examples/example_map_sampling.py        # Usage examples (600+ lines)
test_map_sampling.py                          # Comprehensive tests (650+ lines)
```

### Modified Files:
```
pateda/sampling/__init__.py                    # Added MAP sampling exports
```

### Total Lines Added: ~2,800 lines of code + documentation

---

## Testing & Validation

All implementations have been validated:

✅ **Unit Tests**: MAP inference correctness verified
✅ **Integration Tests**: Works with MN-FDA and MOA
✅ **Benchmark Tests**: Performance on OneMax, Trap-5, high-cardinality
✅ **Comparison Tests**: Statistical comparison with baseline methods
✅ **Example Verification**: All 7 examples run successfully

Quick validation:
```bash
# Quick functionality test
python quick_test_map.py

# Full test suite
python test_map_sampling.py

# Example demonstrations
python pateda/examples/example_map_sampling.py
```

---

## Future Enhancements

Potential extensions (not implemented):

1. **Advanced Junction Tree Methods**
   - Full junction tree construction from scratch
   - Variable elimination ordering optimization
   - Clique tree message passing

2. **Max-Flow Propagation for k-MAP**
   - Implement Nilsson (1998) divide-and-conquer algorithm
   - More efficient k-MAP for large k
   - Exact k-MAP ranking

3. **Adaptive MAP Sampling**
   - Dynamic selection of MAP method based on network properties
   - Adaptive template_prob based on convergence
   - Hybrid with variable number of MAP inserts

4. **MAP-Elitism**
   - Always preserve MAP in next generation
   - MAP-based replacement strategies
   - MAP-guided local search

5. **Parallel MAP Computation**
   - Parallel belief propagation
   - Distributed k-MAP computation
   - Multi-threaded sampling

---

## References

1. **Santana, R. (2013)**. "Message Passing Methods for Estimation of Distribution Algorithms Based on Markov Networks". In: *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*, Springer.

2. **Nilsson, D. (1998)**. "An efficient algorithm for finding the M most probable configurations in probabilistic expert systems". *Statistics and Computing*, 8(2), 159-173.

3. **Gonzalez, C., Lozano, J. A., & Larranaga, P. (2002)**. "Analyzing the PBIL Algorithm by Means of Discrete Dynamical Systems". *Complex Systems*, 12(4), 465-479.

4. **Pelikan, M., Goldberg, D. E., & Lobo, F. G. (2002)**. "A survey of optimization by building and using probabilistic models". *Computational Optimization and Applications*, 21(1), 5-20.

---

## Summary

This implementation provides complete, production-ready MAP-based sampling for Markov network EDAs in PATEDA:

- ✅ **Complete**: All three strategies from Santana (2013) implemented
- ✅ **Modular**: Clean integration with existing PATEDA components
- ✅ **Tested**: Comprehensive test suite with benchmarks
- ✅ **Documented**: Extensive examples and documentation
- ✅ **Efficient**: Optimized inference and sampling algorithms
- ✅ **Robust**: Graceful degradation and error handling
- ✅ **Research-Ready**: Suitable for experimental studies and comparisons

The implementation respects PATEDA's architecture principles while adding powerful new capabilities for optimization with Markov network-based EDAs.
