# MN-FDA and MOA Implementation for pateda

## Overview

This implementation adds three new Markov network-based Estimation of Distribution Algorithms (EDAs) to pateda:

1. **MN-FDA**: Markov Network Factorized Distribution Algorithm
2. **MN-FDAG**: MN-FDA with G-test of independence
3. **MOA**: Markovianity Based Optimization Algorithm

These algorithms use undirected graphical models (Markov networks) to represent variable dependencies, providing an alternative to directed models (Bayesian networks) used by algorithms like EBNA and BOA.

## Reference

Santana, R. (2013). "Message Passing Methods for Estimation of Distribution Algorithms Based on Markov Networks". In: *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*, Springer.

## Algorithms

### MN-FDA (Markov Network Factorized Distribution Algorithm)

**Algorithm:**
```
1. Learn independence graph G using chi-square test
2. (Optional) Refine the graph
3. Find maximal cliques of G
4. Construct junction graph from cliques
5. Compute marginal probabilities for cliques
6. Sample using PLS (Probabilistic Logic Sampling)
```

**Characteristics:**
- Uses chi-square test to detect pairwise dependencies
- Builds factorized model from maximal cliques
- Supports bounded clique complexity
- Compatible with PLS sampling (fast, exact)

**Parameters:**
- `max_clique_size`: Maximum clique size (default: 3)
- `threshold`: Chi-square significance threshold (default: 0.05)
- `prior`: Use Laplace prior smoothing (default: True)

### MN-FDAG (MN-FDA with G-test)

**Difference from MN-FDA:**
- Uses G-test statistic: `G(Xi, Xj) = 2*N*MI(Xi, Xj)`
- More statistically principled dependency detection
- Accounts for different variable cardinalities
- Degrees of freedom: `df = (card_i - 1) * (card_j - 1)`

**Parameters:**
- `max_clique_size`: Maximum clique size (default: 3)
- `alpha`: Significance level for G-test (default: 0.05)
- `prior`: Use Laplace prior smoothing (default: True)

### MOA (Markovianity Based Optimization Algorithm)

**Algorithm:**
```
1. Compute mutual information matrix
2. For each variable Xi:
   - Find k nearest neighbors with MI > threshold
   - Create clique {Xi, neighbors}
3. Compute conditional probabilities P(Xi | neighbors)
4. Sample using Gibbs sampling
```

**Characteristics:**
- Local Markov structure (one clique per variable)
- Simpler than MN-FDA, faster learning
- Uses Gibbs sampling (MCMC)
- Recommended for harder problems

**Parameters:**
- `k_neighbors`: Max neighbors per variable (default: 3, paper uses 8)
- `threshold_factor`: MI threshold multiplier (default: 1.5)
- Gibbs iterations: `IT * n * ln(n)` where `IT=4` (default)

## File Structure

```
pateda/
├── learning/
│   ├── mnfda.py           # MN-FDA learning
│   ├── mnfdag.py          # MN-FDAG learning
│   ├── moa.py             # MOA learning
│   └── utils/
│       ├── mutual_information.py  # MI computation, G-test
│       ├── markov_network.py      # Graph construction, clique finding
│       └── probability_tables.py  # Probability table computation
├── sampling/
│   └── gibbs.py           # Gibbs sampling (MCMC)
└── examples/
    └── example_markov_edas.py  # Usage examples
```

## Usage

### Example 1: MN-FDA with PLS Sampling

```python
from pateda import EDA, EDAComponents
from pateda.learning import LearnMNFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.seeding import RandomInit
import numpy as np

# Configure components
components = EDAComponents(
    seeding=RandomInit(),
    selection=TruncationSelection(proportion=0.5),
    learning=LearnMNFDA(
        max_clique_size=3,
        threshold=0.05,
        return_factorized=True  # Return FactorizedModel for PLS
    ),
    sampling=SampleFDA(n_samples=100),
    stop_condition=StopCriteriaMaxGen(50),
)

# Create and run EDA
eda = EDA(
    pop_size=100,
    n_vars=30,
    fitness_func=my_fitness_function,
    cardinality=np.full(30, 2),
    components=components,
)

stats, cache = eda.run()
```

### Example 2: MN-FDAG with PLS Sampling

```python
from pateda.learning import LearnMNFDAG

components = EDAComponents(
    seeding=RandomInit(),
    selection=TruncationSelection(proportion=0.3),
    learning=LearnMNFDAG(
        max_clique_size=5,  # Allow larger cliques
        alpha=0.01,         # More conservative
        return_factorized=True
    ),
    sampling=SampleFDA(n_samples=150),
    stop_condition=StopCriteriaMaxGen(100),
)
```

### Example 3: MOA with Gibbs Sampling

```python
from pateda.learning import LearnMOA
from pateda.sampling import SampleGibbs

components = EDAComponents(
    seeding=RandomInit(),
    selection=TruncationSelection(proportion=0.5),
    learning=LearnMOA(
        k_neighbors=5,
        threshold_factor=1.5,
    ),
    sampling=SampleGibbs(
        n_samples=100,
        IT=4,              # Iterations = IT * n * ln(n)
        temperature=1.0,   # Standard Gibbs
        random_order=True,
    ),
    stop_condition=StopCriteriaMaxGen(50),
)
```

## Sampling Methods

### 1. PLS (Probabilistic Logic Sampling)

**Use with:** MN-FDA, MN-FDAG

**Characteristics:**
- Uses existing `SampleFDA` class
- Fast, exact sampling
- Requires ordered clique structure
- Best for small to medium clique sizes

**Configuration:**
```python
learning=LearnMNFDA(return_factorized=True),
sampling=SampleFDA(n_samples=pop_size),
```

### 2. Gibbs Sampling

**Use with:** MOA, MN-FDA, MN-FDAG

**Characteristics:**
- MCMC sampling method
- Works with any Markov network structure
- Slower but more flexible than PLS
- Recommended for MOA

**Parameters:**
- `n_iterations`: Total iterations (if None: `IT * n * ln(n)`)
- `IT`: Iteration factor (default: 4)
- `temperature`: Boltzmann temperature (default: 1.0)
  - `T > 1.0`: more exploration
  - `T < 1.0`: more exploitation
- `random_order`: Randomize variable order (default: True)
- `burnin`: Discard first N iterations (default: 0)

**Configuration:**
```python
learning=LearnMOA(k_neighbors=5),
sampling=SampleGibbs(
    n_samples=pop_size,
    IT=4,
    temperature=1.2,  # Slightly higher for exploration
    burnin=50,
),
```

## Algorithm Comparison

| Feature | MN-FDA | MN-FDAG | MOA |
|---------|--------|---------|-----|
| **Dependency test** | Chi-square | G-test | MI threshold |
| **Structure** | Maximal cliques | Maximal cliques | Local neighborhoods |
| **Complexity** | Medium | Medium | Low |
| **Sampling** | PLS (recommended) | PLS (recommended) | Gibbs (required) |
| **Learning speed** | Medium | Medium | Fast |
| **Sampling speed** | Fast | Fast | Medium |
| **Best for** | General problems | Mixed cardinalities | Hard problems |

## Implementation Details

### Mutual Information Computation

File: `pateda/learning/utils/mutual_information.py`

- `compute_pairwise_mi()`: Compute MI(Xi, Xj) for one pair
- `compute_mutual_information_matrix()`: Compute all pairwise MIs
- `g_test_statistic()`: G-test of independence
- `compute_g_test_matrix()`: G-test for all pairs
- `chi_square_test()`: Chi-square independence test

### Graph Construction

File: `pateda/learning/utils/markov_network.py`

- `build_dependency_graph_threshold()`: Build graph from MI threshold
- `find_maximal_cliques_greedy()`: Find maximal cliques with size bound
- `find_k_neighbors()`: Find k-nearest neighbors for each variable (MOA)
- `order_cliques_for_sampling()`: Topological ordering for PLS

### Probability Tables

File: `pateda/learning/utils/probability_tables.py`

- `compute_clique_table()`: Marginal P(clique variables)
- `compute_conditional_table()`: Conditional P(new | overlap)
- `compute_moa_tables()`: All P(Xi | neighbors) for MOA

## Testing

Run the comprehensive example:

```bash
cd /home/user/pateda
python pateda/examples/example_markov_edas.py
```

This runs:
1. MN-FDA on OneMax
2. MN-FDAG on Trap-5
3. MOA on OneMax
4. MOA on Trap-5
5. Comparison of all three algorithms

## Performance Recommendations

### For OneMax and simple problems:
- Use **MN-FDA** with PLS sampling
- Small clique size (2-3) is sufficient
- Population size: 50-100
- Generations: 30-50

### For Trap-5 and deceptive problems:
- Use **MN-FDAG** (better dependency detection)
- Larger clique size (5) to capture block structure
- Or use **MOA** with more neighbors (k=8)
- Population size: 150-200
- Generations: 100-150

### For hard problems (long dependencies):
- Use **MOA** with Gibbs sampling
- More neighbors (k=8-10)
- Higher temperature for exploration (T=1.2)
- Use burnin for Gibbs (50-100 iterations)
- Larger population and more generations

## Comparison with Existing EDAs

| Algorithm | Model Type | Structure | Sampling |
|-----------|----------|-----------|----------|
| **UMDA** | Univariate | None | Independent |
| **BMDA** | Bivariate | Pairwise | PLS |
| **EBNA** | Bayesian Network (directed) | Tree | PLS |
| **BOA** | Bayesian Network (directed) | DAG | PLS |
| **MN-FDA** | Markov Network (undirected) | Cliques | PLS |
| **MOA** | Markov Network (undirected) | Local | Gibbs |

**Advantages of Markov networks:**
- No need to determine edge directions
- Capture cyclic dependencies
- Often simpler structure than Bayesian networks
- Gibbs sampling provides flexibility

## References

### Paper
- Santana, R. (2013). "Message Passing Methods for Estimation of Distribution Algorithms Based on Markov Networks"

### C++ Implementation
- `cpp_EDAs/mainmoa.cpp`: MOA main loop
- `cpp_EDAs/FDA.cpp`: MN-FDA learning and sampling
- `cpp_EDAs/FactorGraphMethods.cpp`: MAP inference (not yet implemented)

### Python Implementation
Based on MATEDA-2.0 structure but following pateda's modular component-based architecture.

## Future Extensions

Potential additions:
1. **MAP-based sampling** (Insert-MAP, Template-MAP strategies)
2. **Junction tree inference** for exact MAP computation
3. **Belief propagation** for approximate inference
4. **Decimation-based MAP** finding
5. **Graph refinement** using conditional independence tests
6. **Adaptive clique size** based on problem structure
7. **Multi-objective MOA** variants

## License

Same as pateda (MIT License)
