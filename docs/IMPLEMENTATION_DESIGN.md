# MN-FDA and MOA Implementation Design for pateda

## Overview
Implementation plan for Markov Network-based EDAs following pateda's modular architecture.

## Architecture

### Components to Implement

#### 1. Learning Methods (pateda/learning/)

##### **LearnMNFDA** - Base MN-FDA Learning
```python
class LearnMNFDA(LearningMethod):
    """
    Learn Markov network factorization for MN-FDA

    Based on: mainmoa.cpp Markovinit(), FDA.cpp UpdateModel()
    """
    def __init__(self,
                 max_clique_size: int = 3,
                 max_n_cliques: int = None,
                 threshold: float = 0.05,
                 prior: bool = True):
        """
        Args:
            max_clique_size: Maximum clique size (default 3)
            max_n_cliques: Maximum number of cliques (default None = unlimited)
            threshold: Chi-square threshold for dependencies
            prior: Whether to use Laplace prior smoothing
        """

    def learn(self, generation, n_vars, cardinality, population, fitness, **params):
        """
        Learn factorized Markov network from population

        Steps:
        1. Compute pairwise mutual information (MI) matrix
        2. Detect dependencies using chi-square test: chi^2 = 2*N*MI(i,j)
        3. Build dependency graph (adjacency matrix)
        4. Find maximal cliques with bounded size
        5. Compute probability tables for each clique

        Returns:
            MarkovNetworkModel with clique structure and tables
        """
```

##### **LearnMNFDAG** - MN-FDA with G-statistics
```python
class LearnMNFDAG(LearningMethod):
    """
    Learn Markov network using G-test of independence

    Based on: FDA.cpp LearnMatrixGTest()

    Difference from base MN-FDA:
    - Uses G-statistic: G(Xi, Xj) = 2*N*MI(Xi, Xj)
    - Threshold is based on chi-square distribution with df = (card_i-1)*(card_j-1)
    """
    def __init__(self,
                 max_clique_size: int = 3,
                 max_n_cliques: int = None,
                 alpha: float = 0.05,  # Significance level
                 prior: bool = True):
        """
        Args:
            alpha: Significance level for G-test (default 0.05)
        """
```

##### **LearnMOA** - Markovianity-Based Optimization
```python
class LearnMOA(LearningMethod):
    """
    Learn local Markov structure for MOA algorithm

    Based on: mainmoa.cpp MOA(), FDA.cpp FindNeighbors()

    Creates one clique per variable containing:
    - The variable itself
    - Its k nearest neighbors (based on mutual information)
    """
    def __init__(self,
                 k_neighbors: int = 3,
                 threshold_factor: float = 1.5,
                 prior: bool = True):
        """
        Args:
            k_neighbors: Maximum number of neighbors per variable
            threshold_factor: Multiplier for average MI threshold
        """

    def learn(self, generation, n_vars, cardinality, population, fitness, **params):
        """
        Learn local Markov network structure

        Steps:
        1. Compute mutual information matrix using tree method
        2. For each variable Xi:
           - Find neighbors with MI > threshold_factor * avg(MI)
           - Keep top k_neighbors by MI strength
           - Create clique: {Xi, neighbors}
        3. Compute conditional probability tables P(Xi | neighbors)

        Returns:
            MarkovNetworkModel with local neighborhood structure
        """
```

#### 2. Sampling Methods (pateda/sampling/)

##### **SampleGibbs** - Gibbs Sampling for Markov Networks
```python
class SampleGibbs(SamplingMethod):
    """
    Gibbs sampling for Markov network models

    Based on: FDA.cpp GenIndividualMOA()
    """
    def __init__(self,
                 n_samples: int,
                 n_iterations: int = None,  # If None, use IT * n * ln(n)
                 IT: int = 4,
                 temperature: float = 1.0,
                 random_order: bool = True):
        """
        Args:
            n_iterations: Total Gibbs iterations (if None, compute from IT)
            IT: Iteration factor (default 4 from MOA paper)
            temperature: Temperature for Boltzmann sampling (default 1.0)
            random_order: Whether to randomize variable order each iteration
        """

    def sample(self, n_vars, model, cardinality, aux_pop=None, aux_fitness=None, **params):
        """
        Sample using Gibbs sampling

        Algorithm:
        1. Initialize: random configuration or from aux_pop
        2. For each of n_iterations iterations:
           - For each variable (in random or fixed order):
             - Sample Xi ~ P(Xi | X_neighbors)
        3. Return final configurations

        For MOA: Each variable's clique contains Xi and its neighbors
        Conditional: P(Xi | neighbors) from clique probability table
        """
```

##### **SampleMAP** - MAP-based Sampling
```python
class SampleMAP(SamplingMethod):
    """
    Sample using Maximum A Posteriori (MAP) configurations

    Strategies (from paper):
    - Insert-MAP: Add MAP solution to population
    - Template-MAP: Use MAP as template for crossover
    """
    def __init__(self,
                 n_samples: int,
                 strategy: str = "insert",  # "insert" or "template"
                 map_method: str = "bp",     # "bp", "jtree", "decimation"
                 n_map_samples: int = 1,
                 gibbs_sampler: SampleGibbs = None):
        """
        Args:
            strategy: "insert" or "template"
            map_method: MAP inference method
            n_map_samples: Number of MAP samples to use
            gibbs_sampler: Gibbs sampler for remaining population
        """
```

#### 3. Utilities (pateda/learning/utils/)

##### **mutual_information.py**
```python
def compute_mutual_information_matrix(
    population: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute pairwise mutual information matrix

    Uses tree-based efficient computation (Chow-Liu style)
    """

def compute_pairwise_mi(
    population: np.ndarray,
    var_i: int,
    var_j: int,
    card_i: int,
    card_j: int,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute MI(Xi, Xj) for a single pair
    """

def g_test_statistic(
    mi: float,
    n_samples: int,
    card_i: int,
    card_j: int
) -> Tuple[float, float]:
    """
    Compute G-statistic and p-value

    Returns:
        (G, p_value) where G = 2*N*MI(i,j)
    """
```

##### **markov_network.py**
```python
def build_dependency_graph(
    mi_matrix: np.ndarray,
    threshold: float,
    method: str = "threshold"  # "threshold" or "gtest"
) -> np.ndarray:
    """
    Build adjacency matrix from MI matrix
    """

def find_maximal_cliques(
    adjacency: np.ndarray,
    max_clique_size: int,
    max_n_cliques: Optional[int] = None
) -> np.ndarray:
    """
    Find maximal cliques with size constraint

    Uses greedy algorithm or Bron-Kerbosch
    """

def find_k_neighbors(
    mi_matrix: np.ndarray,
    k: int,
    threshold_factor: float = 1.0
) -> List[List[int]]:
    """
    Find k-nearest neighbors for each variable

    Used by MOA
    """

def order_cliques(
    cliques: np.ndarray
) -> np.ndarray:
    """
    Order cliques for sampling (PLS requirement)

    Ensures overlap variables are sampled before new variables
    """
```

##### **probability_tables.py**
```python
def compute_clique_tables(
    population: np.ndarray,
    cliques: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True
) -> List[np.ndarray]:
    """
    Compute probability tables for cliques

    For each clique:
    - If root (no overlap): P(vars)
    - If overlapping: P(new_vars | overlap_vars)
    """
```

#### 4. MAP Inference (pateda/inference/)

##### **map_inference.py**
```python
def compute_map(
    model: MarkovNetworkModel,
    cardinality: np.ndarray,
    method: str = "bp"  # "bp", "jtree", "decimation"
) -> np.ndarray:
    """
    Compute MAP configuration using pgmpy or custom implementation

    Methods:
    - "bp": Belief propagation (max-product)
    - "jtree": Junction tree exact inference
    - "decimation": Iterative variable fixing
    """
```

## File Structure

```
pateda/
├── learning/
│   ├── mnfda.py          # LearnMNFDA
│   ├── mnfdag.py         # LearnMNFDAG
│   ├── moa.py            # LearnMOA
│   └── utils/
│       ├── mutual_information.py
│       ├── markov_network.py
│       └── probability_tables.py
├── sampling/
│   ├── gibbs.py          # SampleGibbs
│   └── map_sampling.py   # SampleMAP
├── inference/
│   └── map_inference.py  # MAP computation utilities
└── examples/
    ├── example_mnfda.py
    ├── example_mnfdag.py
    └── example_moa.py
```

## Implementation Order

1. **Phase 1: Utilities**
   - mutual_information.py
   - markov_network.py
   - probability_tables.py

2. **Phase 2: Base MN-FDA**
   - LearnMNFDA
   - Test with existing SampleFDA (PLS already works)

3. **Phase 3: Gibbs Sampling**
   - SampleGibbs
   - Test with simple Markov networks

4. **Phase 4: MN-FDAG**
   - LearnMNFDAG (extends MN-FDA with G-statistics)

5. **Phase 5: MOA**
   - LearnMOA
   - Test with SampleGibbs

6. **Phase 6: MAP Sampling (Optional)**
   - map_inference.py
   - SampleMAP

## Testing Strategy

### Unit Tests
- Test MI computation against known values
- Test G-statistic calculation
- Test clique finding algorithms
- Test probability table computation
- Test Gibbs sampling on simple networks

### Integration Tests
- Compare with C++ implementation on same problems
- Test on deceptive functions (OneMax, Trap, HIFF)
- Validate convergence behavior

### Benchmarks
- OneMax (n=100, 200)
- Concatenated Trap-5 (k=5, m=20)
- HIFF (hierarchical)

## Pseudocodes from Paper

### Algorithm 1: MN-FDA
```
1  Set t ⇐ 0. Generate N ≫ 0 points randomly.
2  do {
3      Evaluate the points using the fitness function.
4      Select a set D_t^S of k ≤ N points according to a selection method.
5      Learn an undirected graphical model from D_t^S.
6      Generate the new population sampling from the model.
7      t ⇐ t + 1
8  } until Termination criteria are met
```

### Algorithm 2: Model Learning in MN-FDA
```
1  Learn an independence graph G using the G-test.
2  If necessary, refine the graph.
3  Find the set L of all the maximal cliques of G.
4  Construct a labeled JG from L.
5  Find the marginal probabilities for the cliques in the JG.
```
- G-test: G(X_i, X_j) = 2N·MI(X_i, X_j)
- MN-FDA^G uses G-test, base MN-FDA uses chi-square

### Algorithm 3: MOA
```
1  Set t ⇐ 0. Generate M points randomly
2  do {
3      Evaluate the points using the fitness function.
4      Select a set D_t^S of N ≤ M points according to a selection method.
5      Estimate the structure of a Markov network from D_t^S.
6      Estimate the local Markov conditional probabilities, p(x_i|N_i)
7      Generate M new points sampling from the Markov network.
8      t ⇐ t + 1
9  } until Termination criteria are met
```
- Threshold: TR = avg(MI) × 1.5
- Gibbs steps: r = n × ln(n) × IT, IT = 4
- Temperature: T (controls Boltzmann sampling)

### Algorithm 4: Insert-MAP
```
1  Learn the Markov network model MN.
2  Generate the set S of N − |E| − 1 solutions according to generation method.
3  Compute the factor graph FG.
4  Generate a MAP solution from FG.
5  Form the new population with S, E, and the MAP solution.
```

### Algorithm 5: Template-MAP
```
1  Learn the Markov network model MN.
2  Compute the factor graph FG.
3  Obtain MAP solution from FG.
4  for i = 1 to N − |E|
5      Apply uniform crossover between solution x^i and MAP solution.
```

## Sampling Method Flexibility

**Note:** We implement multiple sampling variants for flexibility:

**For MN-FDA:**
- PLS (Probabilistic Logic Sampling) - using existing SampleFDA
- Gibbs sampling - new SampleGibbs
- Insert-MAP - new SampleMAP with strategy="insert"
- Template-MAP - new SampleMAP with strategy="template"

**For MOA:**
- Gibbs sampling (recommended in paper)
- PLS (if junction graph structure is available)
- Insert-MAP / Template-MAP

## References

### C++ Code Locations
- MN-FDA: `cpp_EDAs/mainmoa.cpp:306` (Markovinit)
- MN-FDAG: `cpp_EDAs/FDA.cpp:1610` (LearnMatrixGTest)
- MOA: `cpp_EDAs/mainmoa.cpp:587`, `cpp_EDAs/FDA.cpp:2446`
- Gibbs: `cpp_EDAs/FDA.cpp:2446` (GenIndividualMOA)
- G-test: `cpp_EDAs/FDA.cpp:1610-1632`
- FindNeighbors: `cpp_EDAs/FDA.cpp:1369-1443`

### Paper
- Santana, R. (2013). "Message Passing Methods for Estimation of Distribution Algorithms Based on Markov Networks"
