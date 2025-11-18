# Advanced PATEDA Features

This directory contains examples demonstrating advanced features of PATEDA, including continuous optimization with Gaussian models, multi-objective optimization, and additional test functions.

## Continuous Optimization (Gaussian EDAs)

### Gaussian UMDA
**Example:** `gaussian_umda_sphere.py`

Univariate Gaussian model where each variable is modeled independently:
- **Learning:** Estimates mean and standard deviation for each variable
- **Sampling:** Samples each variable independently from its Gaussian distribution
- **Best for:** Separable problems, initial exploration
- **Benchmark:** Sphere function (f(x) = Σx²)

### Multivariate Gaussian EDA
**Example:** `gaussian_full_rastrigin.py`

Full covariance matrix model capturing dependencies between variables:
- **Learning:** Estimates mean vector and full covariance matrix
- **Sampling:** Samples from multivariate Gaussian distribution
- **Best for:** Problems with variable dependencies
- **Benchmark:** Rastrigin function (highly multimodal)
- **Parameter:** `var_scaling` to control covariance scaling

### Mixture of Gaussians
**Example:** `mixture_gaussian_rosenbrock.py`

Mixture model with multiple Gaussian components using k-means clustering:
- **Learning:** Clusters population and fits Gaussian to each cluster
- **Sampling:** Samples from mixture components proportionally
- **Best for:** Multimodal landscapes, diverse populations
- **Parameters:**
  - `n_clusters`: Number of mixture components
  - `what_to_cluster`: 'vars', 'objs', or 'vars_and_objs'
  - `normalize`: Whether to normalize before clustering
- **Benchmark:** Rosenbrock function (valley-shaped)

## Continuous Benchmark Functions

All available in `pateda.functions.continuous`:

| Function | Domain | Characteristics | Global Minimum |
|----------|--------|----------------|----------------|
| `sphere` | [-5.12, 5.12]ⁿ | Smooth, unimodal, separable | f(0,...,0) = 0 |
| `rastrigin` | [-5.12, 5.12]ⁿ | Highly multimodal | f(0,...,0) = 0 |
| `rosenbrock` | [-5, 10]ⁿ | Valley-shaped, non-convex | f(1,...,1) = 0 |
| `ackley` | [-32.768, 32.768]ⁿ | Nearly flat outer region | f(0,...,0) = 0 |
| `griewank` | [-600, 600]ⁿ | Many local optima | f(0,...,0) = 0 |
| `schwefel` | [-500, 500]ⁿ | Distant global optimum | f(420.97,...) ≈ 0 |
| `levy` | [-10, 10]ⁿ | Multimodal | f(1,...,1) = 0 |
| `michalewicz` | [0, π]ⁿ | Steep valleys | Dimension-dependent |
| `zakharov` | [-5, 10]ⁿ | Unimodal | f(0,...,0) = 0 |
| `sum_function` | User-defined | Linear | At lower bounds |

## Multi-Objective Optimization

### SAT Problems
**Example:** `umda_sat.py`

Solves multi-objective 3-SAT problems:
- **Problem:** Satisfiability with multiple formulas as objectives
- **Selection:** Pareto ranking for multi-objective optimization
- **Functions:**
  - `load_random_3sat()`: Generate random 3-SAT instances
  - `make_var_dep_formulas()`: Create structured SAT with dependencies
  - `SATInstance.evaluate()`: Evaluate clause satisfaction
- **Output:** Pareto front of non-dominated solutions

### uBQP (Unconstrained Binary Quadratic Programming)
**Example:** `tree_eda_ubqp.py`

Solves multi-objective binary quadratic programming:
- **Problem:** Maximize Σw_ij·x_i·x_j for multiple objectives
- **Algorithm:** Tree-based EDA with Pareto ranking
- **Functions:**
  - `generate_random_ubqp()`: Create random instances
  - `load_ubqp_instance()`: Load from file
  - `create_max_cut_ubqp()`: Convert Max-Cut to uBQP
  - `create_set_packing_ubqp()`: Convert Set Packing to uBQP
- **Applications:** Max-Cut, Set Packing, network optimization

## Running Examples

### Continuous Optimization
```bash
# Gaussian UMDA on Sphere
python examples/gaussian_umda_sphere.py

# Full Gaussian on Rastrigin
python examples/gaussian_full_rastrigin.py

# Mixture of Gaussians on Rosenbrock
python examples/mixture_gaussian_rosenbrock.py
```

### Multi-Objective Optimization
```bash
# Multi-objective SAT with UMDA
python examples/umda_sat.py

# Multi-objective uBQP with Tree EDA
python examples/tree_eda_ubqp.py
```

## Using Continuous Benchmarks

```python
from pateda.functions.continuous import sphere, rastrigin, get_function

# Direct function call
value = sphere(np.array([1.0, 2.0, 3.0]))

# Population evaluation (vectorized)
population = np.random.randn(100, 10)
values = rastrigin(population)

# Get function by name
func = get_function('ackley')
value = func(population)
```

## Creating Custom SAT Instances

```python
from pateda.functions.discrete.sat import SATInstance

# Create instance
sat = SATInstance()

# Add clauses (var1, var2, var3, neg1, neg2, neg3)
# Variables are 1-indexed, neg=1 means not negated
formula = [
    (1, 2, 3, 1, 0, 1),  # x1 OR ~x2 OR x3
    (2, 3, 4, 1, 1, 0),  # x2 OR x3 OR ~x4
]
sat.add_formula(formula)

# Evaluate solution
solution = np.array([1, 0, 1, 0])
satisfied = sat.evaluate(solution)
print(f"Clauses satisfied: {satisfied[0]} / {len(formula)}")
```

## Creating Custom uBQP Instances

```python
from pateda.functions.discrete.ubqp import UBQPInstance

# Create instance
ubqp = UBQPInstance(n_vars=10, n_objectives=2)

# Add interactions (objective_idx, i, j, weight)
ubqp.add_interaction(0, 1, 2, 5.0)   # Objective 1: x1*x2*5.0
ubqp.add_interaction(0, 2, 3, -3.0)  # Objective 1: x2*x3*(-3.0)
ubqp.add_interaction(1, 1, 3, 2.0)   # Objective 2: x1*x3*2.0

# Evaluate solution
solution = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
values = ubqp.evaluate(solution)
print(f"Objective values: {values}")
```

## Algorithm Selection Guide

| Problem Type | Recommended Algorithm | Example |
|--------------|----------------------|---------|
| Separable continuous | Gaussian UMDA | Sphere |
| Coupled continuous | Full Gaussian EDA | Rotated functions |
| Multimodal continuous | Mixture of Gaussians | Rastrigin, Ackley |
| Binary separable | UMDA | OneMax |
| Binary coupled | Tree EDA, FDA | NK-landscape |
| Binary highly coupled | EBNA, BOA | Hierarchical problems |
| Multi-objective | Any + Pareto ranking | SAT, uBQP |

## Advanced Features Not Yet Implemented

The following features from MATEDA are planned but not yet implemented:

1. **Gaussian Network EDAs**: Bayesian network structure learning for continuous variables
2. **Affinity-based Factorization**: Using affinity propagation for structure learning
3. **MPE Sampling**: Most probable explanation sampling from Bayesian networks
4. **MOA (Mixture of Ancestors)**: Advanced probabilistic model
5. **Hybrid algorithms**: Combining EDAs with local search

These will be added in future updates.

## Performance Tips

1. **Population size**: Larger for complex problems (200-500)
2. **Selection rate**: Usually 0.3-0.5 of population size
3. **Variance scaling**: For Gaussian EDAs, use 0.3-0.7 to prevent premature convergence
4. **Mixture components**: Start with 3-5 clusters, adjust based on problem
5. **Multi-objective**: Larger populations (300-500) for better Pareto front coverage

## References

- Larrañaga, P., & Lozano, J. A. (Eds.). (2001). *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*. Springer.
- Pelikan, M., Goldberg, D. E., & Lobo, F. G. (2002). A survey of optimization by building and using probabilistic models. *Computational Optimization and Applications*, 21(1), 5-20.
- Santana, R. (2011). *Estimation of Distribution Algorithms: A New Evolutionary Computation Approach for Graph Matching Problems*. PhD Thesis.
