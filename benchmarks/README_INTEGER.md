# Integer Functions Benchmark for Discrete EDAs

This directory contains scripts for benchmarking discrete Estimation of Distribution Algorithms (EDAs) on benchmark functions designed for **integer (multi-valued discrete) representation**.

## Overview

Unlike binary benchmarks that test EDAs on variables with cardinality 2, this benchmark evaluates EDAs on integer-valued variables with cardinality > 2 (typically 4). This tests the ability of EDAs to model dependencies among multi-valued discrete variables.

The benchmark includes 10 different function configurations across 5 categories:

- **Simple**: Basic integer functions testing convergence (onemax, leading blocks)
- **Block**: Functions with block-level dependencies (max blocks, plateau search)
- **Deceptive**: Functions with deceptive structure (multi-level trap, generalized k-deceptive)
- **Dependency**: Functions testing dependency chain modeling (dependency chain, categorical match)
- **Hierarchical**: Multi-level structural functions (hierarchical blocks, parity blocks)

## Integer Representation

In integer representation:
- Each variable can take values in `[0, cardinality-1]`
- Default cardinality is 4, so values are in `{0, 1, 2, 3}`
- The optimal solution often has all variables at maximum value `(cardinality - 1)`
- This generalizes binary problems and tests richer probability distributions

## Integer Benchmark Functions

### Simple Category
- **`integer_onemax`**: Sum of integer values (tests basic convergence)
- **`integer_leading_blocks`**: Leading blocks with max values

### Block Category
- **`integer_max_blocks`**: Max blocks function with bonuses for optimal blocks
- **`integer_plateau_search`**: Tests plateau navigation with block structure

### Deceptive Category
- **`integer_multi_level_trap`**: Multi-level trap with deceptive local optima
- **`gen_k_decep_int`**: Generalized k-deceptive for integer representation

### Dependency Category
- **`integer_dependency_chain`**: Sequential dependency modeling
- **`integer_categorical_match`**: Categorical distribution learning

### Hierarchical Category
- **`integer_hierarchical`**: Multi-level consistency blocks
- **`integer_parity_blocks`**: Non-linear parity relationships

## EDAs Evaluated

The benchmark evaluates three discrete EDAs with increasing model complexity:

1. **UMDA** (`umda`)
   - Univariate Marginal Distribution Algorithm
   - Assumes complete independence between variables
   - Works well for separable integer problems

2. **Tree-EDA** (`tree_eda`)
   - Tree-structured probabilistic model
   - Captures pairwise dependencies
   - Better for problems with pairwise interactions

3. **MN-FDA** (`mnfda`)
   - Markov Network Factorized Distribution Algorithm
   - Learns clique-based factorization
   - Handles higher-order interactions

## Files

- **`integer_functions_benchmark.py`**: Main benchmarking script
- **`README_INTEGER.md`**: This documentation

## Usage

### Quick Test
```python
from benchmarks.integer_functions_benchmark import run_integer_experiment

# Run single experiment
results = run_integer_experiment(
    eda_name='umda',
    function_name='integer_onemax',
    n_vars=30,
    cardinality=4,
    pop_size=100,
    max_gen=50,
    seed=42,
    verbose=True
)
```

### Full Benchmark
```python
from benchmarks.integer_functions_benchmark import run_integer_benchmark

# Run comprehensive benchmark
df = run_integer_benchmark(
    eda_names=['umda', 'tree_eda', 'mnfda'],
    function_names=['integer_onemax', 'integer_max_blocks', 'integer_multi_level_trap'],
    cardinalities=[4],
    n_runs=10,
    pop_size=200,
    max_gen=100,
    output_folder='integer_results',
    verbose=True
)
```

### Command Line
```bash
python benchmarks/integer_functions_benchmark.py
```

## Function Registry

The benchmark uses a registry to manage function configurations:

```python
from benchmarks.integer_functions_benchmark import create_integer_function_registry

registry = create_integer_function_registry(cardinality=4)

# Access function info
func_info = registry['integer_onemax']
print(func_info['description'])  # "Sum of integer values (tests basic convergence)"
print(func_info['sizes'])        # [30, 60, 90]
print(func_info['optimal'](30))  # 90 (for cardinality=4)
```

## IntegerNKLandscape

The benchmark also includes an integer extension of NK landscapes:

```python
from pateda.functions.discrete.integer_functions import IntegerNKLandscape

# Create landscape with 4-valued variables
nk = IntegerNKLandscape(
    n_vars=20,
    k=2,
    cardinality=4,
    random_seed=42
)

# Evaluate a solution
x = np.random.randint(0, 4, 20)
fitness = nk.evaluate(x)
```

## Results

Benchmark results are saved in CSV and pickle formats with summary statistics including:
- Success rate (if optimal is known)
- Mean/median/std fitness
- Mean generations to best solution
- Mean runtime

## References

- Larrañaga, P., & Lozano, J. A. (2002). "Estimation of Distribution Algorithms"
- Mühlenbein, H. (1998). "Scalable Problems for Evolutionary Computation"
- Pelikan, M. (2005). "Hierarchical Bayesian Optimization Algorithm"
