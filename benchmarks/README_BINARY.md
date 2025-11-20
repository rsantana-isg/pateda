# Binary Functions Benchmark for Discrete EDAs

This directory contains scripts for benchmarking discrete Estimation of Distribution Algorithms (EDAs) on a comprehensive set of binary benchmark functions from the additively decomposable family.

## Overview

The benchmark includes 32 different binary function configurations across 6 categories:
- **K-Deceptive**: Functions with k-bit deceptive building blocks
- **Deceptive-3**: Various 3-bit deceptive lookup tables
- **Hard-Deceptive**: Harder 5-bit deceptive functions
- **Hierarchical**: HIFF and fhtrap1 with hierarchical building blocks
- **Polytree**: Functions based on polytree structures
- **Cuban**: Multimodal Cuban functions (Fc2-Fc5)

Each function is tested on appropriate problem sizes (e.g., HIFF on powers of 2, k-deceptive-3 on multiples of 3).

## References

- Mühlenbein, H., & Paass, G. (1996). "From recombination of genes to the estimation of distributions I."
- Watson, R.A. (2002). "Hierarchical Building-Block Problems"
- Goldberg, D.E. (2002). "The Design of Innovation"

## Files

- **`binary_functions_benchmark.py`**: Main benchmarking script
- **`binary_single_example.py`**: Simple examples and comparisons
- **`README_BINARY.md`**: This documentation

## Discrete EDAs Evaluated

The benchmark evaluates three discrete EDAs with increasing model complexity:

1. **UMDA** (`umda`)
   - Univariate Marginal Distribution Algorithm
   - Assumes complete independence between variables
   - Simplest model, fast learning and sampling
   - Good baseline for comparison

2. **Tree-EDA** (`tree_eda`)
   - Tree-structured probabilistic model
   - Captures pairwise dependencies using maximum spanning tree
   - Based on mutual information
   - Good balance between complexity and efficiency

3. **MN-FDA** (`mnfda`)
   - Markov Network Factorized Distribution Algorithm
   - Learns clique-based factorization using chi-square test
   - Can model higher-order interactions
   - More expressive but computationally intensive

## Usage

### Quick Example (Single Function)

```python
python benchmarks/binary_single_example.py
```

This runs UMDA on k-deceptive-3 (n=30) with default parameters.

### Compare EDAs on Single Function

```python
python benchmarks/binary_single_example.py compare
```

This compares UMDA, Tree-EDA, and MN-FDA on the same function.

### List All Available Functions

```python
python benchmarks/binary_single_example.py list
```

### Full Benchmark (All EDAs, Selected Functions)

```python
python benchmarks/binary_functions_benchmark.py
```

**Note**: By default, this runs a subset of functions. Edit `FUNCTION_SUBSET` in the main block to select different functions, or set to `None` to run all functions.

### Custom Benchmark Configuration

```python
from benchmarks.binary_functions_benchmark import run_benchmark

# Run benchmark on specific functions
results_df = run_benchmark(
    eda_names=['umda', 'tree_eda'],  # Select EDAs
    function_names=[                   # Select functions
        'k_deceptive_k3',
        'decep3_no_overlap',
        'hiff_32'
    ],
    n_runs=10,                         # Runs per configuration
    pop_size=200,                      # Population size
    max_gen=200,                       # Maximum generations
    selection_ratio=0.5,               # Truncation ratio
    output_folder='binary_results',    # Output directory
    verbose=True
)
```

### Single Experiment (Programmatic)

```python
from benchmarks.binary_functions_benchmark import run_single_experiment

results = run_single_experiment(
    eda_name='umda',
    function_name='k_deceptive_k3',
    n_vars=30,
    pop_size=100,
    max_gen=100,
    selection_ratio=0.5,
    seed=42,
    verbose=True
)

print(f"Best fitness: {results['best_fitness']}")
print(f"Success: {results['success']}")
print(f"Generations: {results['generation_found']}")
```

## Output Files

The benchmark generates three types of output files in the `binary_results/` directory:

1. **`binary_results_TIMESTAMP.csv`**
   - Full results for all individual runs
   - Columns: eda_name, function_name, category, n_vars, run, best_fitness, success, etc.
   - Suitable for detailed analysis

2. **`binary_results_TIMESTAMP.pkl`**
   - Pickle file with complete results including fitness histories
   - Use for convergence analysis

3. **`binary_summary_TIMESTAMP.csv`**
   - Aggregated statistics per (EDA, function, size) combination
   - Columns: success_rate, mean_fitness, std_fitness, mean_generations, etc.
   - Suitable for statistical comparison

## Binary Functions Registry

### K-Deceptive Functions

| Function | Sizes | Block Size | Optimal |
|----------|-------|------------|---------|
| k_deceptive_k3 | 30, 60, 90 | 3 | n (all 1s) |
| k_deceptive_k4 | 40, 80 | 4 | n (all 1s) |
| k_deceptive_k5 | 50, 100 | 5 | n (all 1s) |
| gen_k_decep_k3 | 30, 60, 90 | 3 | n (all 1s) |
| gen_k_decep_overlap | 30, 60 | 3 (overlap=1) | varies |

### Deceptive-3 Functions

| Function | Sizes | Overlap | Optimal |
|----------|-------|---------|---------|
| decep3_overlap | 30, 60 | Yes (step=2) | (n-2)//2 + 1 |
| decep3_no_overlap | 30, 60, 90 | No (step=3) | n//3 |
| decep_marta3 | 30, 60, 90 | No | varies |
| decep_marta3_new | 30, 60 | No | n//3 * 1.5 |
| decep3_mh | 30, 60, 90 | No | n//3 * 3.0 |
| two_peaks_decep3 | 31, 61 | No | varies |
| decep_venturini | 30, 60, 90 | No | varies |

### Hierarchical Functions

| Function | Sizes | Constraint | Optimal |
|----------|-------|------------|---------|
| hiff_16 | 16 | Power of 2 | n*(log₂(n)+1) |
| hiff_32 | 32 | Power of 2 | n*(log₂(n)+1) |
| hiff_64 | 64 | Power of 2 | n*(log₂(n)+1) |
| hiff_128 | 128 | Power of 2 | n*(log₂(n)+1) |
| fhtrap1_9 | 9 | Power of 3 | varies |
| fhtrap1_27 | 27 | Power of 3 | varies |
| fhtrap1_81 | 81 | Power of 3 | varies |

### Polytree Functions

| Function | Sizes | Block Size | Optimal |
|----------|-------|------------|---------|
| polytree3_no_overlap | 30, 60 | 3 | varies |
| polytree3_overlap | 30, 60 | 3 (overlap) | varies |
| polytree5 | 50, 100 | 5 | varies |

### Cuban Functions

| Function | Sizes | Structure | Optimal |
|----------|-------|-----------|---------|
| fc2 | 50, 100 | F5Muhl | n//5 * 4.0 |
| fc3 | 50, 100 | F5Multimodal | n//5 * 7.0 |
| fc4 | 21, 41, 81 | F5Cuban1 | varies |
| fc5 | 29, 53 | Combined | varies |

### Hard Deceptive

| Function | Sizes | Block Size | Optimal |
|----------|-------|------------|---------|
| hard_decep5 | 50, 100 | 5 | n//5 * 1.0 |

## Performance Metrics

The benchmark tracks:

- **Best fitness**: Best solution found
- **Optimal fitness**: Known optimal (if available)
- **Success**: Whether optimal was found
- **Generation found**: When best solution was discovered
- **Generations run**: Total generations executed
- **Runtime**: Wall-clock time in seconds
- **Fitness history**: Convergence trajectory (compressed)

## Example Results Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('binary_results/binary_results_TIMESTAMP.csv')

# Calculate success rates per EDA and category
success_by_eda_category = df.groupby(['eda_name', 'category'])['success'].mean()
print(success_by_eda_category)

# Plot fitness distribution
df.boxplot(column='best_fitness', by=['eda_name', 'category'], figsize=(12, 6))
plt.ylabel('Best Fitness')
plt.title('Fitness Distribution by EDA and Category')
plt.show()

# Compare on specific function
func_data = df[df['function_name'] == 'k_deceptive_k3']
print(func_data.groupby('eda_name')['success'].describe())
```

## Customization

### Adding New Functions

To add a new binary function:

1. Add it to `BINARY_FUNCTIONS` dictionary:

```python
'my_new_function': {
    'function': my_function_implementation,
    'sizes': [30, 60, 90],  # Appropriate problem sizes
    'optimal': lambda n: compute_optimal(n),  # Or None if unknown
    'category': 'my-category'
}
```

2. The function should accept a 1D numpy array and return a scalar.

### Adding New EDAs

To add a new discrete EDA:

1. Implement learning and sampling components in pateda
2. Add configuration in `get_eda_configuration()`:

```python
elif eda_name == 'my_new_eda':
    components = EDAComponents(
        learning=LearnMyNewEDA(...),
        sampling=SampleMyNewSampling(n_samples=pop_size),
        selection=TruncationSelection(ratio=selection_ratio),
        stop_condition=stop_conditions[0],
    )
```

3. Add to `EDA_NAMES` list

### Adjusting Parameters

Common parameters to adjust:

- **`pop_size`**: Population size (default: 200)
  - Smaller: faster, may fail on harder problems
  - Larger: slower, better exploration

- **`max_gen`**: Maximum generations (default: 200)
  - Adjust based on problem difficulty

- **`selection_ratio`**: Truncation ratio (default: 0.5)
  - Lower: stronger selection pressure
  - Higher: more diversity

- **`n_runs`**: Independent runs (default: 10)
  - More runs: better statistics, longer time

## Testing

Run tests to verify the benchmark:

```bash
# Quick syntax check
python3 -m py_compile benchmarks/binary_functions_benchmark.py

# Run quick example
python benchmarks/binary_single_example.py

# Run test suite (requires pytest)
pytest tests/test_binary_benchmark.py -v

# Run only fast tests
pytest tests/test_binary_benchmark.py -v -m "not slow"
```

## Troubleshooting

### Common Issues

1. **"Invalid size for function"**
   - Check that n_vars is in the function's `sizes` list
   - E.g., HIFF requires powers of 2, k-deceptive-3 requires multiples of 3

2. **Slow execution**
   - Reduce `pop_size` and `max_gen`
   - Test on subset of functions first
   - Use fewer `n_runs`

3. **Poor convergence**
   - Increase `pop_size` for harder problems
   - Increase `max_gen` to allow more time
   - Try different EDAs (Tree-EDA or MN-FDA for problems with dependencies)

4. **Memory issues**
   - Reduce population size
   - Process functions in batches
   - Clear cache between runs

## Expected Performance

### Easy Functions (UMDA should succeed)
- Simple k-deceptive with small k
- Non-overlapping deceptive-3

### Medium Functions (Tree-EDA recommended)
- Larger k-deceptive
- Overlapping deceptive functions
- Small polytree functions

### Hard Functions (MN-FDA or larger populations needed)
- HIFF (hierarchical structure)
- Large fhtrap1
- Complex Cuban functions

## Citation

If you use this benchmark in your research, please cite the relevant papers for the functions you use (see references above) and pateda.

## License

This benchmark implementation follows the same license as pateda.
