# GNBG Benchmark for Continuous EDAs

This directory contains scripts for benchmarking continuous Estimation of Distribution Algorithms (EDAs) on the GNBG (Generalized Numerical Benchmark Generator) test suite.

## Overview

The GNBG benchmark comprises 24 problem instances with diverse characteristics:
- **Multimodality**: Multiple local/global optima
- **Conditioning**: Ill-conditioned landscapes
- **Variable interactions**: Rotated and separable problems
- **Deceptiveness**: Misleading gradients
- **Scalability**: Configurable dimensions

Each instance includes pre-configured parameters, acceptance thresholds, and optimum values for comprehensive performance evaluation.

## References

- D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv preprint arXiv:2312.07083, 2023.

## Files

- **`gnbg_benchmark.py`**: Main benchmarking script
- **`gnbg_single_example.py`**: Simple example for testing
- **`README_GNBG.md`**: This documentation

## Continuous EDAs Evaluated

The benchmark evaluates four Gaussian-based continuous EDAs:

1. **Gaussian UMDA** (`gaussian_umda`)
   - Univariate Gaussian model (assumes independence)
   - Simplest model, good baseline
   - Fast, low memory

2. **Full Gaussian EDA** (`gaussian_full`)
   - Full covariance matrix (models all dependencies)
   - Captures all linear correlations
   - Higher computational cost

3. **Gaussian Mixture EDA** (`gaussian_mixture`)
   - Mixture of Gaussian components
   - Handles multimodal distributions
   - Adaptive number of components

4. **GMRF-EDA** (`gmrf_eda`)
   - Gaussian Markov Random Field
   - Sparse dependency structure via regularization
   - Good balance between univariate and full covariance

## Usage

### Quick Example (Single Problem)

```python
python benchmarks/gnbg_single_example.py
```

This runs Gaussian UMDA on GNBG f1 with reduced population size for quick testing.

### Compare EDAs on Single Problem

```python
python benchmarks/gnbg_single_example.py compare
```

### Full Benchmark (All EDAs, All Problems)

```python
python benchmarks/gnbg_benchmark.py
```

**Warning**: This runs all 4 EDAs on all 24 problems with 5 runs each (480 experiments). This can take several hours depending on your hardware.

### Custom Benchmark Configuration

```python
from benchmarks.gnbg_benchmark import run_benchmark

# Run benchmark on subset of problems
results_df = run_benchmark(
    eda_names=['gaussian_umda', 'gaussian_full'],  # Select EDAs
    problem_indices=[1, 2, 3, 7, 8, 9],            # Select problems
    instances_folder='pateda/functions/GNBG_Instances.Python-main',
    n_runs=3,                                      # Runs per configuration
    pop_size=100,                                  # Population size
    selection_ratio=0.5,                           # Truncation ratio
    output_folder='gnbg_results',                  # Output directory
    verbose=True
)
```

### Single Experiment (Programmatic)

```python
from benchmarks.gnbg_benchmark import run_single_experiment

results = run_single_experiment(
    eda_name='gaussian_umda',
    problem_index=1,
    instances_folder='pateda/functions/GNBG_Instances.Python-main',
    pop_size=100,
    selection_ratio=0.5,
    seed=42,
    verbose=True
)

print(f"Best fitness: {results['best_fitness']}")
print(f"Error: {results['error_from_optimum']}")
print(f"Success: {results['success']}")
```

## Output Files

The benchmark generates three types of output files in the `gnbg_results/` directory:

1. **`gnbg_results_TIMESTAMP.csv`**
   - Full results for all individual runs
   - Columns: eda_name, problem_index, run, best_fitness, error_from_optimum, success, etc.
   - Suitable for detailed analysis and visualization

2. **`gnbg_results_TIMESTAMP.pkl`**
   - Pickle file with complete results including convergence histories
   - Use for detailed convergence analysis

3. **`gnbg_summary_TIMESTAMP.csv`**
   - Aggregated statistics per (EDA, problem) pair
   - Columns: success_rate, mean_error, std_error, mean_fe, etc.
   - Suitable for statistical comparison

## Performance Metrics

The benchmark tracks:

- **Best fitness**: Best solution found
- **Error from optimum**: |best_fitness - optimum_value|
- **Success**: Whether error < acceptance_threshold
- **Function evaluations**: Total number of fitness evaluations
- **Acceptance reach point**: FE when acceptance threshold was reached (if ever)
- **Runtime**: Wall-clock time in seconds
- **Convergence history**: Best fitness over time (sampled)

## Example Results Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('gnbg_results/gnbg_results_TIMESTAMP.csv')

# Calculate success rates per EDA
success_rates = df.groupby('eda_name')['success'].mean()
print(success_rates)

# Plot error distribution
df.boxplot(column='error_from_optimum', by='eda_name', figsize=(10, 6))
plt.yscale('log')
plt.ylabel('Error from Optimum')
plt.title('Error Distribution by EDA')
plt.show()

# Compare performance on specific problem
problem_1 = df[df['problem_index'] == 1]
print(problem_1.groupby('eda_name')['error_from_optimum'].describe())
```

## GNBG Problem Characteristics

| Problem | Dimension | Components | Characteristics |
|---------|-----------|------------|-----------------|
| f1-f6   | 5         | 1          | Low-dimensional, unimodal |
| f7-f12  | 10        | 1-5        | Medium-dimensional, varying modality |
| f13-f18 | 20        | 1-10       | High-dimensional, multimodal |
| f19-f24 | 30        | 5-15       | Very high-dimensional, complex |

## Customization

### Adding New EDAs

To add a new continuous EDA to the benchmark:

1. Implement the learning and sampling components in pateda
2. Add a configuration in `get_eda_configuration()`:

```python
elif eda_name == 'my_new_eda':
    components = EDAComponents(
        learning=LearnMyNewEDA(...),
        sampling=SampleMyNewEDA(n_samples=pop_size),
        selection=TruncationSelection(ratio=selection_ratio),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
```

3. Add to `EDA_NAMES` list in the main block

### Adjusting Parameters

Common parameters to adjust:

- **`pop_size`**: Population size (default: 100)
  - Smaller: faster, may converge prematurely
  - Larger: slower, better exploration

- **`selection_ratio`**: Fraction selected for learning (default: 0.5)
  - Lower: stronger selection pressure
  - Higher: more diversity

- **`n_runs`**: Independent runs per configuration (default: 5)
  - More runs: better statistics, longer runtime

## Testing

Run tests to verify the benchmark:

```bash
# Quick syntax check
python3 -m py_compile benchmarks/gnbg_benchmark.py

# Run quick example
python benchmarks/gnbg_single_example.py

# Run full test suite (requires pytest)
pytest tests/test_gnbg_benchmark.py -v
```

## Troubleshooting

### Common Issues

1. **"No module named scipy"**
   - Install: `pip install scipy`

2. **"GNBG instances not found"**
   - Verify path: `pateda/functions/GNBG_Instances.Python-main/`
   - Check that .mat files (f1.mat - f24.mat) exist

3. **"Singular covariance matrix" error**
   - Increase population size
   - Check regularization parameter in LearnGaussianFull

4. **Very slow execution**
   - Reduce `pop_size`
   - Test on subset of problems first
   - Use fewer `n_runs`

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{yazdani2023gnbg,
  title={GNBG: A Generalized and Configurable Benchmark Generator for Continuous Numerical Optimization},
  author={Yazdani, D. and Omidvar, M. N. and Yazdani, D. and Deb, K. and Gandomi, A. H.},
  journal={arXiv preprint arXiv:2312.07083},
  year={2023}
}
```

## License

This benchmark implementation follows the same license as pateda.
