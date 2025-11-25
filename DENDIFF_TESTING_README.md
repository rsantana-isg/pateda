# Dendiff Distribution Approximation Testing

This document describes the comprehensive testing framework for evaluating the `learn_dendiff` and `sample_dendiff` functions in the PATEDA library.

## Overview

The testing framework evaluates how accurately the denoising diffusion model (dendiff) can approximate various probability distributions. Two main scripts have been created:

1. **`pateda/tests/test_dendiff_distributions.py`** - PyTest-based test suite
2. **`benchmark_dendiff_distributions.py`** - Standalone benchmark script with detailed reporting

## Installation

Before running the tests, ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- numpy >= 1.21.0
- scipy >= 1.7.0
- torch >= 2.0.0
- pytest >= 7.0.0

## Test Distributions

The framework tests dendiff on the following probability distributions:

### 1. Univariate Gaussian Distribution
- **Description**: Independent Gaussian distributions for each variable
- **Parameters**: Different means and standard deviations for each dimension
- **Difficulty**: Easy (baseline test)

### 2. Multivariate Gaussian Distribution
- **Description**: Correlated Gaussian with predetermined eigenvectors
- **Parameters**: Random covariance matrix with controlled eigenvalues
- **Difficulty**: Medium (tests ability to capture correlations)

### 3. Multivariate Cauchy Distribution
- **Description**: Independent Cauchy distributions (heavy-tailed)
- **Parameters**: Different locations and scales
- **Difficulty**: Hard (tests robustness to outliers and heavy tails)

### 4. Gaussian Mixture Model
- **Description**: Mixture of 3 Gaussian components
- **Parameters**: Random means, covariances, and mixture weights
- **Difficulty**: Hard (tests multi-modal distribution learning)

### 5. Uniform Distribution
- **Description**: Independent uniform distributions
- **Parameters**: Different ranges for each dimension
- **Difficulty**: Medium (tests ability to learn sharp boundaries)

## Evaluation Metrics

The framework computes several metrics to compare original and sampled distributions:

### Distance Metrics

1. **Jensen-Shannon (JS) Divergence**
   - Range: [0, 1] (base 2)
   - Symmetric version of KL divergence
   - Interpretation:
     - < 0.1: Excellent approximation
     - 0.1-0.3: Good approximation
     - 0.3-0.5: Moderate approximation
     - > 0.5: Poor approximation

2. **Kullback-Leibler (KL) Divergence**
   - Range: [0, ∞)
   - Measures information loss
   - Interpretation:
     - < 0.1: Excellent approximation
     - 0.1-0.5: Good approximation
     - 0.5-1.0: Moderate approximation
     - > 1.0: Poor approximation

3. **Wasserstein Distance**
   - Optimal transport distance
   - Lower is better
   - Scale-dependent

### Statistical Metrics

1. **Mean Difference**: L2 norm of difference between means
2. **Standard Deviation Difference**: L2 norm of difference between standard deviations
3. **Covariance Difference**: Frobenius norm of difference between covariance matrices
4. **Relative Metrics**: Normalized versions of the above

## Running the Tests

### Option 1: PyTest Test Suite

Run all tests:
```bash
cd /home/user/pateda
pytest pateda/tests/test_dendiff_distributions.py -v -s
```

Run specific test class:
```bash
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffUnivariateGaussian -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffMultivariateGaussian -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffCauchy -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffMixture -v -s
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffUniform -v -s
```

Run parameter variation tests:
```bash
pytest pateda/tests/test_dendiff_distributions.py::TestDendiffParameterVariations -v -s
```

### Option 2: Standalone Benchmark Script

Run comprehensive benchmark:
```bash
cd /home/user/pateda
python benchmark_dendiff_distributions.py
```

This will:
1. Test all 5 distribution types
2. Report detailed metrics for each
3. Run parameter variation experiments
4. Generate a comprehensive summary report

Expected runtime: 5-15 minutes depending on hardware

## Test Structure

### Test Workflow

For each distribution:

1. **Generate Original Data**
   ```python
   original_samples, metadata = generator(n_samples=500, n_vars=10, seed=42)
   ```

2. **Create Fitness Vector**
   ```python
   fitness = np.random.randn(n_samples)  # Random fitness for testing
   ```

3. **Learn Dendiff Model**
   ```python
   model = learn_dendiff(original_samples, fitness, params=params)
   ```

4. **Sample from Learned Model**
   ```python
   sampled_data = sample_dendiff(model, n_samples=500, bounds=bounds, params=params)
   ```

5. **Compare Distributions**
   - Compute JS divergence, KL divergence, Wasserstein distance
   - Compute statistical differences (mean, std, covariance)
   - Report results

### Key Parameters

Default parameters used in tests:
```python
params = {
    'epochs': 100,              # Number of training epochs
    'n_timesteps': 500,         # Number of diffusion timesteps
    'hidden_dims': [128, 64],   # Network architecture
    'batch_size': None,         # Auto-computed based on data
    'learning_rate': 1e-3,      # Adam learning rate
    'beta_schedule': 'linear',  # Noise schedule type
    'time_emb_dim': 32,         # Time embedding dimension
}
```

## Parameter Tuning Recommendations

### For Better Quality

1. **Increase Sample Size**
   - More data → better model learning
   - Recommended: n_samples >= 500

2. **Increase Training Epochs**
   - More training → better convergence
   - Recommended: epochs >= 100
   - For complex distributions: epochs >= 150

3. **Increase Diffusion Timesteps**
   - More steps → smoother denoising process
   - Recommended: n_timesteps >= 500
   - For complex distributions: n_timesteps >= 800

4. **Larger Network Architecture**
   - More capacity → better approximation
   - Recommended: hidden_dims = [128, 64]
   - For complex distributions: hidden_dims = [256, 128] or [256, 128, 64]

5. **Alternative Beta Schedule**
   - Try cosine schedule for better stability
   - Set: beta_schedule='cosine'

### Trade-offs

- **More timesteps**: Better quality but slower sampling
- **Larger networks**: Better capacity but slower training
- **More epochs**: Better fit but risk of overfitting with small samples

## Expected Results

### Typical Performance by Distribution Type

| Distribution | Expected JS Div | Expected KL Div | Difficulty |
|--------------|----------------|-----------------|------------|
| Univariate Gaussian | < 0.3 | < 0.5 | Easy |
| Multivariate Gaussian | < 0.4 | < 0.8 | Medium |
| Cauchy | < 1.0 | < 2.0 | Hard |
| Gaussian Mixture | < 0.6 | < 1.0 | Hard |
| Uniform | < 0.8 | < 1.5 | Medium |

Note: Actual results may vary based on:
- Random seed
- Sample size
- Model parameters
- Hardware/computation

## Understanding the Results

### Good Performance Indicators

✓ JS divergence < 0.4
✓ Mean difference relative < 0.5
✓ Std difference relative < 0.5
✓ Model successfully captures the general shape of the distribution

### Poor Performance Indicators

✗ JS divergence > 0.7
✗ Mean difference relative > 1.0
✗ Sampled distribution looks very different from original
✗ Training loss not converging

### Common Issues

1. **High divergence with all distributions**
   - Solution: Increase epochs, use larger network, more timesteps

2. **Good performance on Gaussian, poor on others**
   - Expected: Dendiff works best with smooth, continuous distributions
   - Solution: Use specialized parameters for complex distributions

3. **Poor correlation capture in multivariate**
   - Solution: Increase network capacity, more training epochs

4. **Mode collapse in mixtures**
   - Solution: Increase timesteps, use cosine schedule

## Example Usage

```python
import numpy as np
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff

# Generate test data (e.g., multivariate Gaussian)
mean = np.zeros(10)
cov = np.eye(10)
original_samples = np.random.multivariate_normal(mean, cov, 500)
fitness = np.random.randn(500)

# Learn model
params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [128, 64]
}
model = learn_dendiff(original_samples, fitness, params=params)

# Sample from model
bounds = np.array([
    original_samples.min(axis=0) - 3,
    original_samples.max(axis=0) + 3
])
sampled_data = sample_dendiff(model, n_samples=500, bounds=bounds)

# Compare distributions
from scipy.spatial.distance import jensenshannon
# ... compute metrics ...
```

## Files Created

1. **`pateda/tests/test_dendiff_distributions.py`**
   - Comprehensive pytest test suite
   - Tests all distribution types
   - Tests parameter variations
   - Includes assertions for quality thresholds

2. **`benchmark_dendiff_distributions.py`**
   - Standalone benchmark script
   - Generates detailed reports
   - Includes interpretation guides
   - Tests parameter variations

3. **`DENDIFF_TESTING_README.md`** (this file)
   - Documentation and usage guide

## References

The dendiff implementation is based on:

1. **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020)
2. **Improved Denoising Diffusion Probabilistic Models** (Nichol & Dhariwal, ICML 2021)
3. **DiffImpute: Tabular Data Imputation with Denoising Diffusion Probabilistic Model** (Nazzal et al., 2024)

## Contact

For questions or issues with the testing framework, please refer to the PATEDA project documentation or create an issue in the project repository.
