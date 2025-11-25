"""
Benchmark script for evaluating dendiff model quality across different distributions.

This script evaluates the ability of learn_dendiff and sample_dendiff functions
to accurately approximate various probability distributions and generates a
comprehensive report of the results.

Usage:
    python benchmark_dendiff_distributions.py
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, Tuple, Callable, List
import sys
import os

# Add pateda to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff


# ============================================================================
# Distribution Generators
# ============================================================================

def generate_univariate_gaussian(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent univariate Gaussian distributions."""
    np.random.seed(seed)
    means = np.random.uniform(-2, 2, n_vars)
    stds = np.random.uniform(0.5, 2.0, n_vars)
    samples = np.random.randn(n_samples, n_vars) * stds + means

    metadata = {
        'type': 'Univariate Gaussian',
        'means': means,
        'stds': stds
    }
    return samples, metadata


def generate_multivariate_gaussian(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from a multivariate Gaussian with predetermined eigenvectors."""
    np.random.seed(seed)
    mean = np.random.uniform(-1, 1, n_vars)
    Q, _ = np.linalg.qr(np.random.randn(n_vars, n_vars))
    eigenvalues = np.random.uniform(0.5, 3.0, n_vars)
    cov = Q @ np.diag(eigenvalues) @ Q.T
    samples = np.random.multivariate_normal(mean, cov, n_samples)

    metadata = {
        'type': 'Multivariate Gaussian',
        'mean': mean,
        'covariance': cov,
        'eigenvalues': eigenvalues
    }
    return samples, metadata


def generate_multivariate_cauchy(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent Cauchy distributions."""
    np.random.seed(seed)
    locations = np.random.uniform(-1, 1, n_vars)
    scales = np.random.uniform(0.3, 1.0, n_vars)

    samples = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        samples[:, i] = stats.cauchy.rvs(
            loc=locations[i],
            scale=scales[i],
            size=n_samples,
            random_state=seed + i
        )
    samples = np.clip(samples, -10, 10)

    metadata = {
        'type': 'Cauchy',
        'locations': locations,
        'scales': scales
    }
    return samples, metadata


def generate_mixture_of_gaussians(
    n_samples: int,
    n_vars: int,
    n_components: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from a Gaussian mixture model."""
    np.random.seed(seed)
    weights = np.random.dirichlet(np.ones(n_components))

    means = []
    covs = []
    for i in range(n_components):
        mean = np.random.uniform(-3, 3, n_vars)
        means.append(mean)
        std = np.random.uniform(0.3, 1.5, n_vars)
        cov = np.diag(std ** 2)
        covs.append(cov)

    samples = []
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    for i in range(n_samples):
        comp = component_assignments[i]
        sample = np.random.multivariate_normal(means[comp], covs[comp])
        samples.append(sample)
    samples = np.array(samples)

    metadata = {
        'type': 'Gaussian Mixture',
        'n_components': n_components,
        'weights': weights,
        'means': means
    }
    return samples, metadata


def generate_uniform(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent uniform distributions."""
    np.random.seed(seed)
    low = np.random.uniform(-3, 0, n_vars)
    high = np.random.uniform(1, 4, n_vars)
    samples = np.random.uniform(low, high, (n_samples, n_vars))

    metadata = {
        'type': 'Uniform',
        'low': low,
        'high': high
    }
    return samples, metadata


# ============================================================================
# Distribution Comparison Metrics
# ============================================================================

def compute_js_divergence(
    samples1: np.ndarray,
    samples2: np.ndarray,
    n_bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence (symmetric version of KL)."""
    n_vars = samples1.shape[1]
    js_divs = []

    for i in range(n_vars):
        data1 = samples1[:, i]
        data2 = samples2[:, i]

        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)

        js = jensenshannon(hist1, hist2, base=2)
        js_divs.append(js)

    return np.mean(js_divs)


def compute_kl_divergence_kde(
    samples1: np.ndarray,
    samples2: np.ndarray,
    n_bins: int = 50
) -> float:
    """Estimate KL divergence between two sample sets using histograms."""
    n_vars = samples1.shape[1]
    kl_divs = []

    for i in range(n_vars):
        data1 = samples1[:, i]
        data2 = samples2[:, i]

        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        kl = np.sum(hist1 * np.log(hist1 / hist2))
        kl_divs.append(kl)

    return np.mean(kl_divs)


def compute_statistical_distance(
    samples1: np.ndarray,
    samples2: np.ndarray
) -> Dict[str, float]:
    """Compute various statistical distance metrics between two sample sets."""
    # Mean difference
    mean1 = np.mean(samples1, axis=0)
    mean2 = np.mean(samples2, axis=0)
    mean_diff = np.linalg.norm(mean1 - mean2)
    mean_diff_relative = mean_diff / (np.linalg.norm(mean1) + 1e-10)

    # Std difference
    std1 = np.std(samples1, axis=0)
    std2 = np.std(samples2, axis=0)
    std_diff = np.linalg.norm(std1 - std2)
    std_diff_relative = std_diff / (np.linalg.norm(std1) + 1e-10)

    # Covariance difference
    cov1 = np.cov(samples1.T)
    cov2 = np.cov(samples2.T)
    cov_diff = np.linalg.norm(cov1 - cov2, 'fro')
    cov_diff_relative = cov_diff / (np.linalg.norm(cov1, 'fro') + 1e-10)

    # Wasserstein distance
    n_vars = samples1.shape[1]
    wasserstein_dists = []
    for i in range(n_vars):
        wd = stats.wasserstein_distance(samples1[:, i], samples2[:, i])
        wasserstein_dists.append(wd)
    mean_wasserstein = np.mean(wasserstein_dists)

    return {
        'mean_diff': mean_diff,
        'mean_diff_relative': mean_diff_relative,
        'std_diff': std_diff,
        'std_diff_relative': std_diff_relative,
        'cov_diff_frobenius': cov_diff,
        'cov_diff_relative': cov_diff_relative,
        'mean_wasserstein': mean_wasserstein
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_dendiff_on_distribution(
    generator: Callable,
    n_samples: int = 500,
    n_vars: int = 10,
    n_samples_test: int = 500,
    params: Dict[str, Any] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate dendiff model on a given distribution.

    Parameters
    ----------
    generator : callable
        Function that generates (samples, metadata)
    n_samples : int
        Number of training samples
    n_vars : int
        Number of variables
    n_samples_test : int
        Number of samples to generate from learned model
    params : dict
        Parameters for learn_dendiff
    seed : int
        Random seed
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        Evaluation results including metrics
    """
    if verbose:
        print(f"  Generating {n_samples} samples from distribution...")

    # Generate original distribution
    original_samples, metadata = generator(n_samples, n_vars, seed)

    # Create associated fitness values (random for this benchmark)
    np.random.seed(seed)
    fitness = np.random.randn(n_samples)

    # Set default params if not provided
    if params is None:
        params = {}

    if verbose:
        print(f"  Learning dendiff model...")

    # Learn dendiff model
    model = learn_dendiff(original_samples, fitness, params=params)

    if verbose:
        print(f"  Sampling {n_samples_test} samples from learned model...")

    # Sample from learned model
    bounds = np.array([
        original_samples.min(axis=0) - 3 * original_samples.std(axis=0),
        original_samples.max(axis=0) + 3 * original_samples.std(axis=0)
    ])

    sampled_data = sample_dendiff(
        model,
        n_samples=n_samples_test,
        bounds=bounds,
        params=params
    )

    if verbose:
        print(f"  Computing metrics...")

    # Compute metrics
    kl_div = compute_kl_divergence_kde(original_samples, sampled_data)
    js_div = compute_js_divergence(original_samples, sampled_data)
    stat_distances = compute_statistical_distance(original_samples, sampled_data)

    results = {
        'distribution_type': metadata['type'],
        'n_vars': n_vars,
        'n_samples': n_samples,
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        **stat_distances
    }

    return results


# ============================================================================
# Main Benchmark Execution
# ============================================================================

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all distributions."""

    print("=" * 80)
    print("DENDIFF DISTRIBUTION APPROXIMATION BENCHMARK")
    print("=" * 80)
    print()

    # Configuration
    n_vars = 10
    n_samples = 500
    n_samples_test = 500

    # Test configurations: (name, generator, params)
    test_configs = [
        (
            "1. Univariate Gaussian",
            generate_univariate_gaussian,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "2. Multivariate Gaussian (with correlations)",
            generate_multivariate_gaussian,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "3. Cauchy Distribution (heavy tails)",
            generate_multivariate_cauchy,
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [128, 64]}
        ),
        (
            "4. Gaussian Mixture (3 components)",
            generate_mixture_of_gaussians,
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [256, 128]}
        ),
        (
            "5. Uniform Distribution",
            generate_uniform,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
    ]

    all_results = []

    for test_name, generator, params in test_configs:
        print(f"\n{test_name}")
        print("-" * 80)

        try:
            results = evaluate_dendiff_on_distribution(
                generator,
                n_samples=n_samples,
                n_vars=n_vars,
                n_samples_test=n_samples_test,
                params=params,
                seed=42,
                verbose=True
            )

            all_results.append((test_name, results))

            print(f"\n  Results:")
            print(f"    KL Divergence:           {results['kl_divergence']:.6f}")
            print(f"    JS Divergence:           {results['js_divergence']:.6f}")
            print(f"    Mean Difference:         {results['mean_diff']:.6f}")
            print(f"    Std Difference:          {results['std_diff']:.6f}")
            print(f"    Mean Wasserstein Dist:   {results['mean_wasserstein']:.6f}")
            print(f"    Relative Mean Diff:      {results['mean_diff_relative']:.4f}")
            print(f"    Relative Std Diff:       {results['std_diff_relative']:.4f}")
            print(f"    Relative Cov Diff:       {results['cov_diff_relative']:.4f}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Distribution':<45} {'JS Div':<12} {'KL Div':<12} {'Wasserstein':<12}")
    print("-" * 80)

    for test_name, results in all_results:
        dist_name = test_name.split('. ', 1)[1] if '. ' in test_name else test_name
        print(f"{dist_name:<45} {results['js_divergence']:>10.4f}  "
              f"{results['kl_divergence']:>10.4f}  "
              f"{results['mean_wasserstein']:>10.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
JS Divergence (Jensen-Shannon):
  - Range: [0, 1] (base 2), where 0 = identical distributions
  - < 0.1: Excellent approximation
  - 0.1-0.3: Good approximation
  - 0.3-0.5: Moderate approximation
  - > 0.5: Poor approximation

KL Divergence (Kullback-Leibler):
  - Range: [0, ∞), where 0 = identical distributions
  - < 0.1: Excellent approximation
  - 0.1-0.5: Good approximation
  - 0.5-1.0: Moderate approximation
  - > 1.0: Poor approximation

Wasserstein Distance:
  - Range: [0, ∞), where 0 = identical distributions
  - Lower is better
  - Depends on scale of the data

Notes:
  - Dendiff models are based on Denoising Diffusion Probabilistic Models
  - Performance depends on: sample size, model capacity, training epochs, and timesteps
  - Gaussian distributions are typically easier to approximate than heavy-tailed or mixture distributions
  - Correlations in multivariate distributions add complexity
""")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
For better approximation quality:
  1. Increase sample size (n_samples > 500)
  2. Increase number of training epochs (epochs > 100)
  3. Increase diffusion timesteps (n_timesteps > 500)
  4. Use larger network architectures (hidden_dims = [256, 128, 64])
  5. For complex distributions (mixtures, heavy tails):
     - Use more training epochs (150-200)
     - Use more timesteps (800-1000)
     - Consider using cosine schedule: beta_schedule='cosine'

Parameter trade-offs:
  - More timesteps = better quality but slower sampling
  - Larger networks = better capacity but slower training
  - More epochs = better fit but risk of overfitting with small samples
""")

    return all_results


# ============================================================================
# Parameter Variation Tests
# ============================================================================

def test_parameter_variations():
    """Test how different parameters affect approximation quality."""

    print("\n\n" + "=" * 80)
    print("PARAMETER VARIATION TESTS")
    print("=" * 80)

    # Test 1: Varying timesteps
    print("\n\nTest 1: Effect of Number of Diffusion Timesteps")
    print("-" * 80)
    print(f"{'Timesteps':<15} {'JS Divergence':<15} {'KL Divergence':<15} {'Wasserstein':<15}")
    print("-" * 80)

    for n_timesteps in [100, 300, 500, 1000]:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_gaussian,
            n_samples=300,
            n_vars=10,
            n_samples_test=300,
            params={'epochs': 80, 'n_timesteps': n_timesteps, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{n_timesteps:<15} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f} {results['mean_wasserstein']:<15.4f}")

    # Test 2: Varying architectures
    print("\n\nTest 2: Effect of Network Architecture")
    print("-" * 80)
    print(f"{'Architecture':<25} {'JS Divergence':<15} {'KL Divergence':<15}")
    print("-" * 80)

    architectures = [
        ([64, 32], "[64, 32]"),
        ([128, 64], "[128, 64]"),
        ([256, 128], "[256, 128]"),
        ([256, 128, 64], "[256, 128, 64]"),
    ]

    for hidden_dims, arch_str in architectures:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_gaussian,
            n_samples=400,
            n_vars=10,
            n_samples_test=400,
            params={'epochs': 80, 'n_timesteps': 500, 'hidden_dims': hidden_dims},
            seed=42,
            verbose=False
        )
        print(f"{arch_str:<25} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f}")

    # Test 3: Varying epochs
    print("\n\nTest 3: Effect of Training Epochs")
    print("-" * 80)
    print(f"{'Epochs':<15} {'JS Divergence':<15} {'KL Divergence':<15}")
    print("-" * 80)

    for epochs in [20, 50, 100, 150]:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_gaussian,
            n_samples=400,
            n_vars=10,
            n_samples_test=400,
            params={'epochs': epochs, 'n_timesteps': 500, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{epochs:<15} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import time

    start_time = time.time()

    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()

    # Run parameter variation tests
    test_parameter_variations()

    elapsed_time = time.time() - start_time

    print("\n\n" + "=" * 80)
    print(f"Benchmark completed in {elapsed_time:.2f} seconds")
    print("=" * 80)
