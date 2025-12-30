"""
Comparison of Standard Dendiff vs ReLU Variant

This example demonstrates the differences between:
1. Standard dendiff: SiLU activation + sinusoidal time embedding
2. ReLU variant: ReLU activation + raw timestep input

The comparison shows:
- Training time
- Model size (number of parameters)
- Sample quality
- Distribution approximation
"""

import sys
import os

# Add parent directory to path for running examples without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import time

# Import both variants
from pateda.learning.dendiff import learn_dendiff
from pateda.learning.dendiff_relu import learn_dendiff_relu
from pateda.sampling.dendiff import sample_dendiff
from pateda.sampling.dendiff_relu import sample_dendiff_relu


def count_parameters(model_state):
    """Count total number of parameters in model."""
    total = 0
    for key, tensor in model_state.items():
        total += tensor.numel()
    return total


def generate_test_data(n_samples=500, n_vars=10):
    """Generate test data from a multivariate Gaussian mixture."""
    # Create a mixture of 3 Gaussians
    n_per_component = n_samples // 3

    # Component 1: centered at origin
    data1 = np.random.randn(n_per_component, n_vars) * 0.5

    # Component 2: shifted
    data2 = np.random.randn(n_per_component, n_vars) * 0.3 + 2.0

    # Component 3: shifted differently
    data3 = np.random.randn(n_samples - 2*n_per_component, n_vars) * 0.4 - 1.5

    data = np.vstack([data1, data2, data3])
    np.random.shuffle(data)

    # Create dummy fitness (not used)
    fitness = np.sum(data**2, axis=1)

    return data, fitness


def compute_kl_divergence_estimate(samples1, samples2):
    """Estimate KL divergence using KDE (simplified)."""
    # Use KS test as a proxy for distribution similarity
    # Lower p-value = more different distributions
    ks_stats = []
    for dim in range(samples1.shape[1]):
        stat, _ = ks_2samp(samples1[:, dim], samples2[:, dim])
        ks_stats.append(stat)
    return np.mean(ks_stats)


def run_comparison():
    """Run comprehensive comparison between standard and ReLU variants."""

    print("=" * 80)
    print("DENDIFF COMPARISON: Standard (SiLU + Sinusoidal) vs ReLU (Simple)")
    print("=" * 80)
    print()

    # Generate test data
    print("Generating test data...")
    n_samples_train = 500
    n_vars = 10
    data, fitness = generate_test_data(n_samples_train, n_vars)

    print(f"  Training data: {data.shape}")
    print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
    print()

    # Common parameters
    common_params = {
        'n_timesteps': 500,
        'hidden_dims': [128, 64],
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3
    }

    # ========================================================================
    # Train Standard Dendiff (SiLU + Sinusoidal)
    # ========================================================================
    print("=" * 80)
    print("TRAINING STANDARD DENDIFF (SiLU + Sinusoidal Time Embedding)")
    print("=" * 80)
    print()

    start_time = time.time()
    model_standard = learn_dendiff(data, fitness, params=common_params)
    time_standard = time.time() - start_time

    params_standard = count_parameters(model_standard['model_state'])

    print(f"\n✓ Training completed in {time_standard:.2f} seconds")
    print(f"  Model parameters: {params_standard:,}")
    print()

    # ========================================================================
    # Train ReLU Variant
    # ========================================================================
    print("=" * 80)
    print("TRAINING RELU VARIANT (ReLU + Raw Timestep)")
    print("=" * 80)
    print()

    start_time = time.time()
    model_relu = learn_dendiff_relu(data, fitness, params=common_params)
    time_relu = time.time() - start_time

    params_relu = count_parameters(model_relu['model_state'])

    print(f"\n✓ Training completed in {time_relu:.2f} seconds")
    print(f"  Model parameters: {params_relu:,}")
    print()

    # ========================================================================
    # Generate Samples
    # ========================================================================
    print("=" * 80)
    print("GENERATING SAMPLES")
    print("=" * 80)
    print()

    n_samples_test = 1000

    print("Sampling from standard dendiff...")
    start_time = time.time()
    samples_standard = sample_dendiff(model_standard, n_samples_test)
    time_sample_standard = time.time() - start_time

    print("Sampling from ReLU variant...")
    start_time = time.time()
    samples_relu = sample_dendiff_relu(model_relu, n_samples_test)
    time_sample_relu = time.time() - start_time

    print(f"\n✓ Standard: {n_samples_test} samples in {time_sample_standard:.2f}s")
    print(f"✓ ReLU:     {n_samples_test} samples in {time_sample_relu:.2f}s")
    print()

    # ========================================================================
    # Evaluate Sample Quality
    # ========================================================================
    print("=" * 80)
    print("SAMPLE QUALITY EVALUATION")
    print("=" * 80)
    print()

    # Compare distributions
    ks_standard = compute_kl_divergence_estimate(data, samples_standard)
    ks_relu = compute_kl_divergence_estimate(data, samples_relu)

    # Compute statistics
    print("Distribution Statistics:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Training Data':<15} {'Standard':<15} {'ReLU':<15}")
    print("-" * 80)
    print(f"{'Mean (dim 0)':<30} {data[:, 0].mean():>14.4f} {samples_standard[:, 0].mean():>14.4f} {samples_relu[:, 0].mean():>14.4f}")
    print(f"{'Std (dim 0)':<30} {data[:, 0].std():>14.4f} {samples_standard[:, 0].std():>14.4f} {samples_relu[:, 0].std():>14.4f}")
    print(f"{'Mean (all dims)':<30} {data.mean():>14.4f} {samples_standard.mean():>14.4f} {samples_relu.mean():>14.4f}")
    print(f"{'Std (all dims)':<30} {data.std():>14.4f} {samples_standard.std():>14.4f} {samples_relu.std():>14.4f}")
    print(f"{'KS Distance (avg)':<30} {'N/A':>14} {ks_standard:>14.4f} {ks_relu:>14.4f}")
    print()

    # ========================================================================
    # Summary Comparison
    # ========================================================================
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)
    print()

    print("Architecture:")
    print("-" * 80)
    print(f"{'Aspect':<35} {'Standard':<25} {'ReLU Variant':<25}")
    print("-" * 80)
    print(f"{'Activation Function':<35} {'SiLU/Swish':<25} {'ReLU':<25}")
    print(f"{'Time Encoding':<35} {'Sinusoidal (32-dim)':<25} {'Raw scalar (normalized)':<25}")
    print(f"{'Additional Embedding Layer':<35} {'Yes':<25} {'No':<25}")
    print(f"{'Total Parameters':<35} {f'{params_standard:,}':<25} {f'{params_relu:,}':<25}")

    param_reduction = ((params_standard - params_relu) / params_standard) * 100
    print(f"{'Parameter Reduction':<35} {'—':<25} {f'{param_reduction:.1f}%':<25}")
    print()

    print("Performance:")
    print("-" * 80)
    print(f"{'Metric':<35} {'Standard':<25} {'ReLU Variant':<25}")
    print("-" * 80)
    print(f"{'Training Time':<35} {f'{time_standard:.2f}s':<25} {f'{time_relu:.2f}s':<25}")

    speedup = ((time_standard - time_relu) / time_standard) * 100
    print(f"{'Training Speedup':<35} {'—':<25} {f'{speedup:+.1f}%':<25}")

    print(f"{'Sampling Time ({n_samples_test} samples)':<35} {f'{time_sample_standard:.2f}s':<25} {f'{time_sample_relu:.2f}s':<25}")

    sample_speedup = ((time_sample_standard - time_sample_relu) / time_sample_standard) * 100
    print(f"{'Sampling Speedup':<35} {'—':<25} {f'{sample_speedup:+.1f}%':<25}")
    print()

    print("Sample Quality:")
    print("-" * 80)
    print(f"{'Metric':<35} {'Standard':<25} {'ReLU Variant':<25}")
    print("-" * 80)
    print(f"{'KS Distance (lower = better)':<35} {f'{ks_standard:.4f}':<25} {f'{ks_relu:.4f}':<25}")

    quality_diff = ((ks_relu - ks_standard) / ks_standard) * 100
    if quality_diff > 0:
        quality_str = f'{quality_diff:+.1f}% worse'
    else:
        quality_str = f'{abs(quality_diff):.1f}% better'
    print(f"{'Quality Difference':<35} {'—':<25} {quality_str:<25}")
    print()

    # ========================================================================
    # Recommendations
    # ========================================================================
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("Use STANDARD DENDIFF (SiLU + Sinusoidal) when:")
    print("  • Maximum sample quality is critical")
    print("  • Working with complex, high-dimensional distributions")
    print("  • Computational resources are not a constraint")
    print("  • Following best practices from diffusion model literature")
    print()

    print("Use RELU VARIANT (ReLU + Raw Timestep) when:")
    print("  • Training/sampling speed is important")
    print("  • Memory is constrained (fewer parameters)")
    print("  • Working with simpler, lower-dimensional problems")
    print("  • Computational efficiency is prioritized over marginal quality gains")
    print()

    print(f"In this test:")
    if param_reduction > 15:
        print(f"  ✓ ReLU variant has {param_reduction:.1f}% fewer parameters")
    if speedup > 5:
        print(f"  ✓ ReLU variant trains {speedup:.1f}% faster")
    if abs(quality_diff) < 10:
        print(f"  ✓ Sample quality is comparable (within {abs(quality_diff):.1f}%)")

    print()
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_comparison()
