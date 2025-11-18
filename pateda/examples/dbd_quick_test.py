"""
Quick test for DbD-EDA implementation

This script performs a simple validation test of the DbD-EDA algorithms.
Run this to verify that the implementation is working correctly.

Requirements:
- torch>=1.9.0
- numpy>=1.21.0
"""

import numpy as np
import time


def sphere_function(x):
    """Simple sphere function for testing"""
    return np.sum(x**2)


def quick_test_dbd():
    """Quick validation test of DbD-EDA."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from learning.dbd import learn_dbd, sample_univariate_gaussian
        from sampling.dbd import sample_dbd

        print("="*80)
        print("DbD-EDA QUICK VALIDATION TEST")
        print("="*80)

        # Test 1: Basic learning and sampling
        print("\nTest 1: Basic learning and sampling")
        print("-" * 40)

        np.random.seed(42)
        p0 = np.random.normal(0, 1, (50, 5))
        p1 = np.random.normal(3, 0.5, (50, 5))

        print("Learning DbD model...")
        start = time.time()
        model = learn_dbd(
            p0, p1,
            params={'epochs': 10, 'batch_size': 16, 'num_alpha_samples': 5}
        )
        learn_time = time.time() - start
        print(f"✓ Model learned in {learn_time:.3f}s")
        print(f"  Input dimension: {model['input_dim']}")
        print(f"  Hidden dims: {model['hidden_dims']}")

        print("\nSampling from DbD model...")
        start = time.time()
        samples = sample_dbd(model, p0, n_samples=20, params={'num_iterations': 5})
        sample_time = time.time() - start
        print(f"✓ Generated {len(samples)} samples in {sample_time:.3f}s")
        print(f"  Sample shape: {samples.shape}")
        print(f"  Sample mean: {np.mean(samples, axis=0)}")
        print(f"  Expected mean ~3.0, actual mean: {np.mean(samples):.3f}")

        # Test 2: Univariate sampling
        print("\nTest 2: Univariate Gaussian sampling")
        print("-" * 40)

        pop = np.random.normal(2, 1, (100, 5))
        uni_samples = sample_univariate_gaussian(pop, 50)
        print(f"✓ Generated {len(uni_samples)} univariate samples")
        print(f"  Original mean: {np.mean(pop, axis=0)}")
        print(f"  Sampled mean: {np.mean(uni_samples, axis=0)}")

        # Test 3: Simple optimization
        print("\nTest 3: Simple optimization with DbD-CS")
        print("-" * 40)

        from examples.dbd_eda_example import DbDEDA

        n_vars = 5
        bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        eda = DbDEDA(
            variant='CS',
            pop_size=50,
            selection_ratio=0.3,
            dbd_params={'epochs': 10, 'batch_size': 16, 'num_iterations': 5}
        )

        print(f"Running DbD-CS for 10 generations on {n_vars}D sphere function...")
        result = eda.optimize(
            sphere_function,
            n_vars,
            bounds,
            n_generations=10,
            verbose=False
        )

        print(f"✓ Optimization completed")
        print(f"  Initial fitness: {result['fitness_history'][0]:.6e}")
        print(f"  Final fitness: {result['best_fitness']:.6e}")
        print(f"  Improvement: {result['fitness_history'][0] / max(result['best_fitness'], 1e-10):.2f}x")

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = quick_test_dbd()
    exit(0 if success else 1)
