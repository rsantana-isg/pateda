"""
Test script for new alternative sampling strategies in Dendiff EDA.

Tests the three new strategies:
1. Straight-Through Estimator (STE)
2. Hard Concrete Distribution
3. Deterministic Softmax

This script verifies basic functionality of learning and sampling.
"""

import sys

import numpy as np
import pytest
import torch

# Import learning functions
from pateda_nn.learning.discrete_dendiff_ste import learn_discrete_dendiff_ste
from pateda_nn.learning.discrete_dendiff_hard_concrete import learn_discrete_dendiff_hard_concrete
from pateda_nn.learning.discrete_dendiff_deterministic import learn_discrete_dendiff_deterministic

# Import sampling functions
from pateda_nn.sampling.discrete_dendiff import (
    sample_discrete_dendiff_ste,
    sample_discrete_dendiff_hard_concrete,
    sample_discrete_dendiff_deterministic
)


def _run_strategy(strategy_name, learn_fn, sample_fn, learning_params=None, sampling_params=None):
    """
    Test a single sampling strategy.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy
    learn_fn : callable
        Learning function
    sample_fn : callable
        Sampling function
    learning_params : dict, optional
        Learning parameters
    sampling_params : dict, optional
        Sampling parameters
    """
    print(f"\n{'='*70}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*70}")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create simple binary population (OneMax-like)
    n_vars = 20
    pop_size = 50
    population = np.random.randint(0, 2, (pop_size, n_vars)).astype(np.float32)
    fitness = np.sum(population, axis=1)
    
    print(f"Population shape: {population.shape}")
    print(f"Fitness range: [{fitness.min():.2f}, {fitness.max():.2f}]")
    
    # Set default learning params
    if learning_params is None:
        learning_params = {
            'n_timesteps': 20,
            'epochs': 10,
            'batch_size': 10,
            'hidden_dims': [32, 16],
            'learning_rate': 1e-3
        }
    
    # Learn model
    print(f"\nLearning model...")
    try:
        model = learn_fn(population, fitness, learning_params)
        print(f"✓ Learning successful")
        print(f"  Model type: {model['type']}")
        print(f"  Input dim: {model['input_dim']}")
        print(f"  Hidden dims: {model['hidden_dims']}")
    except Exception as e:
        print(f"✗ Learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sample from model
    if sampling_params is None:
        sampling_params = {
            'n_steps': 10,
        }
    
    print(f"\nSampling from model...")
    try:
        samples = sample_fn(model, 20, sampling_params)
        print(f"✓ Sampling successful")
        print(f"  Samples shape: {samples.shape}")
        print(f"  Sample values: min={samples.min()}, max={samples.max()}")
        print(f"  Sample dtype: {samples.dtype}")
        
        # Check samples are binary
        unique_values = np.unique(samples)
        if set(unique_values).issubset({0, 1}):
            print(f"  ✓ Samples are binary (values: {unique_values})")
        else:
            print(f"  ✗ Warning: Non-binary values found: {unique_values}")
        
        # Show sample statistics
        sample_fitness = np.sum(samples, axis=1)
        print(f"  Sample fitness: mean={sample_fitness.mean():.2f}, std={sample_fitness.std():.2f}")
        print(f"  Original fitness: mean={fitness.mean():.2f}, std={fitness.std():.2f}")
        
    except Exception as e:
        print(f"✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✓ {strategy_name} test passed!")
    return True


_STRATEGY_CASES = [
    (
        "STE",
        learn_discrete_dendiff_ste,
        sample_discrete_dendiff_ste,
        {"n_timesteps": 20, "schedule": "linear", "noise_start": 0.01,
         "noise_end": 0.5, "epochs": 10, "batch_size": 10, "hidden_dims": [32, 16]},
    ),
    (
        "HardConcrete",
        learn_discrete_dendiff_hard_concrete,
        sample_discrete_dendiff_hard_concrete,
        {"n_timesteps": 20, "schedule": "linear", "beta_start": 0.0001,
         "beta_end": 0.3, "temperature": 0.1, "epochs": 10, "batch_size": 10,
         "hidden_dims": [32, 16]},
    ),
    (
        "Deterministic",
        learn_discrete_dendiff_deterministic,
        sample_discrete_dendiff_deterministic,
        {"n_timesteps": 20, "schedule": "linear", "beta_start": 0.0001,
         "beta_end": 0.3, "epochs": 10, "batch_size": 10, "hidden_dims": [32, 16]},
    ),
]


@pytest.mark.parametrize(
    "name,learn_fn,sample_fn,learning_params",
    _STRATEGY_CASES,
    ids=[c[0] for c in _STRATEGY_CASES],
)
def test_dendiff_strategy(name, learn_fn, sample_fn, learning_params):
    """Learning + sampling round-trips for each new Dendiff strategy."""
    assert _run_strategy(name, learn_fn, sample_fn, learning_params=learning_params)


def main():
    """Run tests for all new strategies."""
    print("\n" + "="*70)
    print("TESTING NEW ALTERNATIVE SAMPLING STRATEGIES")
    print("="*70)
    
    results = {}
    
    # Test 1: Straight-Through Estimator (STE)
    results['STE'] = _run_strategy(
        "Straight-Through Estimator (STE)",
        learn_discrete_dendiff_ste,
        sample_discrete_dendiff_ste,
        learning_params={
            'n_timesteps': 20,
            'schedule': 'linear',
            'noise_start': 0.01,
            'noise_end': 0.5,
            'epochs': 10,
            'batch_size': 10,
            'hidden_dims': [32, 16],
        }
    )
    
    # Test 2: Hard Concrete Distribution
    results['Hard Concrete'] = _run_strategy(
        "Hard Concrete Distribution",
        learn_discrete_dendiff_hard_concrete,
        sample_discrete_dendiff_hard_concrete,
        learning_params={
            'n_timesteps': 20,
            'schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.3,
            'temperature': 0.1,
            'epochs': 10,
            'batch_size': 10,
            'hidden_dims': [32, 16],
        }
    )
    
    # Test 3: Deterministic Softmax
    results['Deterministic'] = _run_strategy(
        "Deterministic Softmax",
        learn_discrete_dendiff_deterministic,
        sample_discrete_dendiff_deterministic,
        learning_params={
            'n_timesteps': 20,
            'schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.3,
            'epochs': 10,
            'batch_size': 10,
            'hidden_dims': [32, 16],
        }
    )
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for strategy, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{strategy:20s}: {status}")
    
    all_passed = all(results.values())
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
