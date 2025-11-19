"""
Simple tests for Generalized Mallows models (without pytest)
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from module files to avoid torch dependency issues
from pateda.learning.mallows import (
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
)
from pateda.sampling.mallows import (
    SampleGeneralizedMallowsKendall,
    SampleGeneralizedMallowsCayley,
)
# Import distance functions
import importlib.util
spec = importlib.util.spec_from_file_location("distances", "pateda/permutation/distances.py")
distances = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distances)
kendall_distance = distances.kendall_distance
cayley_distance = distances.cayley_distance


def test_generalized_mallows_kendall():
    """Test Generalized Mallows with Kendall distance"""
    print("\n=== Testing Generalized Mallows Kendall ===")
    np.random.seed(42)

    # Create a simple population
    n_vars = 5
    pop_size = 20
    population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
    fitness = np.random.rand(pop_size)

    # Learn model
    print("Learning model...")
    learner = LearnGeneralizedMallowsKendall()
    model = learner(
        generation=0,
        n_vars=n_vars,
        cardinality=np.arange(n_vars),
        selected_pop=population,
        selected_fitness=fitness,
        initial_theta=0.1,
        upper_theta=10.0,
        max_iter=100,
        consensus_method="borda",
    )

    print(f"Model type: {model['model_type']}")
    print(f"Consensus: {model['consensus']}")
    print(f"Theta vector: {model['theta']}")
    print(f"V-probs shape: {model['v_probs'].shape}")

    # Check model structure
    assert "v_probs" in model
    assert "consensus" in model
    assert "theta" in model
    assert "psis" in model
    assert model["model_type"] == "generalized_mallows_kendall"

    # Check dimensions
    assert model["v_probs"].shape == (n_vars - 1, n_vars)
    assert len(model["consensus"]) == n_vars
    assert len(model["theta"]) == n_vars - 1
    assert len(model["psis"]) == n_vars - 1

    print("✓ Model structure correct")

    # Sample from model
    print("\nSampling from model...")
    sampler = SampleGeneralizedMallowsKendall()
    sample_size = 30
    new_pop = sampler(
        n_vars=n_vars,
        model=model,
        cardinality=np.arange(n_vars),
        population=population,
        fitness=fitness,
        sample_size=sample_size,
    )

    print(f"Sampled population shape: {new_pop.shape}")
    print(f"First 3 samples:\n{new_pop[:3]}")

    # Check dimensions
    assert new_pop.shape == (sample_size, n_vars)

    # Check that all samples are valid permutations
    for i in range(sample_size):
        assert set(new_pop[i]) == set(range(n_vars)), f"Sample {i} is not a valid permutation"

    print("✓ All samples are valid permutations")

    print("\n✓✓✓ Generalized Mallows Kendall test PASSED ✓✓✓")


def test_generalized_mallows_cayley():
    """Test Generalized Mallows with Cayley distance"""
    print("\n=== Testing Generalized Mallows Cayley ===")
    np.random.seed(42)

    # Create a simple population
    n_vars = 6
    pop_size = 25
    population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
    fitness = np.random.rand(pop_size)

    # Learn model
    print("Learning model...")
    learner = LearnGeneralizedMallowsCayley()
    model = learner(
        generation=0,
        n_vars=n_vars,
        cardinality=np.arange(n_vars),
        selected_pop=population,
        selected_fitness=fitness,
        initial_theta=0.1,
        upper_theta=10.0,
        max_iter=100,
        consensus_method="borda",
    )

    print(f"Model type: {model['model_type']}")
    print(f"Consensus: {model['consensus']}")
    print(f"Theta vector: {model['theta']}")
    print(f"X-probs shape: {model['x_probs'].shape}")

    # Check model structure
    assert "x_probs" in model
    assert "consensus" in model
    assert "theta" in model
    assert "psis" in model
    assert model["model_type"] == "generalized_mallows_cayley"

    # Check dimensions
    assert model["x_probs"].shape == (n_vars - 1, 2)
    assert len(model["consensus"]) == n_vars
    assert len(model["theta"]) == n_vars - 1
    assert len(model["psis"]) == n_vars - 1

    print("✓ Model structure correct")

    # Sample from model
    print("\nSampling from model...")
    sampler = SampleGeneralizedMallowsCayley()
    sample_size = 40
    new_pop = sampler(
        n_vars=n_vars,
        model=model,
        cardinality=np.arange(n_vars),
        population=population,
        fitness=fitness,
        sample_size=sample_size,
    )

    print(f"Sampled population shape: {new_pop.shape}")
    print(f"First 3 samples:\n{new_pop[:3]}")

    # Check dimensions
    assert new_pop.shape == (sample_size, n_vars)

    # Check that all samples are valid permutations
    for i in range(sample_size):
        assert set(new_pop[i]) == set(range(n_vars)), f"Sample {i} is not a valid permutation"

    print("✓ All samples are valid permutations")

    print("\n✓✓✓ Generalized Mallows Cayley test PASSED ✓✓✓")


def main():
    """Run all tests"""
    try:
        test_generalized_mallows_kendall()
        test_generalized_mallows_cayley()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
