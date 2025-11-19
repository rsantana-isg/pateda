"""
Tests for Mallows model with Cayley distance

This test file verifies that the newly implemented Mallows model with
Cayley distance works correctly for permutation-based EDAs.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/pateda')

from pateda.learning.mallows import LearnMallowsCayley
from pateda.sampling.mallows import SampleMallowsCayley
from pateda.permutation.distances import cayley_distance, _x_vector_cycles, _generate_perm_from_x
from pateda.permutation.consensus import find_consensus_borda


def test_x_vector_cycles():
    """Test the cycle-based x-vector computation."""
    print("Testing x-vector cycles...")

    # Test with identity permutation
    perm1 = np.array([0, 1, 2, 3, 4])
    x1 = _x_vector_cycles(perm1)
    print(f"  Identity permutation: {perm1} -> x-vector: {x1}")
    # For identity, all cycles are of length 1, so all largest elements get 0
    assert np.all(x1 == 0), f"Expected all zeros for identity, got {x1}"

    # Test with a simple permutation
    perm2 = np.array([1, 0, 2, 3, 4])  # Single swap of 0 and 1
    x2 = _x_vector_cycles(perm2)
    print(f"  Permutation {perm2} -> x-vector: {x2}")
    # Cycle (0,1) has largest element 1, so x[0]=1, x[1]=0
    # Other positions are in their own cycles

    print("  ✓ x-vector cycles test passed!")


def test_generate_perm_from_x():
    """Test permutation generation from x-vector."""
    print("\nTesting generate_perm_from_x...")

    n = 5
    # Test with all zeros (should give identity)
    x_zeros = np.zeros(n - 1, dtype=int)
    perm_zeros = _generate_perm_from_x(x_zeros, n)
    print(f"  x-vector {x_zeros} -> permutation: {perm_zeros}")
    assert np.all(perm_zeros == np.arange(n)), f"Expected identity, got {perm_zeros}"

    # Test with all ones (should give a non-identity permutation)
    x_ones = np.ones(n - 1, dtype=int)
    perm_ones = _generate_perm_from_x(x_ones, n)
    print(f"  x-vector {x_ones} -> permutation: {perm_ones}")
    # Should be a valid permutation
    assert len(set(perm_ones)) == n, "Not a valid permutation"

    print("  ✓ generate_perm_from_x test passed!")


def test_mallows_cayley_learning():
    """Test Mallows with Cayley distance learning."""
    print("\nTesting Mallows Cayley learning...")

    # Create a simple population of permutations
    pop = np.array([
        [0, 1, 2, 3, 4],
        [0, 2, 1, 3, 4],
        [1, 0, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [1, 2, 0, 3, 4],
    ])

    fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Learn model
    learner = LearnMallowsCayley()
    model = learner(
        generation=0,
        n_vars=5,
        cardinality=np.arange(5),
        selected_pop=pop,
        selected_fitness=fitness,
        initial_theta=0.1,
        upper_theta=10.0,
        max_iter=100,
        consensus_method="borda"
    )

    print(f"  Learned model type: {model['model_type']}")
    print(f"  Consensus: {model['consensus']}")
    print(f"  Theta: {model['theta']:.4f}")
    print(f"  X-probs shape: {model['x_probs'].shape}")
    print(f"  X-probs: {model['x_probs']}")

    # Verify model structure
    assert model['model_type'] == 'mallows_cayley'
    assert 'x_probs' in model
    assert 'consensus' in model
    assert 'theta' in model
    assert 'psis' in model
    assert len(model['x_probs']) == 4  # n-1 for n=5

    print("  ✓ Mallows Cayley learning test passed!")
    return model


def test_mallows_cayley_sampling(model):
    """Test Mallows with Cayley distance sampling."""
    print("\nTesting Mallows Cayley sampling...")

    sampler = SampleMallowsCayley()

    # Sample some permutations
    new_pop = sampler(
        n_vars=5,
        model=model,
        cardinality=np.arange(5),
        population=np.array([[0, 1, 2, 3, 4]]),
        fitness=np.array([1.0]),
        sample_size=10
    )

    print(f"  Sampled {len(new_pop)} permutations")
    print(f"  Sample shape: {new_pop.shape}")
    print(f"  First 3 samples:")
    for i in range(min(3, len(new_pop))):
        print(f"    {new_pop[i]}")

    # Verify all are valid permutations
    for i, perm in enumerate(new_pop):
        assert len(set(perm)) == 5, f"Sample {i} is not a valid permutation: {perm}"
        assert set(perm) == set(range(5)), f"Sample {i} has wrong elements: {perm}"

    print("  ✓ Mallows Cayley sampling test passed!")


def test_full_eda_cycle():
    """Test a complete EDA cycle with Mallows Cayley."""
    print("\nTesting full EDA cycle with Mallows Cayley...")

    # Initialize random population
    np.random.seed(42)
    pop_size = 20
    n_vars = 6

    population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])

    # Simple fitness: minimize Cayley distance to target
    target = np.arange(n_vars)
    fitness = np.array([cayley_distance(perm, target) for perm in population])

    # Select best half
    n_select = pop_size // 2
    best_indices = np.argsort(fitness)[:n_select]
    selected_pop = population[best_indices]
    selected_fitness = fitness[best_indices]

    print(f"  Population size: {pop_size}")
    print(f"  Selected size: {n_select}")
    print(f"  Best fitness in selected: {np.min(selected_fitness):.2f}")
    print(f"  Worst fitness in selected: {np.max(selected_fitness):.2f}")

    # Learn model
    learner = LearnMallowsCayley()
    model = learner(
        generation=0,
        n_vars=n_vars,
        cardinality=np.arange(n_vars),
        selected_pop=selected_pop,
        selected_fitness=selected_fitness,
    )

    print(f"  Learned theta: {model['theta']:.4f}")
    print(f"  Consensus: {model['consensus']}")

    # Sample new population
    sampler = SampleMallowsCayley()
    new_pop = sampler(
        n_vars=n_vars,
        model=model,
        cardinality=np.arange(n_vars),
        population=population,
        fitness=fitness,
        sample_size=pop_size
    )

    # Calculate fitness of new population
    new_fitness = np.array([cayley_distance(perm, target) for perm in new_pop])

    print(f"  New population best fitness: {np.min(new_fitness):.2f}")
    print(f"  New population mean fitness: {np.mean(new_fitness):.2f}")
    print(f"  Original mean fitness: {np.mean(fitness):.2f}")

    # Verify all new permutations are valid
    for i, perm in enumerate(new_pop):
        assert len(set(perm)) == n_vars, f"Sample {i} is not a valid permutation"

    print("  ✓ Full EDA cycle test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MALLOWS WITH CAYLEY DISTANCE IMPLEMENTATION")
    print("=" * 60)

    try:
        # Run all tests
        test_x_vector_cycles()
        test_generate_perm_from_x()
        model = test_mallows_cayley_learning()
        test_mallows_cayley_sampling(model)
        test_full_eda_cycle()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
