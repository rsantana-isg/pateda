"""
Simple test to verify the MaxGenerationsOrOptimum stop condition works correctly
"""

import numpy as np

from pateda.stop_conditions import MaxGenerationsOrOptimum

def test_max_generations_only():
    """Test that it stops when max generations is reached"""
    stop_condition = MaxGenerationsOrOptimum(max_gen=10, optimal_fitness=None)

    # Should not stop before max_gen
    assert not stop_condition.should_stop(5, np.array([]), np.array([0.5]))
    assert not stop_condition.should_stop(9, np.array([]), np.array([0.9]))

    # Should stop at max_gen
    assert stop_condition.should_stop(10, np.array([]), np.array([0.5]))
    assert stop_condition.should_stop(11, np.array([]), np.array([0.5]))

    print("✓ Test 1: Max generations only - PASSED")


def test_optimum_reached():
    """Test that it stops when optimum is reached"""
    stop_condition = MaxGenerationsOrOptimum(max_gen=100, optimal_fitness=20.0)

    # Should not stop when fitness is below optimum
    fitness = np.array([[15.0], [16.0], [17.0]])
    assert not stop_condition.should_stop(5, np.array([]), fitness)

    # Should stop when fitness reaches optimum (within tolerance)
    fitness = np.array([[19.0], [20.0], [18.0]])
    assert stop_condition.should_stop(5, np.array([]), fitness)

    # Should stop even if slightly different due to tolerance
    fitness = np.array([[19.0], [19.9999995], [18.0]])
    assert stop_condition.should_stop(5, np.array([]), fitness)

    print("✓ Test 2: Optimum reached - PASSED")


def test_combined_conditions():
    """Test that it stops when either condition is met"""
    stop_condition = MaxGenerationsOrOptimum(max_gen=10, optimal_fitness=20.0)

    # Should not stop when neither condition is met
    fitness = np.array([[15.0], [16.0], [17.0]])
    assert not stop_condition.should_stop(5, np.array([]), fitness)

    # Should stop when optimum is reached (before max_gen)
    fitness = np.array([[19.0], [20.0], [18.0]])
    assert stop_condition.should_stop(5, np.array([]), fitness)

    # Should stop when max_gen is reached (even if optimum not reached)
    fitness = np.array([[15.0], [16.0], [17.0]])
    assert stop_condition.should_stop(10, np.array([]), fitness)

    print("✓ Test 3: Combined conditions - PASSED")


def test_1d_fitness_array():
    """Test with 1D fitness array"""
    stop_condition = MaxGenerationsOrOptimum(max_gen=100, optimal_fitness=20.0)

    # Should handle 1D fitness array
    fitness = np.array([15.0, 16.0, 17.0])
    assert not stop_condition.should_stop(5, np.array([]), fitness)

    fitness = np.array([15.0, 20.0, 17.0])
    assert stop_condition.should_stop(5, np.array([]), fitness)

    print("✓ Test 4: 1D fitness array - PASSED")


if __name__ == "__main__":
    print("Testing MaxGenerationsOrOptimum stop condition...")
    print()

    test_max_generations_only()
    test_optimum_reached()
    test_combined_conditions()
    test_1d_fitness_array()

    print()
    print("=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)
