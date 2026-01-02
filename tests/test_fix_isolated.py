"""Isolated test for the Boltzmann distribution fix - no external deps needed."""

import numpy as np

def rosenbrock_function(x, shift=None):
    """Rosenbrock function for testing."""
    if shift is not None:
        x = x - shift
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

def test_boltzmann_selection():
    """Test the Boltzmann selection with the fix."""
    print("Testing Boltzmann selection fix...")
    print("=" * 80)

    # Configuration that was failing
    n_initial = 1000
    n_selected = 500
    n_vars = 10
    temperature = 2.0
    bounds = (-2.048, 2.048)

    np.random.seed(42)

    # Generate shift
    shift = np.random.uniform(-1.0, 1.0, n_vars)

    # Generate initial population
    initial_population = np.random.uniform(
        bounds[0], bounds[1],
        (n_initial, n_vars)
    )

    # Evaluate fitness
    fitness = rosenbrock_function(initial_population, shift=shift)

    # Normalize fitness
    fitness_normalized = fitness - np.min(fitness)

    # Compute Boltzmann probabilities
    boltzmann_weights = np.exp(-fitness_normalized / temperature)
    probabilities = boltzmann_weights / np.sum(boltzmann_weights)

    # Count non-zero probabilities (THE FIX)
    non_zero_count = np.sum(probabilities > 0)
    print(f"\nInitial setup:")
    print(f"  n_initial: {n_initial}")
    print(f"  n_selected requested: {n_selected}")
    print(f"  Non-zero probabilities: {non_zero_count}")
    print(f"  Temperature: {temperature}")
    print(f"  Fitness range: [{fitness.min():.2f}, {fitness.max():.2f}]")
    print(f"  Normalized fitness range: [{fitness_normalized.min():.2f}, {fitness_normalized.max():.2f}]")

    # Check if adjustment is needed
    if non_zero_count < n_selected:
        print(f"\n  ⚠ Warning: Adjusting n_selected from {n_selected} to {non_zero_count}")
        n_selected_actual = non_zero_count
    else:
        print(f"\n  ✓ No adjustment needed")
        n_selected_actual = n_selected

    # Try to select (this was failing before the fix)
    try:
        selected_indices = np.random.choice(
            n_initial,
            size=n_selected_actual,
            replace=False,
            p=probabilities
        )
        print(f"\n✓ SUCCESS: Selected {len(selected_indices)} solutions")

        # Get selected solutions
        selected_samples = initial_population[selected_indices]
        selected_fitness = fitness[selected_indices]

        # Sort by fitness
        sorted_indices = np.argsort(selected_fitness)
        selected_samples = selected_samples[sorted_indices]
        selected_fitness = selected_fitness[sorted_indices]

        print(f"  Best fitness: {selected_fitness[0]:.6f}")
        print(f"  Mean fitness: {selected_fitness.mean():.6f}")
        print(f"  Worst fitness: {selected_fitness[-1]:.6f}")

        return True

    except ValueError as e:
        print(f"\n✗ FAILED: {str(e)}")
        return False

def test_rank_based_selection():
    """Test the rank-based selection with the fix."""
    print("\n\nTesting Rank-based selection fix...")
    print("=" * 80)

    # Configuration
    n_initial = 1000
    n_selected = 500
    n_vars = 10
    selection_pressure = 2.5
    bounds = (-2.048, 2.048)

    np.random.seed(42)

    # Generate shift
    shift = np.random.uniform(-1.0, 1.0, n_vars)

    # Generate initial population
    initial_population = np.random.uniform(
        bounds[0], bounds[1],
        (n_initial, n_vars)
    )

    # Evaluate fitness
    fitness = rosenbrock_function(initial_population, shift=shift)

    # Rank solutions
    sorted_indices = np.argsort(fitness)
    ranks = np.empty(n_initial, dtype=int)
    ranks[sorted_indices] = np.arange(1, n_initial + 1)

    # Compute rank-based probabilities
    rank_weights = selection_pressure - 2 * selection_pressure * (ranks - 1) / (n_initial - 1)
    rank_weights = np.maximum(rank_weights, 0)
    probabilities = rank_weights / np.sum(rank_weights)

    # Count non-zero probabilities (THE FIX)
    non_zero_count = np.sum(probabilities > 0)
    print(f"\nInitial setup:")
    print(f"  n_initial: {n_initial}")
    print(f"  n_selected requested: {n_selected}")
    print(f"  Non-zero probabilities: {non_zero_count}")
    print(f"  Selection pressure: {selection_pressure}")

    # Check if adjustment is needed
    if non_zero_count < n_selected:
        print(f"\n  ⚠ Warning: Adjusting n_selected from {n_selected} to {non_zero_count}")
        n_selected_actual = non_zero_count
    else:
        print(f"\n  ✓ No adjustment needed")
        n_selected_actual = n_selected

    # Try to select
    try:
        selected_indices = np.random.choice(
            n_initial,
            size=n_selected_actual,
            replace=False,
            p=probabilities
        )
        print(f"\n✓ SUCCESS: Selected {len(selected_indices)} solutions")

        # Get selected solutions
        selected_fitness = fitness[selected_indices]

        # Sort by fitness
        sorted_indices = np.argsort(selected_fitness)
        selected_fitness = selected_fitness[sorted_indices]

        print(f"  Best fitness: {selected_fitness[0]:.6f}")
        print(f"  Mean fitness: {selected_fitness.mean():.6f}")
        print(f"  Worst fitness: {selected_fitness[-1]:.6f}")

        return True

    except ValueError as e:
        print(f"\n✗ FAILED: {str(e)}")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("TESTING DENDIFF BENCHMARK FIX")
    print("=" * 80)
    print("\nThis test verifies that the fix for 'Fewer non-zero entries in p than size'")
    print("error works correctly by counting non-zero probabilities before sampling.")
    print("\n" + "=" * 80)

    success1 = test_boltzmann_selection()
    success2 = test_rank_based_selection()

    print("\n\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Boltzmann selection: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Rank-based selection: {'✓ PASS' if success2 else '✗ FAIL'}")

    if success1 and success2:
        print("\n✓ All tests PASSED!")
        print("=" * 80)
        exit(0)
    else:
        print("\n✗ Some tests FAILED!")
        print("=" * 80)
        exit(1)
