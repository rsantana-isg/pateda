"""
Test multi-objective functionality in pateda

This script tests the multi-objective support added to pateda.
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.seeding import RandomInit
from pateda.learning import LearnFDA
from pateda.sampling import SampleFDA
from pateda.selection import (
    TruncationSelection,
    NonDominatedSelection,
    ParetoFrontSelection,
)


def multi_objective_onemax(x):
    """
    Multi-objective test function: 2 objectives

    Objective 1: Count 1s in first half
    Objective 2: Count 1s in second half
    """
    n = len(x)
    mid = n // 2
    obj1 = np.sum(x[:mid])
    obj2 = np.sum(x[mid:])
    return np.array([obj1, obj2])


def single_objective_onemax(x):
    """Single objective: count all 1s"""
    return np.sum(x)


def test_single_objective():
    """Test that single-objective problems still work"""
    print("=" * 60)
    print("Testing Single-Objective EDA")
    print("=" * 60)

    n_vars = 20
    pop_size = 100
    n_generations = 50

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnFDA(),
        sampling=SampleFDA(n_samples=pop_size),
        stop_condition=MaxGenerations(n_generations),
    )

    cardinality = np.full(n_vars, 2)

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=single_objective_onemax,
        cardinality=cardinality,
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal best fitness: {stats.best_fitness_overall}")
    print(f"Best individual: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")

    # Verify fitness shape
    assert isinstance(stats.best_fitness_overall, (int, float)), \
        "Single-objective should return scalar fitness"

    print("\n✓ Single-objective test passed!")
    return True


def test_multi_objective_basic():
    """Test multi-objective with standard selection (truncation)"""
    print("\n" + "=" * 60)
    print("Testing Multi-Objective EDA with Truncation Selection")
    print("=" * 60)

    n_vars = 20
    pop_size = 100
    n_generations = 50

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),  # Uses mean fitness
        learning=LearnFDA(),
        sampling=SampleFDA(n_samples=pop_size),
        stop_condition=MaxGenerations(n_generations),
    )

    cardinality = np.full(n_vars, 2)

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=multi_objective_onemax,
        cardinality=cardinality,
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal best fitness: {stats.best_fitness_overall}")
    print(f"Best individual: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")

    # Verify fitness shape
    assert isinstance(stats.best_fitness_overall, np.ndarray), \
        "Multi-objective should return array fitness"
    assert len(stats.best_fitness_overall) == 2, \
        "Should have 2 objectives"

    print("\n✓ Multi-objective basic test passed!")
    return True


def test_multi_objective_pareto():
    """Test multi-objective with Pareto-based selection"""
    print("\n" + "=" * 60)
    print("Testing Multi-Objective EDA with Pareto Front Selection")
    print("=" * 60)

    n_vars = 20
    pop_size = 100
    n_generations = 50

    components = EDAComponents(
        seeding=RandomInit(),
        selection=ParetoFrontSelection(ratio=0.5),
        learning=LearnFDA(),
        sampling=SampleFDA(n_samples=pop_size),
        stop_condition=MaxGenerations(n_generations),
    )

    cardinality = np.full(n_vars, 2)

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=multi_objective_onemax,
        cardinality=cardinality,
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal best fitness: {stats.best_fitness_overall}")
    print(f"Best individual: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")

    # Verify fitness shape
    assert isinstance(stats.best_fitness_overall, np.ndarray), \
        "Multi-objective should return array fitness"
    assert len(stats.best_fitness_overall) == 2, \
        "Should have 2 objectives"

    print("\n✓ Multi-objective Pareto test passed!")
    return True


def test_non_dominated_selection():
    """Test non-dominated selection"""
    print("\n" + "=" * 60)
    print("Testing Multi-Objective EDA with Non-Dominated Selection")
    print("=" * 60)

    n_vars = 20
    pop_size = 100
    n_generations = 50

    components = EDAComponents(
        seeding=RandomInit(),
        selection=NonDominatedSelection(min_select=50),  # Ensure at least 50 selected
        learning=LearnFDA(),
        sampling=SampleFDA(n_samples=pop_size),
        stop_condition=MaxGenerations(n_generations),
    )

    cardinality = np.full(n_vars, 2)

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=multi_objective_onemax,
        cardinality=cardinality,
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal best fitness: {stats.best_fitness_overall}")
    print(f"Best individual: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")

    print("\n✓ Non-dominated selection test passed!")
    return True


def test_pareto_utilities():
    """Test Pareto utility functions directly"""
    print("\n" + "=" * 60)
    print("Testing Pareto Utility Functions")
    print("=" * 60)

    from pateda.selection.utils import (
        find_pareto_set,
        pareto_dominates,
        pareto_ranking,
        fitness_ranking,
    )

    # Test data: 4 solutions with 2 objectives
    fitness = np.array([
        [1.0, 2.0],   # Solution 0
        [2.0, 1.0],   # Solution 1
        [0.5, 0.5],   # Solution 2 (dominated)
        [1.5, 1.5],   # Solution 3 (non-dominated)
    ])

    # Test dominance
    assert pareto_dominates(fitness[0], fitness[2]), "0 should dominate 2"
    assert not pareto_dominates(fitness[0], fitness[1]), "0 should not dominate 1"
    print("✓ Pareto dominance test passed")

    # Test Pareto set identification
    pareto_idx = find_pareto_set(fitness, maximize=True)
    print(f"Pareto front indices: {pareto_idx}")
    assert 2 not in pareto_idx, "Solution 2 should be dominated"
    assert 3 in pareto_idx, "Solution 3 should be in Pareto front"
    print("✓ Pareto set identification passed")

    # Test Pareto ranking
    ranks, order = pareto_ranking(fitness, maximize=True)
    print(f"Pareto ranks: {ranks}")
    print(f"Ordered indices: {order}")
    assert ranks[2] > ranks[0], "Dominated solution should have worse rank"
    print("✓ Pareto ranking passed")

    # Test fitness ranking
    order = fitness_ranking(fitness, maximize=True)
    print(f"Fitness ranking order: {order}")
    print("✓ Fitness ranking passed")

    print("\n✓ All Pareto utility tests passed!")
    return True


if __name__ == "__main__":
    np.random.seed(42)

    try:
        # Run all tests
        test_pareto_utilities()
        test_single_objective()
        test_multi_objective_basic()
        test_multi_objective_pareto()
        test_non_dominated_selection()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
