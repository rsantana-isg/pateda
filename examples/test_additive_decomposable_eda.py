"""
Example: Testing Additive Decomposable Functions with Discrete EDAs

This script demonstrates how to use the newly ported additive decomposable
benchmark functions with discrete EDAs like UMDA and cGA.
"""

import numpy as np
from pateda.eda.discrete.umda import UMDA
from pateda.eda.discrete.compact_ga import CompactGA
from pateda.functions.discrete.additive_decomposable import (
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
    k_deceptive,
    decep3,
    hiff,
    first_polytree3_ochoa,
    hard_decep5,
    fc2,
    fc3,
)


def test_k_deceptive_with_umda():
    """Test K-Deceptive function with UMDA"""
    print("=" * 70)
    print("Test 1: K-Deceptive (k=3) with UMDA")
    print("=" * 70)

    n_vars = 30  # 10 partitions of size 3
    obj_func = create_k_deceptive_function(k=3)

    # Create UMDA instance
    umda = UMDA(
        n_vars=n_vars,
        pop_size=100,
        selection_size=50,
        maximize=True,
        cardinality=2
    )

    # Run optimization
    best_solution, best_fitness, history = umda.run(
        objective_function=obj_func,
        max_generations=50
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Optimal fitness: {n_vars}  (all ones)")
    print(f"Found optimal: {best_fitness >= n_vars}")
    print()

    return best_fitness >= n_vars


def test_decep3_with_compact_ga():
    """Test Deceptive-3 with Compact GA"""
    print("=" * 70)
    print("Test 2: Deceptive-3 (overlapping) with Compact GA")
    print("=" * 70)

    n_vars = 20
    obj_func = create_decep3_function(overlap=True)

    # Create Compact GA instance
    cga = CompactGA(
        n_vars=n_vars,
        pop_size=200,
        maximize=True
    )

    # Run optimization
    best_solution, best_fitness, history = cga.run(
        objective_function=obj_func,
        max_generations=100
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print()

    return True


def test_hiff_with_umda():
    """Test HIFF function with UMDA"""
    print("=" * 70)
    print("Test 3: HIFF (Hierarchical If and only If) with UMDA")
    print("=" * 70)

    n_vars = 64  # Must be power of 2
    obj_func = create_hiff_function()

    # Create UMDA instance with larger population for harder problem
    umda = UMDA(
        n_vars=n_vars,
        pop_size=200,
        selection_size=100,
        maximize=True,
        cardinality=2
    )

    # Run optimization
    best_solution, best_fitness, history = umda.run(
        objective_function=obj_func,
        max_generations=100
    )

    print(f"Best solution sum: {np.sum(best_solution)}")
    print(f"Best fitness: {best_fitness}")

    # Check if solution is uniform (all 0s or all 1s)
    is_uniform = (np.all(best_solution == 0) or np.all(best_solution == 1))
    print(f"Solution is uniform (optimal): {is_uniform}")
    print()

    return is_uniform


def test_polytree3_with_umda():
    """Test Polytree-3 with UMDA"""
    print("=" * 70)
    print("Test 4: Polytree-3 (Ochoa) with UMDA")
    print("=" * 70)

    n_vars = 30  # 10 partitions of size 3
    obj_func = create_polytree3_function(overlap=False)

    # Create UMDA instance
    umda = UMDA(
        n_vars=n_vars,
        pop_size=150,
        selection_size=75,
        maximize=True,
        cardinality=2
    )

    # Run optimization
    best_solution, best_fitness, history = umda.run(
        objective_function=obj_func,
        max_generations=75
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print()

    return True


def test_hard_decep5_with_umda():
    """Test Hard Deceptive-5 with UMDA"""
    print("=" * 70)
    print("Test 5: Hard Deceptive-5 with UMDA")
    print("=" * 70)

    n_vars = 25  # 5 partitions of size 5
    def hard_decep5_obj(pop):
        if pop.ndim == 1:
            return np.array([hard_decep5(pop)])
        return np.array([hard_decep5(ind) for ind in pop])

    # Create UMDA instance
    umda = UMDA(
        n_vars=n_vars,
        pop_size=200,
        selection_size=100,
        maximize=True,
        cardinality=2
    )

    # Run optimization
    best_solution, best_fitness, history = umda.run(
        objective_function=hard_decep5_obj,
        max_generations=100
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Optimal fitness: {n_vars / 5.0}  (all partitions optimal)")
    print()

    return True


def test_fc3_with_umda():
    """Test Fc3 (Multimodal) function with UMDA"""
    print("=" * 70)
    print("Test 6: Fc3 (F5Multimodal) with UMDA")
    print("=" * 70)

    n_vars = 25  # 5 partitions of size 5
    def fc3_obj(pop):
        if pop.ndim == 1:
            return np.array([fc3(pop)])
        return np.array([fc3(ind) for ind in pop])

    # Create UMDA instance
    umda = UMDA(
        n_vars=n_vars,
        pop_size=150,
        selection_size=75,
        maximize=True,
        cardinality=2
    )

    # Run optimization
    best_solution, best_fitness, history = umda.run(
        objective_function=fc3_obj,
        max_generations=75
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print()

    return True


def compare_functions():
    """Compare different functions on same EDA configuration"""
    print("=" * 70)
    print("Test 7: Comparing Multiple Functions with Same EDA Configuration")
    print("=" * 70)

    n_vars = 30
    pop_size = 100
    selection_size = 50
    max_gen = 50

    functions = {
        "K-Deceptive (k=3)": create_k_deceptive_function(k=3),
        "Deceptive-3 (overlap)": create_decep3_function(overlap=True),
        "Polytree-3": create_polytree3_function(overlap=False),
    }

    results = {}

    for func_name, obj_func in functions.items():
        umda = UMDA(
            n_vars=n_vars,
            pop_size=pop_size,
            selection_size=selection_size,
            maximize=True,
            cardinality=2
        )

        best_solution, best_fitness, history = umda.run(
            objective_function=obj_func,
            max_generations=max_gen
        )

        results[func_name] = {
            "best_fitness": best_fitness,
            "avg_fitness_last_gen": np.mean(history[-1]) if len(history) > 0 else 0
        }

    print("\nComparative Results:")
    print("-" * 70)
    for func_name, res in results.items():
        print(f"{func_name:30s} | Best: {res['best_fitness']:8.3f} | "
              f"Avg (last gen): {res['avg_fitness_last_gen']:8.3f}")
    print()

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TESTING ADDITIVE DECOMPOSABLE FUNCTIONS WITH DISCRETE EDAs")
    print("=" * 70 + "\n")

    results = []

    # Run individual tests
    results.append(("K-Deceptive with UMDA", test_k_deceptive_with_umda()))
    results.append(("Decep3 with CompactGA", test_decep3_with_compact_ga()))
    results.append(("HIFF with UMDA", test_hiff_with_umda()))
    results.append(("Polytree-3 with UMDA", test_polytree3_with_umda()))
    results.append(("Hard Decep5 with UMDA", test_hard_decep5_with_umda()))
    results.append(("Fc3 with UMDA", test_fc3_with_umda()))
    results.append(("Function Comparison", compare_functions()))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:40s} : {status}")
    print("=" * 70)

    total_pass = sum(results, key=lambda x: x[1])
    print(f"\nTotal: {len(results)} tests, {total_pass} passed\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
