"""
Example: Testing Additive Decomposable Functions with Discrete EDAs

This script demonstrates how to use the additive decomposable benchmark
functions with discrete EDAs (UMDA and Tree-EDA) using the pateda API.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.learning.umda import LearnUMDA
from pateda.learning.tree import LearnTreeModel
from pateda.sampling.fda import SampleFDA
from pateda.functions.discrete_binary.toy_functions.additive_decomposable import (
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


def run_umda(n_vars, pop_size, selection_size, fitness_func, max_gen, seed=42):
    """Run UMDA (univariate EDA) and return (best_individual, best_fitness, fitness_history)."""
    def _wrapped(x):
        val = fitness_func(x)
        return float(val[0]) if hasattr(val, '__len__') else float(val)

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(n_select=selection_size),
        learning=LearnUMDA(alpha=0.0),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=_wrapped,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    history = [stats.best_fitness[g] for g in range(len(stats.best_fitness))]
    return stats.best_individual, stats.best_fitness_overall, history


def test_k_deceptive_with_umda():
    """Test K-Deceptive function with UMDA"""
    print("=" * 70)
    print("Test 1: K-Deceptive (k=3) with UMDA")
    print("=" * 70)

    n_vars = 30
    obj_func = create_k_deceptive_function(k=3)
    best_solution, best_fitness, history = run_umda(
        n_vars=n_vars, pop_size=100, selection_size=50,
        fitness_func=obj_func, max_gen=50,
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"Optimal fitness: {n_vars}  (all ones)")
    print(f"Found optimal: {best_fitness >= n_vars}")
    print()
    return best_fitness >= n_vars


def test_decep3_with_tree_eda():
    """Test Deceptive-3 with Tree-EDA (replaces CompactGA which is not available)"""
    print("=" * 70)
    print("Test 2: Deceptive-3 (overlapping) with Tree-EDA")
    print("=" * 70)

    n_vars = 20
    obj_func = create_decep3_function(overlap=True)

    def _wrapped(x):
        val = obj_func(x)
        return float(val[0]) if hasattr(val, '__len__') else float(val)

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnTreeModel(alpha=0.1),
        sampling=SampleFDA(n_samples=200),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(100),
    )
    eda = EDA(
        pop_size=200,
        n_vars=n_vars,
        fitness_func=_wrapped,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=42,
    )
    stats, _ = eda.run(verbose=False)

    print(f"Best solution: {stats.best_individual}")
    print(f"Best fitness: {stats.best_fitness_overall:.3f}")
    print()
    return True


def test_hiff_with_umda():
    """Test HIFF function with UMDA"""
    print("=" * 70)
    print("Test 3: HIFF (Hierarchical If and only If) with UMDA")
    print("=" * 70)

    n_vars = 64  # must be power of 2
    obj_func = create_hiff_function()
    best_solution, best_fitness, _ = run_umda(
        n_vars=n_vars, pop_size=200, selection_size=100,
        fitness_func=obj_func, max_gen=100,
    )

    print(f"Best solution sum: {np.sum(best_solution)}")
    print(f"Best fitness: {best_fitness:.3f}")
    is_uniform = (np.all(best_solution == 0) or np.all(best_solution == 1))
    print(f"Solution is uniform (optimal): {is_uniform}")
    print()
    return is_uniform


def test_polytree3_with_umda():
    """Test Polytree-3 with UMDA"""
    print("=" * 70)
    print("Test 4: Polytree-3 (Ochoa) with UMDA")
    print("=" * 70)

    n_vars = 30
    obj_func = create_polytree3_function(overlap=False)
    best_solution, best_fitness, _ = run_umda(
        n_vars=n_vars, pop_size=150, selection_size=75,
        fitness_func=obj_func, max_gen=75,
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.3f}")
    print()
    return True


def test_hard_decep5_with_umda():
    """Test Hard Deceptive-5 with UMDA"""
    print("=" * 70)
    print("Test 5: Hard Deceptive-5 with UMDA")
    print("=" * 70)

    n_vars = 25
    def obj_func(x):
        return hard_decep5(x)

    best_solution, best_fitness, _ = run_umda(
        n_vars=n_vars, pop_size=200, selection_size=100,
        fitness_func=obj_func, max_gen=100,
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"Optimal fitness: {n_vars / 5.0:.1f}  (all partitions optimal)")
    print()
    return True


def test_fc3_with_umda():
    """Test Fc3 (Multimodal) function with UMDA"""
    print("=" * 70)
    print("Test 6: Fc3 (F5Multimodal) with UMDA")
    print("=" * 70)

    n_vars = 25
    def obj_func(x):
        return fc3(x)

    best_solution, best_fitness, _ = run_umda(
        n_vars=n_vars, pop_size=150, selection_size=75,
        fitness_func=obj_func, max_gen=75,
    )

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.3f}")
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
        _, best_fitness, history = run_umda(
            n_vars=n_vars, pop_size=pop_size, selection_size=selection_size,
            fitness_func=obj_func, max_gen=max_gen,
        )
        results[func_name] = {
            "best_fitness": best_fitness,
            "avg_fitness_last_gen": history[-1] if len(history) > 0 else 0,
        }

    print("\nComparative Results:")
    print("-" * 70)
    for func_name, res in results.items():
        print(f"{func_name:30s} | Best: {res['best_fitness']:8.3f} | "
              f"Best (last gen): {res['avg_fitness_last_gen']:8.3f}")
    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TESTING ADDITIVE DECOMPOSABLE FUNCTIONS WITH DISCRETE EDAs")
    print("=" * 70 + "\n")

    results = []
    results.append(("K-Deceptive with UMDA", test_k_deceptive_with_umda()))
    results.append(("Decep3 with Tree-EDA", test_decep3_with_tree_eda()))
    results.append(("HIFF with UMDA", test_hiff_with_umda()))
    results.append(("Polytree-3 with UMDA", test_polytree3_with_umda()))
    results.append(("Hard Decep5 with UMDA", test_hard_decep5_with_umda()))
    results.append(("Fc3 with UMDA", test_fc3_with_umda()))
    results.append(("Function Comparison", compare_functions()))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:40s} : {status}")
    print("=" * 70)

    total_pass = sum(1 for _, success in results if success)
    print(f"\nTotal: {len(results)} tests, {total_pass} passed\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()
