"""
Additive Decomposable Functions Examples

This script demonstrates how to use the newly ported additive decomposable
benchmark functions with discrete EDAs using the pateda framework.

Functions tested:
- K-Deceptive
- Deceptive-3 variants
- HIFF (Hierarchical If and only If)
- Polytree functions
- Cuban functions
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.additive_decomposable import (
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
)


def run_eda_on_function(objective, n_vars, cardinality, pop_size=1000,
                        max_gen=100, function_name="Function"):
    """
    Helper function to run EDA on a given objective function

    Args:
        objective: Objective function to optimize
        n_vars: Number of variables
        cardinality: Cardinality array
        pop_size: Population size
        max_gen: Maximum generations
        function_name: Name of the function for display

    Returns:
        statistics: Dictionary with optimization statistics
    """
    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnUMDA(alpha=1.0),  # Laplace smoothing
        sampling=SampleFDA(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=objective,
        components=components,
        random_seed=42,
    )

    # Run optimization
    print(f"\nRunning EDA on {function_name}...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Maximum generations: {max_gen}")

    stats, cache = eda.run(verbose=True)

    # Print results
    print("=" * 60)
    print(f"Results for {function_name}")
    print("=" * 60)
    print(f"Generations run: {len(stats.best_fitness)}")
    print(f"Best fitness: {stats.best_fitness_overall:.4f}")
    print(f"Mean fitness (final): {stats.mean_fitness[-1]:.4f}")
    print(f"Best solution: {stats.best_individual}")
    print()

    return stats


def example_k_deceptive():
    """Test K-Deceptive function with k=3"""
    print("\n" + "=" * 70)
    print("Example 1: K-Deceptive (k=3)")
    print("=" * 70)

    n_vars = 30  # 10 partitions of size 3
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_k_deceptive_function(k=3)

    print(f"\nFunction: K-Deceptive with k=3")
    print(f"Optimal solution: All 1s")
    print(f"Optimal fitness: {n_vars}")

    stats = run_eda_on_function(
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=50,
        function_name="K-Deceptive (k=3)"
    )

    # Check if optimal was found
    is_optimal = stats['best_fitness'][-1] >= n_vars
    print(f"Optimal solution found: {is_optimal}")

    return stats


def example_decep3():
    """Test Deceptive-3 with overlap"""
    print("\n" + "=" * 70)
    print("Example 2: Deceptive-3 (with overlap)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_decep3_function(overlap=True)

    print(f"\nFunction: Deceptive-3 with overlapping partitions")
    print(f"This function uses overlapping 3-variable subfunctions")

    stats = run_eda_on_function(
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=75,
        function_name="Deceptive-3 (overlap)"
    )

    return stats


def example_hiff():
    """Test HIFF (Hierarchical If and only If)"""
    print("\n" + "=" * 70)
    print("Example 3: HIFF (Hierarchical If and only If)")
    print("=" * 70)

    n_vars = 64  # Must be power of 2
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_hiff_function()

    print(f"\nFunction: HIFF")
    print(f"This is a hierarchical function that rewards building blocks")
    print(f"at multiple scales. Problem size must be a power of 2.")
    print(f"Optimal solutions: All 0s or all 1s (uniform)")

    stats = run_eda_on_function(
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=2000,
        max_gen=100,
        function_name="HIFF"
    )

    # Check if solution is uniform
    best_sol = stats['best_solution']
    is_uniform = (np.all(best_sol == 0) or np.all(best_sol == 1))
    print(f"Solution is uniform (optimal): {is_uniform}")
    print(f"Sum of best solution: {np.sum(best_sol)}")

    return stats


def example_polytree3():
    """Test First Polytree-3 (Ochoa)"""
    print("\n" + "=" * 70)
    print("Example 4: First Polytree-3 (Ochoa)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_polytree3_function(overlap=False)

    print(f"\nFunction: Ochoa's First Polytree-3")
    print(f"This function uses a lookup table for 3-variable subfunctions")

    stats = run_eda_on_function(
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=75,
        function_name="Polytree-3"
    )

    return stats


def example_polytree3_overlap():
    """Test First Polytree-3 with overlap"""
    print("\n" + "=" * 70)
    print("Example 5: First Polytree-3 with Overlap (Ochoa)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_polytree3_function(overlap=True)

    print(f"\nFunction: Ochoa's First Polytree-3 with overlapping partitions")
    print(f"Overlapping partitions make the problem more challenging")

    stats = run_eda_on_function(
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1500,
        max_gen=100,
        function_name="Polytree-3 (overlap)"
    )

    return stats


def compare_k_values():
    """Compare K-Deceptive with different k values"""
    print("\n" + "=" * 70)
    print("Example 6: Comparing K-Deceptive with different k values")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    k_values = [3, 5]

    results = {}

    for k in k_values:
        print(f"\n{'=' * 60}")
        print(f"Testing K-Deceptive with k={k}")
        print('=' * 60)

        objective = create_k_deceptive_function(k=k)

        stats = run_eda_on_function(
            objective=objective,
            n_vars=n_vars,
            cardinality=cardinality,
            pop_size=1000,
            max_gen=75,
            function_name=f"K-Deceptive (k={k})"
        )

        results[f"k={k}"] = {
            "best_fitness": stats['best_fitness'][-1],
            "mean_fitness": stats['mean_fitness'][-1],
            "generations": len(stats['best_fitness'])
        }

    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    for k_label, res in results.items():
        print(f"{k_label:15s} | Best: {res['best_fitness']:8.3f} | "
              f"Mean: {res['mean_fitness']:8.3f} | "
              f"Gens: {res['generations']:3d}")

    return results


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("ADDITIVE DECOMPOSABLE BENCHMARK FUNCTIONS - EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the newly ported additively")
    print("decomposable benchmark functions from the C++ EDA implementation.")
    print()

    # Run examples
    all_stats = {}

    all_stats['k_deceptive'] = example_k_deceptive()
    all_stats['decep3'] = example_decep3()
    all_stats['hiff'] = example_hiff()
    all_stats['polytree3'] = example_polytree3()
    all_stats['polytree3_overlap'] = example_polytree3_overlap()
    all_stats['k_comparison'] = compare_k_values()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY - All Examples Completed Successfully")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - K-Deceptive variants (k_deceptive, gen_k_decep, gen_k_decep_overlap)")
    print("  - Deceptive-3 variants (decep3, decep_marta3, decep3_mh, etc.)")
    print("  - Hard Deceptive-5 (hard_decep5)")
    print("  - Hierarchical functions (hiff, fhtrap1)")
    print("  - Polytree functions (first_polytree3_ochoa, first_polytree5_ochoa)")
    print("  - Cuban functions (fc2, fc3, fc4, fc5)")
    print("\nSee pateda/functions/discrete/additive_decomposable.py for details")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
