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

# Add parent directory to path for running examples without installation

import argparse
import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.additive_decomposable import (
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
)


EDA_NAME_MAP = {
    "umda": "UMDA",
    "tree": "Tree-EDA",
}


def build_learning_method(eda_type):
    """Build learning component for requested EDA type."""
    if eda_type == "umda":
        return LearnUMDA(alpha=1.0)
    if eda_type == "tree":
        return LearnTreeModel(alpha=1.0)
    raise ValueError(f"Unsupported EDA type: {eda_type}")


def run_eda_on_function(objective, n_vars, cardinality, pop_size=1000,
                        max_gen=100, function_name="Function",
                        eda_type="umda", seed=42):
    """
    Helper function to run EDA on a given objective function

    Args:
        objective: Objective function to optimize
        n_vars: Number of variables
        cardinality: Cardinality array
        pop_size: Population size
        max_gen: Maximum generations
        function_name: Name of the function for display
        eda_type: EDA learning model to use ("umda" or "tree")
        seed: Random seed used in EDA

    Returns:
        statistics: Dictionary with optimization statistics
    """
    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=build_learning_method(eda_type),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=objective,
        components=components,
        random_seed=seed,
    )

    # Run optimization
    print(f"\nRunning {EDA_NAME_MAP[eda_type]} on {function_name}...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Maximum generations: {max_gen}")
    print(f"Seed: {seed}")

    stats, cache = eda.run(verbose=True)

    # Print results
    print("=" * 60)
    print(f"Results for {function_name} ({EDA_NAME_MAP[eda_type]})")
    print("=" * 60)
    print(f"Generations run: {len(stats.best_fitness)}")
    print(f"Best fitness: {stats.best_fitness_overall:.4f}")
    print(f"Mean fitness (final): {stats.mean_fitness[-1]:.4f}")
    print(f"Best solution: {stats.best_individual}")
    print()

    return stats


def run_eda_suite(eda_types, **kwargs):
    """Run one benchmark configuration for each selected EDA."""
    results = {}
    for eda_type in eda_types:
        results[eda_type] = run_eda_on_function(eda_type=eda_type, **kwargs)
    return results


def example_k_deceptive(eda_types, seed):
    """Test K-Deceptive function with k=3"""
    print("\n" + "=" * 70)
    print("Example 1: K-Deceptive (k=3)")
    print("=" * 70)

    n_vars = 30  # 10 partitions of size 3
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_k_deceptive_function(k=3)

    print("\nFunction: K-Deceptive with k=3")
    print("Optimal solution: All 1s")
    print(f"Optimal fitness: {n_vars}")

    stats_by_eda = run_eda_suite(
        eda_types=eda_types,
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=50,
        function_name="K-Deceptive (k=3)",
        seed=seed,
    )

    # Check if optimal was found
    for eda_type, stats in stats_by_eda.items():
        is_optimal = stats.best_fitness[-1] >= n_vars
        print(f"Optimal solution found ({EDA_NAME_MAP[eda_type]}): {is_optimal}")

    return stats_by_eda


def example_decep3(eda_types, seed):
    """Test Deceptive-3 with overlap"""
    print("\n" + "=" * 70)
    print("Example 2: Deceptive-3 (with overlap)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_decep3_function(overlap=True)

    print("\nFunction: Deceptive-3 with overlapping partitions")
    print("This function uses overlapping 3-variable subfunctions")

    stats_by_eda = run_eda_suite(
        eda_types=eda_types,
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=75,
        function_name="Deceptive-3 (overlap)",
        seed=seed,
    )

    return stats_by_eda


def example_hiff(eda_types, seed):
    """Test HIFF (Hierarchical If and only If)"""
    print("\n" + "=" * 70)
    print("Example 3: HIFF (Hierarchical If and only If)")
    print("=" * 70)

    n_vars = 64  # Must be power of 2
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_hiff_function()

    print("\nFunction: HIFF")
    print("This is a hierarchical function that rewards building blocks")
    print("at multiple scales. Problem size must be a power of 2.")
    print("Optimal solutions: All 0s or all 1s (uniform)")

    stats_by_eda = run_eda_suite(
        eda_types=eda_types,
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=2000,
        max_gen=100,
        function_name="HIFF",
        seed=seed,
    )

    for eda_type, stats in stats_by_eda.items():
        best_sol = stats.best_individual
        is_uniform = np.all(best_sol == 0) or np.all(best_sol == 1)
        print(f"Solution is uniform ({EDA_NAME_MAP[eda_type]}): {is_uniform}")
        print(f"Sum of best solution ({EDA_NAME_MAP[eda_type]}): {np.sum(best_sol)}")

    return stats_by_eda


def example_polytree3(eda_types, seed):
    """Test First Polytree-3 (Ochoa)"""
    print("\n" + "=" * 70)
    print("Example 4: First Polytree-3 (Ochoa)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_polytree3_function(overlap=False)

    print("\nFunction: Ochoa's First Polytree-3")
    print("This function uses a lookup table for 3-variable subfunctions")

    stats_by_eda = run_eda_suite(
        eda_types=eda_types,
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1000,
        max_gen=75,
        function_name="Polytree-3",
        seed=seed,
    )

    return stats_by_eda


def example_polytree3_overlap(eda_types, seed):
    """Test First Polytree-3 with overlap"""
    print("\n" + "=" * 70)
    print("Example 5: First Polytree-3 with Overlap (Ochoa)")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    objective = create_polytree3_function(overlap=True)

    print("\nFunction: Ochoa's First Polytree-3 with overlapping partitions")
    print("Overlapping partitions make the problem more challenging")

    stats_by_eda = run_eda_suite(
        eda_types=eda_types,
        objective=objective,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=1500,
        max_gen=100,
        function_name="Polytree-3 (overlap)",
        seed=seed,
    )

    return stats_by_eda


def compare_k_values(eda_types, seed):
    """Compare K-Deceptive with different k values"""
    print("\n" + "=" * 70)
    print("Example 6: Comparing K-Deceptive with different k values")
    print("=" * 70)

    n_vars = 30
    cardinality = 2 * np.ones(n_vars, dtype=int)
    k_values = [3, 5]

    results = {eda_type: {} for eda_type in eda_types}

    for k in k_values:
        print(f"\n{'=' * 60}")
        print(f"Testing K-Deceptive with k={k}")
        print('=' * 60)

        objective = create_k_deceptive_function(k=k)

        for eda_type in eda_types:
            stats = run_eda_on_function(
                objective=objective,
                n_vars=n_vars,
                cardinality=cardinality,
                pop_size=1000,
                max_gen=75,
                function_name=f"K-Deceptive (k={k})",
                eda_type=eda_type,
                seed=seed,
            )

            results[eda_type][f"k={k}"] = {
                "best_fitness": stats.best_fitness[-1],
                "mean_fitness": stats.mean_fitness[-1],
                "generations": len(stats.best_fitness)
            }

    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    for eda_type in eda_types:
        print(f"\n{EDA_NAME_MAP[eda_type]}:")
        for k_label, res in results[eda_type].items():
            print(f"{k_label:15s} | Best: {res['best_fitness']:8.3f} | "
                  f"Mean: {res['mean_fitness']:8.3f} | "
                  f"Gens: {res['generations']:3d}")

    return results


def print_eda_final_comparison(all_stats, eda_types):
    """Print final UMDA vs Tree-EDA comparison."""
    if len(eda_types) < 2:
        print("\nOnly one EDA selected; skipping cross-EDA comparison table.")
        return

    print("\n" + "=" * 70)
    print("FINAL EDA COMPARISON (UMDA vs Tree-EDA)")
    print("=" * 70)
    print(f"{'Function':30s} {'UMDA':>12s} {'Tree-EDA':>12s} {'Winner':>12s}")
    print("-" * 70)

    for key, label in [
        ("k_deceptive", "K-Deceptive (k=3)"),
        ("decep3", "Deceptive-3"),
        ("hiff", "HIFF"),
        ("polytree3", "Polytree-3"),
        ("polytree3_overlap", "Polytree-3 overlap"),
    ]:
        umda_best = all_stats[key]["umda"].best_fitness_overall
        tree_best = all_stats[key]["tree"].best_fitness_overall
        if umda_best > tree_best:
            winner = "UMDA"
        elif tree_best > umda_best:
            winner = "Tree-EDA"
        else:
            winner = "Tie"
        print(f"{label:30s} {umda_best:12.4f} {tree_best:12.4f} {winner:>12s}")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run additive decomposable benchmark examples with configurable EDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible runs (default: 42).",
    )
    parser.add_argument(
        "--eda-type",
        type=str,
        choices=["umda", "tree", "both"],
        default="both",
        help="EDA to run: umda, tree, or both (default: both).",
    )
    return parser.parse_args()


def main():
    """Run all examples."""
    args = parse_args()
    eda_types = ["umda", "tree"] if args.eda_type == "both" else [args.eda_type]

    print("\n" + "=" * 70)
    print("ADDITIVE DECOMPOSABLE BENCHMARK FUNCTIONS - EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the newly ported additively")
    print("decomposable benchmark functions from the C++ EDA implementation.")
    print()
    print(f"Selected EDA(s): {', '.join(EDA_NAME_MAP[e] for e in eda_types)}")
    print(f"Seed: {args.seed}")
    print()

    np.random.seed(args.seed)
    # Run examples
    all_stats = {}

    all_stats['k_deceptive'] = example_k_deceptive(eda_types, args.seed)
    all_stats['decep3'] = example_decep3(eda_types, args.seed)
    all_stats['hiff'] = example_hiff(eda_types, args.seed)
    all_stats['polytree3'] = example_polytree3(eda_types, args.seed)
    all_stats['polytree3_overlap'] = example_polytree3_overlap(eda_types, args.seed)
    all_stats['k_comparison'] = compare_k_values(eda_types, args.seed)

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
    print_eda_final_comparison(all_stats, eda_types)


if __name__ == "__main__":
    main()
