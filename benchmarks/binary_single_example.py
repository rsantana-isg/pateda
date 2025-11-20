"""
Binary Functions Single Example

This script demonstrates how to run a single discrete EDA on a single binary
function. This is useful for quick testing and understanding the benchmark.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.binary_functions_benchmark import (
    run_single_experiment,
    BINARY_FUNCTIONS,
    get_eda_configuration
)

from pateda import EDA, EDAComponents
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


def run_simple_example():
    """
    Run a simple example: UMDA on k-deceptive-3 with n=30.
    """
    print("="*80)
    print("Binary Functions Single Example")
    print("="*80)
    print()

    # Configuration
    eda_name = 'umda'
    function_name = 'k_deceptive_k3'
    n_vars = 30
    pop_size = 100
    max_gen = 100
    seed = 42

    print(f"Function: {function_name} (n={n_vars})")
    print(f"EDA: {eda_name}")
    print(f"Population size: {pop_size}")
    print(f"Max generations: {max_gen}")
    print(f"Seed: {seed}")
    print()

    # Run experiment
    results = run_single_experiment(
        eda_name=eda_name,
        function_name=function_name,
        n_vars=n_vars,
        pop_size=pop_size,
        max_gen=max_gen,
        selection_ratio=0.5,
        seed=seed,
        verbose=True
    )

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nFinal results:")
    print(f"  Best fitness: {results['best_fitness']:.6f}")
    if results['optimal_fitness'] is not None:
        print(f"  Optimal fitness: {results['optimal_fitness']:.6f}")
        print(f"  Success: {results['success']}")
    print(f"  Generation found: {results['generation_found']}")
    print(f"  Runtime: {results['runtime_seconds']:.2f} seconds")

    return results


def compare_edas_on_single_function():
    """
    Compare multiple EDAs on a single binary function.
    """
    print("\n" + "="*80)
    print("Comparing EDAs on k-deceptive-3 (n=30)")
    print("="*80)
    print()

    # Configuration
    function_name = 'k_deceptive_k3'
    n_vars = 30
    eda_names = ['umda', 'tree_eda', 'mnfda']
    pop_size = 100
    max_gen = 100
    seed = 42

    results_list = []

    for eda_name in eda_names:
        print(f"\nRunning {eda_name}...")

        results = run_single_experiment(
            eda_name=eda_name,
            function_name=function_name,
            n_vars=n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            selection_ratio=0.5,
            seed=seed,
            verbose=False
        )
        results_list.append(results)

        success_mark = "✓" if results.get('success', False) else " "
        print(f"  {success_mark} Fitness: {results['best_fitness']:.4f}, "
              f"Gen: {results['generation_found']}, "
              f"Time: {results['runtime_seconds']:.2f}s")

    # Print comparison
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"{'EDA':<15} | {'Best Fitness':<15} | {'Success':<10} | {'Gen':<8} | {'Runtime (s)'}")
    print("-"*80)
    for r in results_list:
        success_mark = "✓" if r.get('success', False) else " "
        print(f"{r['eda_name']:<15} | {r['best_fitness']:<15.4f} | {success_mark:<10} | "
              f"{r['generation_found']:<8} | {r['runtime_seconds']:.2f}")


def show_available_functions():
    """
    Display all available binary functions organized by category.
    """
    print("\n" + "="*80)
    print("Available Binary Functions")
    print("="*80)

    # Group by category
    by_category = {}
    for name, info in BINARY_FUNCTIONS.items():
        category = info['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((name, info['sizes']))

    for category, functions in sorted(by_category.items()):
        print(f"\n{category.upper()}:")
        for name, sizes in sorted(functions):
            print(f"  {name:<30} sizes: {sizes}")

    print(f"\nTotal: {len(BINARY_FUNCTIONS)} functions")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'compare':
            # Compare multiple EDAs
            compare_edas_on_single_function()
        elif command == 'list':
            # List available functions
            show_available_functions()
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python binary_single_example.py         # Run simple example")
            print("  python binary_single_example.py compare # Compare EDAs")
            print("  python binary_single_example.py list    # List all functions")
    else:
        # Run simple example
        run_simple_example()
