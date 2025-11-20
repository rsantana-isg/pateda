"""
GNBG Single Problem Example

This script demonstrates how to run a single continuous EDA on a GNBG problem
instance. This is useful for quick testing and understanding the benchmark.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.gnbg_benchmark import (
    run_single_experiment,
    load_gnbg_instance,
    create_gnbg_fitness_wrapper
)

from pateda import EDA, EDAComponents
from pateda.learning import LearnGaussianUnivariate
from pateda.sampling import SampleGaussianUnivariate
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


def run_simple_example():
    """
    Run a simple example: Gaussian UMDA on GNBG f1.
    """
    print("="*80)
    print("GNBG Single Problem Example")
    print("="*80)
    print()

    # Configuration
    instances_folder = str(Path(__file__).parent.parent / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')
    problem_index = 1
    eda_name = 'gaussian_umda'
    pop_size = 50
    selection_ratio = 0.5
    seed = 42

    print(f"Problem: GNBG f{problem_index}")
    print(f"EDA: {eda_name}")
    print(f"Population size: {pop_size}")
    print(f"Seed: {seed}")
    print()

    # Run experiment
    results = run_single_experiment(
        eda_name=eda_name,
        problem_index=problem_index,
        instances_folder=instances_folder,
        pop_size=pop_size,
        selection_ratio=selection_ratio,
        seed=seed,
        verbose=True
    )

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nFinal results:")
    print(f"  Best fitness: {results['best_fitness']:.6e}")
    print(f"  Error from optimum: {results['error_from_optimum']:.6e}")
    print(f"  Success: {results['success']}")
    print(f"  Function evaluations: {results['function_evaluations']}")
    print(f"  Runtime: {results['runtime_seconds']:.2f} seconds")

    return results


def compare_edas_on_single_problem():
    """
    Compare multiple EDAs on a single GNBG problem.
    """
    print("\n" + "="*80)
    print("Comparing EDAs on GNBG f1")
    print("="*80)
    print()

    # Configuration
    instances_folder = str(Path(__file__).parent.parent / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')
    problem_index = 1
    eda_names = ['gaussian_umda', 'gaussian_full']
    pop_size = 50
    seed = 42

    results_list = []

    for eda_name in eda_names:
        print(f"\nRunning {eda_name}...")

        results = run_single_experiment(
            eda_name=eda_name,
            problem_index=problem_index,
            instances_folder=instances_folder,
            pop_size=pop_size,
            selection_ratio=0.5,
            seed=seed,
            verbose=False
        )
        results_list.append(results)

        print(f"  Error: {results['error_from_optimum']:.6e}, "
              f"FE: {results['function_evaluations']}, "
              f"Time: {results['runtime_seconds']:.2f}s")

    # Print comparison
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"{'EDA':<20} | {'Error':<15} | {'FE':<10} | {'Runtime (s)'}")
    print("-"*80)
    for r in results_list:
        print(f"{r['eda_name']:<20} | {r['error_from_optimum']:<15.6e} | "
              f"{r['function_evaluations']:<10} | {r['runtime_seconds']:.2f}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Compare multiple EDAs
        compare_edas_on_single_problem()
    else:
        # Run simple example
        run_simple_example()
