"""
Binary Functions Benchmark for Discrete EDAs

This script evaluates discrete EDAs on a comprehensive set of binary benchmark
functions from the additively decomposable family. These functions are commonly
used for testing Estimation of Distribution Algorithms.

The benchmark includes:
- K-Deceptive variants
- Deceptive-3 variants
- Hierarchical functions (HIFF, fhtrap1)
- Polytree functions
- Cuban functions

References:
- Mühlenbein, H., & Paass, G. (1996). "From recombination of genes to the
  estimation of distributions I."
- Watson, R.A. (2002). "Hierarchical Building-Block Problems"
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

from pateda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.learning.mnfda import LearnMNFDA
from pateda.sampling import SampleFDA
from pateda.sampling.gibbs import SampleGibbs
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations

# Import all binary functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, gen_k_decep, gen_k_decep_overlap,
    decep3, decep_marta3, decep_marta3_new, decep3_mh,
    two_peaks_decep3, decep_venturini, hard_decep5,
    hiff, fhtrap1,
    first_polytree3_ochoa, first_polytree5_ochoa,
    fc2, fc3, fc4, fc5
)


# ============================================================================
# Binary Function Registry with Appropriate Problem Sizes
# ============================================================================

BINARY_FUNCTIONS = {
    # K-Deceptive Functions
    'k_deceptive_k3': {
        'function': lambda x: k_deceptive(x, k=3),
        'sizes': [30, 60, 90],  # Divisible by 3
        'optimal': lambda n: n,  # All 1s gives n
        'category': 'k-deceptive'
    },
    'k_deceptive_k4': {
        'function': lambda x: k_deceptive(x, k=4),
        'sizes': [40, 80],  # Divisible by 4
        'optimal': lambda n: n,
        'category': 'k-deceptive'
    },
    'k_deceptive_k5': {
        'function': lambda x: k_deceptive(x, k=5),
        'sizes': [50, 100],  # Divisible by 5
        'optimal': lambda n: n,
        'category': 'k-deceptive'
    },
    'gen_k_decep_k3': {
        'function': lambda x: gen_k_decep(x, k=3, cardinality=2),
        'sizes': [30, 60, 90],
        'optimal': lambda n: n,  # All 1s
        'category': 'k-deceptive'
    },
    'gen_k_decep_overlap': {
        'function': lambda x: gen_k_decep_overlap(x, k=3, cardinality=2, overlap=1),
        'sizes': [30, 60],
        'optimal': lambda n: None,  # Depends on overlapping structure
        'category': 'k-deceptive'
    },

    # Deceptive-3 Functions
    'decep3_overlap': {
        'function': lambda x: decep3(x, overlap=True),
        'sizes': [30, 60],  # Step=2, any size works
        'optimal': lambda n: (n - 2) // 2 + 1,  # All 111 triplets
        'category': 'deceptive-3'
    },
    'decep3_no_overlap': {
        'function': lambda x: decep3(x, overlap=False),
        'sizes': [30, 60, 90],  # Divisible by 3
        'optimal': lambda n: n // 3,
        'category': 'deceptive-3'
    },
    'decep_marta3': {
        'function': decep_marta3,
        'sizes': [30, 60, 90],  # Divisible by 3
        'optimal': lambda n: None,  # Complex lookup table
        'category': 'deceptive-3'
    },
    'decep_marta3_new': {
        'function': decep_marta3_new,
        'sizes': [30, 60],  # Divisible by 3
        'optimal': lambda n: (n // 3) * 1.5,  # 1.5 per triplet
        'category': 'deceptive-3'
    },
    'decep3_mh': {
        'function': decep3_mh,
        'sizes': [30, 60, 90],  # Divisible by 3
        'optimal': lambda n: (n // 3) * 3.0,  # 3.0 per optimal triplet
        'category': 'deceptive-3'
    },
    'two_peaks_decep3': {
        'function': two_peaks_decep3,
        'sizes': [31, 61],  # n = 3k + 1 for proper coverage
        'optimal': lambda n: None,  # Depends on first bit
        'category': 'deceptive-3'
    },
    'decep_venturini': {
        'function': decep_venturini,
        'sizes': [30, 60, 90],  # Divisible by 3
        'optimal': lambda n: None,  # Complex lookup table
        'category': 'deceptive-3'
    },

    # Hard Deceptive-5
    'hard_decep5': {
        'function': hard_decep5,
        'sizes': [50, 100],  # Divisible by 5
        'optimal': lambda n: n // 5,  # 1.0 per optimal 5-bit block
        'category': 'hard-deceptive'
    },

    # Hierarchical Functions
    'hiff_16': {
        'function': hiff,
        'sizes': [16],  # Power of 2
        'optimal': lambda n: n * (np.log2(n) + 1),
        'category': 'hierarchical'
    },
    'hiff_32': {
        'function': hiff,
        'sizes': [32],
        'optimal': lambda n: n * (np.log2(n) + 1),
        'category': 'hierarchical'
    },
    'hiff_64': {
        'function': hiff,
        'sizes': [64],
        'optimal': lambda n: n * (np.log2(n) + 1),
        'category': 'hierarchical'
    },
    'hiff_128': {
        'function': hiff,
        'sizes': [128],
        'optimal': lambda n: n * (np.log2(n) + 1),
        'category': 'hierarchical'
    },
    'fhtrap1_9': {
        'function': fhtrap1,
        'sizes': [9],  # Power of 3
        'optimal': lambda n: None,  # Complex hierarchical
        'category': 'hierarchical'
    },
    'fhtrap1_27': {
        'function': fhtrap1,
        'sizes': [27],
        'optimal': lambda n: None,
        'category': 'hierarchical'
    },
    'fhtrap1_81': {
        'function': fhtrap1,
        'sizes': [81],
        'optimal': lambda n: None,
        'category': 'hierarchical'
    },

    # Polytree Functions
    'polytree3_no_overlap': {
        'function': lambda x: first_polytree3_ochoa(x, overlap=False),
        'sizes': [30, 60],  # Divisible by 3
        'optimal': lambda n: None,  # Complex lookup table
        'category': 'polytree'
    },
    'polytree3_overlap': {
        'function': lambda x: first_polytree3_ochoa(x, overlap=True),
        'sizes': [30, 60],  # Step=2
        'optimal': lambda n: None,
        'category': 'polytree'
    },
    'polytree5': {
        'function': first_polytree5_ochoa,
        'sizes': [50, 100],  # Divisible by 5
        'optimal': lambda n: None,
        'category': 'polytree'
    },

    # Cuban Functions
    'fc2': {
        'function': fc2,
        'sizes': [50, 100],  # Divisible by 5
        'optimal': lambda n: (n // 5) * 4.0,  # 4.0 per optimal partition
        'category': 'cuban'
    },
    'fc3': {
        'function': fc3,
        'sizes': [50, 100],  # Divisible by 5
        'optimal': lambda n: (n // 5) * 7.0,  # 5 + 2 per optimal partition
        'category': 'cuban'
    },
    'fc4': {
        'function': fc4,
        'sizes': [21, 41, 81],  # (n-1)/4 partitions
        'optimal': lambda n: None,  # Complex structure
        'category': 'cuban'
    },
    'fc5': {
        'function': fc5,
        'sizes': [29, 53],  # (n-5)/8 partitions
        'optimal': lambda n: None,  # Combined structure
        'category': 'cuban'
    },
}


def create_eda_friendly_function(func: Callable, maximize: bool = True):
    """
    Wrap a function to work with pateda's EDA framework.

    Args:
        func: Binary function that takes 1D array and returns scalar
        maximize: If True, return positive fitness; if False, negate for minimization

    Returns:
        Function compatible with pateda EDA
    """
    def eda_function(x: np.ndarray) -> np.ndarray:
        """EDA-compatible fitness function"""
        if x.ndim == 1:
            # Single solution
            fitness = func(x)
            return np.array([fitness if maximize else -fitness])
        else:
            # Population (2D array)
            pop_size = x.shape[0]
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                fitness[i] = func(x[i])
            return fitness if maximize else -fitness

    return eda_function


def get_eda_configuration(
    eda_name: str,
    n_vars: int,
    pop_size: int,
    max_gen: int,
    selection_ratio: float = 0.5,
    target_fitness: Optional[float] = None
) -> EDAComponents:
    """
    Get EDA component configuration for a specific algorithm.

    Args:
        eda_name: Name of EDA ('umda', 'tree_eda', 'mnfda')
        n_vars: Number of variables
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Selection ratio for truncation selection
        target_fitness: Target fitness for early stopping (optional)

    Returns:
        EDAComponents configuration
    """
    # Stop condition (note: target_fitness parameter is ignored in current implementation)
    stop_condition = MaxGenerations(max_gen=max_gen)

    if eda_name == 'umda':
        # UMDA: Univariate (independence assumption)
        components = EDAComponents(
            learning=LearnUMDA(alpha=0.01),  # Small Laplace smoothing
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=stop_condition,
        )

    elif eda_name == 'tree_eda':
        # Tree-EDA: Pairwise dependencies (tree structure)
        components = EDAComponents(
            learning=LearnTreeModel(
                alpha=0.01,
                mi_threshold=0.01,
                normalize_mi=True
            ),
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=stop_condition,
        )

    elif eda_name == 'mnfda':
        # MN-FDA: Markov network factorization
        components = EDAComponents(
            learning=LearnMNFDA(
                max_clique_size=3,
                threshold=0.05,
                prior=True,
                return_factorized=True
            ),
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=stop_condition,
        )

    else:
        raise ValueError(f"Unknown EDA name: {eda_name}")

    return components


def run_single_experiment(
    eda_name: str,
    function_name: str,
    n_vars: int,
    pop_size: int = 100,
    max_gen: int = 100,
    selection_ratio: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single EDA on a single binary function.

    Args:
        eda_name: Name of EDA to run
        function_name: Name of binary function
        n_vars: Problem size (number of variables)
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Selection ratio
        seed: Random seed (optional)
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    if seed is not None:
        np.random.seed(seed)

    # Get function info
    if function_name not in BINARY_FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")

    func_info = BINARY_FUNCTIONS[function_name]
    func = func_info['function']
    category = func_info['category']

    # Verify n_vars is appropriate
    if n_vars not in func_info['sizes']:
        raise ValueError(f"Invalid size {n_vars} for {function_name}. "
                        f"Valid sizes: {func_info['sizes']}")

    # Compute optimal fitness if available
    optimal_fitness = None
    if func_info['optimal'](n_vars) is not None:
        optimal_fitness = func_info['optimal'](n_vars)

    # Create EDA-compatible fitness function
    fitness_func = create_eda_friendly_function(func, maximize=True)

    # Cardinality for binary variables
    cardinality = np.full(n_vars, 2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {eda_name} on {function_name} (n={n_vars})")
        print(f"{'='*80}")
        print(f"Category: {category}")
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_gen}")
        if optimal_fitness is not None:
            print(f"Optimal fitness: {optimal_fitness}")

    # Get EDA configuration
    components = get_eda_configuration(
        eda_name=eda_name,
        n_vars=n_vars,
        pop_size=pop_size,
        max_gen=max_gen,
        selection_ratio=selection_ratio,
        target_fitness=optimal_fitness
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
        random_seed=seed,
    )

    # Measure runtime
    start_time = time.time()
    statistics, cache = eda.run(verbose=verbose)
    runtime = time.time() - start_time

    # Calculate success (reached optimal if known)
    success = False
    if optimal_fitness is not None:
        success = np.isclose(statistics.best_fitness_overall, optimal_fitness, rtol=1e-5)

    # Results dictionary
    results = {
        'eda_name': eda_name,
        'function_name': function_name,
        'category': category,
        'n_vars': n_vars,
        'seed': seed,
        'pop_size': pop_size,
        'selection_ratio': selection_ratio,
        'max_gen': max_gen,

        # Performance metrics
        'best_fitness': float(statistics.best_fitness_overall),
        'optimal_fitness': float(optimal_fitness) if optimal_fitness is not None else None,
        'success': bool(success),
        'generation_found': int(statistics.generation_found),
        'generations_run': len(statistics.best_fitness),
        'runtime_seconds': float(runtime),

        # Convergence statistics
        'mean_fitness_final': float(statistics.mean_fitness[-1]),
        'std_fitness_final': float(statistics.std_fitness[-1]),

        # Fitness history (compressed)
        'best_fitness_history': [float(f) for f in statistics.best_fitness[::max(1, len(statistics.best_fitness)//20)]],
    }

    if verbose:
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"Best fitness: {statistics.best_fitness_overall:.6f}")
        if optimal_fitness is not None:
            print(f"Optimal fitness: {optimal_fitness:.6f}")
            print(f"Success: {'✓' if success else '✗'}")
        print(f"Generation found: {statistics.generation_found}")
        print(f"Runtime: {runtime:.2f} seconds")

    return results


def run_benchmark(
    eda_names: List[str],
    function_names: Optional[List[str]] = None,
    n_runs: int = 10,
    pop_size: int = 100,
    max_gen: int = 100,
    selection_ratio: float = 0.5,
    output_folder: str = 'binary_results',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across multiple EDAs and binary functions.

    Args:
        eda_names: List of EDA names to evaluate
        function_names: List of function names (None = all functions)
        n_runs: Number of independent runs per configuration
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Selection ratio
        output_folder: Folder to save results
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Use all functions if not specified
    if function_names is None:
        function_names = list(BINARY_FUNCTIONS.keys())

    all_results = []
    total_experiments = 0

    # Count total experiments
    for func_name in function_names:
        total_experiments += len(BINARY_FUNCTIONS[func_name]['sizes']) * len(eda_names) * n_runs

    experiment_count = 0

    print(f"\n{'='*80}")
    print("Binary Functions Benchmark for Discrete EDAs")
    print(f"{'='*80}")
    print(f"EDAs: {', '.join(eda_names)}")
    print(f"Functions: {len(function_names)} functions")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*80}\n")

    for eda_name in eda_names:
        for func_name in function_names:
            func_info = BINARY_FUNCTIONS[func_name]

            for n_vars in func_info['sizes']:
                for run in range(n_runs):
                    experiment_count += 1

                    if verbose:
                        print(f"\n[{experiment_count}/{total_experiments}] "
                              f"{eda_name} on {func_name}(n={n_vars}), run {run+1}/{n_runs}")

                    # Run experiment with unique seed
                    seed = hash((func_name, n_vars, eda_name, run)) % (2**31)

                    try:
                        results = run_single_experiment(
                            eda_name=eda_name,
                            function_name=func_name,
                            n_vars=n_vars,
                            pop_size=pop_size,
                            max_gen=max_gen,
                            selection_ratio=selection_ratio,
                            seed=seed,
                            verbose=False
                        )
                        results['run'] = run
                        all_results.append(results)

                        if verbose:
                            status = "✓" if results.get('success', False) else " "
                            print(f"  {status} Fitness: {results['best_fitness']:.4f}, "
                                  f"Gen: {results['generation_found']}, "
                                  f"Time: {results['runtime_seconds']:.2f}s")

                    except Exception as e:
                        print(f"  ✗ ERROR: {str(e)}")
                        continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save full results as CSV
    csv_path = os.path.join(output_folder, f'binary_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save as pickle (includes fitness history)
    pickle_path = os.path.join(output_folder, f'binary_results_{timestamp}.pkl')
    df.to_pickle(pickle_path)
    print(f"Full results (with history) saved to: {pickle_path}")

    # Generate summary statistics
    summary = generate_summary_statistics(df)
    summary_path = os.path.join(output_folder, f'binary_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary
    print_summary(summary)

    return df


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from benchmark results."""
    summary_rows = []

    for eda_name in df['eda_name'].unique():
        for func_name in df['function_name'].unique():
            for n_vars in df[df['function_name'] == func_name]['n_vars'].unique():
                subset = df[(df['eda_name'] == eda_name) &
                           (df['function_name'] == func_name) &
                           (df['n_vars'] == n_vars)]

                if len(subset) == 0:
                    continue

                summary_rows.append({
                    'eda_name': eda_name,
                    'function_name': func_name,
                    'category': subset.iloc[0]['category'],
                    'n_vars': n_vars,
                    'n_runs': len(subset),
                    'success_rate': subset['success'].mean() if 'optimal_fitness' in subset.columns else None,
                    'mean_fitness': subset['best_fitness'].mean(),
                    'std_fitness': subset['best_fitness'].std(),
                    'median_fitness': subset['best_fitness'].median(),
                    'mean_generations': subset['generation_found'].mean(),
                    'std_generations': subset['generation_found'].std(),
                    'mean_runtime': subset['runtime_seconds'].mean(),
                    'std_runtime': subset['runtime_seconds'].std(),
                })

    return pd.DataFrame(summary_rows)


def print_summary(summary: pd.DataFrame):
    """Print summary statistics in a readable format."""
    print(f"\n{'='*80}")
    print("Summary Statistics by Category")
    print(f"{'='*80}\n")

    for category in summary['category'].unique():
        cat_summary = summary[summary['category'] == category]

        print(f"\n{category.upper()}:")
        print(f"  Number of problems: {len(cat_summary['function_name'].unique())}")

        for eda_name in cat_summary['eda_name'].unique():
            eda_cat = cat_summary[cat_summary['eda_name'] == eda_name]

            if 'success_rate' in eda_cat.columns and eda_cat['success_rate'].notna().any():
                success_rate = eda_cat['success_rate'].mean() * 100
                print(f"  {eda_name}: Success rate: {success_rate:.1f}%, "
                      f"Avg gen: {eda_cat['mean_generations'].mean():.1f}")
            else:
                print(f"  {eda_name}: Avg fitness: {eda_cat['mean_fitness'].mean():.4f}, "
                      f"Avg gen: {eda_cat['mean_generations'].mean():.1f}")


if __name__ == '__main__':
    # Default configuration
    EDA_NAMES = [
        'umda',      # Univariate (independence)
        'tree_eda',  # Tree structure (pairwise dependencies)
        'mnfda',     # Markov network factorization
    ]

    # Select subset of functions for quick testing
    # Comment out to run all functions
    FUNCTION_SUBSET = [
        'k_deceptive_k3',
        'decep3_no_overlap',
        'hiff_32',
        'polytree3_no_overlap',
        'fc2',
    ]

    # Benchmark configuration
    N_RUNS = 10           # Independent runs per configuration
    POP_SIZE = 200        # Population size
    MAX_GEN = 200         # Maximum generations
    SELECTION_RATIO = 0.5 # Truncation selection ratio

    # Run benchmark
    results_df = run_benchmark(
        eda_names=EDA_NAMES,
        function_names=FUNCTION_SUBSET,  # Use None for all functions
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        selection_ratio=SELECTION_RATIO,
        output_folder='binary_results',
        verbose=True
    )

    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}\n")
