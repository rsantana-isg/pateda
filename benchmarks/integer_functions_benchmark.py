"""
Integer Functions Benchmark for Discrete EDAs with Integer Representation

This script evaluates discrete EDAs on benchmark functions designed for
integer (multi-valued discrete) representation. These functions test the
ability of EDAs to model dependencies among integer-valued variables
with cardinality > 2.

The benchmark compares different EDA approaches:
- UMDA: Univariate model (assumes independence)
- Tree-EDA: Tree-structured dependencies (pairwise)
- MN-FDA: Markov network factorization (higher-order dependencies)

References:
- Larrañaga, P., & Lozano, J. A. (2002). "Estimation of Distribution Algorithms"
- Mühlenbein, H. (1998). "Scalable Problems for Evolutionary Computation"
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pateda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.learning.mnfda import LearnMNFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit

# Import integer benchmark functions
from pateda.functions.discrete.integer_functions import (
    integer_onemax,
    integer_leading_blocks,
    integer_max_blocks,
    integer_categorical_match,
    integer_plateau_search,
    integer_dependency_chain,
    integer_multi_level_trap,
    integer_parity_blocks,
    integer_hierarchical_blocks,
    IntegerNKLandscape,
    create_integer_nk_objective_function,
)

# Also include generalized k-deceptive from additive_decomposable
from pateda.functions.discrete.additive_decomposable import (
    gen_k_decep,
    gen_k_decep_overlap,
)


# ============================================================================
# Integer Function Registry with Appropriate Problem Sizes
# ============================================================================

def create_integer_function_registry(cardinality: int = 4) -> Dict[str, Dict[str, Any]]:
    """
    Create registry of integer benchmark functions for a given cardinality.

    Args:
        cardinality: Number of possible values for each variable (default: 4)

    Returns:
        Dictionary mapping function names to function info
    """
    max_val = cardinality - 1

    registry = {
        # Simple functions (independence-friendly)
        'integer_onemax': {
            'function': lambda x: integer_onemax(x, cardinality=cardinality),
            'sizes': [30, 60, 90],
            'optimal': lambda n: n * max_val,
            'category': 'simple',
            'description': 'Sum of integer values (tests basic convergence)',
        },
        'integer_leading_blocks': {
            'function': lambda x: integer_leading_blocks(x, cardinality=cardinality, block_size=3),
            'sizes': [30, 60, 90],  # Divisible by 3
            'optimal': lambda n: n // 3,  # All blocks with max values
            'category': 'simple',
            'description': 'Leading blocks with max values (tests sequential optimization)',
        },

        # Block-based functions (tests dependency modeling)
        'integer_max_blocks': {
            'function': lambda x: integer_max_blocks(x, k=3, cardinality=cardinality),
            'sizes': [30, 60, 90],  # Divisible by 3
            'optimal': lambda n: (n // 3) * (3 * max_val + 3),  # All max with bonus
            'category': 'block',
            'description': 'Max blocks function (tests block-level dependencies)',
        },
        'integer_plateau_search': {
            'function': lambda x: integer_plateau_search(x, cardinality=cardinality, block_size=3),
            'sizes': [30, 60, 90],
            'optimal': lambda n: (n // 3) * (3 * max_val + max_val),  # Max blocks with bonus
            'category': 'block',
            'description': 'Plateau search (tests plateau navigation)',
        },

        # Deceptive functions (tests ability to escape local optima)
        'integer_multi_level_trap': {
            'function': lambda x: integer_multi_level_trap(x, k=3, cardinality=cardinality),
            'sizes': [30, 60, 90],  # Divisible by 3
            'optimal': lambda n: (n // 3) * (3 * max_val + 3),
            'category': 'deceptive',
            'description': 'Multi-level trap (tests handling of deceptive structure)',
        },
        'gen_k_decep_int': {
            'function': lambda x: gen_k_decep(x, k=3, cardinality=cardinality),
            'sizes': [30, 60, 90],  # Divisible by 3
            'optimal': lambda n: (n // 3) * (3 * max_val),
            'category': 'deceptive',
            'description': 'Generalized k-deceptive for integers',
        },

        # Dependency functions (tests dependency chain modeling)
        'integer_dependency_chain': {
            'function': lambda x: integer_dependency_chain(x, cardinality=cardinality),
            'sizes': [30, 60, 90],
            'optimal': lambda n: (n - 1) * cardinality + n * max_val,  # All dependencies + max sum
            'category': 'dependency',
            'description': 'Dependency chain (tests sequential dependency modeling)',
        },
        'integer_categorical_match': {
            'function': lambda x: integer_categorical_match(x, cardinality=cardinality),
            'sizes': [30, 60, 90],
            'optimal': lambda n: n,  # All positions match
            'category': 'dependency',
            'description': 'Categorical match (tests categorical distribution learning)',
        },

        # Hierarchical functions (tests multi-level structure)
        'integer_hierarchical': {
            'function': lambda x: integer_hierarchical_blocks(x, cardinality=cardinality),
            'sizes': [16, 32, 64],  # Powers of 2
            'optimal': lambda n: None,  # Complex, depends on structure
            'category': 'hierarchical',
            'description': 'Hierarchical blocks (tests multi-level consistency)',
        },
        'integer_parity_blocks': {
            'function': lambda x: integer_parity_blocks(x, block_size=4, cardinality=cardinality),
            'sizes': [32, 64],  # Divisible by 4
            'optimal': lambda n: None,  # Depends on parity
            'category': 'hierarchical',
            'description': 'Parity blocks (tests non-linear relationships)',
        },
    }

    return registry


# ============================================================================
# EDA Configuration for Integer Problems
# ============================================================================

def get_integer_eda_configuration(
    eda_name: str,
    n_vars: int,
    pop_size: int,
    max_gen: int,
    cardinality: int = 4,
    selection_ratio: float = 0.5,
) -> EDAComponents:
    """
    Get EDA component configuration for integer representation problems.

    Args:
        eda_name: Name of EDA ('umda', 'tree_eda', 'mnfda')
        n_vars: Number of variables
        pop_size: Population size
        max_gen: Maximum generations
        cardinality: Number of possible values per variable
        selection_ratio: Selection ratio for truncation selection

    Returns:
        EDAComponents configuration
    """
    stop_condition = MaxGenerations(max_gen=max_gen)
    seeding = RandomInit()

    if eda_name == 'umda':
        # UMDA: Univariate model (independence assumption)
        components = EDAComponents(
            seeding=seeding,
            learning=LearnUMDA(alpha=0.01),
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=stop_condition,
        )

    elif eda_name == 'tree_eda':
        # Tree-EDA: Tree-structured dependencies
        components = EDAComponents(
            seeding=seeding,
            learning=LearnTreeModel(
                alpha=0.01,
                mi_threshold=0.01
            ),
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=stop_condition,
        )

    elif eda_name == 'mnfda':
        # MN-FDA: Markov network factorization
        components = EDAComponents(
            seeding=seeding,
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


def create_eda_fitness_function(func: Callable, maximize: bool = True) -> Callable:
    """
    Wrap a function to work with pateda's EDA framework.

    Args:
        func: Function that takes 1D array and returns scalar
        maximize: If True, return positive fitness

    Returns:
        Function compatible with pateda EDA
    """
    def eda_function(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            fitness = func(x)
            return np.array([fitness if maximize else -fitness])
        else:
            pop_size = x.shape[0]
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                fitness[i] = func(x[i])
            return fitness if maximize else -fitness

    return eda_function


# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_integer_experiment(
    eda_name: str,
    function_name: str,
    n_vars: int,
    cardinality: int = 4,
    pop_size: int = 200,
    max_gen: int = 100,
    selection_ratio: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single EDA on a single integer function.

    Args:
        eda_name: Name of EDA to run
        function_name: Name of integer function
        n_vars: Problem size (number of variables)
        cardinality: Number of possible values per variable
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Selection ratio
        seed: Random seed (optional)
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    # Get function registry
    registry = create_integer_function_registry(cardinality)

    if function_name not in registry:
        raise ValueError(f"Unknown function: {function_name}. "
                        f"Available: {list(registry.keys())}")

    func_info = registry[function_name]
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
    fitness_func = create_eda_fitness_function(func, maximize=True)

    # Cardinality for integer variables
    var_cardinality = np.full(n_vars, cardinality)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {eda_name} on {function_name} (n={n_vars}, c={cardinality})")
        print(f"{'='*80}")
        print(f"Category: {category}")
        print(f"Description: {func_info['description']}")
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_gen}")
        if optimal_fitness is not None:
            print(f"Optimal fitness: {optimal_fitness}")

    # Get EDA configuration
    components = get_integer_eda_configuration(
        eda_name=eda_name,
        n_vars=n_vars,
        pop_size=pop_size,
        max_gen=max_gen,
        cardinality=cardinality,
        selection_ratio=selection_ratio,
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=var_cardinality,
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
        'cardinality': cardinality,
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


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_integer_benchmark(
    eda_names: List[str],
    function_names: Optional[List[str]] = None,
    cardinalities: List[int] = None,
    n_runs: int = 10,
    pop_size: int = 200,
    max_gen: int = 100,
    selection_ratio: float = 0.5,
    output_folder: str = 'integer_results',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across multiple EDAs and integer functions.

    Args:
        eda_names: List of EDA names to evaluate
        function_names: List of function names (None = all functions)
        cardinalities: List of cardinalities to test (default: [4])
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

    if cardinalities is None:
        cardinalities = [4]

    # Get first registry to determine function names
    first_registry = create_integer_function_registry(cardinalities[0])

    if function_names is None:
        function_names = list(first_registry.keys())

    all_results = []
    total_experiments = 0

    # Count total experiments
    for card in cardinalities:
        registry = create_integer_function_registry(card)
        for func_name in function_names:
            if func_name in registry:
                total_experiments += len(registry[func_name]['sizes']) * len(eda_names) * n_runs

    experiment_count = 0

    print(f"\n{'='*80}")
    print("Integer Functions Benchmark for Discrete EDAs")
    print(f"{'='*80}")
    print(f"EDAs: {', '.join(eda_names)}")
    print(f"Functions: {len(function_names)} functions")
    print(f"Cardinalities: {cardinalities}")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*80}\n")

    for cardinality in cardinalities:
        registry = create_integer_function_registry(cardinality)

        for eda_name in eda_names:
            for func_name in function_names:
                if func_name not in registry:
                    continue

                func_info = registry[func_name]

                for n_vars in func_info['sizes']:
                    for run in range(n_runs):
                        experiment_count += 1

                        if verbose:
                            print(f"\n[{experiment_count}/{total_experiments}] "
                                  f"{eda_name} on {func_name}(n={n_vars}, c={cardinality}), "
                                  f"run {run+1}/{n_runs}")

                        # Run experiment with unique seed
                        seed = hash((func_name, n_vars, cardinality, eda_name, run)) % (2**31)

                        try:
                            results = run_integer_experiment(
                                eda_name=eda_name,
                                function_name=func_name,
                                n_vars=n_vars,
                                cardinality=cardinality,
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

    if len(df) == 0:
        print("No results to save.")
        return df

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save full results as CSV
    csv_path = os.path.join(output_folder, f'integer_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save as pickle (includes fitness history)
    pickle_path = os.path.join(output_folder, f'integer_results_{timestamp}.pkl')
    df.to_pickle(pickle_path)
    print(f"Full results (with history) saved to: {pickle_path}")

    # Generate summary statistics
    summary = generate_integer_summary(df)
    summary_path = os.path.join(output_folder, f'integer_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary
    print_integer_summary(summary)

    return df


def generate_integer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from benchmark results."""
    summary_rows = []

    for eda_name in df['eda_name'].unique():
        for func_name in df['function_name'].unique():
            for n_vars in df[df['function_name'] == func_name]['n_vars'].unique():
                for card in df[(df['function_name'] == func_name) &
                              (df['n_vars'] == n_vars)]['cardinality'].unique():
                    subset = df[(df['eda_name'] == eda_name) &
                               (df['function_name'] == func_name) &
                               (df['n_vars'] == n_vars) &
                               (df['cardinality'] == card)]

                    if len(subset) == 0:
                        continue

                    summary_rows.append({
                        'eda_name': eda_name,
                        'function_name': func_name,
                        'category': subset.iloc[0]['category'],
                        'n_vars': n_vars,
                        'cardinality': card,
                        'n_runs': len(subset),
                        'success_rate': subset['success'].mean() if 'success' in subset.columns else None,
                        'mean_fitness': subset['best_fitness'].mean(),
                        'std_fitness': subset['best_fitness'].std(),
                        'median_fitness': subset['best_fitness'].median(),
                        'mean_generations': subset['generation_found'].mean(),
                        'std_generations': subset['generation_found'].std(),
                        'mean_runtime': subset['runtime_seconds'].mean(),
                        'std_runtime': subset['runtime_seconds'].std(),
                    })

    return pd.DataFrame(summary_rows)


def print_integer_summary(summary: pd.DataFrame):
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


# ============================================================================
# Quick Test Function
# ============================================================================

def quick_test_integer_benchmark():
    """Run a quick test of the integer benchmark."""
    print("Running quick test of integer benchmark...")

    # Test with a single function and small settings
    results = run_integer_experiment(
        eda_name='umda',
        function_name='integer_onemax',
        n_vars=30,
        cardinality=4,
        pop_size=100,
        max_gen=50,
        seed=42,
        verbose=True
    )

    print("\nQuick test completed successfully!")
    return results


if __name__ == '__main__':
    # Default configuration
    EDA_NAMES = [
        'umda',      # Univariate (independence)
        'tree_eda',  # Tree structure (pairwise dependencies)
        'mnfda',     # Markov network factorization
    ]

    # Select subset of functions for quick testing
    FUNCTION_SUBSET = [
        'integer_onemax',
        'integer_max_blocks',
        'integer_multi_level_trap',
        'integer_dependency_chain',
        'gen_k_decep_int',
    ]

    # Benchmark configuration
    CARDINALITIES = [4]   # Test with 4-valued variables
    N_RUNS = 5            # Independent runs per configuration
    POP_SIZE = 200        # Population size
    MAX_GEN = 100         # Maximum generations
    SELECTION_RATIO = 0.5 # Truncation selection ratio

    # Run benchmark
    results_df = run_integer_benchmark(
        eda_names=EDA_NAMES,
        function_names=FUNCTION_SUBSET,
        cardinalities=CARDINALITIES,
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        selection_ratio=SELECTION_RATIO,
        output_folder='benchmarks/results/integer_results',
        verbose=True
    )

    print(f"\n{'='*80}")
    print("Integer Benchmark complete!")
    print(f"{'='*80}\n")
