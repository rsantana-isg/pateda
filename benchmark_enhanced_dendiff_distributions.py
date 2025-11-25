"""
Enhanced benchmark script for evaluating dendiff on shifted objective functions
with Boltzmann and rank-based sampling methods.

This script focuses on:
1. Objective functions with shifted optima (away from origin)
2. Boltzmann distribution-based sampling
3. Rank-based distribution sampling (CMA-ES style)
4. Comparison of mean and standard deviation metrics

Usage:
    python benchmark_enhanced_dendiff_distributions.py
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, Tuple, Callable, List
import sys
import os

# Add pateda to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff
from benchmark_dendiff_distributions import (
    OBJECTIVE_FUNCTIONS,
    generate_empirical_fitness_distribution,
    generate_boltzmann_distribution,
    generate_rank_based_distribution,
    compute_kl_divergence_kde,
    compute_js_divergence,
    compute_statistical_distance
)


# ============================================================================
# Enhanced Evaluation Functions
# ============================================================================

def evaluate_dendiff_on_enhanced_distribution(
    generator_fn: Callable,
    generator_params: Dict[str, Any],
    n_samples_test: int = 500,
    dendiff_params: Dict[str, Any] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate dendiff model on enhanced fitness distribution.

    This evaluates dendiff using the enhanced distribution generators that
    support shifted optima, Boltzmann selection, or rank-based selection.

    Parameters
    ----------
    generator_fn : callable
        Function to generate distribution (e.g., generate_boltzmann_distribution)
    generator_params : dict
        Parameters for the generator function
    n_samples_test : int
        Number of samples to generate from learned model
    dendiff_params : dict
        Parameters for learn_dendiff
    seed : int
        Random seed
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        Evaluation results including metrics and fitness statistics
    """
    if verbose:
        print(f"  Generating distribution with {generator_params.get('n_initial', 'N/A')} initial solutions...")

    # Step 1: Generate MAT using specified method
    MAT, MAT_fitness, metadata = generator_fn(**generator_params)

    if verbose:
        print(f"    Distribution type: {metadata['type']}")
        print(f"    Best fitness in MAT: {metadata['best_fitness']:.6f}")
        print(f"    Mean fitness in MAT: {metadata['mean_fitness']:.6f}")
        print(f"    Std fitness in MAT: {metadata['std_fitness']:.6f}")
        if metadata.get('shift') is not None:
            print(f"    Shifted optimum: {metadata['shifted_optimum'][:3]}... (showing first 3 dims)")

    # Set default params if not provided
    if dendiff_params is None:
        dendiff_params = {}

    if verbose:
        print(f"  Learning dendiff model on MAT...")

    # Step 2: Learn dendiff model
    model = learn_dendiff(MAT, MAT_fitness, params=dendiff_params)

    if verbose:
        print(f"  Sampling {n_samples_test} solutions from learned model...")

    # Step 3: Sample from learned model
    objective_name = metadata['objective_name']
    bounds_info = OBJECTIVE_FUNCTIONS[objective_name]
    bounds = np.array([
        [bounds_info['bounds'][0]] * generator_params['n_vars'],
        [bounds_info['bounds'][1]] * generator_params['n_vars']
    ])

    Sampled_MAT = sample_dendiff(
        model,
        n_samples=n_samples_test,
        bounds=bounds,
        params=dendiff_params
    )

    # Evaluate fitness of sampled solutions
    obj_function = bounds_info['function']
    shift = metadata.get('shift')
    Sampled_MAT_fitness = obj_function(Sampled_MAT, shift=shift)

    if verbose:
        print(f"  Generating independent comparison distribution...")

    # Step 4: Generate MAT_ANOTHER (independent sample for comparison)
    # Use different seed to get independent sample
    generator_params_another = generator_params.copy()
    generator_params_another['seed'] = seed + 1000
    MAT_ANOTHER, MAT_ANOTHER_fitness, metadata_another = generator_fn(**generator_params_another)

    if verbose:
        print(f"  Computing distribution comparison metrics...")

    # Step 5: Compare distributions
    kl_div = compute_kl_divergence_kde(MAT_ANOTHER, Sampled_MAT)
    js_div = compute_js_divergence(MAT_ANOTHER, Sampled_MAT)
    stat_distances = compute_statistical_distance(MAT_ANOTHER, Sampled_MAT)

    # Also compare with original MAT
    kl_div_vs_mat = compute_kl_divergence_kde(MAT, Sampled_MAT)
    js_div_vs_mat = compute_js_divergence(MAT, Sampled_MAT)

    if verbose:
        print(f"  Computing fitness statistics...")

    # Enhanced fitness statistics comparison including std
    fitness_stats = {
        'mat_best_fitness': metadata['best_fitness'],
        'mat_mean_fitness': metadata['mean_fitness'],
        'mat_std_fitness': metadata['std_fitness'],
        'sampled_best_fitness': np.min(Sampled_MAT_fitness),
        'sampled_mean_fitness': np.mean(Sampled_MAT_fitness),
        'sampled_std_fitness': np.std(Sampled_MAT_fitness),
        'mat_another_best_fitness': metadata_another['best_fitness'],
        'mat_another_mean_fitness': metadata_another['mean_fitness'],
        'mat_another_std_fitness': metadata_another['std_fitness'],
        'fitness_mean_diff': abs(np.mean(Sampled_MAT_fitness) - metadata_another['mean_fitness']),
        'fitness_std_diff': abs(np.std(Sampled_MAT_fitness) - metadata_another['std_fitness']),
        'fitness_mean_improvement': (metadata['mean_fitness'] - np.mean(Sampled_MAT_fitness)) / (abs(metadata['mean_fitness']) + 1e-10),
        'fitness_std_ratio': np.std(Sampled_MAT_fitness) / (metadata['std_fitness'] + 1e-10),
    }

    results = {
        'distribution_type': metadata['type'],
        'objective_name': objective_name,
        'objective_description': metadata['objective_description'],
        'n_vars': generator_params['n_vars'],
        'n_initial': generator_params['n_initial'],
        'n_selected': generator_params['n_selected'],
        'selection_ratio': metadata['selection_ratio'],
        'has_shift': shift is not None,
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'kl_divergence_vs_mat': kl_div_vs_mat,
        'js_divergence_vs_mat': js_div_vs_mat,
        **stat_distances,
        **fitness_stats
    }

    # Add method-specific parameters
    if 'temperature' in metadata:
        results['temperature'] = metadata['temperature']
    if 'selection_pressure' in metadata:
        results['selection_pressure'] = metadata['selection_pressure']

    return results


# ============================================================================
# Main Benchmark Execution
# ============================================================================

def run_enhanced_benchmark():
    """Run comprehensive benchmark on enhanced fitness distributions."""

    print("=" * 80)
    print("ENHANCED DENDIFF DISTRIBUTION BENCHMARK")
    print("=" * 80)
    print()
    print("This benchmark evaluates dendiff on distributions with:")
    print("  1. Shifted objective function optima (away from origin)")
    print("  2. Boltzmann distribution-based selection")
    print("  3. Rank-based selection (CMA-ES style)")
    print("  4. Enhanced metrics including std comparison")
    print()

    # Configuration
    n_vars = 10
    n_initial = 1000
    n_selected = 500
    n_samples_test = 500
    seed = 42

    # Common dendiff parameters
    dendiff_params = {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}

    # Test objectives (focus on a subset for comprehensive testing)
    objectives_to_test = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'ellipsoid']

    all_results = []

    # ========================================================================
    # Part 1: Standard Selection with Shifted Optima
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: STANDARD SELECTION WITH SHIFTED OPTIMA")
    print("=" * 80)
    print()

    for objective_name in objectives_to_test:
        print(f"\n{objective_name.upper()} Function - Standard Selection (Shifted)")
        print("-" * 80)

        try:
            generator_params = {
                'objective_name': objective_name,
                'n_initial': n_initial,
                'n_selected': n_selected,
                'n_vars': n_vars,
                'seed': seed,
                'use_shift': True  # Enable shifting
            }

            results = evaluate_dendiff_on_enhanced_distribution(
                generate_empirical_fitness_distribution,
                generator_params,
                n_samples_test=n_samples_test,
                dendiff_params=dendiff_params,
                seed=seed,
                verbose=True
            )

            all_results.append(('Standard (Shifted)', objective_name, results))

            print(f"\n  Results:")
            print(f"    JS Divergence:           {results['js_divergence']:.6f}")
            print(f"    Fitness Mean Diff:       {results['fitness_mean_diff']:.6f}")
            print(f"    Fitness Std Diff:        {results['fitness_std_diff']:.6f}")
            print(f"    Fitness Std Ratio:       {results['fitness_std_ratio']:.4f}")
            print(f"    Mean Improvement:        {results['fitness_mean_improvement']*100:.2f}%")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # Part 2: Boltzmann Distribution Selection
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 2: BOLTZMANN DISTRIBUTION SELECTION")
    print("=" * 80)
    print()
    print("Testing with different temperature parameters:")
    print()

    temperatures = [0.5, 1.0, 2.0]

    for objective_name in objectives_to_test[:3]:  # Test on subset
        for temp in temperatures:
            print(f"\n{objective_name.upper()} - Boltzmann (T={temp})")
            print("-" * 80)

            try:
                generator_params = {
                    'objective_name': objective_name,
                    'n_initial': n_initial,
                    'n_selected': n_selected,
                    'n_vars': n_vars,
                    'temperature': temp,
                    'seed': seed,
                    'use_shift': True
                }

                results = evaluate_dendiff_on_enhanced_distribution(
                    generate_boltzmann_distribution,
                    generator_params,
                    n_samples_test=n_samples_test,
                    dendiff_params=dendiff_params,
                    seed=seed,
                    verbose=True
                )

                all_results.append((f'Boltzmann (T={temp})', objective_name, results))

                print(f"\n  Results:")
                print(f"    JS Divergence:           {results['js_divergence']:.6f}")
                print(f"    Fitness Mean Diff:       {results['fitness_mean_diff']:.6f}")
                print(f"    Fitness Std Diff:        {results['fitness_std_diff']:.6f}")
                print(f"    Fitness Std Ratio:       {results['fitness_std_ratio']:.4f}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    # ========================================================================
    # Part 3: Rank-Based Selection
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 3: RANK-BASED SELECTION (CMA-ES STYLE)")
    print("=" * 80)
    print()
    print("Testing with different selection pressure parameters:")
    print()

    selection_pressures = [1.5, 2.0, 2.5]

    for objective_name in objectives_to_test[:3]:  # Test on subset
        for sp in selection_pressures:
            print(f"\n{objective_name.upper()} - Rank-based (SP={sp})")
            print("-" * 80)

            try:
                generator_params = {
                    'objective_name': objective_name,
                    'n_initial': n_initial,
                    'n_selected': n_selected,
                    'n_vars': n_vars,
                    'selection_pressure': sp,
                    'seed': seed,
                    'use_shift': True
                }

                results = evaluate_dendiff_on_enhanced_distribution(
                    generate_rank_based_distribution,
                    generator_params,
                    n_samples_test=n_samples_test,
                    dendiff_params=dendiff_params,
                    seed=seed,
                    verbose=True
                )

                all_results.append((f'Rank-based (SP={sp})', objective_name, results))

                print(f"\n  Results:")
                print(f"    JS Divergence:           {results['js_divergence']:.6f}")
                print(f"    Fitness Mean Diff:       {results['fitness_mean_diff']:.6f}")
                print(f"    Fitness Std Diff:        {results['fitness_std_diff']:.6f}")
                print(f"    Fitness Std Ratio:       {results['fitness_std_ratio']:.4f}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    # ========================================================================
    # Summary Tables
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print()

    # Group results by method
    methods = {}
    for method, obj_name, results in all_results:
        if method not in methods:
            methods[method] = []
        methods[method].append((obj_name, results))

    # Print summary for each method
    for method in sorted(methods.keys()):
        print(f"\n{method}")
        print("-" * 80)
        print(f"{'Objective':<15} {'JS Div':<12} {'Mean Diff':<12} {'Std Diff':<12} {'Std Ratio':<12}")
        print("-" * 80)

        for obj_name, results in methods[method]:
            print(f"{obj_name:<15} {results['js_divergence']:>10.4f}  "
                  f"{results['fitness_mean_diff']:>10.4f}  "
                  f"{results['fitness_std_diff']:>10.4f}  "
                  f"{results['fitness_std_ratio']:>10.4f}")

    # ========================================================================
    # Interpretation Guide
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Enhanced Metrics Explanation:

1. JS Divergence (Jensen-Shannon):
   - Measures distributional similarity
   - Range: [0, 1], lower is better
   - < 0.3: Good approximation
   - < 0.5: Acceptable approximation

2. Fitness Mean Difference:
   - Absolute difference in mean fitness between sampled and reference
   - Lower is better
   - Should be small relative to fitness scale

3. Fitness Std Difference:
   - Absolute difference in standard deviation
   - Measures if dendiff captures fitness variance
   - Lower is better for pure approximation

4. Fitness Std Ratio:
   - Ratio of sampled std to original std
   - > 1.0: Sampled distribution has MORE variance (better exploration)
   - < 1.0: Sampled distribution has LESS variance (more exploitation)
   - Ideally ≥ 1.0 for EDA applications to promote exploration

Method Comparisons:

Standard Selection (Shifted):
  - Selects top solutions by fitness
  - Shifted optimum avoids origin bias
  - Good baseline for comparison

Boltzmann Distribution:
  - Temperature controls exploration vs exploitation
  - Lower T: More exploitation (favor best solutions)
  - Higher T: More exploration (more uniform selection)
  - Good for maintaining diversity

Rank-Based Selection:
  - Robust to fitness scaling
  - Selection pressure controls selection intensity
  - Similar to CMA-ES recombination weights
  - Good for multimodal problems

Expected Observations:

1. Shifted optima should show different solution distributions
2. Boltzmann with high T should have higher std ratio
3. Rank-based should be robust across different fitness landscapes
4. All methods should capture the general fitness distribution structure
""")

    return all_results


def compare_selection_methods():
    """Direct comparison of the three selection methods on same objectives."""

    print("\n\n" + "=" * 80)
    print("DIRECT METHOD COMPARISON")
    print("=" * 80)
    print()

    n_vars = 10
    n_initial = 1000
    n_selected = 500
    seed = 42
    dendiff_params = {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}

    objectives = ['sphere', 'rastrigin', 'ackley']

    for objective_name in objectives:
        print(f"\n{objective_name.upper()} Function - Method Comparison")
        print("=" * 80)
        print(f"{'Method':<30} {'JS Div':<12} {'Mean Diff':<12} {'Std Ratio':<12} {'Best Fit':<12}")
        print("-" * 80)

        methods_to_compare = [
            ('Standard (Shifted)', generate_empirical_fitness_distribution, {}),
            ('Boltzmann (T=1.0)', generate_boltzmann_distribution, {'temperature': 1.0}),
            ('Rank-based (SP=2.0)', generate_rank_based_distribution, {'selection_pressure': 2.0}),
        ]

        for method_name, generator_fn, extra_params in methods_to_compare:
            generator_params = {
                'objective_name': objective_name,
                'n_initial': n_initial,
                'n_selected': n_selected,
                'n_vars': n_vars,
                'seed': seed,
                'use_shift': True,
                **extra_params
            }

            results = evaluate_dendiff_on_enhanced_distribution(
                generator_fn,
                generator_params,
                n_samples_test=n_selected,
                dendiff_params=dendiff_params,
                seed=seed,
                verbose=False
            )

            print(f"{method_name:<30} {results['js_divergence']:>10.4f}  "
                  f"{results['fitness_mean_diff']:>10.4f}  "
                  f"{results['fitness_std_ratio']:>10.4f}  "
                  f"{results['sampled_best_fitness']:>10.4f}")

    print("\n" + "=" * 80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import time

    start_time = time.time()

    print("=" * 80)
    print("ENHANCED DENDIFF BENCHMARK SUITE")
    print("=" * 80)
    print()
    print("This suite evaluates dendiff on:")
    print("  1. Objective functions with shifted optima")
    print("  2. Boltzmann distribution selection")
    print("  3. Rank-based selection (CMA-ES style)")
    print("  4. Enhanced metrics including std comparison")
    print()

    # Run main benchmark
    results = run_enhanced_benchmark()

    # Run direct comparison
    compare_selection_methods()

    elapsed_time = time.time() - start_time

    print("\n\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nTotal benchmark time: {elapsed_time:.2f} seconds")
    print(f"Total configurations tested: {len(results)}")
    print("\nAll benchmarks completed successfully!")
    print("=" * 80)
