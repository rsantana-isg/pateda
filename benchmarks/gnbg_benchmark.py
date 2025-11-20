"""
GNBG Benchmark for Continuous EDAs

This script evaluates continuous EDAs on the GNBG (Generalized Numerical
Benchmark Generator) test suite, comprising 24 problem instances with varying
characteristics (multimodality, conditioning, variable interactions, etc.).

The GNBG benchmark provides a comprehensive evaluation of continuous optimization
algorithms across diverse problem landscapes.

References:
- D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A
  Generalized and Configurable Benchmark Generator for Continuous Numerical
  Optimization," arXiv preprint arXiv:2312.07083, 2023.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.io import loadmat

# Add enhanced_edas to path for GNBG class
sys.path.insert(0, str(Path(__file__).parent.parent / 'enhanced_edas'))
from GNBG_class import GNBG

from pateda import EDA, EDAComponents
from pateda.learning import (
    LearnGaussianUnivariate,
    LearnGaussianFull,
)
from pateda.learning.mixture_gaussian import LearnMixtureGaussian
from pateda.learning.gmrf_eda import LearnGMRFEDA
from pateda.sampling import (
    SampleGaussianUnivariate,
    SampleGaussianFull,
)
from pateda.sampling.mixture_gaussian import SampleMixtureGaussian
from pateda.sampling.gmrf_eda import SampleGMRFEDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


def load_gnbg_instance(problem_index: int, instances_folder: str) -> Tuple[GNBG, Dict[str, Any]]:
    """
    Load a GNBG problem instance from .mat file.

    Args:
        problem_index: Problem instance number (1-24)
        instances_folder: Path to folder containing GNBG .mat files

    Returns:
        Tuple of (GNBG instance, problem info dict)
    """
    if not (1 <= problem_index <= 24):
        raise ValueError(f'ProblemIndex must be between 1 and 24, got {problem_index}')

    # Load .mat file
    filename = f'f{problem_index}.mat'
    filepath = os.path.join(instances_folder, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GNBG instance file not found: {filepath}")

    GNBG_tmp = loadmat(filepath)['GNBG']

    # Extract parameters
    MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
    AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
    Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
    CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]
    MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
    MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
    CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
    CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
    CompH = np.array(GNBG_tmp['Component_H'][0, 0])
    Mu = np.array(GNBG_tmp['Mu'][0, 0])
    Omega = np.array(GNBG_tmp['Omega'][0, 0])
    Lambda = np.array(GNBG_tmp['lambda'][0, 0])
    RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
    OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
    OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])

    # Create GNBG instance
    gnbg = GNBG(
        MaxEvals, AcceptanceThreshold, Dimension, CompNum,
        MinCoordinate, MaxCoordinate, CompMinPos, CompSigma,
        CompH, Mu, Omega, Lambda, RotationMatrix,
        OptimumValue, OptimumPosition
    )

    # Problem info
    problem_info = {
        'problem_index': problem_index,
        'dimension': int(Dimension),
        'max_evals': int(MaxEvals),
        'acceptance_threshold': float(AcceptanceThreshold),
        'optimum_value': float(OptimumValue),
        'num_components': int(CompNum),
        'bounds': (float(MinCoordinate), float(MaxCoordinate))
    }

    return gnbg, problem_info


def create_gnbg_fitness_wrapper(gnbg: GNBG):
    """
    Create a fitness function wrapper for GNBG that works with pateda.

    Args:
        gnbg: GNBG instance

    Returns:
        Fitness function compatible with pateda
    """
    def fitness_function(x: np.ndarray) -> np.ndarray:
        """
        Fitness function wrapper for GNBG.

        Args:
            x: Solution(s) to evaluate (1D or 2D array)

        Returns:
            Fitness value(s) (minimization)
        """
        # GNBG expects 2D input
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Evaluate fitness
        fitness = gnbg.fitness(x)

        # Return scalar for single solution, array for multiple
        if len(fitness) == 1:
            return fitness[0]
        return fitness

    return fitness_function


def get_eda_configuration(
    eda_name: str,
    n_vars: int,
    pop_size: int,
    max_gen: int,
    selection_ratio: float = 0.5
) -> EDAComponents:
    """
    Get EDA component configuration for a specific algorithm.

    Args:
        eda_name: Name of EDA ('gaussian_umda', 'gaussian_full', 'gaussian_mixture', 'gmrf_eda')
        n_vars: Number of variables
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Selection ratio for truncation selection

    Returns:
        EDAComponents configuration
    """
    if eda_name == 'gaussian_umda':
        # Gaussian UMDA: Univariate Gaussian (independence)
        components = EDAComponents(
            learning=LearnGaussianUnivariate(alpha=1e-10),
            sampling=SampleGaussianUnivariate(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

    elif eda_name == 'gaussian_full':
        # Full Gaussian EDA: Full covariance matrix
        components = EDAComponents(
            learning=LearnGaussianFull(regularization=1e-6),
            sampling=SampleGaussianFull(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

    elif eda_name == 'gaussian_mixture':
        # Gaussian Mixture EDA: Mixture of Gaussians (multimodality)
        n_components = min(3, max(2, pop_size // 20))  # Adaptive number of components
        components = EDAComponents(
            learning=LearnMixtureGaussian(
                n_components=n_components,
                covariance_type='full',
                use_em=True
            ),
            sampling=SampleMixtureGaussian(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

    elif eda_name == 'gmrf_eda':
        # GMRF-EDA: Sparse dependencies via regularization
        components = EDAComponents(
            learning=LearnGMRFEDA(
                regularization='lasso',
                alpha=0.01,
                clustering_method='affinity'
            ),
            sampling=SampleGMRFEDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=selection_ratio),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

    else:
        raise ValueError(f"Unknown EDA name: {eda_name}")

    return components


def run_single_experiment(
    eda_name: str,
    problem_index: int,
    instances_folder: str,
    pop_size: int = 100,
    selection_ratio: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single EDA on a single GNBG problem instance.

    Args:
        eda_name: Name of EDA to run
        problem_index: GNBG problem index (1-24)
        instances_folder: Path to GNBG instances folder
        pop_size: Population size
        selection_ratio: Selection ratio
        seed: Random seed (optional)
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    if seed is not None:
        np.random.seed(seed)

    # Load GNBG instance
    gnbg, problem_info = load_gnbg_instance(problem_index, instances_folder)

    # Create fitness function wrapper
    fitness_func = create_gnbg_fitness_wrapper(gnbg)

    # Problem parameters
    n_vars = problem_info['dimension']
    max_evals = problem_info['max_evals']
    max_gen = max_evals // pop_size
    bounds = problem_info['bounds']

    # Cardinality for continuous variables (bounds)
    cardinality = np.array([[bounds[0]] * n_vars, [bounds[1]] * n_vars])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {eda_name} on GNBG f{problem_index}")
        print(f"{'='*80}")
        print(f"Dimension: {n_vars}")
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_gen}")
        print(f"Max evaluations: {max_evals}")
        print(f"Optimum value: {problem_info['optimum_value']:.6e}")
        print(f"Acceptance threshold: {problem_info['acceptance_threshold']:.6e}")

    # Get EDA configuration
    components = get_eda_configuration(
        eda_name=eda_name,
        n_vars=n_vars,
        pop_size=pop_size,
        max_gen=max_gen,
        selection_ratio=selection_ratio
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
    )

    # Measure runtime
    start_time = time.time()
    statistics, cache = eda.run(verbose=verbose)
    runtime = time.time() - start_time

    # Extract results from GNBG tracking
    best_fitness_gnbg = gnbg.BestFoundResult
    acceptance_reach_point = gnbg.AcceptanceReachPoint
    fe_history = gnbg.FEhistory

    # Calculate error from optimum
    error_from_optimum = abs(best_fitness_gnbg - problem_info['optimum_value'])
    success = error_from_optimum < problem_info['acceptance_threshold']

    # Results dictionary
    results = {
        'eda_name': eda_name,
        'problem_index': problem_index,
        'dimension': n_vars,
        'seed': seed,
        'pop_size': pop_size,
        'selection_ratio': selection_ratio,

        # Problem info
        'optimum_value': problem_info['optimum_value'],
        'acceptance_threshold': problem_info['acceptance_threshold'],
        'max_evals': max_evals,

        # Performance metrics
        'best_fitness': float(best_fitness_gnbg),
        'error_from_optimum': float(error_from_optimum),
        'success': bool(success),
        'acceptance_reach_point': int(acceptance_reach_point) if not np.isinf(acceptance_reach_point) else None,
        'function_evaluations': int(gnbg.FE),
        'runtime_seconds': float(runtime),

        # EDA statistics
        'generations_run': len(statistics.best_fitness),
        'best_fitness_eda': float(statistics.best_fitness_overall),
        'generation_found': int(statistics.generation_found),

        # Convergence history (compressed)
        'convergence_history': [float(f) for f in fe_history[::max(1, len(fe_history)//100)]],  # Sample up to 100 points
    }

    if verbose:
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"Best fitness: {best_fitness_gnbg:.6e}")
        print(f"Error from optimum: {error_from_optimum:.6e}")
        print(f"Success: {'✓' if success else '✗'}")
        if not np.isinf(acceptance_reach_point):
            print(f"Acceptance reached at FE: {acceptance_reach_point}")
        print(f"Total function evaluations: {gnbg.FE}")
        print(f"Runtime: {runtime:.2f} seconds")

    return results


def run_benchmark(
    eda_names: List[str],
    problem_indices: List[int],
    instances_folder: str,
    n_runs: int = 5,
    pop_size: int = 100,
    selection_ratio: float = 0.5,
    output_folder: str = 'gnbg_results',
    verbose: bool = True,
    save_individual_runs: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive GNBG benchmark across multiple EDAs and problems.

    Args:
        eda_names: List of EDA names to evaluate
        problem_indices: List of GNBG problem indices to test
        instances_folder: Path to GNBG instances folder
        n_runs: Number of independent runs per (EDA, problem) pair
        pop_size: Population size
        selection_ratio: Selection ratio
        output_folder: Folder to save results
        verbose: Print progress
        save_individual_runs: Save individual run data

    Returns:
        DataFrame with all results
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    all_results = []
    total_experiments = len(eda_names) * len(problem_indices) * n_runs
    experiment_count = 0

    print(f"\n{'='*80}")
    print("GNBG Benchmark for Continuous EDAs")
    print(f"{'='*80}")
    print(f"EDAs: {', '.join(eda_names)}")
    print(f"Problems: f{min(problem_indices)}-f{max(problem_indices)} ({len(problem_indices)} instances)")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*80}\n")

    for eda_name in eda_names:
        for problem_index in problem_indices:
            for run in range(n_runs):
                experiment_count += 1

                if verbose:
                    print(f"\n[{experiment_count}/{total_experiments}] "
                          f"{eda_name} on f{problem_index}, run {run+1}/{n_runs}")

                # Run experiment with unique seed
                seed = problem_index * 1000 + run

                try:
                    results = run_single_experiment(
                        eda_name=eda_name,
                        problem_index=problem_index,
                        instances_folder=instances_folder,
                        pop_size=pop_size,
                        selection_ratio=selection_ratio,
                        seed=seed,
                        verbose=False
                    )
                    results['run'] = run
                    all_results.append(results)

                    if verbose:
                        status = "✓" if results['success'] else "✗"
                        print(f"  {status} Error: {results['error_from_optimum']:.6e}, "
                              f"FE: {results['function_evaluations']}, "
                              f"Time: {results['runtime_seconds']:.2f}s")

                except Exception as e:
                    print(f"  ✗ ERROR: {str(e)}")
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save full results as CSV
    csv_path = os.path.join(output_folder, f'gnbg_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save as pickle (includes convergence history)
    pickle_path = os.path.join(output_folder, f'gnbg_results_{timestamp}.pkl')
    df.to_pickle(pickle_path)
    print(f"Full results (with convergence) saved to: {pickle_path}")

    # Generate summary statistics
    summary = generate_summary_statistics(df)
    summary_path = os.path.join(output_folder, f'gnbg_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary
    print_summary(summary)

    return df


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics from benchmark results.

    Args:
        df: DataFrame with individual run results

    Returns:
        DataFrame with summary statistics
    """
    summary_rows = []

    for eda_name in df['eda_name'].unique():
        for problem_index in df['problem_index'].unique():
            subset = df[(df['eda_name'] == eda_name) & (df['problem_index'] == problem_index)]

            if len(subset) == 0:
                continue

            summary_rows.append({
                'eda_name': eda_name,
                'problem_index': problem_index,
                'n_runs': len(subset),
                'success_rate': subset['success'].mean(),
                'mean_error': subset['error_from_optimum'].mean(),
                'std_error': subset['error_from_optimum'].std(),
                'median_error': subset['error_from_optimum'].median(),
                'min_error': subset['error_from_optimum'].min(),
                'max_error': subset['error_from_optimum'].max(),
                'mean_fe': subset['function_evaluations'].mean(),
                'std_fe': subset['function_evaluations'].std(),
                'mean_runtime': subset['runtime_seconds'].mean(),
                'std_runtime': subset['runtime_seconds'].std(),
            })

    return pd.DataFrame(summary_rows)


def print_summary(summary: pd.DataFrame):
    """Print summary statistics in a readable format."""
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")

    for eda_name in summary['eda_name'].unique():
        eda_summary = summary[summary['eda_name'] == eda_name]

        print(f"\n{eda_name}:")
        print(f"  Overall success rate: {eda_summary['success_rate'].mean()*100:.1f}%")
        print(f"  Average error: {eda_summary['mean_error'].mean():.6e} ± {eda_summary['std_error'].mean():.6e}")
        print(f"  Average FE: {eda_summary['mean_fe'].mean():.0f} ± {eda_summary['std_fe'].mean():.0f}")
        print(f"  Average runtime: {eda_summary['mean_runtime'].mean():.2f}s ± {eda_summary['std_runtime'].mean():.2f}s")


if __name__ == '__main__':
    # Default configuration
    INSTANCES_FOLDER = str(Path(__file__).parent.parent / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')

    # EDAs to benchmark
    EDA_NAMES = [
        'gaussian_umda',      # Univariate Gaussian (independence)
        'gaussian_full',      # Full covariance matrix
        'gaussian_mixture',   # Mixture of Gaussians (multimodality)
        'gmrf_eda',          # Sparse dependencies (regularization)
    ]

    # GNBG problems to test (can test all 24 or a subset)
    PROBLEM_INDICES = list(range(1, 25))  # All 24 problems
    # PROBLEM_INDICES = [1, 2, 3, 7, 8, 9]  # Quick test: subset of problems

    # Benchmark configuration
    N_RUNS = 5            # Independent runs per configuration
    POP_SIZE = 100        # Population size
    SELECTION_RATIO = 0.5 # Truncation selection ratio

    # Run benchmark
    results_df = run_benchmark(
        eda_names=EDA_NAMES,
        problem_indices=PROBLEM_INDICES,
        instances_folder=INSTANCES_FOLDER,
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        selection_ratio=SELECTION_RATIO,
        output_folder='gnbg_results',
        verbose=True
    )

    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}\n")
