#!/usr/bin/env python3
"""
Parametric Script for Running Gaussian-Based EDAs on GNBG Benchmark
====================================================================

This script runs a specified Gaussian-based EDA (including mixtures) on a GNBG
benchmark function and saves the results to a CSV file. It is designed for
batch execution on computer clusters.

Usage:
    python run_gaussian_eda_gnbg.py --eda_name <eda> --function_index <1-24> --seed <seed>

Example:
    python run_gaussian_eda_gnbg.py --eda_name gaussian_univariate --function_index 1 --seed 42
    python run_gaussian_eda_gnbg.py --eda_name mixture_gaussian_em --function_index 15 --seed 123

Supported EDAs:
    Basic Gaussian:
        - gaussian_univariate: Univariate Gaussian (UMDA-c)
        - gaussian_full: Full covariance Gaussian (EMNA)

    Mixture Gaussian:
        - mixture_gaussian_univariate: Mixture with independent components
        - mixture_gaussian_full: Mixture with full covariance (k-means init)
        - mixture_gaussian_em: GMM with EM algorithm (sklearn)

    GMRF-EDA (Gaussian Markov Random Field):
        - gmrf_eda: Basic GMRF with affinity propagation clustering
        - gmrf_eda_lasso: GMRF with LASSO (L1) regularization
        - gmrf_eda_elasticnet: GMRF with Elastic Net (L1+L2)
        - gmrf_eda_lars: GMRF with LARS

    Vine Copula:
        - vine_copula_cvine: Canonical vine copula
        - vine_copula_dvine: Drawable vine copula
        - vine_copula_auto: Automatic vine structure selection

Output:
    - CSV file with run statistics saved to results/ folder
    - Filename format: {eda_name}_f{function_index}_seed{seed}.csv
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'enhanced_edas'))
sys.path.insert(0, str(PROJECT_ROOT / 'pateda'))

# Import GNBG class
from GNBG_class import GNBG
from scipy.io import loadmat

# Basic Gaussian learning functions
from pateda.learning.basic_gaussian import learn_gaussian_univariate, learn_gaussian_full

# Mixture Gaussian learning functions
from pateda.learning.mixture_gaussian import (
    learn_mixture_gaussian_univariate,
    learn_mixture_gaussian_full,
    learn_mixture_gaussian_em
)

# GMRF-EDA learning functions
from pateda.learning.gmrf_eda import (
    learn_gmrf_eda,
    learn_gmrf_eda_lasso,
    learn_gmrf_eda_elasticnet,
    learn_gmrf_eda_lars
)

# Vine Copula learning functions
from pateda.learning.vine_copula import (
    learn_vine_copula_cvine,
    learn_vine_copula_dvine,
    learn_vine_copula_auto
)

# Basic Gaussian sampling functions
from pateda.sampling.basic_gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full
)

# Mixture Gaussian sampling functions
from pateda.sampling.mixture_gaussian import (
    sample_mixture_gaussian_univariate,
    sample_mixture_gaussian_full,
    sample_mixture_gaussian_em
)

# GMRF-EDA sampling function
from pateda.sampling.gmrf_eda import sample_gmrf_eda

# Vine Copula sampling function
from pateda.sampling.vine_copula import sample_vine_copula

# Default paths
DEFAULT_INSTANCES_FOLDER = str(PROJECT_ROOT / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')
DEFAULT_OUTPUT_FOLDER = str(PROJECT_ROOT / 'benchmarks' / 'results')

# Supported EDAs for continuous optimization (GNBG)
# Note: GNBG is a MINIMIZATION problem (lower fitness is better)
SUPPORTED_EDAS = [
    # Basic Gaussian
    'gaussian_univariate', 'gaussian_full',
    # Mixture Gaussian
    'mixture_gaussian_univariate', 'mixture_gaussian_full', 'mixture_gaussian_em',
    # GMRF-EDA variants
    'gmrf_eda', 'gmrf_eda_lasso', 'gmrf_eda_elasticnet', 'gmrf_eda_lars',
    # Vine Copula variants
    'vine_copula_cvine', 'vine_copula_dvine', 'vine_copula_auto'
]


def load_gnbg_instance(problem_index: int, instances_folder: str):
    """
    Load a GNBG problem instance from .mat file.

    Parameters
    ----------
    problem_index : int
        Problem instance number (1-24)
    instances_folder : str
        Path to folder containing GNBG .mat files

    Returns
    -------
    gnbg : GNBG
        GNBG problem instance
    problem_info : dict
        Dictionary with problem information
    """
    if not (1 <= problem_index <= 24):
        raise ValueError(f'function_index must be between 1 and 24, got {problem_index}')

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

    gnbg = GNBG(
        MaxEvals, AcceptanceThreshold, Dimension, CompNum,
        MinCoordinate, MaxCoordinate, CompMinPos, CompSigma,
        CompH, Mu, Omega, Lambda, RotationMatrix,
        OptimumValue, OptimumPosition
    )

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


def create_fitness_wrapper(gnbg: GNBG) -> Callable:
    """
    Create a fitness function wrapper for GNBG.

    Parameters
    ----------
    gnbg : GNBG
        GNBG problem instance

    Returns
    -------
    fitness_function : callable
        Function that evaluates solutions
    """
    def fitness_function(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        fitness = gnbg.fitness(x)
        if len(fitness) == 1:
            return fitness[0]
        return fitness

    return fitness_function


def get_default_params(eda_name: str, n_vars: int, pop_size: int) -> Dict[str, Any]:
    """
    Get default parameters for the specified Gaussian EDA.

    Parameters
    ----------
    eda_name : str
        Name of the EDA
    n_vars : int
        Number of variables
    pop_size : int
        Population size

    Returns
    -------
    params : dict
        Default parameters for the EDA
    """
    # Default number of components for mixture models
    n_components = max(2, min(5, pop_size // 20))

    defaults = {
        # Basic Gaussian
        'gaussian_univariate': {
            'min_std': 1e-6  # Minimum standard deviation to prevent collapse
        },
        'gaussian_full': {
            'regularization': 1e-6  # Regularization for covariance matrix
        },
        # Mixture Gaussian
        'mixture_gaussian_univariate': {
            'n_components': n_components,
            'min_std': 1e-6
        },
        'mixture_gaussian_full': {
            'n_components': n_components,
            'regularization': 1e-6
        },
        'mixture_gaussian_em': {
            'n_components': n_components,
            'max_iter': 100,
            'tol': 1e-3,
            'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical'
            'reg_covar': 1e-6
        },
        # GMRF-EDA variants
        'gmrf_eda': {
            'regularization': 'lasso',
            'alpha': 0.1,
            'min_cluster_size': 2
        },
        'gmrf_eda_lasso': {
            'alpha': 0.1,
            'min_cluster_size': 2
        },
        'gmrf_eda_elasticnet': {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'min_cluster_size': 2
        },
        'gmrf_eda_lars': {
            'n_nonzero_coefs': max(1, n_vars // 4),
            'min_cluster_size': 2
        },
        # Vine Copula variants
        'vine_copula_cvine': {
            'copula_family': 'gaussian'  # 'gaussian', 'student', 'clayton', 'gumbel', 'frank'
        },
        'vine_copula_dvine': {
            'copula_family': 'gaussian'
        },
        'vine_copula_auto': {
            'criterion': 'aic'  # 'aic', 'bic'
        }
    }

    return defaults.get(eda_name, {})


def learn_model(eda_name: str, selected_population: np.ndarray,
                selected_fitness: np.ndarray, params: Dict[str, Any]):
    """
    Learn a model using the specified Gaussian EDA.

    Parameters
    ----------
    eda_name : str
        Name of the EDA
    selected_population : np.ndarray
        Selected population
    selected_fitness : np.ndarray
        Fitness of selected population
    params : dict
        EDA parameters

    Returns
    -------
    model : dict
        Learned model
    """
    # Basic Gaussian
    if eda_name == 'gaussian_univariate':
        return learn_gaussian_univariate(selected_population, selected_fitness, params=params)

    elif eda_name == 'gaussian_full':
        return learn_gaussian_full(selected_population, selected_fitness, params=params)

    # Mixture Gaussian
    elif eda_name == 'mixture_gaussian_univariate':
        return learn_mixture_gaussian_univariate(selected_population, selected_fitness, params=params)

    elif eda_name == 'mixture_gaussian_full':
        return learn_mixture_gaussian_full(selected_population, selected_fitness, params=params)

    elif eda_name == 'mixture_gaussian_em':
        return learn_mixture_gaussian_em(selected_population, selected_fitness, params=params)

    # GMRF-EDA variants
    elif eda_name == 'gmrf_eda':
        return learn_gmrf_eda(selected_population, selected_fitness, params=params)

    elif eda_name == 'gmrf_eda_lasso':
        return learn_gmrf_eda_lasso(selected_population, selected_fitness, params=params)

    elif eda_name == 'gmrf_eda_elasticnet':
        return learn_gmrf_eda_elasticnet(selected_population, selected_fitness, params=params)

    elif eda_name == 'gmrf_eda_lars':
        return learn_gmrf_eda_lars(selected_population, selected_fitness, params=params)

    # Vine Copula variants
    elif eda_name == 'vine_copula_cvine':
        return learn_vine_copula_cvine(selected_population, selected_fitness, params=params)

    elif eda_name == 'vine_copula_dvine':
        return learn_vine_copula_dvine(selected_population, selected_fitness, params=params)

    elif eda_name == 'vine_copula_auto':
        return learn_vine_copula_auto(selected_population, selected_fitness, params=params)

    else:
        raise ValueError(f"Unknown EDA: {eda_name}")


def sample_model(eda_name: str, model, n_samples: int, bounds: np.ndarray,
                 params: Dict[str, Any] = None) -> np.ndarray:
    """
    Sample from a model using the specified Gaussian EDA.

    Parameters
    ----------
    eda_name : str
        Name of the EDA
    model : dict
        Learned model
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray
        Variable bounds
    params : dict, optional
        Sampling parameters

    Returns
    -------
    samples : np.ndarray
        Generated samples
    """
    if params is None:
        params = {}

    # Basic Gaussian
    if eda_name == 'gaussian_univariate':
        return sample_gaussian_univariate(model, n_samples=n_samples, bounds=bounds, params=params)

    elif eda_name == 'gaussian_full':
        return sample_gaussian_full(model, n_samples=n_samples, bounds=bounds, params=params)

    # Mixture Gaussian
    elif eda_name == 'mixture_gaussian_univariate':
        return sample_mixture_gaussian_univariate(model, n_samples=n_samples, bounds=bounds, params=params)

    elif eda_name == 'mixture_gaussian_full':
        return sample_mixture_gaussian_full(model, n_samples=n_samples, bounds=bounds, params=params)

    elif eda_name == 'mixture_gaussian_em':
        return sample_mixture_gaussian_em(model, n_samples=n_samples, bounds=bounds, params=params)

    # GMRF-EDA variants (all use same sampling function)
    elif eda_name in ['gmrf_eda', 'gmrf_eda_lasso', 'gmrf_eda_elasticnet', 'gmrf_eda_lars']:
        return sample_gmrf_eda(model, n_samples=n_samples, bounds=bounds, params=params)

    # Vine Copula variants (all use same sampling function)
    elif eda_name in ['vine_copula_cvine', 'vine_copula_dvine', 'vine_copula_auto']:
        return sample_vine_copula(model, n_samples=n_samples, bounds=bounds, params=params)

    else:
        raise ValueError(f"Unknown EDA: {eda_name}")


def run_gaussian_eda(
    eda_name: str,
    function_index: int,
    seed: int,
    instances_folder: str = DEFAULT_INSTANCES_FOLDER,
    pop_size: int = 100,
    selection_ratio: float = 0.3,
    eda_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a Gaussian-based EDA on a GNBG benchmark function.

    Parameters
    ----------
    eda_name : str
        Name of the EDA to run
    function_index : int
        GNBG function index (1-24)
    seed : int
        Random seed
    instances_folder : str
        Path to GNBG instance files
    pop_size : int
        Population size
    selection_ratio : float
        Ratio of population to select
    eda_params : dict, optional
        Parameters for the EDA
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary with all run statistics
    """
    # Validate inputs
    if eda_name not in SUPPORTED_EDAS:
        raise ValueError(f"Unknown EDA: {eda_name}. Supported: {SUPPORTED_EDAS}")

    if not (1 <= function_index <= 24):
        raise ValueError(f"function_index must be between 1 and 24, got {function_index}")

    # Set random seeds
    np.random.seed(seed)

    # Load GNBG problem
    gnbg, problem_info = load_gnbg_instance(function_index, instances_folder)
    fitness_func = create_fitness_wrapper(gnbg)

    # Problem parameters
    n_vars = problem_info['dimension']
    max_evals = problem_info['max_evals']
    max_gen = max_evals // pop_size
    bounds_tuple = problem_info['bounds']
    bounds = np.array([[bounds_tuple[0]] * n_vars, [bounds_tuple[1]] * n_vars])

    # Selection size
    selection_size = int(pop_size * selection_ratio)

    # Get default parameters if not provided
    if eda_params is None:
        eda_params = get_default_params(eda_name, n_vars, pop_size)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {eda_name.upper()}-EDA on GNBG f{function_index}")
        print(f"{'='*80}")
        print(f"Dimension: {n_vars}")
        print(f"Population size: {pop_size}")
        print(f"Selection size: {selection_size}")
        print(f"Max generations: {max_gen}")
        print(f"Max evaluations: {max_evals}")
        print(f"Seed: {seed}")
        print(f"Optimum value: {problem_info['optimum_value']:.6e}")
        print(f"Acceptance threshold: {problem_info['acceptance_threshold']:.6e}")
        print(f"{'='*80}\n")

    # Compute precision for error display based on acceptance threshold
    acceptance_threshold = problem_info['acceptance_threshold']
    optimum_value = problem_info['optimum_value']
    if acceptance_threshold > 0:
        error_precision = max(1, int(-np.floor(np.log10(acceptance_threshold))) + 1)
    else:
        error_precision = 10

    # Initialize population
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    # Tracking variables
    best_fitness_overall = np.inf
    best_solution_overall = None
    generation_found = 0
    fitness_history = []
    mean_fitness_history = []
    learning_times = []
    sampling_times = []
    n_evaluations = 0

    # Start timing
    start_time = time.time()

    # Main EDA loop
    for gen in range(max_gen):
        # Evaluate population
        fitness = fitness_func(population)
        n_evaluations += len(population)

        # Track statistics
        gen_best_idx = np.argmin(fitness)
        gen_best_fitness = fitness[gen_best_idx]
        gen_mean_fitness = np.mean(fitness)

        fitness_history.append(float(gen_best_fitness))
        mean_fitness_history.append(float(gen_mean_fitness))

        # Update global best
        if gen_best_fitness < best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_solution_overall = population[gen_best_idx].copy()
            generation_found = gen

        if verbose and (gen % 10 == 0 or gen == max_gen - 1):
            current_error = abs(best_fitness_overall - optimum_value)
            error_str = f"{current_error:.{error_precision}e}"
            threshold_status = "< threshold" if current_error < acceptance_threshold else ">= threshold"
            print(f"Gen {gen+1:4d}: Best = {gen_best_fitness:.6e}, "
                  f"Error = {error_str} ({threshold_status})")

        # Check if max evaluations reached
        if n_evaluations >= max_evals:
            break

        # Select best individuals
        idx = np.argsort(fitness)[:selection_size]
        selected_population = population[idx]
        selected_fitness = fitness[idx]

        # Learn model
        learn_start = time.time()
        try:
            model = learn_model(eda_name, selected_population, selected_fitness, eda_params)
        except Exception as e:
            if verbose:
                print(f"  Warning: Learning failed at gen {gen+1}: {e}")
            # Random restart on failure
            population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))
            continue

        learning_times.append(time.time() - learn_start)

        # Sample new population
        sample_start = time.time()
        try:
            new_population = sample_model(eda_name, model, pop_size, bounds, eda_params)
        except Exception as e:
            if verbose:
                print(f"  Warning: Sampling failed at gen {gen+1}: {e}")
            # Random restart on failure
            population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))
            continue

        sampling_times.append(time.time() - sample_start)

        # Elitism: preserve the best individual from current population
        elite_idx = np.argmin(fitness)  # GNBG is minimization
        elite_solution = population[elite_idx].copy()
        new_population[-1] = elite_solution

        # Update population
        population = new_population

    # Total runtime
    total_runtime = time.time() - start_time

    # Get final results from GNBG tracking
    best_fitness_gnbg = gnbg.BestFoundResult
    acceptance_reach_point = gnbg.AcceptanceReachPoint

    # Calculate error from optimum
    error_from_optimum = abs(best_fitness_gnbg - problem_info['optimum_value'])
    success = error_from_optimum < problem_info['acceptance_threshold']

    # Compile results
    results = {
        # Run identification
        'eda_name': eda_name,
        'function_index': function_index,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),

        # Problem info
        'dimension': n_vars,
        'optimum_value': problem_info['optimum_value'],
        'acceptance_threshold': problem_info['acceptance_threshold'],
        'max_evals': max_evals,
        'num_components': problem_info['num_components'],

        # Configuration
        'pop_size': pop_size,
        'selection_ratio': selection_ratio,
        'max_generations': max_gen,

        # Performance metrics
        'best_fitness': float(best_fitness_gnbg),
        'best_fitness_eda': float(best_fitness_overall),
        'error_from_optimum': float(error_from_optimum),
        'success': bool(success),
        'acceptance_reach_point': int(acceptance_reach_point) if not np.isinf(acceptance_reach_point) else -1,
        'function_evaluations': int(n_evaluations),
        'generations_run': len(fitness_history),
        'generation_found': int(generation_found),

        # Timing
        'runtime_seconds': float(total_runtime),
        'avg_learning_time': float(np.mean(learning_times)) if learning_times else 0.0,
        'avg_sampling_time': float(np.mean(sampling_times)) if sampling_times else 0.0,

        # Convergence history
        'initial_fitness': fitness_history[0] if fitness_history else None,
        'final_fitness': fitness_history[-1] if fitness_history else None,
        'fitness_history': fitness_history,
        'mean_fitness_history': mean_fitness_history,
    }

    if verbose:
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"Best fitness (GNBG): {best_fitness_gnbg:.6e}")
        print(f"Best fitness (EDA):  {best_fitness_overall:.6e}")
        print(f"Optimum value:       {optimum_value:.6e}")
        error_str = f"{error_from_optimum:.{error_precision}e}"
        print(f"Error from optimum:  {error_str}")
        print(f"Acceptance threshold: {acceptance_threshold:.{error_precision}e}")
        print(f"Success: {'YES' if success else 'NO'} (error {'<' if success else '>='} threshold)")
        if not np.isinf(acceptance_reach_point):
            print(f"Acceptance reached at FE: {acceptance_reach_point}")
        print(f"Total function evaluations: {n_evaluations}")
        print(f"Generation found: {generation_found}")
        print(f"Runtime: {total_runtime:.2f} seconds")
        print(f"{'='*80}\n")

    return results


def save_results(results: Dict[str, Any], output_folder: str) -> str:
    """
    Save results to a CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary
    output_folder : str
        Output folder path

    Returns
    -------
    filepath : str
        Path to saved CSV file
    """
    os.makedirs(output_folder, exist_ok=True)

    # Create filename
    filename = f"{results['eda_name']}_f{results['function_index']}_seed{results['seed']}.csv"
    filepath = os.path.join(output_folder, filename)

    # Create a flat dictionary for CSV (exclude arrays)
    flat_results = {k: v for k, v in results.items()
                    if not isinstance(v, (list, np.ndarray))}

    # Add summary of fitness history
    if results.get('fitness_history'):
        flat_results['initial_fitness'] = results['fitness_history'][0]
        flat_results['final_fitness'] = results['fitness_history'][-1]
        flat_results['min_fitness'] = min(results['fitness_history'])

    # Save to CSV
    df = pd.DataFrame([flat_results])
    df.to_csv(filepath, index=False)

    # Also save full results with history as separate file
    history_filepath = filepath.replace('.csv', '_history.csv')
    history_df = pd.DataFrame({
        'generation': range(len(results.get('fitness_history', []))),
        'best_fitness': results.get('fitness_history', []),
        'mean_fitness': results.get('mean_fitness_history', [])
    })
    history_df.to_csv(history_filepath, index=False)

    return filepath


def main():
    """Main entry point for the parametric script."""
    parser = argparse.ArgumentParser(
        description='Run Gaussian-Based EDA on GNBG Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_gaussian_eda_gnbg.py --eda_name gaussian_univariate --function_index 1 --seed 42
    python run_gaussian_eda_gnbg.py --eda_name mixture_gaussian_em --function_index 15 --seed 123
    python run_gaussian_eda_gnbg.py --eda_name gmrf_eda_lasso -f 5 -s 999 --pop_size 200

Supported EDAs:
    Basic Gaussian:
        gaussian_univariate   - Univariate Gaussian (UMDA-c)
        gaussian_full         - Full covariance Gaussian (EMNA)

    Mixture Gaussian:
        mixture_gaussian_univariate - Mixture with independent components
        mixture_gaussian_full       - Mixture with full covariance
        mixture_gaussian_em         - GMM with EM algorithm

    GMRF-EDA:
        gmrf_eda              - Basic GMRF with affinity propagation
        gmrf_eda_lasso        - GMRF with LASSO regularization
        gmrf_eda_elasticnet   - GMRF with Elastic Net
        gmrf_eda_lars         - GMRF with LARS

    Vine Copula:
        vine_copula_cvine     - Canonical vine copula
        vine_copula_dvine     - Drawable vine copula
        vine_copula_auto      - Automatic vine selection
        """
    )

    parser.add_argument(
        '--eda_name', '-e', type=str, required=True,
        choices=SUPPORTED_EDAS,
        help='Name of the Gaussian-based EDA to run'
    )

    parser.add_argument(
        '--function_index', '-f', type=int, required=True,
        choices=range(1, 25), metavar='[1-24]',
        help='GNBG function index (1-24)'
    )

    parser.add_argument(
        '--seed', '-s', type=int, required=True,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--pop_size', type=int, default=100,
        help='Population size (default: 100)'
    )

    parser.add_argument(
        '--selection_ratio', type=float, default=0.3,
        help='Selection ratio (default: 0.3)'
    )

    parser.add_argument(
        '--instances_folder', type=str, default=DEFAULT_INSTANCES_FOLDER,
        help='Path to GNBG instance files'
    )

    parser.add_argument(
        '--output_folder', '-o', type=str, default=DEFAULT_OUTPUT_FOLDER,
        help='Output folder for results'
    )

    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run the EDA
    results = run_gaussian_eda(
        eda_name=args.eda_name,
        function_index=args.function_index,
        seed=args.seed,
        instances_folder=args.instances_folder,
        pop_size=args.pop_size,
        selection_ratio=args.selection_ratio,
        verbose=not args.quiet
    )

    # Save results
    output_path = save_results(results, args.output_folder)

    if not args.quiet:
        print(f"Results saved to: {output_path}")

    # Print summary line for cluster job logs
    threshold = results['acceptance_threshold']
    if threshold > 0:
        err_prec = max(1, int(-np.floor(np.log10(threshold))) + 1)
    else:
        err_prec = 10
    status = "SUCCESS" if results['success'] else "FAILURE"
    error_val = results['error_from_optimum']
    print(f"[{status}] {args.eda_name} f{args.function_index} seed={args.seed}: "
          f"error={error_val:.{err_prec}e} (threshold={threshold:.{err_prec}e}), "
          f"FE={results['function_evaluations']}, "
          f"time={results['runtime_seconds']:.2f}s")

    # Return exit code based on success
    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
