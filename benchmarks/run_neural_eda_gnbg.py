#!/usr/bin/env python3
"""
Parametric Script for Running Neural Network-Based EDAs on GNBG Benchmark
=========================================================================

This script runs a specified continuous neural network-based EDA on a GNBG
benchmark function and saves the results to a CSV file. It is designed for
batch execution on computer clusters.

Usage:
    python run_neural_eda_gnbg.py --eda_name <eda> --function_index <1-24> --seed <seed>

Example:
    python run_neural_eda_gnbg.py --eda_name vae --function_index 1 --seed 42
    python run_neural_eda_gnbg.py --eda_name gan --function_index 15 --seed 123

Supported EDAs (continuous optimization):
    - vae: Variational Autoencoder EDA
    - extended_vae: Extended VAE with fitness predictor
    - conditional_vae: Conditional Extended VAE (fitness-conditioned)
    - gan: Generative Adversarial Network EDA
    - backdrive: Backdrive Network Inversion EDA

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

# Neural network learning functions
from pateda.learning.vae import learn_vae, learn_extended_vae, learn_conditional_extended_vae
from pateda.learning.gan import learn_gan
from pateda.learning.backdrive import learn_backdrive

# Neural network sampling functions
from pateda.sampling.vae import sample_vae, sample_extended_vae, sample_conditional_extended_vae
from pateda.sampling.gan import sample_gan
from pateda.sampling.backdrive import sample_backdrive

# Default paths
DEFAULT_INSTANCES_FOLDER = str(PROJECT_ROOT / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')
DEFAULT_OUTPUT_FOLDER = str(PROJECT_ROOT / 'benchmarks' / 'results')

# Supported EDAs for continuous optimization (GNBG)
# Note: RBM with softmax is designed for discrete optimization and is not included here
SUPPORTED_EDAS = ['vae', 'extended_vae', 'conditional_vae', 'gan', 'backdrive']


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
    Get default parameters for the specified EDA.

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
    # Common parameters
    latent_dim = max(2, n_vars // 4)
    hidden_dims = [max(32, n_vars), max(16, n_vars // 2)]

    defaults = {
        'vae': {
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'epochs': 50,
            'batch_size': min(32, pop_size // 2),
            'learning_rate': 0.001
        },
        'extended_vae': {
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'epochs': 50,
            'batch_size': min(32, pop_size // 2),
            'learning_rate': 0.001
        },
        'conditional_vae': {
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'epochs': 50,
            'batch_size': min(32, pop_size // 2),
            'learning_rate': 0.001
        },
        'gan': {
            'latent_dim': latent_dim,
            'hidden_dims_g': hidden_dims,
            'hidden_dims_d': list(reversed(hidden_dims)),
            'epochs': 100,
            'batch_size': min(32, pop_size // 2),
            'learning_rate': 0.0002
        },
        'backdrive': {
            'hidden_layers': [100, 100],
            'activation': 'tanh',
            'epochs': 50,
            'batch_size': min(32, pop_size // 2),
            'learning_rate': 0.001,
            'backdrive_iterations': 100,
            'backdrive_lr': 0.1
        }
    }

    return defaults.get(eda_name, {})


def learn_model(eda_name: str, generation: int, n_vars: int, bounds: np.ndarray,
                selected_population: np.ndarray, selected_fitness: np.ndarray,
                params: Dict[str, Any], previous_model=None):
    """
    Learn a model using the specified EDA.

    Parameters
    ----------
    eda_name : str
        Name of the EDA
    generation : int
        Current generation number
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Variable bounds
    selected_population : np.ndarray
        Selected population
    selected_fitness : np.ndarray
        Fitness of selected population
    params : dict
        EDA parameters
    previous_model : optional
        Previous model for transfer learning (backdrive)

    Returns
    -------
    model : dict or object
        Learned model
    """
    if eda_name == 'vae':
        return learn_vae(selected_population, selected_fitness, params=params)

    elif eda_name == 'extended_vae':
        return learn_extended_vae(selected_population, selected_fitness, params=params)

    elif eda_name == 'conditional_vae':
        return learn_conditional_extended_vae(selected_population, selected_fitness, params=params)

    elif eda_name == 'gan':
        # GAN uses simpler signature: population, fitness, params
        return learn_gan(selected_population, selected_fitness, params=params)

    elif eda_name == 'backdrive':
        # Backdrive uses: generation, n_vars, cardinality, population, fitness, params
        bd_params = params.copy()
        if previous_model is not None:
            bd_params['previous_model'] = previous_model
        return learn_backdrive(generation, n_vars, bounds, selected_population,
                              selected_fitness, params=bd_params)

    else:
        raise ValueError(f"Unknown EDA: {eda_name}")


def sample_model(eda_name: str, model, n_samples: int, n_vars: int, bounds: np.ndarray,
                 selected_population: np.ndarray, selected_fitness: np.ndarray,
                 params: Dict[str, Any] = None) -> np.ndarray:
    """
    Sample from a model using the specified EDA.

    Parameters
    ----------
    eda_name : str
        Name of the EDA
    model : dict or object
        Learned model
    n_samples : int
        Number of samples to generate
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Variable bounds
    selected_population : np.ndarray
        Selected population (needed for backdrive)
    selected_fitness : np.ndarray
        Fitness of selected population (for conditional sampling)
    params : dict, optional
        Sampling parameters

    Returns
    -------
    samples : np.ndarray
        Generated samples
    """
    if params is None:
        params = {}

    if eda_name == 'vae':
        return sample_vae(model, n_samples=n_samples, bounds=bounds, params=params)

    elif eda_name == 'extended_vae':
        return sample_extended_vae(model, n_samples=n_samples, bounds=bounds,
                                   params={'use_predictor': False})

    elif eda_name == 'conditional_vae':
        target_fitness = np.min(selected_fitness)
        return sample_conditional_extended_vae(model, n_samples=n_samples, bounds=bounds,
                                               params={'target_fitness': target_fitness,
                                                      'fitness_noise': 0.1})

    elif eda_name == 'gan':
        return sample_gan(model, n_samples=n_samples, bounds=bounds, params=params)

    elif eda_name == 'backdrive':
        # Backdrive uses: n_vars, model, cardinality, current_population, current_fitness, params
        bd_params = params.copy() if params else {}
        bd_params['n_samples'] = n_samples
        return sample_backdrive(n_vars, model, bounds, selected_population,
                               selected_fitness, params=bd_params)

    else:
        raise ValueError(f"Unknown EDA: {eda_name}")


def run_neural_eda(
    eda_name: str,
    function_index: int,
    seed: int,
    instances_folder: str = DEFAULT_INSTANCES_FOLDER,
    pop_size: int = 100,
    selection_ratio: float = 0.3,
    neural_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a neural network-based EDA on a GNBG benchmark function.

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
    neural_params : dict, optional
        Parameters for the neural network
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
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

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
    if neural_params is None:
        neural_params = get_default_params(eda_name, n_vars, pop_size)

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
    previous_model = None

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
            print(f"Gen {gen+1:4d}: Best = {gen_best_fitness:.6e}, "
                  f"Mean = {gen_mean_fitness:.6e}, "
                  f"Overall = {best_fitness_overall:.6e}")

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
            model = learn_model(
                eda_name, gen, n_vars, bounds,
                selected_population, selected_fitness,
                neural_params, previous_model
            )
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
            new_population = sample_model(
                eda_name, model, pop_size, n_vars, bounds,
                selected_population, selected_fitness
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Sampling failed at gen {gen+1}: {e}")
            # Random restart on failure
            population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))
            continue

        sampling_times.append(time.time() - sample_start)

        # Update population
        population = new_population
        previous_model = model

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

        # Convergence history (sampled)
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
        print(f"Error from optimum:  {error_from_optimum:.6e}")
        print(f"Success: {'YES' if success else 'NO'}")
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
        description='Run Neural Network-Based EDA on GNBG Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_neural_eda_gnbg.py --eda_name vae --function_index 1 --seed 42
    python run_neural_eda_gnbg.py --eda_name gan --function_index 15 --seed 123
    python run_neural_eda_gnbg.py --eda_name conditional_vae -f 5 -s 999 --pop_size 200

Supported EDAs (continuous optimization):
    vae             - Variational Autoencoder EDA
    extended_vae    - Extended VAE with fitness predictor
    conditional_vae - Conditional Extended VAE (fitness-conditioned)
    gan             - Generative Adversarial Network EDA
    backdrive       - Backdrive Network Inversion EDA
        """
    )

    parser.add_argument(
        '--eda_name', '-e', type=str, required=True,
        choices=SUPPORTED_EDAS,
        help='Name of the neural network-based EDA to run'
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
    results = run_neural_eda(
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
    status = "SUCCESS" if results['success'] else "FAILURE"
    print(f"[{status}] {args.eda_name} f{args.function_index} seed={args.seed}: "
          f"error={results['error_from_optimum']:.6e}, "
          f"FE={results['function_evaluations']}, "
          f"time={results['runtime_seconds']:.2f}s")

    # Return exit code based on success
    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
