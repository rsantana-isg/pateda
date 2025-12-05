"""
Comprehensive Comparison: NN-EDA vs UMDA on GNBG Benchmark (n=30)

This script compares the performance of two EDAs on the GNBG benchmark suite:
1. NN-EDA: Fitness-weighted autoencoder with in-training evaluation
2. UMDA (Gaussian): Univariate marginal distribution algorithm

Key Differences:
- NN-EDA evaluates solutions during learning (local optimization)
- UMDA only evaluates during sampling
- NN-EDA uses elitism by default
- Both use the same population size and selection pressure

GNBG Benchmark:
- 24 diverse test functions
- Problem dimension: n=30 variables
- Various characteristics: unimodal, multimodal, separable, non-separable
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import time

# Add pateda parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GNBG class from enhanced_edas
sys.path.insert(0, str(Path(__file__).parent.parent / 'enhanced_edas'))
from GNBG_class import GNBG
from scipy.io import loadmat

# Import EDA modules
from pateda.learning.nn_eda import learn_nn_eda
from pateda.sampling.nn_eda import sample_nn_eda_hybrid, get_training_statistics
from pateda.learning.basic_gaussian import learn_gaussian_univariate
from pateda.sampling.basic_gaussian import sample_gaussian_univariate


# ============================================================================
# GNBG Loading Functions
# ============================================================================

def load_gnbg_instance(problem_index: int, instances_folder: str = None):
    """Load a GNBG problem instance from .mat file."""
    if instances_folder is None:
        instances_folder = str(Path(__file__).parent / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')

    if not (1 <= problem_index <= 24):
        raise ValueError(f'Problem index must be between 1 and 24, got {problem_index}')

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


def create_gnbg_fitness_function(problem_index: int):
    """Create a fitness function wrapper for a GNBG problem instance."""
    gnbg, info = load_gnbg_instance(problem_index)

    def fitness_function(x: np.ndarray) -> np.ndarray:
        """Fitness function wrapper for GNBG."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        fitness = gnbg.fitness(x)
        return fitness

    return fitness_function, info


# Problem descriptions
GNBG_DESCRIPTIONS = {
    1: 'Unimodal, separable',
    2: 'Unimodal, separable',
    3: 'Unimodal, non-separable',
    4: 'Unimodal, non-separable',
    5: 'Multimodal, separable',
    6: 'Multimodal, separable',
    7: 'Multimodal, non-separable',
    8: 'Multimodal, non-separable',
    9: 'Multimodal, non-separable',
    10: 'Multimodal, non-separable',
    11: 'Multimodal, non-separable',
    12: 'Multimodal, non-separable',
    13: 'Highly multimodal, separable',
    14: 'Highly multimodal, separable',
    15: 'Highly multimodal, non-separable',
    16: 'Highly multimodal, non-separable',
    17: 'Highly multimodal, non-separable',
    18: 'Highly multimodal, non-separable',
    19: 'Highly multimodal, non-separable',
    20: 'Highly multimodal, non-separable',
    21: 'Hybrid composition',
    22: 'Hybrid composition',
    23: 'Hybrid composition',
    24: 'Hybrid composition'
}


# ============================================================================
# EDA Implementations
# ============================================================================

def run_nn_eda(
    objective_function,
    bounds: Tuple[float, float],
    n_vars: int,
    max_evals: int,
    pop_size: int,
    selection_ratio: float,
    seed: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run NN-EDA on a given objective function.

    Returns:
    - best_fitness: Best fitness found
    - best_solution: Best solution found
    - n_evaluations: Total evaluations used
    - convergence_curve: Best fitness per generation
    - time_elapsed: Total time in seconds
    """
    np.random.seed(seed)
    x_min, x_max = bounds

    # Initialize population
    population = np.random.uniform(x_min, x_max, size=(pop_size, n_vars))
    fitness = objective_function(population)
    n_evaluations = pop_size

    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)].copy()
    convergence_curve = [best_fitness]

    generation = 0
    start_time = time.time()

    if verbose:
        print(f"  Gen 0: Best = {best_fitness:.6f}, Evals = {n_evaluations}")

    while n_evaluations < max_evals:
        generation += 1

        # Selection
        n_selected = int(pop_size * selection_ratio)
        selected_indices = np.argsort(fitness)[:n_selected]
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        # Learn NN-EDA model (evaluates solutions during training)
        nn_params = {
            'hidden_dims': [64, 32],
            'epochs': 30,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'elitism': True,  # Enable elitism
            'store_all_solutions': True,
            'verbose': False
        }

        model_data = learn_nn_eda(
            population=selected_pop,
            fitness=selected_fitness,
            objective_function=objective_function,
            params=nn_params
        )

        # Update evaluation count (training evaluations)
        n_evaluations += model_data['n_evaluations']

        # Check if we've found better solutions during training
        if len(model_data['evaluated_fitness']) > 0:
            training_best = np.min(model_data['evaluated_fitness'])
            if training_best < best_fitness:
                best_fitness = training_best
                best_idx = np.argmin(model_data['evaluated_fitness'])
                best_solution = model_data['evaluated_solutions'][best_idx].copy()

        # Sample new population (hybrid: training + generated solutions)
        # Elitism is automatically handled in hybrid sampling
        # Pass current population so autoencoder transforms it (not random solutions)
        new_population = sample_nn_eda_hybrid(
            model_data,
            n_samples=pop_size,
            fraction_from_training=0.6,
            source_population=population,
            source_fitness=fitness,
            params={'add_noise': True, 'noise_std': 0.05}
        )

        # Evaluate new population (but many are from training, so already evaluated)
        # For simplicity, we evaluate all and count only new ones
        # In practice, you'd track which were already evaluated
        new_fitness = objective_function(new_population)
        n_evaluations += pop_size  # Simplified: count all

        # Update best
        gen_best = np.min(new_fitness)
        if gen_best < best_fitness:
            best_fitness = gen_best
            best_solution = new_population[np.argmin(new_fitness)].copy()

        population = new_population
        fitness = new_fitness
        convergence_curve.append(best_fitness)

        if verbose and generation % 5 == 0:
            print(f"  Gen {generation}: Best = {best_fitness:.6f}, Evals = {n_evaluations}")

        # Check if we've exceeded budget
        if n_evaluations >= max_evals:
            break

    time_elapsed = time.time() - start_time

    return {
        'best_fitness': best_fitness,
        'best_solution': best_solution,
        'n_evaluations': n_evaluations,
        'convergence_curve': convergence_curve,
        'time_elapsed': time_elapsed,
        'generations': generation
    }


def run_umda(
    objective_function,
    bounds: Tuple[float, float],
    n_vars: int,
    max_evals: int,
    pop_size: int,
    selection_ratio: float,
    seed: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run UMDA (Gaussian univariate) on a given objective function.

    Returns:
    - best_fitness: Best fitness found
    - best_solution: Best solution found
    - n_evaluations: Total evaluations used
    - convergence_curve: Best fitness per generation
    - time_elapsed: Total time in seconds
    """
    np.random.seed(seed)
    x_min, x_max = bounds

    # Initialize population
    population = np.random.uniform(x_min, x_max, size=(pop_size, n_vars))
    fitness = objective_function(population)
    n_evaluations = pop_size

    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)].copy()
    elite_solution = best_solution.copy()  # Elitism
    elite_fitness = best_fitness
    convergence_curve = [best_fitness]

    generation = 0
    start_time = time.time()

    if verbose:
        print(f"  Gen 0: Best = {best_fitness:.6f}, Evals = {n_evaluations}")

    bounds_array = np.array([[x_min] * n_vars, [x_max] * n_vars])

    while n_evaluations < max_evals:
        generation += 1

        # Selection
        n_selected = int(pop_size * selection_ratio)
        selected_indices = np.argsort(fitness)[:n_selected]
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        # Learn Gaussian UMDA model
        model = learn_gaussian_univariate(selected_pop, selected_fitness)

        # Sample new population
        new_population = sample_gaussian_univariate(
            model,
            n_samples=pop_size - 1,  # -1 for elite
            bounds=bounds_array
        )

        # Add elite solution (elitism)
        new_population = np.vstack([elite_solution.reshape(1, -1), new_population])

        # Evaluate new population
        new_fitness = objective_function(new_population)
        n_evaluations += pop_size

        # Update best and elite
        gen_best = np.min(new_fitness)
        gen_best_idx = np.argmin(new_fitness)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_solution = new_population[gen_best_idx].copy()

        # Update elite
        if gen_best < elite_fitness:
            elite_fitness = gen_best
            elite_solution = new_population[gen_best_idx].copy()

        population = new_population
        fitness = new_fitness
        convergence_curve.append(best_fitness)

        if verbose and generation % 5 == 0:
            print(f"  Gen {generation}: Best = {best_fitness:.6f}, Evals = {n_evaluations}")

        if n_evaluations >= max_evals:
            break

    time_elapsed = time.time() - start_time

    return {
        'best_fitness': best_fitness,
        'best_solution': best_solution,
        'n_evaluations': n_evaluations,
        'convergence_curve': convergence_curve,
        'time_elapsed': time_elapsed,
        'generations': generation
    }


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_on_problem(
    problem_index: int,
    n_vars: int = 30,
    pop_size: int = 100,
    selection_ratio: float = 0.5,
    max_evals: int = 10000,
    n_runs: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare NN-EDA and UMDA on a single GNBG problem.

    Returns statistics across multiple runs.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"GNBG f{problem_index}: {GNBG_DESCRIPTIONS[problem_index]}")
        print(f"{'='*80}")
        print(f"Dimension: {n_vars}, Pop size: {pop_size}, Max evals: {max_evals}, Runs: {n_runs}")
        print()

    # Load problem
    objective_function, info = create_gnbg_fitness_function(problem_index)
    bounds = info['bounds']

    nn_eda_results = []
    umda_results = []

    for run in range(n_runs):
        if verbose:
            print(f"Run {run + 1}/{n_runs}:")

        # Run NN-EDA
        if verbose:
            print("  Running NN-EDA...")
        nn_result = run_nn_eda(
            objective_function,
            bounds,
            n_vars,
            max_evals,
            pop_size,
            selection_ratio,
            seed=42 + run,
            verbose=False
        )
        nn_eda_results.append(nn_result)

        # Run UMDA
        if verbose:
            print("  Running UMDA...")
        umda_result = run_umda(
            objective_function,
            bounds,
            n_vars,
            max_evals,
            pop_size,
            selection_ratio,
            seed=42 + run,
            verbose=False
        )
        umda_results.append(umda_result)

        if verbose:
            print(f"  NN-EDA: {nn_result['best_fitness']:.6f} ({nn_result['time_elapsed']:.2f}s)")
            print(f"  UMDA:   {umda_result['best_fitness']:.6f} ({umda_result['time_elapsed']:.2f}s)")

    # Compute statistics
    nn_eda_fitness = [r['best_fitness'] for r in nn_eda_results]
    umda_fitness = [r['best_fitness'] for r in umda_results]
    nn_eda_time = [r['time_elapsed'] for r in nn_eda_results]
    umda_time = [r['time_elapsed'] for r in umda_results]

    stats = {
        'problem_index': problem_index,
        'description': GNBG_DESCRIPTIONS[problem_index],
        'nn_eda': {
            'mean_fitness': np.mean(nn_eda_fitness),
            'std_fitness': np.std(nn_eda_fitness),
            'best_fitness': np.min(nn_eda_fitness),
            'worst_fitness': np.max(nn_eda_fitness),
            'median_fitness': np.median(nn_eda_fitness),
            'mean_time': np.mean(nn_eda_time),
            'all_results': nn_eda_results
        },
        'umda': {
            'mean_fitness': np.mean(umda_fitness),
            'std_fitness': np.std(umda_fitness),
            'best_fitness': np.min(umda_fitness),
            'worst_fitness': np.max(umda_fitness),
            'median_fitness': np.median(umda_fitness),
            'mean_time': np.mean(umda_time),
            'all_results': umda_results
        }
    }

    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Mean':<12} {'Std':<12} {'Best':<12} {'Median':<12} {'Time (s)':<10}")
        print(f"{'-'*80}")
        print(f"{'NN-EDA':<15} {stats['nn_eda']['mean_fitness']:<12.6f} {stats['nn_eda']['std_fitness']:<12.6f} "
              f"{stats['nn_eda']['best_fitness']:<12.6f} {stats['nn_eda']['median_fitness']:<12.6f} "
              f"{stats['nn_eda']['mean_time']:<10.2f}")
        print(f"{'UMDA':<15} {stats['umda']['mean_fitness']:<12.6f} {stats['umda']['std_fitness']:<12.6f} "
              f"{stats['umda']['best_fitness']:<12.6f} {stats['umda']['median_fitness']:<12.6f} "
              f"{stats['umda']['mean_time']:<10.2f}")

        # Compute improvement
        improvement = ((stats['umda']['mean_fitness'] - stats['nn_eda']['mean_fitness']) /
                      abs(stats['umda']['mean_fitness'])) * 100
        print(f"\nNN-EDA improvement over UMDA: {improvement:.2f}%")

    return stats


def run_comprehensive_comparison(
    problems: list = None,
    n_vars: int = 30,
    pop_size: int = 100,
    selection_ratio: float = 0.5,
    max_evals: int = 10000,
    n_runs: int = 10
):
    """
    Run comprehensive comparison across multiple GNBG problems.
    """
    if problems is None:
        # Test on representative subset
        problems = [1, 3, 5, 7, 13, 15, 21, 23]

    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: NN-EDA vs UMDA on GNBG Benchmark (n=30)")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Dimension: {n_vars}")
    print(f"  Population size: {pop_size}")
    print(f"  Selection ratio: {selection_ratio}")
    print(f"  Max evaluations: {max_evals}")
    print(f"  Runs per problem: {n_runs}")
    print(f"  Problems: {problems}")
    print()

    all_stats = []

    for problem_idx in problems:
        try:
            stats = compare_on_problem(
                problem_idx,
                n_vars=n_vars,
                pop_size=pop_size,
                selection_ratio=selection_ratio,
                max_evals=max_evals,
                n_runs=n_runs,
                verbose=True
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\nError on problem {problem_idx}: {str(e)}")
            continue

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY ACROSS ALL PROBLEMS")
    print("=" * 80)
    print()

    nn_eda_wins = 0
    umda_wins = 0
    ties = 0

    print(f"{'Problem':<10} {'Type':<35} {'NN-EDA':<15} {'UMDA':<15} {'Winner':<10}")
    print("-" * 80)

    for stats in all_stats:
        nn_mean = stats['nn_eda']['mean_fitness']
        umda_mean = stats['umda']['mean_fitness']

        if nn_mean < umda_mean * 0.99:  # 1% margin
            winner = "NN-EDA"
            nn_eda_wins += 1
        elif umda_mean < nn_mean * 0.99:
            winner = "UMDA"
            umda_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"f{stats['problem_index']:<9} {stats['description']:<35} "
              f"{nn_mean:<15.6f} {umda_mean:<15.6f} {winner:<10}")

    print()
    print(f"Results: NN-EDA wins: {nn_eda_wins}, UMDA wins: {umda_wins}, Ties: {ties}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    # Run comparison on a subset of GNBG problems
    # Covering different problem types
    problems_to_test = [
        1,   # Unimodal, separable
        3,   # Unimodal, non-separable
        5,   # Multimodal, separable
        7,   # Multimodal, non-separable
        13,  # Highly multimodal, separable
        15,  # Highly multimodal, non-separable
        21,  # Hybrid composition
        23   # Hybrid composition
    ]

    run_comprehensive_comparison(
        problems=problems_to_test,
        n_vars=30,
        pop_size=100,
        selection_ratio=0.5,
        max_evals=10000,
        n_runs=10
    )
