"""
Population statistics for Estimation of Distribution Algorithms.

This module provides functions to compute and track relevant statistics
about EDA populations during optimization, including fitness metrics,
population diversity, and computational performance.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from scipy import stats as sp_stats


def simple_population_statistics(
    generation: int,
    population: np.ndarray,
    fitness_values: np.ndarray,
    time_operations: np.ndarray,
    number_evaluations: np.ndarray,
    all_statistics: Dict[int, Dict[str, Any]],
    find_best_method: Optional[Callable] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Compute relevant statistics about EDA population in current generation.

    This function computes comprehensive statistics including fitness metrics,
    population diversity, variable statistics, and computational performance
    metrics.

    Parameters
    ----------
    generation : int
        Current generation number (0-indexed or 1-indexed).
    population : np.ndarray
        Current population matrix of shape (pop_size, n_variables).
    fitness_values : np.ndarray
        Matrix of fitness evaluations, shape (pop_size, n_objectives).
        For single-objective problems, shape is (pop_size,) or (pop_size, 1).
    time_operations : np.ndarray
        Matrix with time in seconds for main EDA steps, shape (generation+1, n_ops).
        Columns typically represent: [sampling, repairing, evaluation,
        local_optimization, replacement, selection, learning, total].
    number_evaluations : np.ndarray
        Vector with number of evaluations per generation, shape (generation+1,).
    all_statistics : dict
        Dictionary containing statistics from previous generations.
        This is updated in-place with current generation's statistics.
    find_best_method : callable, optional
        Function to find best individual(s) from population.
        Signature: find_best_method(population, fitness_values) -> indices
        If None, uses default fitness ordering (minimization).

    Returns
    -------
    dict
        Updated all_statistics dictionary with current generation's statistics.
        For each generation k, all_statistics[k] contains:
        - 'fitness_stats': Statistics of fitness values (max, mean, median, min, std)
        - 'best_individual': Best individual in population
        - 'n_unique': Number of unique individuals
        - 'variable_stats': Statistics of variable values (max, mean, median, min, std)
        - 'n_evaluations': Number of evaluations in this generation
        - 'time_operations': Time spent in each operation

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.random.rand(100, 10)
    >>> fitness = np.random.rand(100)
    >>> time_ops = np.array([[0.1, 0.2, 0.3, 0.0, 0.1, 0.05, 0.15, 1.0]])
    >>> n_evals = np.array([100])
    >>> stats = {}
    >>> stats = simple_population_statistics(
    ...     0, pop, fitness, time_ops, n_evals, stats
    ... )
    >>> print(stats[0]['fitness_stats'])

    Notes
    -----
    - For multi-objective problems, statistics are computed per objective
    - Standard deviation is used instead of variance for better interpretability
    - Population diversity is measured by counting unique individuals
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    # Ensure fitness_values is 2D
    if fitness_values.ndim == 1:
        fitness_values = fitness_values.reshape(-1, 1)

    # Compute fitness statistics (per objective)
    fitness_stats = np.vstack([
        np.max(fitness_values, axis=0),      # Maximum
        np.mean(fitness_values, axis=0),     # Mean
        np.median(fitness_values, axis=0),   # Median
        np.min(fitness_values, axis=0),      # Minimum
        np.std(fitness_values, axis=0)       # Standard deviation
    ])

    # Find best individual
    if find_best_method is None:
        # Default: minimization, use first objective if multi-objective
        best_idx = np.argmin(fitness_values[:, 0])
    else:
        indices = find_best_method(population, fitness_values)
        best_idx = indices[0] if hasattr(indices, '__iter__') else indices

    best_individual = population[best_idx, :].copy()

    # Count unique individuals (measure of diversity)
    n_unique = len(np.unique(population, axis=0))

    # Compute variable statistics (per variable)
    variable_stats = np.vstack([
        np.max(population, axis=0),      # Maximum
        np.mean(population, axis=0),     # Mean
        np.median(population, axis=0),   # Median
        np.min(population, axis=0),      # Minimum
        np.std(population, axis=0)       # Standard deviation
    ])

    # Store all statistics for this generation
    all_statistics[generation] = {
        'fitness_stats': fitness_stats,
        'best_individual': best_individual,
        'n_unique': n_unique,
        'variable_stats': variable_stats,
        'n_evaluations': number_evaluations[generation],
        'time_operations': time_operations[generation, :].copy()
    }

    return all_statistics


def compute_fitness_statistics(fitness_values: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute descriptive statistics for fitness values.

    Parameters
    ----------
    fitness_values : np.ndarray
        Fitness values, shape (pop_size,) or (pop_size, n_objectives).

    Returns
    -------
    dict
        Dictionary with keys: 'max', 'mean', 'median', 'min', 'std', 'var'.
        Each value is an array with one element per objective.

    Examples
    --------
    >>> import numpy as np
    >>> fitness = np.random.rand(100, 2)
    >>> stats = compute_fitness_statistics(fitness)
    >>> print(stats['mean'])
    """
    if fitness_values.ndim == 1:
        fitness_values = fitness_values.reshape(-1, 1)

    return {
        'max': np.max(fitness_values, axis=0),
        'mean': np.mean(fitness_values, axis=0),
        'median': np.median(fitness_values, axis=0),
        'min': np.min(fitness_values, axis=0),
        'std': np.std(fitness_values, axis=0),
        'var': np.var(fitness_values, axis=0)
    }


def compute_population_diversity(population: np.ndarray) -> Dict[str, float]:
    """
    Compute diversity metrics for the population.

    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_variables).

    Returns
    -------
    dict
        Dictionary with diversity metrics:
        - 'n_unique': Number of unique individuals
        - 'uniqueness_ratio': Ratio of unique to total individuals
        - 'entropy': Entropy-based diversity measure (for discrete variables)

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.random.randint(0, 2, (100, 20))
    >>> diversity = compute_population_diversity(pop)
    >>> print(f"Uniqueness: {diversity['uniqueness_ratio']:.2%}")
    """
    pop_size = population.shape[0]
    unique_individuals = np.unique(population, axis=0)
    n_unique = len(unique_individuals)
    uniqueness_ratio = n_unique / pop_size

    # Compute entropy for discrete populations (binary or integer)
    entropy = 0.0
    if np.all(population == population.astype(int)):
        # For each variable, compute entropy
        entropies = []
        for var_idx in range(population.shape[1]):
            values, counts = np.unique(population[:, var_idx], return_counts=True)
            probabilities = counts / pop_size
            var_entropy = sp_stats.entropy(probabilities)
            entropies.append(var_entropy)
        entropy = np.mean(entropies)

    return {
        'n_unique': n_unique,
        'uniqueness_ratio': uniqueness_ratio,
        'entropy': entropy
    }


def compute_convergence_metrics(
    all_statistics: Dict[int, Dict[str, Any]],
    objective_idx: int = 0
) -> Dict[str, np.ndarray]:
    """
    Compute convergence metrics across generations.

    Parameters
    ----------
    all_statistics : dict
        Dictionary of statistics across all generations.
    objective_idx : int, default=0
        Index of objective to analyze (for multi-objective problems).

    Returns
    -------
    dict
        Dictionary with convergence metrics:
        - 'best_fitness': Best fitness per generation
        - 'mean_fitness': Mean fitness per generation
        - 'improvement': Improvement in best fitness per generation
        - 'generations': Generation numbers

    Examples
    --------
    >>> # Assume stats were collected over 50 generations
    >>> convergence = compute_convergence_metrics(all_statistics)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(convergence['generations'], convergence['best_fitness'])
    >>> plt.xlabel('Generation')
    >>> plt.ylabel('Best Fitness')
    >>> plt.show()
    """
    generations = sorted(all_statistics.keys())
    best_fitness = []
    mean_fitness = []

    for gen in generations:
        stats = all_statistics[gen]
        fitness_stats = stats['fitness_stats']

        # Extract best and mean fitness for specified objective
        if fitness_stats.ndim > 1 and fitness_stats.shape[1] > objective_idx:
            best_fitness.append(fitness_stats[3, objective_idx])  # min (row 3)
            mean_fitness.append(fitness_stats[1, objective_idx])  # mean (row 1)
        else:
            best_fitness.append(fitness_stats[3])
            mean_fitness.append(fitness_stats[1])

    best_fitness = np.array(best_fitness)
    mean_fitness = np.array(mean_fitness)

    # Compute improvement (difference from previous generation)
    improvement = np.zeros_like(best_fitness)
    if len(best_fitness) > 1:
        improvement[1:] = best_fitness[:-1] - best_fitness[1:]

    return {
        'best_fitness': best_fitness,
        'mean_fitness': mean_fitness,
        'improvement': improvement,
        'generations': np.array(generations)
    }
