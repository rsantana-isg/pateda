"""
Fitness-related measures for knowledge extraction from EDAs.

This module implements various measures to analyze fitness evolution and
selection dynamics in Estimation of Distribution Algorithms. These measures
help understand how the algorithm explores and exploits the search space.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Optional, Tuple, Any


def response_to_selection(
    fitness_before: np.ndarray,
    fitness_after: np.ndarray,
    objective_idx: int = 0,
    minimize: bool = True
) -> Dict[str, float]:
    """
    Compute response to selection between two generations.

    Response to selection measures the change in mean fitness from one
    generation to the next due to selection. It quantifies the effectiveness
    of selection pressure in improving the population.

    Parameters
    ----------
    fitness_before : np.ndarray
        Fitness values before selection, shape (pop_size,) or (pop_size, n_objectives).
    fitness_after : np.ndarray
        Fitness values after selection, shape (selected_size,) or (selected_size, n_objectives).
    objective_idx : int, default=0
        Index of objective to analyze (for multi-objective problems).
    minimize : bool, default=True
        Whether the problem is minimization (True) or maximization (False).

    Returns
    -------
    dict
        Dictionary containing:
        - 'response': Response to selection (mean_after - mean_before)
        - 'mean_before': Mean fitness before selection
        - 'mean_after': Mean fitness after selection
        - 'std_before': Standard deviation before selection
        - 'std_after': Standard deviation after selection
        - 'improvement': Relative improvement in fitness

    Examples
    --------
    >>> import numpy as np
    >>> fitness_pop = np.random.rand(100)
    >>> fitness_selected = np.random.rand(50) * 0.5  # Better individuals
    >>> result = response_to_selection(fitness_pop, fitness_selected)
    >>> print(f"Response: {result['response']:.4f}")

    Notes
    -----
    - Higher absolute response indicates stronger selection effect
    - For minimization, negative response indicates improvement
    - For maximization, positive response indicates improvement
    - Based on quantitative genetics theory
    - Original concept from MATEDA 2.0 (Section 8.1)
    """
    # Ensure 1D for single objective
    if fitness_before.ndim > 1:
        fitness_before = fitness_before[:, objective_idx]
    if fitness_after.ndim > 1:
        fitness_after = fitness_after[:, objective_idx]

    mean_before = np.mean(fitness_before)
    mean_after = np.mean(fitness_after)
    std_before = np.std(fitness_before, ddof=1)
    std_after = np.std(fitness_after, ddof=1)

    response = mean_after - mean_before

    # Compute relative improvement based on optimization direction
    if minimize:
        improvement = (mean_before - mean_after) / abs(mean_before) if mean_before != 0 else 0.0
    else:
        improvement = (mean_after - mean_before) / abs(mean_before) if mean_before != 0 else 0.0

    return {
        'response': float(response),
        'mean_before': float(mean_before),
        'mean_after': float(mean_after),
        'std_before': float(std_before),
        'std_after': float(std_after),
        'improvement': float(improvement)
    }


def amount_of_selection(
    fitness_values: np.ndarray,
    selected_indices: np.ndarray,
    objective_idx: int = 0
) -> Dict[str, float]:
    """
    Compute amount of selection (selection differential).

    The selection differential is the difference between the mean fitness
    of selected individuals and the mean fitness of the entire population.
    It measures the intensity of selection pressure.

    Parameters
    ----------
    fitness_values : np.ndarray
        Fitness values for entire population, shape (pop_size,) or (pop_size, n_objectives).
    selected_indices : np.ndarray
        Indices of selected individuals.
    objective_idx : int, default=0
        Index of objective to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - 'selection_differential': S = mean_selected - mean_population
        - 'standardized_differential': S / std_population
        - 'selection_intensity': i = S / std (standardized selection differential)
        - 'proportion_selected': Proportion of population selected
        - 'mean_population': Mean fitness of entire population
        - 'mean_selected': Mean fitness of selected individuals

    Examples
    --------
    >>> import numpy as np
    >>> fitness = np.random.rand(100)
    >>> selected = np.argsort(fitness)[:20]  # Select top 20
    >>> result = amount_of_selection(fitness, selected)
    >>> print(f"Selection differential: {result['selection_differential']:.4f}")

    Notes
    -----
    - Larger absolute differential indicates stronger selection
    - Selection intensity is normalized by population standard deviation
    - Standardized differential allows comparison across different scales
    - Based on quantitative genetics and breeding theory
    - Original concept from MATEDA 2.0 (Section 8.1)
    """
    # Ensure 1D for single objective
    if fitness_values.ndim > 1:
        fitness_values = fitness_values[:, objective_idx]

    pop_size = len(fitness_values)
    selected_size = len(selected_indices)

    mean_population = np.mean(fitness_values)
    std_population = np.std(fitness_values, ddof=1)
    mean_selected = np.mean(fitness_values[selected_indices])

    selection_differential = mean_selected - mean_population
    standardized_differential = selection_differential / std_population if std_population > 0 else 0.0
    proportion_selected = selected_size / pop_size

    return {
        'selection_differential': float(selection_differential),
        'standardized_differential': float(standardized_differential),
        'selection_intensity': float(standardized_differential),  # Common alias
        'proportion_selected': float(proportion_selected),
        'mean_population': float(mean_population),
        'mean_selected': float(mean_selected),
        'std_population': float(std_population)
    }


def realized_heritability(
    fitness_parents: np.ndarray,
    fitness_offspring: np.ndarray,
    fitness_population: np.ndarray,
    objective_idx: int = 0,
    method: str = 'regression'
) -> Dict[str, float]:
    """
    Compute realized heritability from parent-offspring fitness relationships.

    Realized heritability (h²) estimates the proportion of fitness variance
    that is heritable. It measures how well offspring inherit the fitness
    characteristics of their parents.

    Parameters
    ----------
    fitness_parents : np.ndarray
        Fitness values of selected parents, shape (n_parents,) or (n_parents, n_objectives).
    fitness_offspring : np.ndarray
        Fitness values of offspring, shape (n_offspring,) or (n_offspring, n_objectives).
    fitness_population : np.ndarray
        Fitness values of entire population before selection.
    objective_idx : int, default=0
        Index of objective to analyze.
    method : str, default='regression'
        Method to compute heritability:
        - 'regression': Slope of offspring vs. parent regression
        - 'ratio': Response / Selection differential

    Returns
    -------
    dict
        Dictionary containing:
        - 'heritability': Realized heritability estimate (h²)
        - 'response': Response to selection (R)
        - 'selection_differential': Selection differential (S)
        - 'correlation': Parent-offspring correlation (if method='regression')
        - 'method': Method used

    Examples
    --------
    >>> import numpy as np
    >>> pop_fitness = np.random.rand(100)
    >>> parent_fitness = np.random.rand(30) * 0.5
    >>> offspring_fitness = parent_fitness + np.random.normal(0, 0.1, 30)
    >>> result = realized_heritability(parent_fitness, offspring_fitness, pop_fitness)
    >>> print(f"Heritability: {result['heritability']:.4f}")

    Notes
    -----
    - h² ranges from 0 (no heritability) to 1 (perfect heritability)
    - In EDAs, high heritability suggests the model captures important structure
    - Low heritability may indicate noise or model limitations
    - Based on the breeder's equation: R = h² * S
    - Original concept from MATEDA 2.0 (Section 8.1)
    """
    # Ensure 1D for single objective
    if fitness_parents.ndim > 1:
        fitness_parents = fitness_parents[:, objective_idx]
    if fitness_offspring.ndim > 1:
        fitness_offspring = fitness_offspring[:, objective_idx]
    if fitness_population.ndim > 1:
        fitness_population = fitness_population[:, objective_idx]

    mean_population = np.mean(fitness_population)
    mean_parents = np.mean(fitness_parents)
    mean_offspring = np.mean(fitness_offspring)

    # Selection differential (S)
    selection_differential = mean_parents - mean_population

    # Response to selection (R)
    response = mean_offspring - mean_population

    # Compute heritability based on method
    if method == 'regression':
        # Heritability as parent-offspring regression slope
        # Need paired parent-offspring data
        if len(fitness_parents) == len(fitness_offspring):
            # Linear regression: offspring = h² * parent + constant
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                fitness_parents, fitness_offspring
            )
            heritability = slope
            correlation = r_value
        else:
            # If not paired, use ratio method
            heritability = response / selection_differential if selection_differential != 0 else 0.0
            correlation = None
    else:  # method == 'ratio'
        # Heritability as R/S
        heritability = response / selection_differential if selection_differential != 0 else 0.0
        correlation = None

    # Clip to valid range [0, 1] for interpretability
    heritability = np.clip(heritability, 0.0, 1.0)

    result = {
        'heritability': float(heritability),
        'response': float(response),
        'selection_differential': float(selection_differential),
        'method': method,
        'mean_population': float(mean_population),
        'mean_parents': float(mean_parents),
        'mean_offspring': float(mean_offspring)
    }

    if correlation is not None:
        result['correlation'] = float(correlation)

    return result


def compute_objective_distribution(
    fitness_values: np.ndarray,
    objective_idx: int = 0,
    n_bins: int = 20
) -> Dict[str, Any]:
    """
    Analyze the distribution of objective function values.

    Computes descriptive statistics and distribution characteristics of
    fitness values to understand population diversity and convergence.

    Parameters
    ----------
    fitness_values : np.ndarray
        Fitness values, shape (pop_size,) or (pop_size, n_objectives).
    objective_idx : int, default=0
        Index of objective to analyze.
    n_bins : int, default=20
        Number of bins for histogram.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': Mean fitness
        - 'median': Median fitness
        - 'std': Standard deviation
        - 'var': Variance
        - 'min': Minimum fitness
        - 'max': Maximum fitness
        - 'range': Range (max - min)
        - 'iqr': Interquartile range
        - 'skewness': Distribution skewness
        - 'kurtosis': Distribution kurtosis
        - 'histogram': Tuple of (counts, bin_edges)
        - 'normality_test': Shapiro-Wilk test results (statistic, pvalue)

    Examples
    --------
    >>> import numpy as np
    >>> fitness = np.random.normal(0.5, 0.1, 100)
    >>> dist_info = compute_objective_distribution(fitness)
    >>> print(f"Skewness: {dist_info['skewness']:.4f}")
    >>> print(f"Is normal? p={dist_info['normality_test'][1]:.4f}")

    Notes
    -----
    - Skewness near 0 indicates symmetric distribution
    - Kurtosis near 0 indicates normal-like tails
    - IQR is robust to outliers compared to range
    - Histogram useful for visualizing fitness landscape exploration
    - Original concept from MATEDA 2.0 (Section 8.1)
    """
    # Ensure 1D for single objective
    if fitness_values.ndim > 1:
        fitness_values = fitness_values[:, objective_idx]

    # Basic statistics
    mean = np.mean(fitness_values)
    median = np.median(fitness_values)
    std = np.std(fitness_values, ddof=1)
    var = np.var(fitness_values, ddof=1)
    min_val = np.min(fitness_values)
    max_val = np.max(fitness_values)
    range_val = max_val - min_val

    # Quartiles and IQR
    q1, q3 = np.percentile(fitness_values, [25, 75])
    iqr = q3 - q1

    # Distribution shape
    skewness = sp_stats.skew(fitness_values)
    kurtosis = sp_stats.kurtosis(fitness_values)

    # Histogram
    hist_counts, hist_edges = np.histogram(fitness_values, bins=n_bins)

    # Normality test (Shapiro-Wilk)
    if len(fitness_values) >= 3:
        shapiro_stat, shapiro_p = sp_stats.shapiro(fitness_values)
    else:
        shapiro_stat, shapiro_p = None, None

    return {
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'var': float(var),
        'min': float(min_val),
        'max': float(max_val),
        'range': float(range_val),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'histogram': (hist_counts, hist_edges),
        'normality_test': (shapiro_stat, shapiro_p) if shapiro_stat is not None else None
    }


def analyze_fitness_evolution(
    all_statistics: Dict[int, Dict[str, Any]],
    objective_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze fitness evolution across all generations.

    Computes comprehensive measures of how fitness changes over generations,
    including response to selection, selection intensity, and convergence patterns.

    Parameters
    ----------
    all_statistics : dict
        Dictionary of statistics per generation from simple_population_statistics.
    objective_idx : int, default=0
        Index of objective to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - 'generations': Array of generation numbers
        - 'mean_fitness': Mean fitness per generation
        - 'best_fitness': Best fitness per generation
        - 'std_fitness': Standard deviation per generation
        - 'responses': Response to selection per generation
        - 'improvements': Improvement rate per generation
        - 'diversity_loss': Change in diversity over time

    Examples
    --------
    >>> # Assume all_statistics collected during EDA run
    >>> evolution = analyze_fitness_evolution(all_statistics)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(evolution['generations'], evolution['best_fitness'])
    >>> plt.xlabel('Generation')
    >>> plt.ylabel('Best Fitness')
    >>> plt.show()

    Notes
    -----
    - Provides high-level view of optimization progress
    - Useful for diagnosing premature convergence
    - Can identify phases of exploration vs. exploitation
    """
    generations = sorted(all_statistics.keys())
    n_gens = len(generations)

    mean_fitness = np.zeros(n_gens)
    best_fitness = np.zeros(n_gens)
    std_fitness = np.zeros(n_gens)
    diversity = np.zeros(n_gens)

    for i, gen in enumerate(generations):
        stats = all_statistics[gen]
        fitness_stats = stats['fitness_stats']

        # Extract statistics for this generation
        if fitness_stats.ndim > 1:
            mean_fitness[i] = fitness_stats[1, objective_idx]  # Mean (row 1)
            best_fitness[i] = fitness_stats[3, objective_idx]  # Min (row 3)
            std_fitness[i] = fitness_stats[4, objective_idx]   # Std (row 4)
        else:
            mean_fitness[i] = fitness_stats[1]
            best_fitness[i] = fitness_stats[3]
            std_fitness[i] = fitness_stats[4]

        diversity[i] = stats.get('n_unique', 0)

    # Compute generation-to-generation changes
    responses = np.zeros(n_gens)
    improvements = np.zeros(n_gens)

    if n_gens > 1:
        responses[1:] = mean_fitness[1:] - mean_fitness[:-1]
        improvements[1:] = best_fitness[:-1] - best_fitness[1:]  # For minimization

    # Diversity loss rate
    diversity_loss = np.zeros(n_gens)
    if n_gens > 1:
        diversity_loss[1:] = diversity[:-1] - diversity[1:]

    return {
        'generations': np.array(generations),
        'mean_fitness': mean_fitness,
        'best_fitness': best_fitness,
        'std_fitness': std_fitness,
        'diversity': diversity,
        'responses': responses,
        'improvements': improvements,
        'diversity_loss': diversity_loss
    }
