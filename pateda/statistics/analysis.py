"""
Statistical analysis utilities for EDA results.

This module provides advanced statistical analysis functions for
evaluating and comparing EDA performance across multiple runs.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Tuple, Optional, Any
import warnings


def analyze_multiple_runs(
    all_runs_statistics: List[Dict[int, Dict[str, Any]]],
    objective_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze statistics across multiple independent runs.

    Parameters
    ----------
    all_runs_statistics : list of dict
        List where each element is a statistics dictionary from one run.
    objective_idx : int, default=0
        Objective index for multi-objective problems.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_best_fitness': Mean best fitness across runs per generation
        - 'std_best_fitness': Std of best fitness across runs per generation
        - 'median_best_fitness': Median best fitness across runs per generation
        - 'success_rate': Proportion of runs reaching target (if applicable)
        - 'generations': Generation numbers

    Examples
    --------
    >>> run_stats_list = [run1_stats, run2_stats, run3_stats]
    >>> analysis = analyze_multiple_runs(run_stats_list)
    >>> print(f"Mean best fitness: {analysis['mean_best_fitness'][-1]:.6f}")
    """
    n_runs = len(all_runs_statistics)
    if n_runs == 0:
        raise ValueError("No runs provided for analysis")

    # Find common generations across all runs
    all_gens = [set(run_stats.keys()) for run_stats in all_runs_statistics]
    common_gens = sorted(set.intersection(*all_gens))

    if not common_gens:
        raise ValueError("No common generations found across runs")

    # Extract best fitness for each run and generation
    best_fitness_array = np.zeros((n_runs, len(common_gens)))

    for run_idx, run_stats in enumerate(all_runs_statistics):
        for gen_idx, gen in enumerate(common_gens):
            fitness_stats = run_stats[gen]['fitness_stats']
            if fitness_stats.ndim > 1:
                best_fitness_array[run_idx, gen_idx] = fitness_stats[3, objective_idx]
            else:
                best_fitness_array[run_idx, gen_idx] = fitness_stats[3]

    # Compute aggregate statistics
    mean_best = np.mean(best_fitness_array, axis=0)
    std_best = np.std(best_fitness_array, axis=0)
    median_best = np.median(best_fitness_array, axis=0)
    min_best = np.min(best_fitness_array, axis=0)
    max_best = np.max(best_fitness_array, axis=0)

    return {
        'mean_best_fitness': mean_best,
        'std_best_fitness': std_best,
        'median_best_fitness': median_best,
        'min_best_fitness': min_best,
        'max_best_fitness': max_best,
        'generations': np.array(common_gens),
        'n_runs': n_runs,
        'raw_data': best_fitness_array
    }


def compute_statistical_tests(
    results1: np.ndarray,
    results2: np.ndarray,
    test: str = 'mannwhitneyu',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical significance tests between two sets of results.

    Parameters
    ----------
    results1 : np.ndarray
        Results from first algorithm/configuration.
    results2 : np.ndarray
        Results from second algorithm/configuration.
    test : str, default='mannwhitneyu'
        Statistical test to use:
        - 'mannwhitneyu': Mann-Whitney U test (non-parametric)
        - 'ttest': Independent t-test (parametric)
        - 'wilcoxon': Wilcoxon signed-rank test (paired, non-parametric)
        - 'ttest_paired': Paired t-test (paired, parametric)
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': Test statistic value
        - 'pvalue': P-value
        - 'significant': Whether difference is significant at alpha level
        - 'effect_size': Effect size measure (Cohen's d or rank-biserial)

    Examples
    --------
    >>> results_alg1 = np.array([0.5, 0.6, 0.55, 0.58, 0.62])
    >>> results_alg2 = np.array([0.7, 0.75, 0.72, 0.68, 0.73])
    >>> test_result = compute_statistical_tests(results_alg1, results_alg2)
    >>> if test_result['significant']:
    ...     print("Difference is statistically significant")
    """
    if test == 'mannwhitneyu':
        statistic, pvalue = sp_stats.mannwhitneyu(results1, results2, alternative='two-sided')
        # Compute rank-biserial correlation as effect size
        n1, n2 = len(results1), len(results2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

    elif test == 'ttest':
        statistic, pvalue = sp_stats.ttest_ind(results1, results2)
        # Compute Cohen's d
        pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1, ddof=1) +
                               (len(results2) - 1) * np.var(results2, ddof=1)) /
                              (len(results1) + len(results2) - 2))
        effect_size = (np.mean(results1) - np.mean(results2)) / pooled_std

    elif test == 'wilcoxon':
        statistic, pvalue = sp_stats.wilcoxon(results1, results2)
        # Effect size: rank-biserial correlation
        n = len(results1)
        effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)

    elif test == 'ttest_paired':
        statistic, pvalue = sp_stats.ttest_rel(results1, results2)
        # Cohen's d for paired samples
        diff = results1 - results2
        effect_size = np.mean(diff) / np.std(diff, ddof=1)

    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'test': test,
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha,
        'effect_size': float(effect_size),
        'alpha': alpha
    }


def compute_success_rate(
    all_runs_statistics: List[Dict[int, Dict[str, Any]]],
    target_fitness: float,
    objective_idx: int = 0,
    comparison: str = 'less_equal'
) -> Dict[str, Any]:
    """
    Compute success rate across multiple runs.

    Parameters
    ----------
    all_runs_statistics : list of dict
        List of statistics from multiple runs.
    target_fitness : float
        Target fitness value to achieve.
    objective_idx : int, default=0
        Objective index for multi-objective problems.
    comparison : str, default='less_equal'
        Comparison operator: 'less', 'less_equal', 'greater', 'greater_equal', 'equal'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'success_rate': Proportion of successful runs
        - 'n_successful': Number of successful runs
        - 'n_total': Total number of runs
        - 'successful_generations': Average generation when target was reached

    Examples
    --------
    >>> success_info = compute_success_rate(all_runs, target_fitness=0.01)
    >>> print(f"Success rate: {success_info['success_rate']:.1%}")
    """
    n_runs = len(all_runs_statistics)
    successful_runs = 0
    successful_generations = []

    for run_stats in all_runs_statistics:
        generations = sorted(run_stats.keys())

        for gen in generations:
            fitness_stats = run_stats[gen]['fitness_stats']
            best_fitness = fitness_stats[3, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[3]

            # Check if target is reached
            reached = False
            if comparison == 'less':
                reached = best_fitness < target_fitness
            elif comparison == 'less_equal':
                reached = best_fitness <= target_fitness
            elif comparison == 'greater':
                reached = best_fitness > target_fitness
            elif comparison == 'greater_equal':
                reached = best_fitness >= target_fitness
            elif comparison == 'equal':
                reached = np.isclose(best_fitness, target_fitness)

            if reached:
                successful_runs += 1
                successful_generations.append(gen)
                break

    success_rate = successful_runs / n_runs if n_runs > 0 else 0.0
    avg_success_gen = np.mean(successful_generations) if successful_generations else None

    return {
        'success_rate': success_rate,
        'n_successful': successful_runs,
        'n_total': n_runs,
        'successful_generations': successful_generations,
        'avg_successful_generation': avg_success_gen
    }


def compute_auc(
    generations: np.ndarray,
    fitness_values: np.ndarray
) -> float:
    """
    Compute area under the curve for convergence analysis.

    Uses trapezoidal rule to compute AUC of fitness over generations.
    Lower AUC typically indicates faster convergence (for minimization).

    Parameters
    ----------
    generations : np.ndarray
        Generation numbers.
    fitness_values : np.ndarray
        Fitness values at each generation.

    Returns
    -------
    float
        Area under the curve.

    Examples
    --------
    >>> gens = np.array([0, 1, 2, 3, 4, 5])
    >>> fitness = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.3])
    >>> auc = compute_auc(gens, fitness)
    >>> print(f"AUC: {auc:.2f}")
    """
    return np.trapz(fitness_values, generations)


def detect_convergence(
    fitness_history: np.ndarray,
    window: int = 10,
    threshold: float = 1e-6
) -> Tuple[bool, Optional[int]]:
    """
    Detect if algorithm has converged.

    Parameters
    ----------
    fitness_history : np.ndarray
        Array of best fitness values over generations.
    window : int, default=10
        Number of recent generations to check for convergence.
    threshold : float, default=1e-6
        Maximum allowed change to consider converged.

    Returns
    -------
    converged : bool
        Whether the algorithm has converged.
    generation : int or None
        Generation at which convergence was detected, or None if not converged.

    Examples
    --------
    >>> fitness = np.array([10, 5, 2, 1, 0.5, 0.501, 0.500, 0.501, 0.500, 0.501])
    >>> converged, gen = detect_convergence(fitness, window=5, threshold=0.01)
    >>> if converged:
    ...     print(f"Converged at generation {gen}")
    """
    if len(fitness_history) < window:
        return False, None

    # Check if improvement in last 'window' generations is below threshold
    recent_fitness = fitness_history[-window:]
    fitness_range = np.max(recent_fitness) - np.min(recent_fitness)

    if fitness_range < threshold:
        # Find when convergence started
        for i in range(len(fitness_history) - window + 1):
            window_fitness = fitness_history[i:i + window]
            if np.max(window_fitness) - np.min(window_fitness) < threshold:
                return True, i

    return False, None
