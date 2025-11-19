"""
EDA-specific knowledge extraction strategies for PATEDA.

This module provides specialized knowledge extraction methods tailored to
different types of EDAs implemented in PATEDA, including discrete, continuous,
and deep learning-based EDAs.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from pateda.knowledge_extraction.fitness_measures import (
    response_to_selection,
    amount_of_selection,
    realized_heritability,
    analyze_fitness_evolution
)
from pateda.knowledge_extraction.dependency_analysis import (
    compute_correlation_matrix,
    learn_bayesian_network,
    learn_gaussian_network
)


def extract_bayesian_network_evolution(
    cache: Any,
    selected_generations: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Extract and analyze Bayesian network structures across generations.

    For EDAs that learn Bayesian networks (BOA, EBNA, etc.), this function
    extracts the learned structures across generations and analyzes how
    they evolve.

    Parameters
    ----------
    cache : Cache object
        EDA cache containing models from each generation.
    selected_generations : list of int, optional
        Specific generations to analyze. If None, uses all generations.

    Returns
    -------
    dict
        Dictionary containing:
        - 'structures': List of adjacency matrices per generation
        - 'edge_frequencies': Matrix showing how often each edge appears
        - 'n_edges_per_gen': Number of edges per generation
        - 'stable_edges': Edges that appear in most generations
        - 'emerging_edges': Edges that appear in later generations
        - 'disappearing_edges': Edges that disappear in later generations

    Examples
    --------
    >>> stats, cache = eda.run()
    >>> network_evolution = extract_bayesian_network_evolution(cache)
    >>> print(f"Stable edges: {len(network_evolution['stable_edges'])}")

    Notes
    -----
    - Assumes models have 'adjacency_matrix' or 'structure' attribute
    - Useful for understanding which dependencies are consistently learned
    - Can reveal problem structure from evolutionary learning
    """
    if not hasattr(cache, 'models') or len(cache.models) == 0:
        raise ValueError("Cache does not contain models")

    models = cache.models

    if selected_generations is None:
        selected_generations = list(range(len(models)))

    # Extract structures
    structures = []
    n_vars = None

    for gen_idx in selected_generations:
        if gen_idx >= len(models):
            continue

        model = models[gen_idx]

        # Try to extract adjacency matrix
        if hasattr(model, 'adjacency_matrix'):
            adj_matrix = model.adjacency_matrix
        elif hasattr(model, 'structure'):
            adj_matrix = model.structure
        elif hasattr(model, 'graph'):
            adj_matrix = model.graph
        else:
            warnings.warn(f"Model at generation {gen_idx} has no structure attribute")
            continue

        if adj_matrix is not None:
            structures.append(adj_matrix.copy())
            if n_vars is None:
                n_vars = adj_matrix.shape[0]

    if len(structures) == 0:
        raise ValueError("No structures found in models")

    # Compute edge frequencies
    edge_freq_matrix = np.zeros((n_vars, n_vars))
    n_edges_per_gen = []

    for struct in structures:
        edge_freq_matrix += (struct > 0).astype(int)
        n_edges_per_gen.append(np.sum(struct > 0))

    edge_freq_matrix = edge_freq_matrix / len(structures)

    # Identify edge categories
    stable_edges = []  # Appear in >= 80% of generations
    emerging_edges = []  # Appear more in later half
    disappearing_edges = []  # Appear more in earlier half

    half_point = len(structures) // 2

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue

            freq_overall = edge_freq_matrix[i, j]
            freq_early = np.mean([struct[i, j] > 0 for struct in structures[:half_point]])
            freq_late = np.mean([struct[i, j] > 0 for struct in structures[half_point:]])

            if freq_overall >= 0.8:
                stable_edges.append((i, j, freq_overall))

            if freq_late > freq_early + 0.3 and freq_late >= 0.5:
                emerging_edges.append((i, j, freq_early, freq_late))

            if freq_early > freq_late + 0.3 and freq_early >= 0.5:
                disappearing_edges.append((i, j, freq_early, freq_late))

    return {
        'structures': structures,
        'edge_frequencies': edge_freq_matrix,
        'n_edges_per_gen': np.array(n_edges_per_gen),
        'stable_edges': stable_edges,
        'emerging_edges': emerging_edges,
        'disappearing_edges': disappearing_edges,
        'n_generations': len(structures),
        'n_variables': n_vars
    }


def extract_gaussian_parameters_evolution(
    cache: Any,
    selected_generations: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Extract and analyze Gaussian parameters across generations.

    For continuous EDAs (FDA, EMNA-global, etc.), extracts means and
    covariances to understand how the algorithm focuses on promising regions.

    Parameters
    ----------
    cache : Cache object
        EDA cache containing models.
    selected_generations : list of int, optional
        Generations to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - 'means': List of mean vectors per generation
        - 'covariances': List of covariance matrices per generation
        - 'mean_trajectory': Evolution of means over generations
        - 'variance_reduction': How variances decrease over time
        - 'correlation_evolution': How correlations change

    Examples
    --------
    >>> gaussian_evolution = extract_gaussian_parameters_evolution(cache)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(gaussian_evolution['variance_reduction'])
    >>> plt.title('Variance Reduction Over Generations')
    >>> plt.show()
    """
    if not hasattr(cache, 'models') or len(cache.models) == 0:
        raise ValueError("Cache does not contain models")

    models = cache.models

    if selected_generations is None:
        selected_generations = list(range(len(models)))

    means = []
    covariances = []
    stds = []

    for gen_idx in selected_generations:
        if gen_idx >= len(models):
            continue

        model = models[gen_idx]

        # Extract mean
        if hasattr(model, 'mean'):
            mean = model.mean
        elif hasattr(model, 'mu'):
            mean = model.mu
        else:
            mean = None

        # Extract covariance/std
        if hasattr(model, 'covariance'):
            cov = model.covariance
        elif hasattr(model, 'cov'):
            cov = model.cov
        elif hasattr(model, 'sigma'):
            if model.sigma.ndim == 1:
                # Diagonal covariance
                cov = np.diag(model.sigma ** 2)
            else:
                cov = model.sigma
        else:
            cov = None

        if mean is not None:
            means.append(mean.copy())

        if cov is not None:
            covariances.append(cov.copy())
            stds.append(np.sqrt(np.diag(cov)))

    if len(means) == 0:
        raise ValueError("No Gaussian parameters found in models")

    # Analyze evolution
    mean_trajectory = np.array(means)
    n_gens, n_vars = mean_trajectory.shape

    # Variance reduction: mean of diagonal variances over generations
    variance_reduction = []
    for std in stds:
        variance_reduction.append(np.mean(std ** 2))

    variance_reduction = np.array(variance_reduction)

    # Correlation evolution (if covariances available)
    correlation_evolution = []
    if len(covariances) > 0:
        for cov in covariances:
            # Convert to correlation matrix
            std_dev = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std_dev, std_dev)
            # Mean absolute correlation
            mean_abs_corr = np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)]))
            correlation_evolution.append(mean_abs_corr)

    return {
        'means': means,
        'covariances': covariances,
        'stds': stds,
        'mean_trajectory': mean_trajectory,
        'variance_reduction': variance_reduction,
        'correlation_evolution': np.array(correlation_evolution) if correlation_evolution else None,
        'n_generations': n_gens,
        'n_variables': n_vars
    }


def extract_probability_distribution_evolution(
    cache: Any,
    selected_generations: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Extract probability distributions for discrete univariate EDAs.

    For UMDA, PBIL, and other univariate discrete EDAs, extracts the
    marginal probability distributions to see how they converge.

    Parameters
    ----------
    cache : Cache object
        EDA cache containing models.
    selected_generations : list of int, optional
        Generations to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - 'probabilities': List of probability matrices per generation
        - 'entropy_per_var': Entropy of each variable over generations
        - 'convergence_speed': Rate of entropy decrease
        - 'converged_variables': Variables that have converged (entropy < threshold)

    Examples
    --------
    >>> prob_evolution = extract_probability_distribution_evolution(cache)
    >>> print(f"Converged variables: {prob_evolution['converged_variables'][-1]}")
    """
    if not hasattr(cache, 'models') or len(cache.models) == 0:
        raise ValueError("Cache does not contain models")

    models = cache.models

    if selected_generations is None:
        selected_generations = list(range(len(models)))

    probabilities = []

    for gen_idx in selected_generations:
        if gen_idx >= len(models):
            continue

        model = models[gen_idx]

        # Extract probabilities
        if hasattr(model, 'probabilities'):
            probs = model.probabilities
        elif hasattr(model, 'p'):
            probs = model.p
        elif hasattr(model, 'marginals'):
            probs = model.marginals
        else:
            continue

        if probs is not None:
            probabilities.append(probs.copy())

    if len(probabilities) == 0:
        raise ValueError("No probability distributions found")

    n_gens = len(probabilities)
    n_vars = probabilities[0].shape[0]

    # Compute entropy per variable over generations
    entropy_per_var = np.zeros((n_gens, n_vars))

    for gen_idx, probs in enumerate(probabilities):
        for var_idx in range(n_vars):
            var_probs = probs[var_idx, :]
            # Remove zeros to avoid log(0)
            var_probs = var_probs[var_probs > 0]
            entropy = -np.sum(var_probs * np.log2(var_probs))
            entropy_per_var[gen_idx, var_idx] = entropy

    # Convergence analysis
    convergence_threshold = 0.1  # Low entropy threshold
    converged_variables = []

    for gen_idx in range(n_gens):
        converged_at_gen = np.where(entropy_per_var[gen_idx, :] < convergence_threshold)[0]
        converged_variables.append(converged_at_gen.tolist())

    # Convergence speed (entropy decrease rate)
    convergence_speed = np.zeros(n_vars)
    if n_gens > 1:
        for var_idx in range(n_vars):
            # Linear fit of entropy over generations
            gens = np.arange(n_gens)
            entropies = entropy_per_var[:, var_idx]
            if len(gens) > 1:
                slope = np.polyfit(gens, entropies, 1)[0]
                convergence_speed[var_idx] = -slope  # Positive means decreasing

    return {
        'probabilities': probabilities,
        'entropy_per_var': entropy_per_var,
        'convergence_speed': convergence_speed,
        'converged_variables': converged_variables,
        'n_generations': n_gens,
        'n_variables': n_vars
    }


def generate_comprehensive_report(
    cache: Any,
    statistics: Any,
    eda_type: str = 'auto'
) -> Dict[str, Any]:
    """
    Generate comprehensive knowledge extraction report for an EDA run.

    Automatically detects EDA type and extracts all relevant knowledge,
    including fitness evolution, selection dynamics, model evolution, and
    population diversity.

    Parameters
    ----------
    cache : Cache object
        EDA cache from run.
    statistics : Statistics object
        EDA statistics from run.
    eda_type : str, default='auto'
        Type of EDA: 'auto', 'discrete_univariate', 'bayesian_network',
        'gaussian', 'copula', etc.

    Returns
    -------
    dict
        Comprehensive report with all extracted knowledge organized by category.

    Examples
    --------
    >>> stats, cache = eda.run()
    >>> report = generate_comprehensive_report(cache, stats)
    >>> print(report['summary'])

    Notes
    -----
    - Provides one-stop knowledge extraction for any EDA
    - Useful for post-run analysis and comparison
    - Can be extended with custom extractors
    """
    report = {
        'metadata': {
            'eda_type': eda_type,
            'n_generations': len(cache.populations) if hasattr(cache, 'populations') else 0,
            'pop_size': cache.populations[0].shape[0] if hasattr(cache, 'populations') and len(cache.populations) > 0 else None,
            'n_vars': cache.populations[0].shape[1] if hasattr(cache, 'populations') and len(cache.populations) > 0 else None
        },
        'fitness_evolution': {},
        'selection_dynamics': {},
        'model_evolution': {},
        'population_diversity': {},
        'summary': {}
    }

    # Fitness evolution
    if hasattr(statistics, 'best_fitness'):
        report['fitness_evolution'] = {
            'best_fitness': np.array(statistics.best_fitness),
            'mean_fitness': np.array(statistics.mean_fitness),
            'std_fitness': np.array(statistics.std_fitness),
            'final_best': statistics.best_fitness_overall,
            'generation_found': statistics.generation_found
        }

    # Selection dynamics (if we have population data)
    if hasattr(cache, 'populations') and hasattr(cache, 'fitness_values'):
        if len(cache.populations) > 1:
            # Compute response to selection between first and last generation
            initial_fitness = cache.fitness_values[0]
            final_fitness = cache.fitness_values[-1]

            response = response_to_selection(initial_fitness, final_fitness)
            report['selection_dynamics']['overall_response'] = response

    # Model evolution (detect type and extract accordingly)
    model_evolution = {}

    if eda_type == 'auto' and hasattr(cache, 'models') and len(cache.models) > 0:
        # Try to auto-detect
        first_model = cache.models[0]

        if hasattr(first_model, 'adjacency_matrix') or hasattr(first_model, 'structure'):
            eda_type = 'bayesian_network'
        elif hasattr(first_model, 'mean') or hasattr(first_model, 'mu'):
            eda_type = 'gaussian'
        elif hasattr(first_model, 'probabilities') or hasattr(first_model, 'p'):
            eda_type = 'discrete_univariate'

    try:
        if eda_type == 'bayesian_network':
            model_evolution = extract_bayesian_network_evolution(cache)
        elif eda_type == 'gaussian':
            model_evolution = extract_gaussian_parameters_evolution(cache)
        elif eda_type == 'discrete_univariate':
            model_evolution = extract_probability_distribution_evolution(cache)
    except Exception as e:
        warnings.warn(f"Could not extract model evolution: {e}")

    report['model_evolution'] = model_evolution

    # Population diversity
    if hasattr(cache, 'populations'):
        diversity_over_time = []
        for pop in cache.populations:
            unique_individuals = len(np.unique(pop, axis=0))
            diversity_ratio = unique_individuals / pop.shape[0]
            diversity_over_time.append(diversity_ratio)

        report['population_diversity'] = {
            'diversity_over_time': np.array(diversity_over_time),
            'final_diversity': diversity_over_time[-1] if diversity_over_time else None
        }

    # Generate summary
    summary_text = f"EDA Run Summary\n{'='*50}\n"
    summary_text += f"EDA Type: {eda_type}\n"
    summary_text += f"Generations: {report['metadata']['n_generations']}\n"
    summary_text += f"Population Size: {report['metadata']['pop_size']}\n"
    summary_text += f"Variables: {report['metadata']['n_vars']}\n"

    if report['fitness_evolution']:
        summary_text += f"\nBest Fitness: {report['fitness_evolution']['final_best']}\n"
        summary_text += f"Found at Generation: {report['fitness_evolution']['generation_found']}\n"

    if report['population_diversity']:
        summary_text += f"\nFinal Diversity: {report['population_diversity']['final_diversity']:.2%}\n"

    if model_evolution:
        if 'stable_edges' in model_evolution:
            summary_text += f"\nStable Dependencies: {len(model_evolution['stable_edges'])}\n"
        if 'variance_reduction' in model_evolution:
            initial_var = model_evolution['variance_reduction'][0]
            final_var = model_evolution['variance_reduction'][-1]
            summary_text += f"\nVariance Reduction: {initial_var:.4f} → {final_var:.4f}\n"

    report['summary'] = summary_text

    return report


def compare_eda_runs(
    reports: List[Dict[str, Any]],
    metric: str = 'best_fitness'
) -> Dict[str, Any]:
    """
    Compare multiple EDA runs using extracted knowledge.

    Parameters
    ----------
    reports : list of dict
        List of comprehensive reports from different EDA runs.
    metric : str
        Primary metric for comparison: 'best_fitness', 'convergence_speed',
        'diversity', etc.

    Returns
    -------
    dict
        Comparison results with rankings and statistical analysis.

    Examples
    --------
    >>> reports = [generate_comprehensive_report(cache1, stats1),
    ...           generate_comprehensive_report(cache2, stats2)]
    >>> comparison = compare_eda_runs(reports)
    >>> print(comparison['rankings'])
    """
    n_runs = len(reports)

    if n_runs == 0:
        raise ValueError("No reports provided")

    # Extract metrics for comparison
    best_fitnesses = []
    convergence_gens = []
    final_diversities = []

    for report in reports:
        if 'fitness_evolution' in report and report['fitness_evolution']:
            best_fitnesses.append(report['fitness_evolution']['final_best'])
            convergence_gens.append(report['fitness_evolution']['generation_found'])

        if 'population_diversity' in report and report['population_diversity']:
            final_diversities.append(report['population_diversity']['final_diversity'])

    # Rankings
    if best_fitnesses:
        best_fitnesses = np.array(best_fitnesses)
        fitness_rankings = np.argsort(-best_fitnesses)  # Descending order
    else:
        fitness_rankings = None

    return {
        'n_runs': n_runs,
        'best_fitnesses': best_fitnesses if best_fitnesses else None,
        'convergence_generations': convergence_gens if convergence_gens else None,
        'final_diversities': final_diversities if final_diversities else None,
        'fitness_rankings': fitness_rankings,
        'best_run': int(fitness_rankings[0]) if fitness_rankings is not None else None
    }
