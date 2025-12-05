"""
Benchmark script for evaluating dendiff model quality across different distributions.

This script evaluates the ability of learn_dendiff and sample_dendiff functions
to accurately approximate various probability distributions and generates a
comprehensive report of the results.

Usage:
    python benchmark_dendiff_distributions.py
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, Tuple, Callable, List
import sys
import os

# Add pateda parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff


# ============================================================================
# Objective Functions for Fitness-Based Distributions
# ============================================================================

def sphere_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Sphere function (minimization).
    Global minimum: f(shift, ..., shift) = 0

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from origin

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    return np.sum(x**2, axis=1)


def rastrigin_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Rastrigin function (minimization).
    Global minimum: f(shift, ..., shift) = 0
    Highly multimodal with many local optima.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from origin

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    n_vars = x.shape[1]
    return 10 * n_vars + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def rosenbrock_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Rosenbrock function (minimization).
    Global minimum: f(1+shift, ..., 1+shift) = 0
    Valley-shaped, difficult to optimize.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from (1, ..., 1)

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def ackley_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Ackley function (minimization).
    Global minimum: f(shift, ..., shift) = 0
    Highly multimodal with nearly flat outer region.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from origin

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    n_vars = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n_vars))
    term2 = -np.exp(sum_cos / n_vars)

    return term1 + term2 + 20 + np.e


def griewank_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Griewank function (minimization).
    Global minimum: f(shift, ..., shift) = 0
    Many local optima with strong interactions between variables.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from origin

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    n_vars = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)

    prod_term = np.ones(x.shape[0])
    for i in range(n_vars):
        prod_term *= np.cos(x[:, i] / np.sqrt(i + 1))

    return sum_sq / 4000 - prod_term + 1


def schwefel_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Schwefel function (minimization).
    Global minimum: f(420.9687+shift, ..., 420.9687+shift) = 0
    Deceptive with many local optima far from global optimum.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from (420.9687, ...)

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    n_vars = x.shape[1]
    return 418.9829 * n_vars - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def ellipsoid_function(x: np.ndarray, shift: np.ndarray = None) -> np.ndarray:
    """
    Ellipsoid function (minimization).
    Global minimum: f(shift, ..., shift) = 0
    Unimodal, axis-parallel, with different sensitivities along axes.

    Parameters
    ----------
    x : np.ndarray
        Solutions of shape (n_samples, n_vars)
    shift : np.ndarray, optional
        Translation vector to shift the optimum away from origin

    Returns
    -------
    fitness : np.ndarray
        Fitness values (lower is better)
    """
    if shift is not None:
        x = x - shift
    n_vars = x.shape[1]
    weights = np.arange(1, n_vars + 1)
    return np.sum(weights * x**2, axis=1)


# Dictionary of objective functions with their bounds
OBJECTIVE_FUNCTIONS = {
    'sphere': {
        'function': sphere_function,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'description': 'Unimodal, smooth, separable',
        'shift_range': (-2.0, 2.0)  # Range for random shift
    },
    'rastrigin': {
        'function': rastrigin_function,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'description': 'Highly multimodal, many local optima',
        'shift_range': (-2.0, 2.0)
    },
    'rosenbrock': {
        'function': rosenbrock_function,
        'bounds': (-2.048, 2.048),
        'optimum': 0.0,
        'description': 'Valley-shaped, non-separable',
        'shift_range': (-1.0, 1.0)
    },
    'ackley': {
        'function': ackley_function,
        'bounds': (-32.768, 32.768),
        'optimum': 0.0,
        'description': 'Multimodal with nearly flat outer region',
        'shift_range': (-10.0, 10.0)
    },
    'griewank': {
        'function': griewank_function,
        'bounds': (-600, 600),
        'optimum': 0.0,
        'description': 'Many local optima, product term creates interaction',
        'shift_range': (-200.0, 200.0)
    },
    'schwefel': {
        'function': schwefel_function,
        'bounds': (-500, 500),
        'optimum': 0.0,
        'description': 'Deceptive, many local optima',
        'shift_range': (-100.0, 100.0)
    },
    'ellipsoid': {
        'function': ellipsoid_function,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'description': 'Unimodal, axis-parallel, ill-conditioned',
        'shift_range': (-2.0, 2.0)
    }
}


# ============================================================================
# Empirical Fitness Distribution Generators
# ============================================================================

def generate_empirical_fitness_distribution(
    objective_name: str,
    n_initial: int = 1000,
    n_selected: int = 500,
    n_vars: int = 10,
    seed: int = 42,
    use_shift: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate empirical fitness distribution by sampling and selecting top solutions.

    This simulates the selection step in an EDA where we:
    1. Generate n_initial random solutions
    2. Evaluate them using the objective function (with optional shift)
    3. Select the top n_selected solutions with best fitness

    Parameters
    ----------
    objective_name : str
        Name of the objective function (e.g., 'sphere', 'rastrigin')
    n_initial : int
        Number of initial random solutions to generate
    n_selected : int
        Number of top solutions to select
    n_vars : int
        Number of variables
    seed : int
        Random seed
    use_shift : bool
        Whether to shift the optimum away from origin (default: True)

    Returns
    -------
    selected_samples : np.ndarray
        Selected solutions of shape (n_selected, n_vars)
    selected_fitness : np.ndarray
        Fitness values of selected solutions
    metadata : dict
        Distribution metadata including shift information
    """
    if objective_name not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Unknown objective function: {objective_name}")

    np.random.seed(seed)

    obj_info = OBJECTIVE_FUNCTIONS[objective_name]
    obj_function = obj_info['function']
    bounds = obj_info['bounds']

    # Generate random shift if requested
    shift = None
    if use_shift and 'shift_range' in obj_info:
        shift_range = obj_info['shift_range']
        shift = np.random.uniform(shift_range[0], shift_range[1], n_vars)

    # Generate initial random population within bounds
    initial_population = np.random.uniform(
        bounds[0], bounds[1],
        (n_initial, n_vars)
    )

    # Evaluate fitness (with shift if applicable)
    fitness = obj_function(initial_population, shift=shift)

    # Select top n_selected solutions (lower fitness is better)
    sorted_indices = np.argsort(fitness)
    selected_indices = sorted_indices[:n_selected]

    selected_samples = initial_population[selected_indices]
    selected_fitness = fitness[selected_indices]

    metadata = {
        'type': f'Empirical Fitness ({objective_name})',
        'objective_name': objective_name,
        'objective_description': obj_info['description'],
        'n_initial': n_initial,
        'n_selected': n_selected,
        'selection_ratio': n_selected / n_initial,
        'bounds': bounds,
        'optimum': obj_info['optimum'],
        'shift': shift,
        'shifted_optimum': shift if shift is not None else np.zeros(n_vars),
        'best_fitness': selected_fitness[0],
        'worst_fitness': selected_fitness[-1],
        'mean_fitness': np.mean(selected_fitness),
        'std_fitness': np.std(selected_fitness)
    }

    return selected_samples, selected_fitness, metadata


def generate_boltzmann_distribution(
    objective_name: str,
    n_initial: int = 1000,
    n_selected: int = 500,
    n_vars: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
    use_shift: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate distribution using Boltzmann selection based on fitness.

    In Boltzmann selection, solutions are selected with probability proportional
    to exp(-fitness / temperature). Better fitness (lower value) gives higher
    probability. This promotes both exploitation of good solutions and
    exploration of the fitness landscape.

    Parameters
    ----------
    objective_name : str
        Name of the objective function
    n_initial : int
        Number of initial random solutions to generate
    n_selected : int
        Number of solutions to select using Boltzmann probabilities
    n_vars : int
        Number of variables
    temperature : float
        Temperature parameter for Boltzmann distribution (higher = more exploration)
    seed : int
        Random seed
    use_shift : bool
        Whether to shift the optimum away from origin

    Returns
    -------
    selected_samples : np.ndarray
        Selected solutions of shape (n_selected, n_vars)
    selected_fitness : np.ndarray
        Fitness values of selected solutions
    metadata : dict
        Distribution metadata
    """
    if objective_name not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Unknown objective function: {objective_name}")

    np.random.seed(seed)

    obj_info = OBJECTIVE_FUNCTIONS[objective_name]
    obj_function = obj_info['function']
    bounds = obj_info['bounds']

    # Generate random shift if requested
    shift = None
    if use_shift and 'shift_range' in obj_info:
        shift_range = obj_info['shift_range']
        shift = np.random.uniform(shift_range[0], shift_range[1], n_vars)

    # Generate initial random population
    initial_population = np.random.uniform(
        bounds[0], bounds[1],
        (n_initial, n_vars)
    )

    # Evaluate fitness
    fitness = obj_function(initial_population, shift=shift)

    # Normalize fitness to avoid numerical issues
    # For minimization, lower fitness should have higher probability
    # Use negative fitness in exponential
    fitness_normalized = fitness - np.min(fitness)

    # Compute Boltzmann probabilities
    boltzmann_weights = np.exp(-fitness_normalized / temperature)
    probabilities = boltzmann_weights / np.sum(boltzmann_weights)

    # Count non-zero probabilities to ensure we can sample without replacement
    non_zero_count = np.sum(probabilities > 0)
    if non_zero_count < n_selected:
        # If not enough non-zero probabilities, adjust n_selected
        print(f"    Warning: Only {non_zero_count} non-zero probabilities, adjusting n_selected from {n_selected} to {non_zero_count}")
        n_selected_actual = non_zero_count
    else:
        n_selected_actual = n_selected

    # Select solutions according to Boltzmann probabilities
    selected_indices = np.random.choice(
        n_initial,
        size=n_selected_actual,
        replace=False,
        p=probabilities
    )

    selected_samples = initial_population[selected_indices]
    selected_fitness = fitness[selected_indices]

    # Sort by fitness for consistency with other methods
    sorted_indices = np.argsort(selected_fitness)
    selected_samples = selected_samples[sorted_indices]
    selected_fitness = selected_fitness[sorted_indices]

    metadata = {
        'type': f'Boltzmann Distribution ({objective_name})',
        'objective_name': objective_name,
        'objective_description': obj_info['description'],
        'n_initial': n_initial,
        'n_selected': n_selected_actual,
        'selection_ratio': n_selected_actual / n_initial,
        'temperature': temperature,
        'bounds': bounds,
        'optimum': obj_info['optimum'],
        'shift': shift,
        'shifted_optimum': shift if shift is not None else np.zeros(n_vars),
        'best_fitness': selected_fitness[0],
        'worst_fitness': selected_fitness[-1],
        'mean_fitness': np.mean(selected_fitness),
        'std_fitness': np.std(selected_fitness)
    }

    return selected_samples, selected_fitness, metadata


def generate_rank_based_distribution(
    objective_name: str,
    n_initial: int = 1000,
    n_selected: int = 500,
    n_vars: int = 10,
    selection_pressure: float = 2.0,
    seed: int = 42,
    use_shift: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate distribution using rank-based selection (similar to CMA-ES).

    In rank-based selection, solutions are ranked by fitness and selection
    probability depends only on rank, not the actual fitness values. This
    makes the method robust to fitness scaling issues.

    The probability of selecting a solution with rank r (1 = best) is
    proportional to: (selection_pressure - 2*selection_pressure*(r-1)/(n-1))

    Parameters
    ----------
    objective_name : str
        Name of the objective function
    n_initial : int
        Number of initial random solutions to generate
    n_selected : int
        Number of solutions to select using rank-based probabilities
    n_vars : int
        Number of variables
    selection_pressure : float
        Selection pressure parameter (typically 2.0)
        Higher values favor better-ranked solutions more strongly
    seed : int
        Random seed
    use_shift : bool
        Whether to shift the optimum away from origin

    Returns
    -------
    selected_samples : np.ndarray
        Selected solutions of shape (n_selected, n_vars)
    selected_fitness : np.ndarray
        Fitness values of selected solutions
    metadata : dict
        Distribution metadata
    """
    if objective_name not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Unknown objective function: {objective_name}")

    np.random.seed(seed)

    obj_info = OBJECTIVE_FUNCTIONS[objective_name]
    obj_function = obj_info['function']
    bounds = obj_info['bounds']

    # Generate random shift if requested
    shift = None
    if use_shift and 'shift_range' in obj_info:
        shift_range = obj_info['shift_range']
        shift = np.random.uniform(shift_range[0], shift_range[1], n_vars)

    # Generate initial random population
    initial_population = np.random.uniform(
        bounds[0], bounds[1],
        (n_initial, n_vars)
    )

    # Evaluate fitness
    fitness = obj_function(initial_population, shift=shift)

    # Rank solutions (1 = best, n_initial = worst)
    sorted_indices = np.argsort(fitness)
    ranks = np.empty(n_initial, dtype=int)
    ranks[sorted_indices] = np.arange(1, n_initial + 1)

    # Compute rank-based selection probabilities (linear ranking)
    # p(rank) = (selection_pressure - 2*selection_pressure*(rank-1)/(n-1)) / n
    rank_weights = selection_pressure - 2 * selection_pressure * (ranks - 1) / (n_initial - 1)
    rank_weights = np.maximum(rank_weights, 0)  # Ensure non-negative
    probabilities = rank_weights / np.sum(rank_weights)

    # Count non-zero probabilities to ensure we can sample without replacement
    non_zero_count = np.sum(probabilities > 0)
    if non_zero_count < n_selected:
        # If not enough non-zero probabilities, adjust n_selected
        print(f"    Warning: Only {non_zero_count} non-zero probabilities, adjusting n_selected from {n_selected} to {non_zero_count}")
        n_selected_actual = non_zero_count
    else:
        n_selected_actual = n_selected

    # Select solutions according to rank-based probabilities
    selected_indices = np.random.choice(
        n_initial,
        size=n_selected_actual,
        replace=False,
        p=probabilities
    )

    selected_samples = initial_population[selected_indices]
    selected_fitness = fitness[selected_indices]

    # Sort by fitness for consistency
    sorted_indices = np.argsort(selected_fitness)
    selected_samples = selected_samples[sorted_indices]
    selected_fitness = selected_fitness[sorted_indices]

    metadata = {
        'type': f'Rank-based Distribution ({objective_name})',
        'objective_name': objective_name,
        'objective_description': obj_info['description'],
        'n_initial': n_initial,
        'n_selected': n_selected_actual,
        'selection_ratio': n_selected_actual / n_initial,
        'selection_pressure': selection_pressure,
        'bounds': bounds,
        'optimum': obj_info['optimum'],
        'shift': shift,
        'shifted_optimum': shift if shift is not None else np.zeros(n_vars),
        'best_fitness': selected_fitness[0],
        'worst_fitness': selected_fitness[-1],
        'mean_fitness': np.mean(selected_fitness),
        'std_fitness': np.std(selected_fitness)
    }

    return selected_samples, selected_fitness, metadata


# ============================================================================
# Distribution Generators
# ============================================================================

def generate_univariate_gaussian(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent univariate Gaussian distributions."""
    np.random.seed(seed)
    means = np.random.uniform(-2, 2, n_vars)
    stds = np.random.uniform(0.5, 2.0, n_vars)
    samples = np.random.randn(n_samples, n_vars) * stds + means

    metadata = {
        'type': 'Univariate Gaussian',
        'means': means,
        'stds': stds
    }
    return samples, metadata


def generate_multivariate_gaussian(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from a multivariate Gaussian with predetermined eigenvectors."""
    np.random.seed(seed)
    mean = np.random.uniform(-1, 1, n_vars)
    Q, _ = np.linalg.qr(np.random.randn(n_vars, n_vars))
    eigenvalues = np.random.uniform(0.5, 3.0, n_vars)
    cov = Q @ np.diag(eigenvalues) @ Q.T
    samples = np.random.multivariate_normal(mean, cov, n_samples)

    metadata = {
        'type': 'Multivariate Gaussian',
        'mean': mean,
        'covariance': cov,
        'eigenvalues': eigenvalues
    }
    return samples, metadata


def generate_multivariate_cauchy(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent Cauchy distributions."""
    np.random.seed(seed)
    locations = np.random.uniform(-1, 1, n_vars)
    scales = np.random.uniform(0.3, 1.0, n_vars)

    samples = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        samples[:, i] = stats.cauchy.rvs(
            loc=locations[i],
            scale=scales[i],
            size=n_samples,
            random_state=seed + i
        )
    samples = np.clip(samples, -10, 10)

    metadata = {
        'type': 'Cauchy',
        'locations': locations,
        'scales': scales
    }
    return samples, metadata


def generate_mixture_of_gaussians(
    n_samples: int,
    n_vars: int,
    n_components: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from a Gaussian mixture model."""
    np.random.seed(seed)
    weights = np.random.dirichlet(np.ones(n_components))

    means = []
    covs = []
    for i in range(n_components):
        mean = np.random.uniform(-3, 3, n_vars)
        means.append(mean)
        std = np.random.uniform(0.3, 1.5, n_vars)
        cov = np.diag(std ** 2)
        covs.append(cov)

    samples = []
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    for i in range(n_samples):
        comp = component_assignments[i]
        sample = np.random.multivariate_normal(means[comp], covs[comp])
        samples.append(sample)
    samples = np.array(samples)

    metadata = {
        'type': 'Gaussian Mixture',
        'n_components': n_components,
        'weights': weights,
        'means': means
    }
    return samples, metadata


def generate_uniform(
    n_samples: int,
    n_vars: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate samples from independent uniform distributions."""
    np.random.seed(seed)
    low = np.random.uniform(-3, 0, n_vars)
    high = np.random.uniform(1, 4, n_vars)
    samples = np.random.uniform(low, high, (n_samples, n_vars))

    metadata = {
        'type': 'Uniform',
        'low': low,
        'high': high
    }
    return samples, metadata


# ============================================================================
# Distribution Comparison Metrics
# ============================================================================

def compute_js_divergence(
    samples1: np.ndarray,
    samples2: np.ndarray,
    n_bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence (symmetric version of KL)."""
    n_vars = samples1.shape[1]
    js_divs = []

    for i in range(n_vars):
        data1 = samples1[:, i]
        data2 = samples2[:, i]

        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)

        js = jensenshannon(hist1, hist2, base=2)
        js_divs.append(js)

    return np.mean(js_divs)


def compute_kl_divergence_kde(
    samples1: np.ndarray,
    samples2: np.ndarray,
    n_bins: int = 50
) -> float:
    """Estimate KL divergence between two sample sets using histograms."""
    n_vars = samples1.shape[1]
    kl_divs = []

    for i in range(n_vars):
        data1 = samples1[:, i]
        data2 = samples2[:, i]

        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        kl = np.sum(hist1 * np.log(hist1 / hist2))
        kl_divs.append(kl)

    return np.mean(kl_divs)


def compute_statistical_distance(
    samples1: np.ndarray,
    samples2: np.ndarray
) -> Dict[str, float]:
    """Compute various statistical distance metrics between two sample sets."""
    # Mean difference
    mean1 = np.mean(samples1, axis=0)
    mean2 = np.mean(samples2, axis=0)
    mean_diff = np.linalg.norm(mean1 - mean2)
    mean_diff_relative = mean_diff / (np.linalg.norm(mean1) + 1e-10)

    # Std difference
    std1 = np.std(samples1, axis=0)
    std2 = np.std(samples2, axis=0)
    std_diff = np.linalg.norm(std1 - std2)
    std_diff_relative = std_diff / (np.linalg.norm(std1) + 1e-10)

    # Covariance difference
    cov1 = np.cov(samples1.T)
    cov2 = np.cov(samples2.T)
    cov_diff = np.linalg.norm(cov1 - cov2, 'fro')
    cov_diff_relative = cov_diff / (np.linalg.norm(cov1, 'fro') + 1e-10)

    # Wasserstein distance
    n_vars = samples1.shape[1]
    wasserstein_dists = []
    for i in range(n_vars):
        wd = stats.wasserstein_distance(samples1[:, i], samples2[:, i])
        wasserstein_dists.append(wd)
    mean_wasserstein = np.mean(wasserstein_dists)

    return {
        'mean_diff': mean_diff,
        'mean_diff_relative': mean_diff_relative,
        'std_diff': std_diff,
        'std_diff_relative': std_diff_relative,
        'cov_diff_frobenius': cov_diff,
        'cov_diff_relative': cov_diff_relative,
        'mean_wasserstein': mean_wasserstein
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_dendiff_on_distribution(
    generator: Callable,
    n_samples: int = 500,
    n_vars: int = 10,
    n_samples_test: int = 500,
    params: Dict[str, Any] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate dendiff model on a given distribution.

    Parameters
    ----------
    generator : callable
        Function that generates (samples, metadata)
    n_samples : int
        Number of training samples
    n_vars : int
        Number of variables
    n_samples_test : int
        Number of samples to generate from learned model
    params : dict
        Parameters for learn_dendiff
    seed : int
        Random seed
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        Evaluation results including metrics
    """
    if verbose:
        print(f"  Generating {n_samples} samples from distribution...")

    # Generate original distribution
    original_samples, metadata = generator(n_samples, n_vars, seed)

    # Create associated fitness values (random for this benchmark)
    np.random.seed(seed)
    fitness = np.random.randn(n_samples)

    # Set default params if not provided
    if params is None:
        params = {}

    if verbose:
        print(f"  Learning dendiff model...")

    # Learn dendiff model
    model = learn_dendiff(original_samples, fitness, params=params)

    if verbose:
        print(f"  Sampling {n_samples_test} samples from learned model...")

    # Sample from learned model
    bounds = np.array([
        original_samples.min(axis=0) - 3 * original_samples.std(axis=0),
        original_samples.max(axis=0) + 3 * original_samples.std(axis=0)
    ])

    sampled_data = sample_dendiff(
        model,
        n_samples=n_samples_test,
        bounds=bounds,
        params=params
    )

    if verbose:
        print(f"  Computing metrics...")

    # Compute metrics
    kl_div = compute_kl_divergence_kde(original_samples, sampled_data)
    js_div = compute_js_divergence(original_samples, sampled_data)
    stat_distances = compute_statistical_distance(original_samples, sampled_data)

    results = {
        'distribution_type': metadata['type'],
        'n_vars': n_vars,
        'n_samples': n_samples,
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        **stat_distances
    }

    return results


def evaluate_dendiff_on_fitness_distribution(
    objective_name: str,
    n_initial: int = 1000,
    n_selected: int = 500,
    n_vars: int = 10,
    n_samples_test: int = 500,
    params: Dict[str, Any] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate dendiff model on empirical fitness distribution.

    This follows the complete workflow:
    1. Generate MAT: Sample n_initial solutions, select top n_selected (empirical fitness dist)
    2. Learn dendiff model on MAT
    3. Sample from learned model
    4. Generate MAT_ANOTHER: Independent sample from same empirical fitness dist
    5. Compare sampled distribution with MAT_ANOTHER

    Parameters
    ----------
    objective_name : str
        Name of objective function ('sphere', 'rastrigin', etc.)
    n_initial : int
        Number of initial random solutions to generate
    n_selected : int
        Number of top solutions to select
    n_vars : int
        Number of variables
    n_samples_test : int
        Number of samples to generate from learned model
    params : dict
        Parameters for learn_dendiff
    seed : int
        Random seed for MAT generation
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        Evaluation results including metrics and fitness statistics
    """
    if verbose:
        print(f"  Generating MAT: {n_initial} solutions, selecting top {n_selected}...")

    # Step 1: Generate MAT (empirical fitness distribution)
    MAT, MAT_fitness, metadata = generate_empirical_fitness_distribution(
        objective_name, n_initial, n_selected, n_vars, seed
    )

    if verbose:
        print(f"    Best fitness in MAT: {metadata['best_fitness']:.6f}")
        print(f"    Worst fitness in MAT: {metadata['worst_fitness']:.6f}")
        print(f"    Mean fitness in MAT: {metadata['mean_fitness']:.6f}")

    # Set default params if not provided
    if params is None:
        params = {}

    if verbose:
        print(f"  Learning dendiff model on MAT...")

    # Step 2: Learn dendiff model
    model = learn_dendiff(MAT, MAT_fitness, params=params)

    if verbose:
        print(f"  Sampling {n_samples_test} solutions from learned model...")

    # Step 3: Sample from learned model
    bounds_info = OBJECTIVE_FUNCTIONS[objective_name]
    bounds = np.array([
        [bounds_info['bounds'][0]] * n_vars,
        [bounds_info['bounds'][1]] * n_vars
    ])

    Sampled_MAT = sample_dendiff(
        model,
        n_samples=n_samples_test,
        bounds=bounds,
        params=params
    )

    # Evaluate fitness of sampled solutions
    obj_function = bounds_info['function']
    Sampled_MAT_fitness = obj_function(Sampled_MAT)

    if verbose:
        print(f"  Generating MAT_ANOTHER: independent sample from same fitness distribution...")

    # Step 4: Generate MAT_ANOTHER (another independent sample)
    # Use different seed to get independent sample
    MAT_ANOTHER, MAT_ANOTHER_fitness, metadata_another = generate_empirical_fitness_distribution(
        objective_name, n_initial, n_selected, n_vars, seed + 1000
    )

    if verbose:
        print(f"  Computing distribution comparison metrics...")

    # Step 5: Compare distributions
    # Compare Sampled_MAT with MAT_ANOTHER (both should represent the fitness distribution)
    kl_div = compute_kl_divergence_kde(MAT_ANOTHER, Sampled_MAT)
    js_div = compute_js_divergence(MAT_ANOTHER, Sampled_MAT)
    stat_distances = compute_statistical_distance(MAT_ANOTHER, Sampled_MAT)

    # Also compare with original MAT for reference
    kl_div_vs_mat = compute_kl_divergence_kde(MAT, Sampled_MAT)
    js_div_vs_mat = compute_js_divergence(MAT, Sampled_MAT)

    if verbose:
        print(f"  Computing fitness statistics...")

    # Fitness statistics comparison
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
    }

    results = {
        'distribution_type': metadata['type'],
        'objective_name': objective_name,
        'objective_description': metadata['objective_description'],
        'n_vars': n_vars,
        'n_initial': n_initial,
        'n_selected': n_selected,
        'selection_ratio': metadata['selection_ratio'],
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'kl_divergence_vs_mat': kl_div_vs_mat,
        'js_divergence_vs_mat': js_div_vs_mat,
        **stat_distances,
        **fitness_stats
    }

    return results


# ============================================================================
# Main Benchmark Execution
# ============================================================================

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all distributions."""

    print("=" * 80)
    print("DENDIFF DISTRIBUTION APPROXIMATION BENCHMARK")
    print("=" * 80)
    print()

    # Configuration
    n_vars = 10
    n_samples = 500
    n_samples_test = 500

    # Test configurations: (name, generator, params)
    test_configs = [
        (
            "1. Univariate Gaussian",
            generate_univariate_gaussian,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "2. Multivariate Gaussian (with correlations)",
            generate_multivariate_gaussian,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "3. Cauchy Distribution (heavy tails)",
            generate_multivariate_cauchy,
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [128, 64]}
        ),
        (
            "4. Gaussian Mixture (3 components)",
            generate_mixture_of_gaussians,
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [256, 128]}
        ),
        (
            "5. Uniform Distribution",
            generate_uniform,
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
    ]

    all_results = []

    for test_name, generator, params in test_configs:
        print(f"\n{test_name}")
        print("-" * 80)

        try:
            results = evaluate_dendiff_on_distribution(
                generator,
                n_samples=n_samples,
                n_vars=n_vars,
                n_samples_test=n_samples_test,
                params=params,
                seed=42,
                verbose=True
            )

            all_results.append((test_name, results))

            print(f"\n  Results:")
            print(f"    KL Divergence:           {results['kl_divergence']:.6f}")
            print(f"    JS Divergence:           {results['js_divergence']:.6f}")
            print(f"    Mean Difference:         {results['mean_diff']:.6f}")
            print(f"    Std Difference:          {results['std_diff']:.6f}")
            print(f"    Mean Wasserstein Dist:   {results['mean_wasserstein']:.6f}")
            print(f"    Relative Mean Diff:      {results['mean_diff_relative']:.4f}")
            print(f"    Relative Std Diff:       {results['std_diff_relative']:.4f}")
            print(f"    Relative Cov Diff:       {results['cov_diff_relative']:.4f}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Distribution':<45} {'JS Div':<12} {'KL Div':<12} {'Wasserstein':<12}")
    print("-" * 80)

    for test_name, results in all_results:
        dist_name = test_name.split('. ', 1)[1] if '. ' in test_name else test_name
        print(f"{dist_name:<45} {results['js_divergence']:>10.4f}  "
              f"{results['kl_divergence']:>10.4f}  "
              f"{results['mean_wasserstein']:>10.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
JS Divergence (Jensen-Shannon):
  - Range: [0, 1] (base 2), where 0 = identical distributions
  - < 0.1: Excellent approximation
  - 0.1-0.3: Good approximation
  - 0.3-0.5: Moderate approximation
  - > 0.5: Poor approximation

KL Divergence (Kullback-Leibler):
  - Range: [0, ∞), where 0 = identical distributions
  - < 0.1: Excellent approximation
  - 0.1-0.5: Good approximation
  - 0.5-1.0: Moderate approximation
  - > 1.0: Poor approximation

Wasserstein Distance:
  - Range: [0, ∞), where 0 = identical distributions
  - Lower is better
  - Depends on scale of the data

Notes:
  - Dendiff models are based on Denoising Diffusion Probabilistic Models
  - Performance depends on: sample size, model capacity, training epochs, and timesteps
  - Gaussian distributions are typically easier to approximate than heavy-tailed or mixture distributions
  - Correlations in multivariate distributions add complexity
""")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
For better approximation quality:
  1. Increase sample size (n_samples > 500)
  2. Increase number of training epochs (epochs > 100)
  3. Increase diffusion timesteps (n_timesteps > 500)
  4. Use larger network architectures (hidden_dims = [256, 128, 64])
  5. For complex distributions (mixtures, heavy tails):
     - Use more training epochs (150-200)
     - Use more timesteps (800-1000)
     - Consider using cosine schedule: beta_schedule='cosine'

Parameter trade-offs:
  - More timesteps = better quality but slower sampling
  - Larger networks = better capacity but slower training
  - More epochs = better fit but risk of overfitting with small samples
""")

    return all_results


# ============================================================================
# Parameter Variation Tests
# ============================================================================

def test_parameter_variations():
    """Test how different parameters affect approximation quality."""

    print("\n\n" + "=" * 80)
    print("PARAMETER VARIATION TESTS")
    print("=" * 80)

    # Test 1: Varying timesteps
    print("\n\nTest 1: Effect of Number of Diffusion Timesteps")
    print("-" * 80)
    print(f"{'Timesteps':<15} {'JS Divergence':<15} {'KL Divergence':<15} {'Wasserstein':<15}")
    print("-" * 80)

    for n_timesteps in [100, 300, 500, 1000]:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_gaussian,
            n_samples=300,
            n_vars=20,
            n_samples_test=300,
            params={'epochs': 80, 'n_timesteps': n_timesteps, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{n_timesteps:<15} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f} {results['mean_wasserstein']:<15.4f}")

    # Test 2: Varying architectures
    print("\n\nTest 2: Effect of Network Architecture")
    print("-" * 80)
    print(f"{'Architecture':<25} {'JS Divergence':<15} {'KL Divergence':<15}")
    print("-" * 80)

    #architectures = [
    #    ([64, 32], "[64, 32]"),
    #    ([128, 64], "[128, 64]"),
    #    ([256, 128], "[256, 128]"),
    #    ([256, 128, 64], "[256, 128, 64]"),
    #]

    architectures = [
        ([16, 16], "[16, 16]"),
        ([32, 32], "[32, 32]"),
        ([100, 100], "[100, 100]"),
        ([64, 32, 64], "[64, 32, 64]"),
    ]

    for hidden_dims, arch_str in architectures:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_cauchy,
            #generate_multivariate_gaussian,
            n_samples=400,
            n_vars=10,
            n_samples_test=400,
            params={'epochs': 80, 'n_timesteps': 500, 'hidden_dims': hidden_dims},
            seed=42,
            verbose=False
        )
        print(f"{arch_str:<25} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f}")

    # Test 3: Varying epochs
    print("\n\nTest 3: Effect of Training Epochs")
    print("-" * 80)
    print(f"{'Epochs':<15} {'JS Divergence':<15} {'KL Divergence':<15}")
    print("-" * 80)

    for epochs in [20, 50, 100, 150]:
        results = evaluate_dendiff_on_distribution(
            generate_multivariate_gaussian,
            n_samples=400,
            n_vars=10,
            n_samples_test=400,
            params={'epochs': epochs, 'n_timesteps': 500, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{epochs:<15} {results['js_divergence']:<15.4f} "
              f"{results['kl_divergence']:<15.4f}")


def run_fitness_benchmark():
    """Run comprehensive benchmark on empirical fitness distributions."""

    print("\n\n" + "=" * 80)
    print("EMPIRICAL FITNESS DISTRIBUTION BENCHMARK")
    print("=" * 80)
    print()
    print("This benchmark evaluates dendiff on distributions obtained from")
    print("selecting top solutions based on objective function fitness.")
    print()

    # Configuration
    n_vars = 10
    n_initial = 1000
    n_selected = 500
    n_samples_test = 500

    # Test configurations: (objective_name, params)
    test_configs = [
        (
            "sphere",
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "ellipsoid",
            {'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]}
        ),
        (
            "rastrigin",
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [256, 128]}
        ),
        (
            "rosenbrock",
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [256, 128]}
        ),
        (
            "ackley",
            {'epochs': 150, 'n_timesteps': 800, 'hidden_dims': [256, 128]}
        ),
    ]

    all_results = []

    for objective_name, params in test_configs:
        obj_info = OBJECTIVE_FUNCTIONS[objective_name]
        print(f"\n{objective_name.upper()} Function")
        print("-" * 80)
        print(f"Description: {obj_info['description']}")
        print(f"Bounds: {obj_info['bounds']}")
        print(f"Optimum: {obj_info['optimum']}")
        print()

        try:
            results = evaluate_dendiff_on_fitness_distribution(
                objective_name,
                n_initial=n_initial,
                n_selected=n_selected,
                n_vars=n_vars,
                n_samples_test=n_samples_test,
                params=params,
                seed=42,
                verbose=True
            )

            all_results.append((objective_name, results))

            print(f"\n  Distribution Comparison Results:")
            print(f"    JS Divergence (vs MAT_ANOTHER):  {results['js_divergence']:.6f}")
            print(f"    KL Divergence (vs MAT_ANOTHER):  {results['kl_divergence']:.6f}")
            print(f"    JS Divergence (vs MAT):          {results['js_divergence_vs_mat']:.6f}")
            print(f"    Mean Wasserstein Dist:           {results['mean_wasserstein']:.6f}")
            print(f"    Relative Mean Diff:              {results['mean_diff_relative']:.4f}")
            print(f"    Relative Std Diff:               {results['std_diff_relative']:.4f}")

            print(f"\n  Fitness Statistics Comparison:")
            print(f"    MAT Mean Fitness:                {results['mat_mean_fitness']:.6f}")
            print(f"    Sampled Mean Fitness:            {results['sampled_mean_fitness']:.6f}")
            print(f"    MAT_ANOTHER Mean Fitness:        {results['mat_another_mean_fitness']:.6f}")
            print(f"    Fitness Mean Difference:         {results['fitness_mean_diff']:.6f}")
            print(f"    Fitness Std Difference:          {results['fitness_std_diff']:.6f}")

            print(f"\n  Best Fitness Values:")
            print(f"    MAT Best:                        {results['mat_best_fitness']:.6f}")
            print(f"    Sampled Best:                    {results['sampled_best_fitness']:.6f}")
            print(f"    MAT_ANOTHER Best:                {results['mat_another_best_fitness']:.6f}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("EMPIRICAL FITNESS DISTRIBUTION SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Objective':<15} {'JS Div':<12} {'Fitness Diff':<15} {'Best Fit (MAT)':<18} {'Best Fit (Sampled)':<18}")
    print("-" * 80)

    for objective_name, results in all_results:
        print(f"{objective_name:<15} {results['js_divergence']:>10.4f}  "
              f"{results['fitness_mean_diff']:>13.6f}  "
              f"{results['mat_best_fitness']:>16.6f}  "
              f"{results['sampled_best_fitness']:>16.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION FOR FITNESS-BASED DISTRIBUTIONS")
    print("=" * 80)
    print("""
Key Metrics:

1. JS Divergence (vs MAT_ANOTHER):
   - Compares sampled distribution with independent sample from same fitness distribution
   - This measures how well dendiff captured the empirical fitness distribution
   - Lower is better: < 0.3 = good, < 0.5 = acceptable

2. Fitness Mean Difference:
   - Difference in mean fitness between sampled and MAT_ANOTHER
   - Should be small if dendiff captured the fitness landscape well
   - Lower is better

3. Best Fitness Comparison:
   - MAT Best: Best solution in original selected set
   - Sampled Best: Best solution from dendiff sampling
   - If dendiff is good, sampled best should be similar to or better than MAT best

Expected Behavior:
  - Unimodal functions (sphere, ellipsoid): Should have low JS divergence (< 0.4)
  - Multimodal functions (rastrigin, ackley): May have higher JS divergence (< 0.6)
  - Non-separable functions (rosenbrock): May be more challenging

Quality Indicators:
  ✓ JS divergence < 0.5
  ✓ Fitness mean difference < 10% of MAT mean fitness
  ✓ Sampled best fitness similar to or better than MAT best
  ✓ Similar fitness distributions between sampled and MAT_ANOTHER
""")

    return all_results


def test_fitness_parameter_variations():
    """Test parameter variations on fitness-based distributions."""

    print("\n\n" + "=" * 80)
    print("FITNESS DISTRIBUTION PARAMETER VARIATION TESTS")
    print("=" * 80)

    # Test on sphere function with different parameters
    objective_name = "sphere"
    n_vars = 10
    n_initial = 1000
    n_selected = 500

    # Test 1: Varying selection ratio
    print("\n\nTest 1: Effect of Selection Ratio on Sphere Function")
    print("-" * 80)
    print(f"{'Selection Ratio':<20} {'n_selected':<15} {'JS Divergence':<15} {'Fitness Diff':<15}")
    print("-" * 80)

    for ratio in [0.3, 0.5, 0.7]:
        n_sel = int(n_initial * ratio)
        results = evaluate_dendiff_on_fitness_distribution(
            objective_name,
            n_initial=n_initial,
            n_selected=n_sel,
            n_vars=n_vars,
            n_samples_test=n_sel,
            params={'epochs': 80, 'n_timesteps': 500, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{ratio:<20.2f} {n_sel:<15} {results['js_divergence']:<15.4f} {results['fitness_mean_diff']:<15.6f}")

    # Test 2: Varying timesteps on multimodal function
    print("\n\nTest 2: Effect of Timesteps on Rastrigin Function")
    print("-" * 80)
    print(f"{'Timesteps':<15} {'JS Divergence':<15} {'Fitness Diff':<15}")
    print("-" * 80)

    for n_timesteps in [300, 500, 800, 1000]:
        results = evaluate_dendiff_on_fitness_distribution(
            "rastrigin",
            n_initial=1000,
            n_selected=500,
            n_vars=10,
            n_samples_test=500,
            params={'epochs': 100, 'n_timesteps': n_timesteps, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        print(f"{n_timesteps:<15} {results['js_divergence']:<15.4f} {results['fitness_mean_diff']:<15.6f}")

    # Test 3: Compare different objective functions with same parameters
    print("\n\nTest 3: Comparison Across Different Objective Functions")
    print("-" * 80)
    print(f"{'Objective':<15} {'Type':<30} {'JS Div':<12} {'Fitness Diff':<15}")
    print("-" * 80)

    objectives_to_test = ['sphere', 'ellipsoid', 'rastrigin', 'rosenbrock', 'ackley']
    for obj_name in objectives_to_test:
        results = evaluate_dendiff_on_fitness_distribution(
            obj_name,
            n_initial=1000,
            n_selected=500,
            n_vars=10,
            n_samples_test=500,
            params={'epochs': 100, 'n_timesteps': 500, 'hidden_dims': [128, 64]},
            seed=42,
            verbose=False
        )
        obj_desc = OBJECTIVE_FUNCTIONS[obj_name]['description']
        print(f"{obj_name:<15} {obj_desc:<30} {results['js_divergence']:>10.4f}  {results['fitness_mean_diff']:>13.6f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import time

    start_time = time.time()

    print("=" * 80)
    print("DENDIFF COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print()
    print("This suite includes:")
    print("  1. Standard probability distributions (Gaussian, Cauchy, mixtures, etc.)")
    print("  2. Empirical fitness distributions (from optimization objectives)")
    print("  3. Parameter variation experiments")
    print()

    # Run comprehensive benchmark on standard distributions
    print("\n" + "=" * 80)
    print("PART 1: STANDARD PROBABILITY DISTRIBUTIONS")
    print("=" * 80)
    results = run_comprehensive_benchmark()

    # Run parameter variation tests on standard distributions
    test_parameter_variations()

    # Run fitness-based distribution benchmarks
    print("\n" + "=" * 80)
    print("PART 2: EMPIRICAL FITNESS DISTRIBUTIONS")
    print("=" * 80)
    fitness_results = run_fitness_benchmark()

    # Run parameter variation tests on fitness distributions
    test_fitness_parameter_variations()

    elapsed_time = time.time() - start_time

    print("\n\n" + "=" * 80)
    print("COMPLETE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nTotal benchmark time: {elapsed_time:.2f} seconds")
    print(f"\nStandard distributions tested: {len(results)}")
    print(f"Fitness distributions tested: {len(fitness_results)}")
    print("\nAll benchmarks completed successfully!")
    print("=" * 80)
