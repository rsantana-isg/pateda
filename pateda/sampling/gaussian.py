"""
Gaussian Model Sampling for Continuous EDAs

This module provides sampling algorithms for various Gaussian-based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


def sample_gaussian_univariate(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a univariate Gaussian model.

    Parameters
    ----------
    model : dict
        Model containing 'means' and 'stds'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters (not used)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    means = model['means']
    stds = model['stds']
    n_vars = len(means)

    # Sample from normal distribution
    population = np.random.normal(
        loc=np.tile(means, (n_samples, 1)),
        scale=np.tile(stds, (n_samples, 1))
    )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_gaussian_full(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a full multivariate Gaussian model.

    Parameters
    ----------
    model : dict
        Model containing 'mean' and 'cov'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'var_scaling': scaling factor for covariance (default: 1.0)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    mean = model['mean']
    cov = model['cov']

    # Apply variance scaling if provided
    if params is not None:
        var_scaling = params.get('var_scaling', 1.0)
        cov = cov * var_scaling

    # Sample from multivariate normal distribution
    population = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_mixture_gaussian_univariate(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a mixture of univariate Gaussian models.

    Parameters
    ----------
    model : dict
        Model containing 'components' and 'n_clusters'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters (not used)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    components = model['components']
    n_clusters = model['n_clusters']

    # Extract component weights
    weights = np.array([comp['weight'] for comp in components])
    weights = weights / weights.sum()  # Normalize

    # Sample component assignments
    component_assignments = np.random.choice(
        n_clusters,
        size=n_samples,
        p=weights
    )

    # Get dimensions
    n_vars = len(components[0]['means'])
    population = np.zeros((n_samples, n_vars))

    # Sample from each component
    for i in range(n_clusters):
        mask = component_assignments == i
        n_comp_samples = np.sum(mask)

        if n_comp_samples > 0:
            means = components[i]['means']
            stds = components[i]['stds']

            population[mask] = np.random.normal(
                loc=np.tile(means, (n_comp_samples, 1)),
                scale=np.tile(stds, (n_comp_samples, 1))
            )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_mixture_gaussian_full(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a mixture of full multivariate Gaussian models.

    Parameters
    ----------
    model : dict
        Model containing 'components' and 'n_clusters'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'var_scaling': scaling factor for covariance (default: 1.0)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    components = model['components']
    n_clusters = model['n_clusters']

    # Extract component weights
    weights = np.array([comp['weight'] for comp in components])
    weights = weights / weights.sum()  # Normalize

    # Sample component assignments
    component_assignments = np.random.choice(
        n_clusters,
        size=n_samples,
        p=weights
    )

    # Get dimensions
    n_vars = len(components[0]['mean'])
    population = np.zeros((n_samples, n_vars))

    # Apply variance scaling if provided
    var_scaling = 1.0
    if params is not None:
        var_scaling = params.get('var_scaling', 1.0)

    # Sample from each component
    for i in range(n_clusters):
        mask = component_assignments == i
        n_comp_samples = np.sum(mask)

        if n_comp_samples > 0:
            mean = components[i]['mean']
            cov = components[i]['cov'] * var_scaling

            population[mask] = np.random.multivariate_normal(
                mean, cov, size=n_comp_samples
            )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_weighted_gaussian_univariate(
def sample_gmrf_eda(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a weighted univariate Gaussian model.

    Parameters
    ----------
    model : dict
        Model containing 'means' and 'stds'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters (not used)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    means = model['means']
    stds = model['stds']
    n_vars = len(means)

    # Sample from normal distribution
    population = np.random.normal(
        loc=np.tile(means, (n_samples, 1)),
        scale=np.tile(stds, (n_samples, 1))
    )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_weighted_gaussian_full(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a weighted full multivariate Gaussian model.
    Sample from a GMRF-EDA (Gaussian Markov Random Field) model.

    Samples independently from each clique's multivariate Gaussian distribution
    and combines them to form complete solution vectors.

    Parameters
    ----------
    model : dict
        Model containing 'mean' and 'cov'
        Model containing 'cliques' and 'clique_models'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'var_scaling': scaling factor for covariance (default: 1.0)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    mean = model['mean']
    cov = model['cov']

    # Apply variance scaling if provided
    if params is not None:
        var_scaling = params.get('var_scaling', 1.0)
        cov = cov * var_scaling

    # Sample from multivariate normal distribution
    population = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_gaussian_with_diversity_trigger(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a Gaussian model with diversity-triggered variance expansion.

    Automatically increases variance when diversity drops below a threshold
    to prevent premature convergence.

    Parameters
    ----------
    model : dict
        Model containing 'mean' and 'cov' (for full) or 'means' and 'stds' (for univariate)
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'diversity_threshold': threshold for triggering variance expansion (default: -1, disabled)
        - 'diversity_scaling': scaling factor when diversity is low (default: 2.0)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    diversity_threshold = params.get('diversity_threshold', -1)
    diversity_scaling = params.get('diversity_scaling', 2.0)

    model_type = model.get('type', '')

    # Check if diversity trigger is enabled
    if diversity_threshold > 0:
        if 'cov' in model:
            # Full covariance model
            mean = model['mean']
            cov = model['cov']
            mean_std = np.mean(np.sqrt(np.diag(cov)))

            # Trigger diversity expansion if mean std is below threshold
            if mean_std < diversity_threshold:
                cov = (1 + mean_std) * diversity_scaling * cov

            population = np.random.multivariate_normal(mean, cov, size=n_samples)
        else:
            # Univariate model
            means = model['means']
            stds = model['stds']
            mean_std = np.mean(stds)

            # Trigger diversity expansion if mean std is below threshold
            if mean_std < diversity_threshold:
                stds = (1 + mean_std) * diversity_scaling * stds

            population = np.random.normal(
                loc=np.tile(means, (n_samples, 1)),
                scale=np.tile(stds, (n_samples, 1))
            )
    else:
        # No diversity trigger, use standard sampling
        if 'cov' in model:
            population = np.random.multivariate_normal(model['mean'], model['cov'], size=n_samples)
        else:
            population = np.random.normal(
                loc=np.tile(model['means'], (n_samples, 1)),
                scale=np.tile(model['stds'], (n_samples, 1))
            )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_mixture_gaussian_em(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a Gaussian mixture model learned via EM.

    Parameters
    ----------
    model : dict
        Model containing 'gm_model' (sklearn GaussianMixture object)
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters (not used)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    gm_model = model['gm_model']

    # Use sklearn's built-in sampling method
    population, _ = gm_model.sample(n_samples)
    cliques = model['cliques']
    clique_models = model['clique_models']

    # Determine total number of variables
    n_vars = sum(len(clique) for clique in cliques)

    # Initialize population
    population = np.zeros((n_samples, n_vars))

    # Apply variance scaling if provided
    var_scaling = 1.0
    if params is not None:
        var_scaling = params.get('var_scaling', 1.0)

    # Sample from each clique independently
    for clique, clique_model in zip(cliques, clique_models):
        mean = clique_model['mean']
        cov = clique_model['cov'] * var_scaling

        if len(clique) == 1:
            # Univariate case
            population[:, clique[0]] = np.random.normal(
                loc=mean[0],
                scale=np.sqrt(cov[0, 0]),
                size=n_samples
            )
        else:
            # Multivariate case
            clique_samples = np.random.multivariate_normal(
                mean, cov, size=n_samples
            )
            population[:, clique] = clique_samples

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population
