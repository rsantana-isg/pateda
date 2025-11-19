"""
Basic Gaussian Model Sampling for Continuous EDAs

This module provides sampling algorithms for basic Gaussian-based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional


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
