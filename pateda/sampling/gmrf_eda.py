"""
Gaussian Markov Random Field (GMRF) EDA Sampling

This module provides sampling for GMRF-EDA models learned using
regularized estimation and clique-based factorization.
"""

import numpy as np
from typing import Dict, Any, Optional


def sample_gmrf_eda(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a GMRF-EDA (Gaussian Markov Random Field) model.

    Samples independently from each clique's multivariate Gaussian distribution
    and combines them to form complete solution vectors.

    Parameters
    ----------
    model : dict
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
