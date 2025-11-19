"""
Basic Gaussian Model Learning for Continuous EDAs

This module provides learning algorithms for basic Gaussian-based probabilistic
models used in continuous optimization.

Types of Gaussian Models:
1. Univariate Gaussian (Gaussian UMDA):
   - Each variable is independent with its own mean and variance
   - Joint distribution: p(x) = ∏ᵢ N(xᵢ | μᵢ, σᵢ²)
   - Simplest model, assumes no variable dependencies
   - Equivalent to continuous UMDA

2. Full Multivariate Gaussian:
   - Models all pairwise correlations via full covariance matrix
   - Joint distribution: p(x) = N(x | μ, Σ)
   - Can capture all linear dependencies
   - Requires O(n²) parameters for n variables

References:
- Larrañaga, P., & Lozano, J.A. (Eds.). (2002). "Estimation of Distribution
  Algorithms: A New Tool for Evolutionary Computation." Kluwer Academic Publishers.
- MATEDA-2.0 User Guide, Section 4.2.3: "Gaussian network based factorizations"
"""

import numpy as np
from typing import Dict, Any, Optional


def learn_gaussian_univariate(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a univariate Gaussian model (independent variables).

    This corresponds to Gaussian UMDA, where each variable is modeled
    independently with its own mean and standard deviation.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in this learning method)
    params : dict, optional
        Additional parameters (not used for this method)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'means': array of means for each variable
        - 'stds': array of standard deviations for each variable
        - 'type': 'gaussian_univariate'
    """
    means = np.mean(population, axis=0)
    stds = np.std(population, axis=0)

    # Prevent zero standard deviation
    stds = np.maximum(stds, 1e-10)

    return {
        'means': means,
        'stds': stds,
        'type': 'gaussian_univariate'
    }


def learn_gaussian_full(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a full multivariate Gaussian model.

    Models dependencies between all variables using a full covariance matrix.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in this learning method)
    params : dict, optional
        Additional parameters (not used for this method)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'mean': mean vector
        - 'cov': covariance matrix
        - 'type': 'gaussian_full'
    """
    mean = np.mean(population, axis=0)
    cov = np.cov(population, rowvar=False)

    # Ensure positive definiteness by adding small regularization
    n_vars = population.shape[1]
    cov += np.eye(n_vars) * 1e-6

    return {
        'mean': mean,
        'cov': cov,
        'type': 'gaussian_full'
    }
