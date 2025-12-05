"""
Basic Gaussian Model Sampling for Continuous EDAs

This module provides sampling algorithms for basic Gaussian-based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, GaussianModel


def sample_gaussian_univariate(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
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
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if rng is None:
        rng = np.random.default_rng()

    means = model['means']
    stds = model['stds']
    n_vars = len(means)

    # Sample from normal distribution
    population = rng.normal(
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
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
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
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if rng is None:
        rng = np.random.default_rng()

    mean = model['mean']
    cov = model['cov']

    # Apply variance scaling if provided
    if params is not None:
        var_scaling = params.get('var_scaling', 1.0)
        cov = cov * var_scaling

    # Sample from multivariate normal distribution
    population = rng.multivariate_normal(mean, cov, size=n_samples)

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_gaussian_with_diversity_trigger(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
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
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if rng is None:
        rng = np.random.default_rng()

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

            population = rng.multivariate_normal(mean, cov, size=n_samples)
        else:
            # Univariate model
            means = model['means']
            stds = model['stds']
            mean_std = np.mean(stds)

            # Trigger diversity expansion if mean std is below threshold
            if mean_std < diversity_threshold:
                stds = (1 + mean_std) * diversity_scaling * stds

            population = rng.normal(
                loc=np.tile(means, (n_samples, 1)),
                scale=np.tile(stds, (n_samples, 1))
            )
    else:
        # No diversity trigger, use standard sampling
        if 'cov' in model:
            population = rng.multivariate_normal(model['mean'], model['cov'], size=n_samples)
        else:
            population = rng.normal(
                loc=np.tile(model['means'], (n_samples, 1)),
                scale=np.tile(model['stds'], (n_samples, 1))
            )

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


# ===================================================================
# Class-based wrappers for component architecture
# ===================================================================


class SampleGaussianUnivariate(SamplingMethod):
    """
    Class-based wrapper for univariate Gaussian sampling.

    Samples from independent Gaussian distributions for each variable.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    """

    def __init__(self, n_samples: int):
        """
        Initialize Gaussian UMDA sampling

        Args:
            n_samples: Number of samples to generate
        """
        self.n_samples = n_samples

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample from univariate Gaussian model

        Args:
            n_vars: Number of variables
            model: GaussianModel with univariate structure
            cardinality: Variable bounds (2, n_vars) or (n_vars, 2) array
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness values (not used)
            rng: Random number generator (optional)
            **params: Additional parameters

        Returns:
            Sampled population of shape (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract parameters from model
        if isinstance(model, GaussianModel):
            means = model.parameters['means']
            stds = model.parameters['stds']
        else:
            # Fallback to dict-style model
            means = model['means']
            stds = model['stds']

        # Determine bounds from cardinality
        # cardinality can be (2, n_vars) or (n_vars, 2)
        if cardinality.shape[0] == 2:
            bounds = cardinality  # [lower, upper]
        else:
            bounds = cardinality.T  # Transpose to get [lower, upper]

        # Sample from normal distribution
        population = rng.normal(
            loc=np.tile(means, (self.n_samples, 1)),
            scale=np.tile(stds, (self.n_samples, 1))
        )

        # Apply bounds
        population = np.clip(population, bounds[0], bounds[1])

        return population


class SampleGaussianFull(SamplingMethod):
    """
    Class-based wrapper for full multivariate Gaussian sampling.

    Samples from a full multivariate Gaussian distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    var_scaling : float
        Scaling factor for covariance matrix (default: 1.0)
    """

    def __init__(self, n_samples: int, var_scaling: float = 1.0):
        """
        Initialize full Gaussian sampling

        Args:
            n_samples: Number of samples to generate
            var_scaling: Scaling factor for variance/covariance
        """
        self.n_samples = n_samples
        self.var_scaling = var_scaling

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample from full multivariate Gaussian model

        Args:
            n_vars: Number of variables
            model: GaussianModel with full covariance structure
            cardinality: Variable bounds (2, n_vars) or (n_vars, 2) array
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness values (not used)
            rng: Random number generator (optional)
            **params: Additional parameters
                      - 'var_scaling': override instance var_scaling

        Returns:
            Sampled population of shape (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract parameters from model
        if isinstance(model, GaussianModel):
            mean = model.parameters['mean']
            cov = model.parameters['cov']
        else:
            # Fallback to dict-style model
            mean = model['mean']
            cov = model['cov']

        # Apply variance scaling
        var_scaling = params.get('var_scaling', self.var_scaling)
        cov_scaled = cov * var_scaling

        # Determine bounds from cardinality
        if cardinality.shape[0] == 2:
            bounds = cardinality  # [lower, upper]
        else:
            bounds = cardinality.T  # Transpose to get [lower, upper]

        # Sample from multivariate normal distribution
        population = rng.multivariate_normal(mean, cov_scaled, size=self.n_samples)

        # Apply bounds
        population = np.clip(population, bounds[0], bounds[1])

        return population
