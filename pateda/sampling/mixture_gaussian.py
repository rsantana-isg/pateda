"""
Mixture Gaussian Model Sampling for Continuous EDAs

This module provides sampling algorithms for mixture of Gaussian models,
which can model multimodal distributions.
"""

import numpy as np
from typing import Dict, Any, Optional

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MixtureModel


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

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


# ===================================================================
# Class-based wrappers for component architecture
# ===================================================================


class SampleMixtureGaussian(SamplingMethod):
    """
    Class-based wrapper for mixture of Gaussians sampling.

    Samples from a mixture of Gaussian distributions. First selects
    a component based on mixture weights, then samples from that component.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    var_scaling : float
        Scaling factor for covariance (only for full covariance), default: 1.0
    """

    def __init__(self, n_samples: int, var_scaling: float = 1.0):
        """
        Initialize mixture Gaussian sampling

        Args:
            n_samples: Number of samples to generate
            var_scaling: Scaling factor for covariance matrices
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
        **params: Any,
    ) -> np.ndarray:
        """
        Sample from mixture of Gaussians model

        Args:
            n_vars: Number of variables
            model: MixtureModel with Gaussian components
            cardinality: Variable bounds (2, n_vars) or (n_vars, 2) array
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness values (not used)
            **params: Additional parameters
                      - 'var_scaling': override instance var_scaling

        Returns:
            Sampled population of shape (n_samples, n_vars)
        """
        # Extract parameters from model
        if isinstance(model, MixtureModel):
            components = model.parameters['components']
            model_type = model.parameters['type']
        else:
            # Fallback to dict-style model
            components = model['components']
            model_type = model['type']

        # Determine bounds from cardinality
        if cardinality.shape[0] == 2:
            bounds = cardinality  # [lower, upper]
        else:
            bounds = cardinality.T  # Transpose to get [lower, upper]

        # Create params dict for sampling functions
        sampling_params = {'var_scaling': params.get('var_scaling', self.var_scaling)}

        # Use the appropriate functional sampling method
        model_dict = {
            'components': components,
            'n_clusters': len(components),
            'type': model_type
        }

        if 'univariate' in model_type:
            population = sample_mixture_gaussian_univariate(
                model_dict, self.n_samples, bounds, sampling_params
            )
        else:
            population = sample_mixture_gaussian_full(
                model_dict, self.n_samples, bounds, sampling_params
            )

        return population
