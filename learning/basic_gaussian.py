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

from pateda.core.components import LearningMethod
from pateda.core.models import GaussianModel


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


# ===================================================================
# Class-based wrappers for component architecture
# ===================================================================


class LearnGaussianUnivariate(LearningMethod):
    """
    Class-based wrapper for univariate Gaussian learning.

    This corresponds to Gaussian UMDA for continuous optimization,
    where each variable is modeled independently with its own mean
    and standard deviation.

    Parameters
    ----------
    alpha : float
        Minimum standard deviation to prevent collapse (default: 1e-10)
    """

    def __init__(self, alpha: float = 1e-10):
        """
        Initialize Gaussian UMDA learning

        Args:
            alpha: Minimum standard deviation to prevent collapse
        """
        self.alpha = alpha

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> GaussianModel:
        """
        Learn univariate Gaussian model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable bounds (2, n_vars) array with [lower, upper] bounds
            population: Selected population to learn from
            fitness: Fitness values (not used for this learning method)
            **params: Additional parameters

        Returns:
            Learned GaussianModel with univariate structure
        """
        means = np.mean(population, axis=0)
        stds = np.std(population, axis=0)

        # Prevent zero standard deviation
        stds = np.maximum(stds, self.alpha)

        return GaussianModel(
            structure=None,  # No dependencies for univariate model
            parameters={
                'means': means,
                'stds': stds,
                'type': 'gaussian_univariate'
            },
            metadata={
                'generation': generation,
                'model_type': 'Gaussian UMDA',
            }
        )


class LearnGaussianFull(LearningMethod):
    """
    Class-based wrapper for full multivariate Gaussian learning.

    Models dependencies between all variables using a full covariance matrix.

    Parameters
    ----------
    regularization : float
        Regularization term added to diagonal for positive definiteness
        (default: 1e-6)
    """

    def __init__(self, regularization: float = 1e-6):
        """
        Initialize full Gaussian learning

        Args:
            regularization: Diagonal regularization for covariance matrix
        """
        self.regularization = regularization

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> GaussianModel:
        """
        Learn full multivariate Gaussian model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable bounds (2, n_vars) array with [lower, upper] bounds
            population: Selected population to learn from
            fitness: Fitness values (not used for this learning method)
            **params: Additional parameters

        Returns:
            Learned GaussianModel with full covariance structure
        """
        mean = np.mean(population, axis=0)
        cov = np.cov(population, rowvar=False)

        # Ensure positive definiteness by adding small regularization
        cov += np.eye(n_vars) * self.regularization

        return GaussianModel(
            structure=None,  # Full covariance has implicit all-to-all structure
            parameters={
                'mean': mean,
                'cov': cov,
                'type': 'gaussian_full'
            },
            metadata={
                'generation': generation,
                'model_type': 'Full Gaussian',
            }
        )
