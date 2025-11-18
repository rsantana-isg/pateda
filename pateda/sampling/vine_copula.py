"""
Vine Copula Model Sampling for Continuous EDAs

This module provides sampling algorithms for vine copula-based
probabilistic models used in continuous optimization.

The sampling process involves:
1. Sampling from the vine copula in uniform [0,1] space
2. Transforming samples back to the original variable space using bounds
3. Optionally applying bound constraints

References
----------
- Soto et al. (2011): "Vine Estimation of Distribution Algorithms with
  Application to Molecular Docking"
- pyvinecopulib documentation: https://vinecopulib.github.io/pyvinecopulib/
"""

import numpy as np
from typing import Dict, Any, Optional

try:
    import pyvinecopulib as pv
    PYVINECOPULIB_AVAILABLE = True
except ImportError:
    PYVINECOPULIB_AVAILABLE = False


def sample_vine_copula(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a vine copula model.

    This function works with models learned by any of the vine copula
    learning functions (C-vine, D-vine, or auto).

    Parameters
    ----------
    model : dict
        Model dictionary containing:
        - 'vine_model': pyvinecopulib Vinecop object
        - 'bounds': array of shape (2, n_vars) with [min, max] for rescaling
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Override bounds from model. Array of shape (2, n_vars) with [min, max]
        for each variable. If None, uses bounds from model.
    params : dict, optional
        Additional sampling parameters:
        - 'seeds': list of random seeds for reproducibility
        - 'use_inverse_rosenblatt': bool, use inverse Rosenblatt transform
          instead of simulation (default: False)
        - 'clip_bounds': bool, clip samples to bounds (default: True)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_auto
    >>> from pateda.sampling.vine_copula import sample_vine_copula
    >>> # Learn model
    >>> pop = np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]], size=100)
    >>> model = learn_vine_copula_auto(pop, np.sum(pop**2, axis=1))
    >>> # Sample from model
    >>> new_pop = sample_vine_copula(model, n_samples=50)
    >>> print(new_pop.shape)
    (50, 2)
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract vine model and bounds
    vine_model = model['vine_model']
    model_bounds = model.get('bounds')
    n_vars = model.get('n_vars', len(vine_model.order))

    # Use provided bounds or model bounds
    if bounds is None:
        if model_bounds is None:
            raise ValueError("No bounds available. Provide bounds parameter or ensure model contains bounds.")
        bounds = model_bounds

    # Extract sampling parameters
    seeds = params.get('seeds', None)
    use_inverse_rosenblatt = params.get('use_inverse_rosenblatt', False)
    clip_bounds_flag = params.get('clip_bounds', True)

    # Sample from vine copula in uniform [0,1] space
    if use_inverse_rosenblatt:
        # Use inverse Rosenblatt transform with user-provided uniform samples
        u = np.random.random((n_samples, n_vars))
        u_sim = vine_model.inverse_rosenblatt(u)
    else:
        # Direct simulation from vine copula
        if seeds is not None:
            u_sim = vine_model.simulate(n=n_samples, seeds=seeds)
        else:
            u_sim = vine_model.simulate(n=n_samples)

    # Transform from uniform [0,1] to original variable space
    population = np.copy(u_sim)
    for i in range(n_vars):
        population[:, i] = bounds[0, i] + (bounds[1, i] - bounds[0, i]) * u_sim[:, i]

    # Clip to bounds if requested
    if clip_bounds_flag:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_vine_copula_biased(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a vine copula model with bias towards better solutions.

    This function biases the sampling towards regions with better fitness
    by modifying the first variable's uniform distribution before applying
    the inverse Rosenblatt transform.

    Parameters
    ----------
    model : dict
        Model dictionary containing 'vine_model' and 'bounds'
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Override bounds from model
    params : dict, optional
        Additional sampling parameters:
        - 'exploit_factor': float, exploitation factor for biasing (default: 0.1)
        - 'clip_bounds': bool, clip samples to bounds (default: True)

    Returns
    -------
    population : np.ndarray
        Biased sampled population of shape (n_samples, n_vars)

    Notes
    -----
    This is an experimental feature based on the biased sampling idea
    from the enhanced_edas implementation. The first variable's uniform
    distribution is compressed to favor lower values, which can help
    with exploitation in optimization.

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_auto
    >>> from pateda.sampling.vine_copula import sample_vine_copula_biased
    >>> # Learn model
    >>> pop = np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]], size=100)
    >>> model = learn_vine_copula_auto(pop, np.sum(pop**2, axis=1))
    >>> # Sample with bias
    >>> new_pop = sample_vine_copula_biased(model, n_samples=50,
    ...                                      params={'exploit_factor': 0.1})
    >>> print(new_pop.shape)
    (50, 2)
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract vine model and bounds
    vine_model = model['vine_model']
    model_bounds = model.get('bounds')
    n_vars = model.get('n_vars', len(vine_model.order))

    # Use provided bounds or model bounds
    if bounds is None:
        if model_bounds is None:
            raise ValueError("No bounds available. Provide bounds parameter or ensure model contains bounds.")
        bounds = model_bounds

    # Extract sampling parameters
    exploit_factor = params.get('exploit_factor', 0.1)
    clip_bounds_flag = params.get('clip_bounds', True)

    # Create biased uniform samples
    u = np.random.random((n_samples, n_vars))
    # Bias the first variable towards lower values for exploitation
    u[:, 0] = u[:, 0] * exploit_factor

    # Apply inverse Rosenblatt transform
    u_sim = vine_model.inverse_rosenblatt(u)

    # Transform from uniform [0,1] to original variable space
    population = np.copy(u_sim)
    for i in range(n_vars):
        population[:, i] = bounds[0, i] + (bounds[1, i] - bounds[0, i]) * u_sim[:, i]

    # Clip to bounds if requested
    if clip_bounds_flag:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_vine_copula_conditional(
    model: Dict[str, Any],
    n_samples: int,
    fixed_vars: Dict[int, float],
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a vine copula model with some variables fixed.

    This function allows conditional sampling where certain variables
    are fixed to specific values and the remaining variables are sampled
    conditionally.

    Parameters
    ----------
    model : dict
        Model dictionary containing 'vine_model' and 'bounds'
    n_samples : int
        Number of samples to generate
    fixed_vars : dict
        Dictionary mapping variable indices (0-based) to fixed values
        Example: {0: 0.5, 2: 1.0} fixes variables 0 and 2
    bounds : np.ndarray, optional
        Override bounds from model
    params : dict, optional
        Additional sampling parameters:
        - 'clip_bounds': bool, clip samples to bounds (default: True)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars) with fixed variables

    Notes
    -----
    This uses the vine copula's conditional distribution properties.
    Fixed variables are set to their specified values, and remaining
    variables are sampled from the conditional distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_auto
    >>> from pateda.sampling.vine_copula import sample_vine_copula_conditional
    >>> # Learn model
    >>> pop = np.random.multivariate_normal([0,0,0], np.eye(3), size=100)
    >>> model = learn_vine_copula_auto(pop, np.sum(pop**2, axis=1))
    >>> # Sample with first variable fixed
    >>> new_pop = sample_vine_copula_conditional(model, n_samples=50,
    ...                                           fixed_vars={0: 0.5})
    >>> print(np.all(new_pop[:, 0] == 0.5))
    True
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract vine model and bounds
    vine_model = model['vine_model']
    model_bounds = model.get('bounds')
    n_vars = model.get('n_vars', len(vine_model.order))

    # Use provided bounds or model bounds
    if bounds is None:
        if model_bounds is None:
            raise ValueError("No bounds available. Provide bounds parameter or ensure model contains bounds.")
        bounds = model_bounds

    # Extract sampling parameters
    clip_bounds_flag = params.get('clip_bounds', True)

    # Sample normally first
    population = sample_vine_copula(model, n_samples, bounds, params)

    # Fix specified variables
    for var_idx, var_value in fixed_vars.items():
        if 0 <= var_idx < n_vars:
            population[:, var_idx] = var_value
        else:
            raise ValueError(f"Variable index {var_idx} out of range [0, {n_vars})")

    # Clip to bounds if requested
    if clip_bounds_flag:
        population = np.clip(population, bounds[0], bounds[1])

    return population
