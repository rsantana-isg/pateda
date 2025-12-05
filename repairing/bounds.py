"""
Bounds-based repairing for continuous optimization problems.

This module provides constraint repair mechanisms for continuous solutions
where variables must be within specified bounds. Two strategies are available:
clipping to bounds or replacement with random values within bounds.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Optional


def set_in_bounds_repairing(
    pop: np.ndarray,
    range_values: np.ndarray,
    repairing_params: Optional[dict] = None
) -> np.ndarray:
    """
    Repair continuous solutions by clipping out-of-bounds values to the nearest bound.

    For a problem with continuous representation, this function modifies solutions
    by setting variables that are below (above) the constraints to the minimum
    (respectively maximum) bounds. This is a simple clipping operation.

    Parameters
    ----------
    pop : np.ndarray
        Population matrix of shape (n_individuals, n_variables) with continuous values.
    range_values : np.ndarray
        Range of values for each variable, shape (2, n_variables) where:
        - range_values[0, :]: Lower bounds for each variable
        - range_values[1, :]: Upper bounds for each variable
    repairing_params : Optional[dict], default=None
        Not used for this function. Included for API consistency.

    Returns
    -------
    np.ndarray
        Repaired population with the same shape as input, with all values
        clipped to their respective bounds.

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.array([[-1.0, 5.0, 3.0],
    ...                  [2.0, -0.5, 11.0]])
    >>> range_values = np.array([[0.0, 0.0, 0.0],
    ...                           [4.0, 4.0, 10.0]])
    >>> repaired = set_in_bounds_repairing(pop, range_values)
    >>> # Values will be clipped: [[0.0, 4.0, 3.0], [2.0, 0.0, 10.0]]

    Notes
    -----
    - Uses numpy clipping for efficient boundary enforcement
    - Deterministic operation (no randomness)
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    n_individuals, n_variables = pop.shape
    new_pop = pop.copy()

    for i in range(n_variables):
        # Clip values below the lower bound
        under_val = pop[:, i] < range_values[0, i]
        new_pop[under_val, i] = range_values[0, i]

        # Clip values above the upper bound
        over_val = pop[:, i] > range_values[1, i]
        new_pop[over_val, i] = range_values[1, i]

    return new_pop


def set_within_bounds_repairing(
    pop: np.ndarray,
    range_values: np.ndarray,
    repairing_params: Optional[dict] = None
) -> np.ndarray:
    """
    Repair continuous solutions by replacing out-of-bounds values with random values.

    For a problem with continuous representation, this function modifies solutions
    by replacing variables that are below or above the constraints with random
    values uniformly sampled within the valid range. This introduces more diversity
    compared to simple clipping.

    Parameters
    ----------
    pop : np.ndarray
        Population matrix of shape (n_individuals, n_variables) with continuous values.
    range_values : np.ndarray
        Range of values for each variable, shape (2, n_variables) where:
        - range_values[0, :]: Lower bounds for each variable
        - range_values[1, :]: Upper bounds for each variable
    repairing_params : Optional[dict], default=None
        Not used for this function. Included for API consistency.

    Returns
    -------
    np.ndarray
        Repaired population with the same shape as input, with out-of-bounds
        values replaced by random values within the valid range.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> pop = np.array([[-1.0, 5.0, 3.0],
    ...                  [2.0, -0.5, 11.0]])
    >>> range_values = np.array([[0.0, 0.0, 0.0],
    ...                           [4.0, 4.0, 10.0]])
    >>> repaired = set_within_bounds_repairing(pop, range_values)
    >>> # Out-of-bounds values replaced with random values in valid range

    Notes
    -----
    - Uses uniform random sampling for replacement values
    - Introduces more diversity than clipping
    - Non-deterministic operation
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    n_individuals, n_variables = pop.shape
    new_pop = pop.copy()

    for i in range(n_variables):
        # Find and replace values below the lower bound
        under_val = pop[:, i] < range_values[0, i]
        n_under = np.sum(under_val)
        if n_under > 0:
            new_pop[under_val, i] = (
                range_values[0, i]
                + np.random.rand(n_under) * (range_values[1, i] - range_values[0, i])
            )

        # Find and replace values above the upper bound
        over_val = pop[:, i] > range_values[1, i]
        n_over = np.sum(over_val)
        if n_over > 0:
            new_pop[over_val, i] = (
                range_values[0, i]
                + np.random.rand(n_over) * (range_values[1, i] - range_values[0, i])
            )

    return new_pop
