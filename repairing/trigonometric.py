"""
Trigonometric repairing for continuous optimization problems.

This module provides constraint repair mechanisms for continuous solutions
where variables should be within [0, 2π] using modular arithmetic.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Optional


def trigonometric_repairing(
    pop: np.ndarray,
    range_values: np.ndarray,
    repairing_params: Optional[dict] = None
) -> np.ndarray:
    """
    Repair continuous solutions to wrap values into [0, 2π] using modular arithmetic.

    For a problem with continuous representation where solutions should be in [0, 2π],
    this function modifies unfeasible solutions by wrapping them using the modulo
    operator. Values below the lower bound are wrapped by adding 2π to the remainder,
    while values above the upper bound are wrapped directly.

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
        wrapped into the valid range [0, 2π].

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.array([[-1.0, 7.0, 3.0],
    ...                  [2.0, -0.5, 8.0]])
    >>> range_values = np.array([[0.0, 0.0, 0.0],
    ...                           [2*np.pi, 2*np.pi, 2*np.pi]])
    >>> repaired = trigonometric_repairing(pop, range_values)
    >>> # All values will be wrapped into [0, 2π]

    Notes
    -----
    - Uses modular arithmetic (remainder operation) for wrapping
    - For values below the lower bound: rem(x, 2π) + 2π
    - For values above the upper bound: rem(x, 2π)
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    n_individuals, n_variables = pop.shape
    new_pop = pop.copy()

    for i in range(n_variables):
        # Find values below the lower bound
        under_val = pop[:, i] < range_values[0, i]
        if np.any(under_val):
            new_pop[under_val, i] = np.remainder(pop[under_val, i], 2 * np.pi) + 2 * np.pi

        # Find values above the upper bound
        over_val = pop[:, i] > range_values[1, i]
        if np.any(over_val):
            new_pop[over_val, i] = np.remainder(pop[over_val, i], 2 * np.pi)

    return new_pop
