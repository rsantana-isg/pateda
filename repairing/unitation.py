"""
Unitation-based repairing for binary optimization problems.

This module provides constraint repair mechanisms for binary solutions
where the number of ones must be within a specified range.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Tuple, Optional


def unitation_repairing(
    pop: np.ndarray,
    range_values: Tuple[int, int],
    repairing_params: Optional[dict] = None
) -> np.ndarray:
    """
    Repair binary solutions to ensure the number of ones is within specified bounds.

    For a problem with binary representation, this function modifies the solutions
    to ensure the number of ones is within the minimum and maximum bounds. If a
    solution has too few ones, random zeros are flipped to ones. If it has too
    many ones, random ones are flipped to zeros.

    Parameters
    ----------
    pop : np.ndarray
        Population matrix of shape (n_individuals, n_variables) with binary values.
    range_values : Tuple[int, int]
        Tuple where:
        - range_values[0]: Minimum number of ones allowed
        - range_values[1]: Maximum number of ones allowed
    repairing_params : Optional[dict], default=None
        Not used for this function. Included for API consistency.

    Returns
    -------
    np.ndarray
        Repaired population with the same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.array([[1, 0, 0, 1, 0],
    ...                  [1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0]])
    >>> range_values = (2, 3)
    >>> repaired = unitation_repairing(pop, range_values)
    >>> # Each solution will have between 2 and 3 ones

    Notes
    -----
    - The function processes each individual in the population independently
    - Random selection is used when multiple variables could be flipped
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    n_individuals, n_variables = pop.shape
    new_pop = pop.copy()

    for i in range(n_individuals):
        n_ones = np.sum(pop[i, :])
        n_zeros = n_variables - n_ones

        if n_ones < range_values[0]:
            # There are fewer ones than needed
            n_new_ones = range_values[0] - n_ones
            pos_zeros = np.where(pop[i, :] == 0)[0]

            # Randomly select which zeros to flip to ones
            selected_zeros = np.random.choice(pos_zeros, size=n_new_ones, replace=False)
            new_pop[i, selected_zeros] = 1

        elif n_ones > range_values[1]:
            # There are more ones than needed
            n_new_zeros = n_ones - range_values[1]
            pos_ones = np.where(pop[i, :] == 1)[0]

            # Randomly select which ones to flip to zeros
            selected_ones = np.random.choice(pos_ones, size=n_new_zeros, replace=False)
            new_pop[i, selected_ones] = 0

    return new_pop
