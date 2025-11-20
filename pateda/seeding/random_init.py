"""
Random population initialization

Equivalent to MATEDA's RandomInit.m
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SeedingMethod


class RandomInit(SeedingMethod):
    """
    Random initialization of the population

    For discrete variables: Samples uniformly from [0, Card[i]-1] for each variable
    For continuous variables: Samples uniformly from [min[i], max[i]] for each variable
    """

    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Generate initial population with random values

        Args:
            n_vars: Number of variables
            pop_size: Population size
            cardinality: For discrete: 1D array of cardinalities
                        For continuous: 2D array with shape (2, n_vars)
                                       where row 0 = min values, row 1 = max values
            rng: Random number generator (None = create new generator)
            **params: Additional parameters (unused)

        Returns:
            Random population as (pop_size, n_vars) array
        """
        if rng is None:
            rng = np.random.default_rng()

        if cardinality.ndim == 1:
            # Discrete case
            # Generate random integers in [0, Card[i]) for each variable
            new_pop = np.zeros((pop_size, n_vars), dtype=int)
            for i in range(n_vars):
                new_pop[:, i] = rng.integers(0, cardinality[i], size=pop_size)
            return new_pop
        else:
            # Continuous case
            # cardinality has shape (2, n_vars): [min_values; max_values]
            min_values = cardinality[0, :]
            max_values = cardinality[1, :]
            ranges = max_values - min_values

            # Generate random values in [min, max] for each variable
            new_pop = (
                np.tile(min_values, (pop_size, 1))
                + np.tile(ranges, (pop_size, 1)) * rng.random(size=(pop_size, n_vars))
            )
            return new_pop
