"""
Pre-seeded population initialization

Equivalent to MATEDA's seed_thispop.m
"""

from typing import Any
import numpy as np

from pateda.core.components import SeedingMethod


class SeedThisPop(SeedingMethod):
    """
    Initialize the algorithm from a pre-seeded population

    This method allows starting the evolutionary algorithm with a specific
    initial population rather than generating random individuals.

    Example:
        >>> # Create a specific initial population
        >>> initial_pop = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
        >>> seeder = SeedThisPop()
        >>> pop = seeder.seed(3, 3, np.ones(3)*2, initial_population=initial_pop)
        >>> # Returns the provided initial population
    """

    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Return the pre-seeded population

        Args:
            n_vars: Number of variables (should match initial_population shape)
            pop_size: Population size (should match initial_population shape)
            cardinality: Variable cardinalities (for validation)
            **params: Additional parameters
                initial_population (np.ndarray): The pre-seeded population
                                                 Shape should be (pop_size, n_vars)

        Returns:
            The provided initial population

        Raises:
            ValueError: If initial_population is not provided or has wrong shape
        """
        if 'initial_population' not in params:
            raise ValueError("initial_population parameter is required for SeedThisPop")

        initial_pop = params['initial_population']

        if not isinstance(initial_pop, np.ndarray):
            initial_pop = np.array(initial_pop)

        # Validate shape
        if initial_pop.shape[1] != n_vars:
            raise ValueError(
                f"initial_population has {initial_pop.shape[1]} variables, "
                f"expected {n_vars}"
            )

        # If fewer individuals than pop_size, we could either:
        # 1. Raise an error
        # 2. Return what we have
        # 3. Duplicate individuals
        # For now, just return what we have (matches MATLAB behavior)
        return initial_pop
