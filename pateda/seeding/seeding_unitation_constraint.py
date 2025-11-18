"""
Unitation-constrained seeding

Equivalent to MATEDA's seeding_unitation_constraint.m
"""

from typing import Any
import numpy as np

from pateda.core.components import SeedingMethod


class SeedingUnitationConstraint(SeedingMethod):
    """
    Generate a population of binary vectors with a fixed number of ones

    All generated vectors will have exactly the same number of ones,
    specified by the num_ones parameter.

    Example:
        >>> seeder = SeedingUnitationConstraint()
        >>> pop = seeder.seed(10, 15, np.ones(10)*2, num_ones=4)
        >>> # Creates population of 15 individuals with 10 binary variables
        >>> # where each individual has exactly 4 ones
    """

    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Generate initial population with unitation constraint

        Args:
            n_vars: Number of variables
            pop_size: Population size
            cardinality: Vector with the cardinality of all variables (should be 2 for binary)
            **params: Additional parameters
                num_ones (int): Number of ones in each binary solution (must be in [1, n_vars-1])

        Returns:
            Population as (pop_size, n_vars) array where each row has exactly num_ones ones

        Raises:
            ValueError: If num_ones is not provided or is out of valid range
        """
        if 'num_ones' not in params:
            raise ValueError("num_ones parameter is required for SeedingUnitationConstraint")

        num_ones = params['num_ones']

        # Validate num_ones
        if not (1 <= num_ones < n_vars):
            raise ValueError(
                f"num_ones must be in [1, {n_vars-1}], got {num_ones}"
            )

        # Initialize population with zeros
        new_pop = np.zeros((pop_size, n_vars), dtype=int)

        # For each individual, randomly place num_ones ones
        for i in range(pop_size):
            # Generate random permutation of variable indices
            perm = np.random.permutation(n_vars)
            # Set the first num_ones positions to 1
            new_pop[i, perm[:num_ones]] = 1

        return new_pop
