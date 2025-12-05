"""
Biased population initialization

Equivalent to MATEDA's Bias_Init.m
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SeedingMethod


class BiasInit(SeedingMethod):
    """
    Biased initialization of a population of binary vectors

    Generates binary vectors where the probability of generating 1 is p
    and the probability of generating 0 is (1-p).

    Example:
        >>> seeder = BiasInit()
        >>> pop = seeder.seed(10, 15, np.ones(10)*2, p=0.8)
        >>> # Creates population of 15 individuals with 10 binary variables
        >>> # where each variable has 80% probability of being 1
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
        Generate initial population with biased random values

        Args:
            n_vars: Number of variables
            pop_size: Population size
            cardinality: Vector with the cardinality of all variables (should be 2 for binary)
            rng: Random number generator (None = create new generator)
            **params: Additional parameters
                p (float): Probability of generating 1 (default: 0.5)

        Returns:
            Biased random population as (pop_size, n_vars) array with binary values
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract probability parameter
        p = params.get('p', 0.5)

        # Generate binary population where each variable is 1 with probability p
        new_pop = (rng.random(size=(pop_size, n_vars)) <= p).astype(int)

        return new_pop
