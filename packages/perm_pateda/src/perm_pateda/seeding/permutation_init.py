"""
Permutation population initialization

Generates a population of random permutations of range(n_vars).
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SeedingMethod


class PermutationInit(SeedingMethod):
    """
    Random initialization for permutation-based EDAs.

    Generates pop_size random permutations of range(n_vars), each being
    a valid permutation (i.e., contains each of 0..n_vars-1 exactly once).
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
        Generate initial population of random permutations.

        Args:
            n_vars: Number of variables (permutation length)
            pop_size: Population size
            cardinality: Not used for permutations (kept for API compatibility)
            rng: Random number generator (None = create new generator)
            **params: Additional parameters (unused)

        Returns:
            Population as (pop_size, n_vars) integer array where each row is
            a random permutation of range(n_vars).
        """
        if rng is None:
            rng = np.random.default_rng()

        population = np.zeros((pop_size, n_vars), dtype=int)
        base = np.arange(n_vars)
        for i in range(pop_size):
            population[i] = rng.permutation(base)
        return population
