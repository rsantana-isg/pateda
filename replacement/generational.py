"""
Generational replacement

Equivalent to MATEDA's generational replacement (complete replacement)
"""

from typing import Any, Tuple
import numpy as np

from pateda.core.components import ReplacementMethod


class GenerationalReplacement(ReplacementMethod):
    """
    Generational replacement: completely replace old population with new

    This is the standard generational model where the entire population
    is replaced each generation.
    """

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replace old population with new population

        Args:
            old_pop: Previous generation population
            old_fitness: Previous generation fitness
            new_pop: Newly sampled population
            new_fitness: New population fitness
            **params: Additional parameters (unused)

        Returns:
            Tuple of (new_pop, new_fitness) - complete replacement
        """
        return new_pop, new_fitness
