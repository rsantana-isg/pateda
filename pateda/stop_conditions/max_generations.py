"""
Maximum generations stopping condition

Equivalent to MATEDA's max_gen.m
"""

from typing import Any
import numpy as np

from pateda.core.components import StopCondition


class MaxGenerations(StopCondition):
    """
    Stop after a maximum number of generations

    This is the most common stopping condition for EDAs.
    """

    def __init__(self, max_gen: int):
        """
        Initialize stopping condition

        Args:
            max_gen: Maximum number of generations to run
        """
        self.max_gen = max_gen

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        """
        Check if maximum generations reached

        Args:
            generation: Current generation number
            population: Current population (unused)
            fitness: Current fitness values (unused)
            **params: Additional parameters (unused)

        Returns:
            True if generation >= max_gen, False otherwise
        """
        return generation >= self.max_gen

    def reset(self) -> None:
        """Reset stopping condition (nothing to reset for max_gen)"""
        pass
