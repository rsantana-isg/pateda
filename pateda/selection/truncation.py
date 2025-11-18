"""
Truncation selection

Equivalent to MATEDA's truncation_selection.m
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class TruncationSelection(SelectionMethod):
    """
    Truncation selection: Select top N individuals by fitness

    This is the most common selection method for EDAs, selecting the best
    individuals based on their fitness values.
    """

    def __init__(self, ratio: float = 0.5, n_select: Optional[int] = None):
        """
        Initialize truncation selection

        Args:
            ratio: Fraction of population to select (used if n_select is None)
            n_select: Exact number of individuals to select (overrides ratio)
        """
        self.ratio = ratio
        self.n_select = n_select

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top individuals by fitness

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,)
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - ratio: Override instance ratio

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        pop_size = population.shape[0]

        # Determine number to select
        if n_select is None:
            n_select = self.n_select

        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))

        # Ensure we don't select more than available
        n_select = min(n_select, pop_size)

        # Get indices of top individuals (sorted by fitness, descending)
        sorted_indices = np.argsort(fitness)[::-1]
        selected_indices = sorted_indices[:n_select]

        # Select individuals
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
