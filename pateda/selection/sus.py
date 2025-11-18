"""
Stochastic Universal Sampling (SUS)

Equivalent to MATEDA's stochastic universal sampling methods
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class StochasticUniversalSampling(SelectionMethod):
    """
    Stochastic Universal Sampling (SUS)

    Similar to proportional selection but uses a single random value and
    equally spaced pointers. This provides lower variance and ensures
    sampling with minimum spread.
    """

    def __init__(self, n_select: Optional[int] = None, ratio: float = 0.5):
        """
        Initialize SUS

        Args:
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
        """
        self.n_select = n_select
        self.ratio = ratio

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals using Stochastic Universal Sampling

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,)
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - ratio: Override instance ratio
                     - offset: Value to add to all fitness values (for negative fitness)

        Returns:
            Tuple of (selected_population, selected_fitness)

        Note:
            If fitness values are negative or zero, an offset is automatically
            applied to make all values positive.
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

        # Handle negative or zero fitness values
        offset = params.get("offset", None)
        if offset is None:
            min_fitness = np.min(fitness)
            if min_fitness <= 0:
                offset = abs(min_fitness) + 1e-10
            else:
                offset = 0

        # Calculate selection probabilities
        adjusted_fitness = fitness + offset
        total_fitness = np.sum(adjusted_fitness)

        if total_fitness == 0:
            # All fitness values are equal, use uniform selection
            selected_indices = np.random.choice(pop_size, size=n_select, replace=False)
        else:
            # Calculate cumulative fitness
            cumulative_fitness = np.cumsum(adjusted_fitness)

            # Distance between pointers
            pointer_distance = total_fitness / n_select

            # Random start position
            start = np.random.uniform(0, pointer_distance)

            # Generate equally spaced pointers
            pointers = start + np.arange(n_select) * pointer_distance

            # Select individuals
            selected_indices = []
            for pointer in pointers:
                # Find first individual whose cumulative fitness exceeds pointer
                idx = np.searchsorted(cumulative_fitness, pointer, side="right")
                # Ensure index is valid
                idx = min(idx, pop_size - 1)
                selected_indices.append(idx)

            selected_indices = np.array(selected_indices)

        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
