"""
Proportional selection (Roulette Wheel Selection)

Equivalent to MATEDA's proportional selection methods
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class ProportionalSelection(SelectionMethod):
    """
    Proportional selection (Roulette Wheel Selection)

    Individuals are selected with probability proportional to their fitness.
    This is also known as fitness proportionate selection.
    """

    def __init__(
        self, n_select: Optional[int] = None, ratio: float = 0.5, replacement: bool = True
    ):
        """
        Initialize proportional selection

        Args:
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
            replacement: Whether to allow selecting same individual multiple times
        """
        self.n_select = n_select
        self.ratio = ratio
        self.replacement = replacement

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals proportionally to their fitness

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,) or (pop_size, n_objectives)
                    For multi-objective, uses mean fitness across objectives
            n_select: Number to select (overrides instance n_select)
            rng: Random number generator (None = create default generator)
            **params: Additional parameters
                     - ratio: Override instance ratio
                     - replacement: Override instance replacement
                     - offset: Value to add to all fitness values (for negative fitness)

        Returns:
            Tuple of (selected_population, selected_fitness)

        Note:
            If fitness values are negative or zero, an offset is automatically
            applied to make all values positive.
        """
        if rng is None:
            rng = np.random.default_rng()

        pop_size = population.shape[0]

        # Determine number to select
        if n_select is None:
            n_select = self.n_select

        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))

        replacement = params.get("replacement", self.replacement)

        # Ensure we don't select more than available without replacement
        if not replacement:
            n_select = min(n_select, pop_size)

        # Handle multi-objective fitness by taking mean
        if fitness.ndim == 2 and fitness.shape[1] > 1:
            fitness_for_selection = np.mean(fitness, axis=1)
        elif fitness.ndim == 2:
            fitness_for_selection = fitness[:, 0]
        else:
            fitness_for_selection = fitness

        # Handle negative or zero fitness values
        offset = params.get("offset", None)
        if offset is None:
            min_fitness = np.min(fitness_for_selection)
            if min_fitness <= 0:
                offset = abs(min_fitness) + 1e-10
            else:
                offset = 0

        # Calculate selection probabilities
        adjusted_fitness = fitness_for_selection + offset
        total_fitness = np.sum(adjusted_fitness)

        if total_fitness == 0:
            # All fitness values are equal, use uniform selection
            probabilities = np.ones(pop_size) / pop_size
        else:
            probabilities = adjusted_fitness / total_fitness

        # Select individuals
        selected_indices = rng.choice(
            pop_size, size=n_select, replace=replacement, p=probabilities
        )

        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
