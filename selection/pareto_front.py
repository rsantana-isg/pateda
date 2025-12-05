"""
Pareto front selection with truncation

Equivalent to MATEDA's ParetoFront_selection.m
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod
from pateda.selection.utils.pareto import pareto_ranking


class ParetoFrontSelection(SelectionMethod):
    """
    Pareto front selection: Select individuals by Pareto fronts with truncation

    This selection method orders individuals by Pareto fronts (non-dominated sorting)
    and selects the top individuals. This is similar to NSGA-II style selection.

    For multi-objective problems: Uses Pareto ranking
    For single-objective problems: Falls back to fitness-based selection
    """

    def __init__(
        self,
        ratio: float = 0.5,
        n_select: Optional[int] = None,
        maximize: bool = True,
    ):
        """
        Initialize Pareto front selection

        Args:
            ratio: Fraction of population to select (used if n_select is None)
            n_select: Exact number of individuals to select (overrides ratio)
            maximize: If True, assume maximization problem. If False, minimization.
        """
        self.ratio = ratio
        self.n_select = n_select
        self.maximize = maximize

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals using Pareto front ranking

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size, n_objectives)
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - ratio: Override instance ratio
                     - maximize: Override instance maximize

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        pop_size = population.shape[0]
        maximize = params.get("maximize", self.maximize)

        # Determine number to select
        if n_select is None:
            n_select = self.n_select

        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))

        # Ensure we don't select more than available
        n_select = min(n_select, pop_size)

        # Handle both 1D and 2D fitness arrays
        if fitness.ndim == 1:
            fitness = fitness.reshape(-1, 1)

        # Single objective: simple fitness-based selection
        if fitness.shape[1] == 1:
            if maximize:
                sorted_indices = np.argsort(fitness[:, 0])[::-1]
            else:
                sorted_indices = np.argsort(fitness[:, 0])
            selected_indices = sorted_indices[:n_select]
        else:
            # Multi-objective: use Pareto ranking
            ranks, ordered_indices = pareto_ranking(fitness, maximize=maximize)
            selected_indices = ordered_indices[:n_select]

        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
