"""
Non-dominated selection for multi-objective optimization

Equivalent to MATEDA's NonDominated_selection.m
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod
from pateda.selection.utils.pareto import find_pareto_set


class NonDominatedSelection(SelectionMethod):
    """
    Non-dominated selection: Select only non-dominated (Pareto optimal) individuals

    This selection method identifies and returns all non-dominated solutions
    from the population. It's particularly useful for multi-objective optimization
    where we want to maintain diversity in the Pareto front.

    Note:
        For single-objective problems, this returns only the best individual.
        The number of selected individuals can vary depending on the Pareto front size.
    """

    def __init__(self, maximize: bool = True, min_select: Optional[int] = None):
        """
        Initialize non-dominated selection

        Args:
            maximize: If True, assume maximization problem. If False, minimization.
            min_select: Minimum number of individuals to select. If the Pareto front
                       is smaller, additional individuals from the next front will
                       be included. If None, only the Pareto front is selected.
        """
        self.maximize = maximize
        self.min_select = min_select

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select non-dominated individuals

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size, n_objectives)
            **params: Additional parameters
                     - maximize: Override instance maximize
                     - min_select: Override instance min_select

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        maximize = params.get("maximize", self.maximize)
        min_select = params.get("min_select", self.min_select)

        # Handle both 1D and 2D fitness arrays
        if fitness.ndim == 1:
            fitness = fitness.reshape(-1, 1)

        # Find non-dominated solutions
        pareto_indices = find_pareto_set(fitness, maximize=maximize, return_mask=False)

        # If we need more individuals than the Pareto front provides
        if min_select is not None and len(pareto_indices) < min_select:
            from pateda.selection.utils.pareto import pareto_ranking

            # Use Pareto ranking to get more individuals
            ranks, ordered_indices = pareto_ranking(fitness, maximize=maximize)
            # Take first min_select individuals
            pareto_indices = ordered_indices[:min_select]

        selected_pop = population[pareto_indices]
        selected_fitness = fitness[pareto_indices]

        return selected_pop, selected_fitness
