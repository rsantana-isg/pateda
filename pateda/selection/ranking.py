"""
Ranking selection

Equivalent to MATEDA's ranking selection methods
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class RankingSelection(SelectionMethod):
    """
    Ranking selection

    Selection probabilities are based on fitness rank rather than raw fitness values.
    This reduces the impact of fitness scaling and helps maintain diversity.

    Supports both linear and exponential ranking.
    """

    def __init__(
        self,
        n_select: Optional[int] = None,
        ratio: float = 0.5,
        selection_pressure: float = 1.5,
        ranking_type: str = "linear",
        replacement: bool = True,
    ):
        """
        Initialize ranking selection

        Args:
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
            selection_pressure: Selection pressure parameter
                               For linear: should be in [1.0, 2.0], controls advantage of best
                               For exponential: controls decay rate
            ranking_type: Type of ranking ("linear" or "exponential")
            replacement: Whether to allow selecting same individual multiple times
        """
        self.n_select = n_select
        self.ratio = ratio
        self.selection_pressure = selection_pressure
        self.ranking_type = ranking_type
        self.replacement = replacement

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals based on fitness ranking

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,)
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - ratio: Override instance ratio
                     - selection_pressure: Override instance selection_pressure
                     - ranking_type: Override instance ranking_type
                     - replacement: Override instance replacement

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

        selection_pressure = params.get("selection_pressure", self.selection_pressure)
        ranking_type = params.get("ranking_type", self.ranking_type)
        replacement = params.get("replacement", self.replacement)

        # Ensure we don't select more than available without replacement
        if not replacement:
            n_select = min(n_select, pop_size)

        # Get ranks (best individual has rank pop_size, worst has rank 1)
        ranks = np.argsort(np.argsort(fitness)) + 1

        # Calculate selection probabilities based on ranking type
        if ranking_type == "linear":
            # Linear ranking: P(i) = (2 - SP + 2*(SP-1)*(rank-1)/(N-1)) / N
            # where SP is selection pressure
            if pop_size == 1:
                probabilities = np.ones(1)
            else:
                probabilities = (
                    2 - selection_pressure
                    + 2 * (selection_pressure - 1) * (ranks - 1) / (pop_size - 1)
                ) / pop_size

        elif ranking_type == "exponential":
            # Exponential ranking: P(i) = (c^(N-rank)) / sum(c^(N-rank))
            # where c is selection_pressure (typically < 1)
            c = selection_pressure
            probabilities = c ** (pop_size - ranks)
            probabilities = probabilities / np.sum(probabilities)

        else:
            raise ValueError(
                f"Unknown ranking_type: {ranking_type}. "
                "Must be 'linear' or 'exponential'"
            )

        # Normalize probabilities (in case of numerical errors)
        probabilities = probabilities / np.sum(probabilities)

        # Select individuals
        selected_indices = np.random.choice(
            pop_size, size=n_select, replace=replacement, p=probabilities
        )

        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
