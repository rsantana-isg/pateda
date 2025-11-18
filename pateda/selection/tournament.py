"""
Tournament selection

Equivalent to MATEDA's tournament selection methods
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class TournamentSelection(SelectionMethod):
    """
    Tournament selection: Select individuals by running tournaments

    In each tournament, k individuals are randomly selected and the best
    one is chosen. This process is repeated until n_select individuals
    are selected.
    """

    def __init__(
        self,
        tournament_size: int = 2,
        n_select: Optional[int] = None,
        ratio: float = 0.5,
        replacement: bool = True,
    ):
        """
        Initialize tournament selection

        Args:
            tournament_size: Number of individuals per tournament
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
            replacement: Whether to allow selecting same individual multiple times
        """
        self.tournament_size = tournament_size
        self.n_select = n_select
        self.ratio = ratio
        self.replacement = replacement

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals using tournament selection

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,)
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - tournament_size: Override instance tournament_size
                     - ratio: Override instance ratio
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

        tournament_size = params.get("tournament_size", self.tournament_size)
        replacement = params.get("replacement", self.replacement)

        # Ensure tournament size is valid
        tournament_size = min(tournament_size, pop_size)

        selected_indices = []

        if replacement:
            # With replacement: can select same individual multiple times
            for _ in range(n_select):
                # Randomly select tournament participants
                tournament_indices = np.random.choice(
                    pop_size, size=tournament_size, replace=False
                )

                # Get best individual from tournament
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
        else:
            # Without replacement: each individual can only be selected once
            available = set(range(pop_size))

            for _ in range(min(n_select, pop_size)):
                if len(available) < tournament_size:
                    # Not enough individuals left for full tournament
                    tournament_indices = list(available)
                else:
                    tournament_indices = np.random.choice(
                        list(available), size=tournament_size, replace=False
                    )

                # Get best individual from tournament
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
                available.remove(winner_idx)

        selected_indices = np.array(selected_indices)
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
