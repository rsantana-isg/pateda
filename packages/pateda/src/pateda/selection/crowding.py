"""
NSGA-II crowding-distance selection (Pareto-based paradigm with diversity).

This selection method completes pateda's Pareto-based support by adding the
diversity-preservation step that plain :class:`ParetoFrontSelection` lacks.
Individuals are sorted into non-dominated fronts; complete fronts are accepted
in order, and the front that would overflow ``n_select`` is truncated by
crowding distance (keeping the least-crowded, most spread-out solutions).

Like every other selection method it works with any probabilistic model, so it
turns any pateda EDA into an NSGA-II-style Pareto MOEDA.
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod
from pateda.selection.utils.pareto import find_pareto_set
from pateda.multiobjective.crowding import crowding_distance

__all__ = ["CrowdingDistanceSelection"]


class CrowdingDistanceSelection(SelectionMethod):
    """Non-dominated sorting + crowding-distance truncation (NSGA-II)."""

    def __init__(
        self,
        ratio: float = 0.5,
        n_select: Optional[int] = None,
        maximize: bool = True,
    ):
        """
        Args:
            ratio: Fraction of the population to select (used if ``n_select`` is
                ``None``).
            n_select: Exact number to select (overrides ``ratio``).
            maximize: Direction of optimisation.
        """
        self.ratio = ratio
        self.n_select = n_select
        self.maximize = maximize

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pop_size = population.shape[0]
        maximize = params.get("maximize", self.maximize)

        if n_select is None:
            n_select = self.n_select
        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))
        n_select = min(n_select, pop_size)

        if fitness.ndim == 1:
            fitness = fitness.reshape(-1, 1)

        # Single objective: fall back to plain fitness ordering.
        if fitness.shape[1] == 1:
            order = np.argsort(fitness[:, 0])
            if maximize:
                order = order[::-1]
            selected = order[:n_select]
            return population[selected], fitness[selected]

        # Multi-objective: iterative front extraction with crowding truncation.
        selected_indices: list = []
        remaining = np.ones(pop_size, dtype=bool)

        while len(selected_indices) < n_select and remaining.any():
            rem_idx = np.where(remaining)[0]
            front_mask = find_pareto_set(
                fitness[rem_idx], maximize=maximize, return_mask=True
            )
            front = rem_idx[front_mask]

            if len(selected_indices) + len(front) <= n_select:
                selected_indices.extend(front.tolist())
                remaining[front] = False
            else:
                # Truncate this front by crowding distance (largest kept first).
                cd = crowding_distance(fitness[front])
                room = n_select - len(selected_indices)
                keep = front[np.argsort(cd)[::-1][:room]]
                selected_indices.extend(keep.tolist())
                break

        selected = np.array(selected_indices)
        return population[selected], fitness[selected]
