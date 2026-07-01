"""
External Pareto archive for multi-objective optimisation.

The archive maintains a set of mutually non-dominated solutions discovered
during a run.  It is representation-agnostic: solutions may be discrete
vectors, continuous vectors, permutations, etc. -- only their objective
vectors are used for dominance decisions.

An optional capacity bound keeps the archive size manageable on problems with
large or continuous Pareto fronts; when exceeded, the most crowded members
(smallest crowding distance) are pruned, preserving the extremes.
"""

from typing import List, Optional, Tuple
import numpy as np

from pateda.selection.utils.pareto import pareto_dominates
from pateda.multiobjective.crowding import crowding_distance

__all__ = ["ParetoArchive"]


class ParetoArchive:
    """Bounded set of non-dominated solutions.

    Args:
        maximize: If ``True`` larger objective values are better.
        capacity: Optional maximum number of stored solutions.  ``None`` means
            unbounded.
        tol: Tolerance for treating two objective vectors as duplicates.
    """

    def __init__(
        self,
        maximize: bool = True,
        capacity: Optional[int] = None,
        tol: float = 1e-9,
    ):
        self.maximize = maximize
        self.capacity = capacity
        self.tol = tol
        self.solutions: List[np.ndarray] = []
        self.objectives: List[np.ndarray] = []

    def add(self, solution: np.ndarray, obj_values: np.ndarray) -> bool:
        """Try to insert ``solution`` with objective vector ``obj_values``.

        The solution is rejected if it is dominated by, or duplicates, an
        existing member.  Members dominated by the new solution are removed.

        Returns:
            ``True`` if the solution was inserted.
        """
        obj_values = np.asarray(obj_values, dtype=float)
        to_remove = []
        for i, existing in enumerate(self.objectives):
            if np.allclose(existing, obj_values, atol=self.tol, rtol=0.0):
                return False
            if pareto_dominates(existing, obj_values, self.maximize):
                return False
            if pareto_dominates(obj_values, existing, self.maximize):
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            self.solutions.pop(i)
            self.objectives.pop(i)

        self.solutions.append(np.array(solution).copy())
        self.objectives.append(obj_values.copy())

        if self.capacity is not None and len(self.solutions) > self.capacity:
            self._prune()
        return True

    def add_population(self, population: np.ndarray, fitness: np.ndarray) -> int:
        """Add every individual of a population; return number inserted."""
        fitness = np.atleast_2d(fitness)
        added = 0
        for sol, obj in zip(population, fitness):
            if self.add(sol, obj):
                added += 1
        return added

    def _prune(self) -> None:
        """Remove the most crowded member until capacity is satisfied."""
        while len(self.solutions) > self.capacity:
            objs = np.array(self.objectives)
            cd = crowding_distance(objs)
            # Never drop the extremes (infinite crowding distance).
            victim = int(np.argmin(cd))
            self.solutions.pop(victim)
            self.objectives.pop(victim)

    def get_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(solutions, objectives)`` as arrays (possibly empty)."""
        if not self.solutions:
            return np.array([]), np.array([])
        return np.array(self.solutions), np.array(self.objectives)

    @property
    def size(self) -> int:
        return len(self.solutions)

    def __len__(self) -> int:
        return len(self.solutions)
