"""
Pareto dominance and archive management.

Tools for comparing multi-objective solutions, maintaining non-dominated
sets, and computing hypervolume indicators.
"""

import numpy as np
from typing import List, Optional, Tuple


def dominates(
    a: np.ndarray,
    b: np.ndarray,
    minimize: bool = True,
) -> bool:
    """Check whether objective vector *a* dominates *b*.

    *a* dominates *b* if a is at least as good in every objective
    and strictly better in at least one.

    Args:
        a: Objective vector of shape (m,).
        b: Objective vector of shape (m,).
        minimize: If True, smaller values are better.

    Returns:
        True if a dominates b.
    """
    if minimize:
        return bool(np.all(a <= b) and np.any(a < b))
    else:
        return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(
    objectives: np.ndarray,
    minimize: bool = True,
) -> np.ndarray:
    """Extract the non-dominated (Pareto-front) indices.

    Args:
        objectives: Array of shape (pop_size, m) with objective vectors.
        minimize: If True, smaller values are better.

    Returns:
        1-D array of indices corresponding to non-dominated solutions.
    """
    n = objectives.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(objectives[j], objectives[i], minimize):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


class ParetoArchive:
    """Maintains a set of non-dominated solutions.

    Supports adding new solutions and removing dominated ones.
    """

    def __init__(self, minimize: bool = True):
        """
        Args:
            minimize: If True, smaller objective values are better.
        """
        self.minimize = minimize
        self.solutions: List[np.ndarray] = []     # permutations
        self.objectives: List[np.ndarray] = []     # objective vectors

    def add(self, solution: np.ndarray, obj_values: np.ndarray) -> bool:
        """Try to add a solution to the archive.

        The solution is added if it is not dominated by any existing
        member.  Any existing members dominated by the new solution
        are removed. Solutions with duplicate objective vectors (within
        numerical tolerance) are not added.

        Args:
            solution: A permutation.
            obj_values: Its objective vector of shape (m,).

        Returns:
            True if the solution was added to the archive.
        """
        # Check if new solution is dominated or has duplicate objectives
        to_remove = []
        for i, existing_obj in enumerate(self.objectives):
            # Check for duplicates first (faster than dominance check)
            # Uses np.allclose with default tolerances: rtol=1e-5, atol=1e-8
            if np.allclose(existing_obj, obj_values):
                return False
            if dominates(existing_obj, obj_values, self.minimize):
                return False
            if dominates(obj_values, existing_obj, self.minimize):
                to_remove.append(i)

        # Remove dominated solutions (in reverse to preserve indices)
        for i in sorted(to_remove, reverse=True):
            self.solutions.pop(i)
            self.objectives.pop(i)

        self.solutions.append(solution.copy())
        self.objectives.append(obj_values.copy())
        return True

    def get_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current Pareto front.

        Returns:
            Tuple of (solutions, objectives) arrays.
        """
        if len(self.solutions) == 0:
            return np.array([]), np.array([])
        return np.array(self.solutions), np.array(self.objectives)

    @property
    def size(self) -> int:
        """Number of solutions in the archive."""
        return len(self.solutions)
