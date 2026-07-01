"""
Pareto-dominance helpers for the multi-objective toolkit.

These thin wrappers re-expose the dominance / non-dominated-set utilities that
already live in :mod:`pateda.selection.utils.pareto` so that the multi-objective
subpackage offers a single, self-contained import surface.  All functions follow
the package-wide convention that ``maximize`` controls the direction of
optimisation (``True`` -> larger objective values are better).
"""

from typing import Sequence
import numpy as np

from pateda.selection.utils.pareto import (
    pareto_dominates,
    find_pareto_set,
    pareto_ranking,
)

__all__ = [
    "pareto_dominates",
    "find_pareto_set",
    "pareto_ranking",
    "non_dominated_front",
]


def non_dominated_front(
    objectives: np.ndarray,
    maximize: bool = True,
) -> np.ndarray:
    """Return the objective vectors of the non-dominated solutions.

    Convenience wrapper that returns the *fitness* of the Pareto front rather
    than its indices.

    Args:
        objectives: Array of shape ``(n, m)`` with one objective vector per row.
        maximize: If ``True`` larger values are better.

    Returns:
        Array of shape ``(k, m)`` with the non-dominated objective vectors.
    """
    objectives = np.atleast_2d(objectives)
    idx = find_pareto_set(objectives, maximize=maximize, return_mask=False)
    return objectives[idx]
