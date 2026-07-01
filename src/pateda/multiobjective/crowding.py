"""
Crowding-distance computation (NSGA-II diversity metric).

The crowding distance estimates the density of solutions surrounding a given
point in objective space: it is the sum, over objectives, of the normalised
gap between a solution's two nearest neighbours along that objective.  Boundary
solutions (best/worst on some objective) receive an infinite distance so they
are always preserved.

The metric is direction-independent (it uses sorted positions only), so no
``maximize`` flag is required here.
"""

import numpy as np

__all__ = ["crowding_distance"]


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Compute the crowding distance of each solution.

    Args:
        objectives: Array of shape ``(n, m)`` of objective vectors.

    Returns:
        1-D array of length ``n`` with the crowding distance of each solution
        (``np.inf`` for boundary solutions).
    """
    objectives = np.atleast_2d(objectives)
    n, m = objectives.shape
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    distance = np.zeros(n)
    for obj in range(m):
        order = np.argsort(objectives[:, obj])
        values = objectives[order, obj]
        # Boundary points are always kept.
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[-1] - values[0]
        if span <= 0:
            continue
        # Interior gaps normalised by the objective's range.
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance
