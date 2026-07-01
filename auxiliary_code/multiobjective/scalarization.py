"""
Scalarization functions for multi-objective optimization.

Convert a vector of objective values into a single scalar value using
weight vectors.  Used in decomposition-based methods like MOEA/D.
"""

import numpy as np


def weighted_sum(
    objectives: np.ndarray,
    weights: np.ndarray,
    reference_point: np.ndarray = None,
    minimize: bool = True,
) -> float:
    """Weighted sum scalarization.

    g(x | w) = sum_i  w_i * f_i(x)

    When a reference point z* is provided:
    g(x | w, z*) = sum_i  w_i * |f_i(x) - z*_i|

    Args:
        objectives: Objective vector of shape (m,).
        weights: Weight vector of shape (m,), must sum to 1.
        reference_point: Optional reference (ideal) point.
        minimize: If True, lower scalar values are better.

    Returns:
        Scalar fitness value.
    """
    if reference_point is not None:
        diff = np.abs(objectives - reference_point)
        return float(np.dot(weights, diff))
    return float(np.dot(weights, objectives))


def tchebycheff(
    objectives: np.ndarray,
    weights: np.ndarray,
    reference_point: np.ndarray,
    minimize: bool = True,
) -> float:
    """Tchebycheff scalarization.

    g(x | w, z*) = max_i  w_i * |f_i(x) - z*_i|

    This approach guarantees that every Pareto-optimal solution can be
    found by varying the weight vector (unlike weighted sum).

    Args:
        objectives: Objective vector of shape (m,).
        weights: Weight vector of shape (m,).
        reference_point: Reference (ideal) point.
        minimize: If True, lower scalar values are better.

    Returns:
        Scalar fitness value.
    """
    diff = np.abs(objectives - reference_point)
    # Avoid zero weights causing issues
    w = np.where(weights > 1e-10, weights, 1e-10)
    return float(np.max(w * diff))
