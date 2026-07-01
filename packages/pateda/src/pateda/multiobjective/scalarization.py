"""
Scalarizing (aggregation) functions for decomposition-based optimisation.

A scalarizing function turns an objective vector ``f(x)`` into a single scalar
sub-problem value given a weight vector ``lambda`` and (optionally) a reference
point ``z`` (the *ideal* point: the component-wise best objective values seen so
far).

To keep the rest of the toolkit direction-agnostic, **every scalarizing
function here returns a cost where lower is always better**, regardless of
whether the underlying objectives are maximised or minimised.  The ``maximize``
flag controls how raw objective values are turned into a per-objective
*deficit* with respect to the ideal point.

Supported methods (selectable by the string names used by MOEA/D):

* ``"weighted_sum"`` -- linear aggregation.
* ``"tchebycheff"``  -- weighted Chebyshev distance to the ideal point.
* ``"pbi"``          -- penalty-based boundary intersection.
"""

from typing import Optional
import numpy as np

__all__ = ["scalarize", "weighted_sum", "tchebycheff", "pbi", "SCALARIZATIONS"]

_EPS = 1e-12


def _deficit(objectives: np.ndarray, ideal: np.ndarray, maximize: bool) -> np.ndarray:
    """Non-negative gap between the objectives and the ideal point.

    For maximisation the ideal point holds the largest values, so the deficit
    is ``ideal - f`` (>= 0).  For minimisation it is ``f - ideal`` (>= 0).
    """
    if maximize:
        return ideal - objectives
    return objectives - ideal


def weighted_sum(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: Optional[np.ndarray] = None,
    maximize: bool = True,
    theta: float = 5.0,
) -> float:
    """Weighted-sum scalarisation returned as a cost (lower is better)."""
    objectives = np.asarray(objectives, dtype=float)
    value = float(np.dot(weights, objectives))
    return -value if maximize else value


def tchebycheff(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
    maximize: bool = True,
    theta: float = 5.0,
) -> float:
    """Weighted Tchebycheff distance to the ideal point (lower is better)."""
    objectives = np.asarray(objectives, dtype=float)
    d = _deficit(objectives, ideal, maximize)
    w = np.where(weights > _EPS, weights, _EPS)
    return float(np.max(w * np.abs(d)))


def pbi(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
    maximize: bool = True,
    theta: float = 5.0,
) -> float:
    """Penalty-based boundary intersection scalarisation (lower is better).

    ``g = d1 + theta * d2`` where ``d1`` is the projection of the deficit onto
    the (normalised) weight direction and ``d2`` is the perpendicular distance.
    """
    objectives = np.asarray(objectives, dtype=float)
    d = _deficit(objectives, ideal, maximize)
    norm_w = np.linalg.norm(weights)
    if norm_w < _EPS:
        return float(np.linalg.norm(d))
    w_unit = weights / norm_w
    d1 = float(np.dot(d, w_unit))
    perp = d - d1 * w_unit
    d2 = float(np.linalg.norm(perp))
    return d1 + theta * d2


SCALARIZATIONS = {
    "weighted_sum": weighted_sum,
    "tchebycheff": tchebycheff,
    "pbi": pbi,
}


def scalarize(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: Optional[np.ndarray] = None,
    method: str = "tchebycheff",
    maximize: bool = True,
    theta: float = 5.0,
) -> float:
    """Dispatch to the requested scalarising function.

    Args:
        objectives: Objective vector ``f(x)`` of shape ``(m,)``.
        weights: Weight vector ``lambda`` of shape ``(m,)``.
        ideal: Ideal (reference) point; required for ``tchebycheff`` and
            ``pbi``.  May be ``None`` for ``weighted_sum``.
        method: One of ``"weighted_sum"``, ``"tchebycheff"``, ``"pbi"``.
        maximize: Whether the objectives are maximised.
        theta: Penalty parameter for PBI (ignored otherwise).

    Returns:
        Scalar cost (lower is better).
    """
    try:
        fn = SCALARIZATIONS[method]
    except KeyError:
        raise ValueError(
            f"Unknown scalarization '{method}'. "
            f"Choose from {sorted(SCALARIZATIONS)}."
        )
    if method != "weighted_sum" and ideal is None:
        raise ValueError(f"Scalarization '{method}' requires an ideal point.")
    return fn(objectives, weights, ideal, maximize, theta)
