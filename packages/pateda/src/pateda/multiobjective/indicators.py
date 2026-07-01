"""
Quality indicators for multi-objective optimisation.

These metrics quantify the convergence and/or diversity of an approximation
set.  They support both the *evaluation* of results and the *search* itself
(the indicator-based paradigm uses them to drive selection).

Provided indicators:

* :func:`hypervolume` -- exact dominated hypervolume (any number of objectives).
* :func:`hypervolume_contributions` -- per-solution exclusive HV contribution
  (used by SMS-EMOA / hypervolume-based selection).
* :func:`additive_epsilon_matrix` / :func:`ibea_fitness` -- the binary additive
  epsilon indicator and the IBEA fitness assignment built on it.
* :func:`igd` -- inverted generational distance to a reference front.
* :func:`reference_point_from` -- helper to derive a sensible HV reference point.

All functions take a ``maximize`` flag; internally everything is converted to a
minimisation problem so a single implementation serves both directions.
"""

from typing import Optional
import numpy as np

__all__ = [
    "reference_point_from",
    "hypervolume",
    "hypervolume_contributions",
    "additive_epsilon_matrix",
    "ibea_fitness",
    "igd",
]


def _to_min(points: np.ndarray, maximize: bool) -> np.ndarray:
    """Convert objective vectors to a minimisation convention."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return -points if maximize else points


def reference_point_from(
    points: np.ndarray,
    maximize: bool = True,
    margin: float = 0.1,
) -> np.ndarray:
    """Derive an HV reference point from the (nadir of the) given set.

    The reference is placed slightly *worse* than the worst observed value on
    every objective so that boundary solutions enclose a positive volume.

    Args:
        points: Objective vectors of shape ``(n, m)``.
        maximize: Direction of optimisation.
        margin: Fraction of each objective's range used as the safety offset
            (a flat offset of ``margin`` is used for degenerate ranges).

    Returns:
        Reference point of shape ``(m,)`` in the original objective space.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    spans = points.max(axis=0) - points.min(axis=0)
    offset = np.where(spans > 0, spans * margin, margin)
    if maximize:
        return points.min(axis=0) - offset
    return points.max(axis=0) + offset


def _hv_min(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact dominated hypervolume for a minimisation set, recursive slicing.

    ``points`` and ``ref`` are in the minimisation convention; every point is
    assumed to dominate the box ``[point, ref]``.
    """
    n, m = points.shape
    if n == 0:
        return 0.0
    if m == 1:
        return float(max(0.0, ref[0] - points[:, 0].min()))

    order = np.argsort(points[:, 0])
    pts = points[order]
    total = 0.0
    for i in range(n):
        upper = pts[i + 1, 0] if i + 1 < n else ref[0]
        width = upper - pts[i, 0]
        if width <= 0:
            continue
        total += width * _hv_min(pts[: i + 1, 1:], ref[1:])
    return total


def hypervolume(
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    maximize: bool = True,
) -> float:
    """Exact dominated hypervolume of an approximation set.

    Args:
        points: Objective vectors of shape ``(n, m)``.
        reference: Reference point in the *original* objective space.  When
            ``None`` it is derived via :func:`reference_point_from`.
        maximize: Direction of optimisation.

    Returns:
        The hypervolume dominated by ``points`` and bounded by ``reference``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.size == 0:
        return 0.0
    if reference is None:
        reference = reference_point_from(points, maximize=maximize)

    pmin = _to_min(points, maximize)
    rmin = _to_min(np.atleast_2d(reference), maximize)[0]

    # Keep only points that beat the reference on every objective; clip to ref.
    feasible = np.all(pmin < rmin, axis=1)
    pmin = pmin[feasible]
    if pmin.shape[0] == 0:
        return 0.0
    pmin = np.minimum(pmin, rmin)
    return _hv_min(pmin, rmin)


def hypervolume_contributions(
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    maximize: bool = True,
) -> np.ndarray:
    """Per-solution exclusive hypervolume contribution.

    The contribution of solution ``i`` is ``HV(S) - HV(S \\ {i})``: the volume
    that would be lost if ``i`` were removed.  This drives SMS-EMOA-style and
    hypervolume-based selection.

    Returns:
        1-D array of length ``n`` of contributions (``>= 0``).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = points.shape[0]
    if n == 0:
        return np.array([])
    if reference is None:
        reference = reference_point_from(points, maximize=maximize)

    total = hypervolume(points, reference, maximize)
    contrib = np.zeros(n)
    if n == 1:
        contrib[0] = total
        return contrib
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        contrib[i] = total - hypervolume(points[keep], reference, maximize)
        keep[i] = True
    return contrib


def additive_epsilon_matrix(
    points: np.ndarray,
    maximize: bool = True,
) -> np.ndarray:
    """Binary additive epsilon indicator matrix ``I`` with ``I[i, j] = I_eps(i, j)``.

    ``I_eps(i, j)`` is the smallest amount by which solution ``i`` must be
    shifted (in the improving direction) to weakly dominate solution ``j``.  In
    the minimisation convention ``I_eps(i, j) = max_k (f_i^k - f_j^k)``.

    Returns:
        Array of shape ``(n, n)``.
    """
    pmin = _to_min(points, maximize)
    # I[i, j] = max_k (pmin[i, k] - pmin[j, k])
    diff = pmin[:, None, :] - pmin[None, :, :]
    return diff.max(axis=2)


def ibea_fitness(
    points: np.ndarray,
    kappa: float = 0.05,
    maximize: bool = True,
) -> np.ndarray:
    """IBEA fitness values based on the additive epsilon indicator.

    ``fitness(i) = sum_{j != i} -exp(-I_eps(j, i) / (kappa * c))`` where ``c`` is
    the maximal absolute indicator value used to scale the exponent (as in the
    original IBEA paper).  **Larger fitness is better** under this convention.

    Returns:
        1-D array of length ``n``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = points.shape[0]
    if n <= 1:
        return np.zeros(n)
    indicator = additive_epsilon_matrix(points, maximize=maximize)
    c = np.max(np.abs(indicator))
    if c < 1e-12:
        c = 1.0
    # contribution of j to i is -exp(-I(j, i) / (kappa * c))
    contrib = -np.exp(-indicator / (kappa * c))  # contrib[j, i]
    np.fill_diagonal(contrib, 0.0)
    return contrib.sum(axis=0)


def igd(
    points: np.ndarray,
    reference_front: np.ndarray,
    maximize: bool = True,
) -> float:
    """Inverted generational distance to a known reference front.

    For each reference point, the distance to the nearest solution is computed;
    IGD is the mean of these distances (lower is better).  Direction does not
    affect Euclidean distances, so ``maximize`` is accepted for API symmetry.

    Args:
        points: Approximation set, shape ``(n, m)``.
        reference_front: Reference Pareto front, shape ``(r, m)``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    reference_front = np.atleast_2d(np.asarray(reference_front, dtype=float))
    if points.size == 0 or reference_front.size == 0:
        return float("inf")
    dists = np.linalg.norm(
        reference_front[:, None, :] - points[None, :, :], axis=2
    )
    return float(np.mean(dists.min(axis=1)))
