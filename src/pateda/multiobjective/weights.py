"""
Weight-vector generation for decomposition-based multi-objective optimisation.

Decomposition methods (MOEA/D and decomposition-based EDAs) translate an
``m``-objective problem into a set of scalar sub-problems, each defined by a
weight vector ``lambda`` lying on the unit simplex (``sum_i lambda_i = 1``).

Two generators are provided:

* :func:`uniform_weights` -- evenly spaced vectors for the bi-objective case.
* :func:`das_dennis_weights` -- the simplex-lattice (Das & Dennis) design used
  by MOEA/D for an arbitrary number of objectives.

:func:`generate_weights` dispatches between them and is the function the rest of
the toolkit calls.
"""

from itertools import combinations
from math import comb
from typing import Optional
import numpy as np

__all__ = [
    "uniform_weights",
    "das_dennis_weights",
    "generate_weights",
    "weight_neighbourhoods",
]


def uniform_weights(n_weights: int) -> np.ndarray:
    """Generate ``n_weights`` evenly spaced bi-objective weight vectors.

    Args:
        n_weights: Number of weight vectors (sub-problems).

    Returns:
        Array of shape ``(n_weights, 2)`` whose rows sum to 1.
    """
    if n_weights < 1:
        raise ValueError("n_weights must be >= 1")
    if n_weights == 1:
        return np.array([[0.5, 0.5]])
    w = np.linspace(0.0, 1.0, n_weights)
    return np.column_stack([w, 1.0 - w])


def das_dennis_weights(n_obj: int, n_partitions: int) -> np.ndarray:
    """Simplex-lattice (Das-Dennis) weight vectors.

    Produces all vectors of the form ``(k_1/H, ..., k_m/H)`` with
    ``sum_i k_i = H`` and ``k_i`` non-negative integers.  The number of
    generated vectors is ``C(H + m - 1, m - 1)``.

    Args:
        n_obj: Number of objectives ``m`` (>= 2).
        n_partitions: Number of partitions ``H`` along each axis.

    Returns:
        Array of shape ``(C(H+m-1, m-1), m)`` whose rows sum to 1.
    """
    if n_obj < 2:
        raise ValueError("n_obj must be >= 2")
    if n_partitions < 1:
        raise ValueError("n_partitions must be >= 1")

    # Classic "boundary intersection" enumeration of integer compositions of H
    # into m parts using the stars-and-bars trick on combination indices.
    weights = []
    for cuts in combinations(range(n_partitions + n_obj - 1), n_obj - 1):
        prev = -1
        parts = []
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(n_partitions + n_obj - 1 - prev - 1)
        weights.append(parts)
    w = np.asarray(weights, dtype=float) / float(n_partitions)
    return w


def _partitions_for_target(n_obj: int, target: int) -> int:
    """Smallest ``H`` whose Das-Dennis design has at least ``target`` vectors."""
    h = 1
    while comb(h + n_obj - 1, n_obj - 1) < target:
        h += 1
    return h


def generate_weights(
    n_obj: int,
    n_weights: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate roughly ``n_weights`` weight vectors for ``n_obj`` objectives.

    * For ``n_obj == 2`` the evenly spaced :func:`uniform_weights` design is
      returned with exactly ``n_weights`` vectors.
    * For ``n_obj >= 3`` a Das-Dennis simplex lattice is used; because the
      lattice size is quantised, the partition count ``H`` is chosen so that the
      design has at least ``n_weights`` vectors, and the result is then
      sub-sampled (without replacement) down to ``n_weights``.

    Args:
        n_obj: Number of objectives.
        n_weights: Desired number of weight vectors.
        seed: Random seed used only when sub-sampling a larger lattice.

    Returns:
        Array of shape ``(n_weights, n_obj)`` whose rows sum to 1.
    """
    if n_obj == 2:
        return uniform_weights(n_weights)

    h = _partitions_for_target(n_obj, n_weights)
    w = das_dennis_weights(n_obj, h)
    if len(w) > n_weights:
        rng = np.random.default_rng(seed)
        # Always keep the axis (single-objective) vectors, then fill the rest.
        axis_mask = np.isclose(w.max(axis=1), 1.0)
        axis_idx = np.where(axis_mask)[0]
        rest_idx = np.where(~axis_mask)[0]
        n_extra = max(0, n_weights - len(axis_idx))
        chosen_rest = rng.choice(rest_idx, size=min(n_extra, len(rest_idx)),
                                 replace=False)
        idx = np.concatenate([axis_idx, chosen_rest])[:n_weights]
        w = w[np.sort(idx)]
    return w


def weight_neighbourhoods(weights: np.ndarray, neighbourhood_size: int) -> np.ndarray:
    """Compute, for every weight vector, the indices of its closest neighbours.

    Neighbourhoods are defined by Euclidean distance between weight vectors, as
    in the original MOEA/D paper.

    Args:
        weights: Array of shape ``(N, m)``.
        neighbourhood_size: Number ``T`` of neighbours per sub-problem.

    Returns:
        Integer array of shape ``(N, T)`` of neighbour indices (each row begins
        with the sub-problem itself).
    """
    n = len(weights)
    t = min(neighbourhood_size, n)
    diff = weights[:, None, :] - weights[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return np.argsort(dists, axis=1)[:, :t]
