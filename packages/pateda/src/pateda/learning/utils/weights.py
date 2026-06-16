"""
Per-sample probability weights for model learning.

All count-based EDA learning methods can be generalised so that each of the
``N`` selected individuals carries a probability ``p_i`` (with
``sum_i p_i == 1``) rather than the implicit uniform weight ``1/N``.  This is
the mechanism behind *customized selection* (Santana, Mendiburu & Lozano,
2014): instead of (or in addition to) truncation selection, the relative
importance of each selected solution is encoded directly in ``p``.

Two scales are used throughout the library and this module makes the
conversion between them explicit:

* **Probability scale** — ``p`` sums to 1.  This is what
  :mod:`bayes_nets` expects as its ``sample_weights`` argument.
* **Count scale** — ``N * p`` sums to ``N``.  This is what the pateda
  count-based utilities (marginal tables, FDA parameters, clique tables)
  expect, because they accumulate *weighted counts* that are then combined
  with Laplace smoothing exactly as raw integer counts would be:

      approximate_count(config) = N * sum_{i : x_i == config} p_i
      smoothed_count(config)    = approximate_count(config) + alpha

When ``p`` is ``None`` every routine falls back to the uniform distribution
``p_i = 1/N``, in which case the count-scale weights are all ``1`` and the
behaviour is identical to the original unweighted implementation.

References
----------
- Santana, R., Mendiburu, A., & Lozano, J. A. (2014). "Customized selection
  in estimation of distribution algorithms." SEAL 2014, pp. 94-105.
- Santana, R. (2003). "Factorized distribution algorithms: Selection without
  selected population." ESM 2003, pp. 91-97.
"""

from typing import Optional

import numpy as np


def normalize_probabilities(
    p: Optional[np.ndarray], n_samples: int, atol: float = 1e-6
) -> Optional[np.ndarray]:
    """
    Validate and normalise a per-sample probability vector.

    Args:
        p: Probability vector of length ``n_samples`` (``sum == 1``), or
           ``None`` to request the uniform distribution.
        n_samples: Expected number of samples (= length of ``p``).
        atol: Absolute tolerance used when checking that ``p`` sums to 1.

    Returns:
        A 1-D float array of length ``n_samples`` summing to exactly 1, or
        ``None`` when ``p`` is ``None`` (uniform) or degenerate (all zeros).

    Raises:
        ValueError: If ``p`` has the wrong length or contains negative values.
    """
    if p is None:
        return None

    p = np.asarray(p, dtype=float).reshape(-1)

    if p.shape[0] != n_samples:
        raise ValueError(
            f"Probability vector p has length {p.shape[0]} but the selected "
            f"population has {n_samples} individuals."
        )

    if np.any(p < -atol):
        raise ValueError("Probability vector p contains negative values.")

    # Clip tiny negative values arising from numerical noise.
    p = np.clip(p, 0.0, None)

    total = p.sum()
    if total <= 0:
        # Degenerate: behave as uniform.
        return None

    # Renormalise defensively (caller may pass approximately-normalised p).
    if not np.isclose(total, 1.0, atol=atol):
        p = p / total
    else:
        p = p / total

    return p


def count_weights_from_p(
    p: Optional[np.ndarray], n_samples: int
) -> Optional[np.ndarray]:
    """
    Convert a probability vector to count-scale weights (``N * p``).

    The result is suitable for the pateda count-based utilities, where a
    sample weight plays the role of a (fractional) occurrence count.  When
    ``p`` is ``None`` (uniform) the function returns ``None`` so that callers
    keep their original, identical unweighted code path.

    Args:
        p: Probability vector (sums to 1) or ``None`` for uniform.
        n_samples: Number of selected individuals ``N``.

    Returns:
        Array of length ``n_samples`` summing to ``n_samples`` (the weighted
        counts), or ``None`` for the uniform case.
    """
    p_norm = normalize_probabilities(p, n_samples)
    if p_norm is None:
        return None
    return p_norm * n_samples


def weighted_univariate_counts(
    column: np.ndarray, cardinality: int, count_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Count occurrences of each value of a single variable, optionally weighted.

    Args:
        column: 1-D array of integer values (one variable across the
                population).
        cardinality: Number of distinct values the variable can take.
        count_weights: Count-scale weights (``N * p``) of the same length as
                       ``column``, or ``None`` for raw integer counts.

    Returns:
        Float array of length ``cardinality`` with the (weighted) count of
        each value.
    """
    column = np.asarray(column).astype(int)
    k = int(cardinality)
    if count_weights is None:
        return np.bincount(column, minlength=k)[:k].astype(float)
    w = np.asarray(count_weights, dtype=float)
    return np.bincount(column, weights=w, minlength=k)[:k].astype(float)


def weighted_bivariate_counts(
    col_i: np.ndarray,
    col_j: np.ndarray,
    card_i: int,
    card_j: int,
    count_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Joint (weighted) counts for a pair of variables.

    Args:
        col_i, col_j: 1-D integer arrays for the two variables.
        card_i, card_j: Cardinalities of the two variables.
        count_weights: Count-scale weights (``N * p``) or ``None``.

    Returns:
        2-D float array of shape ``(card_i, card_j)`` with joint counts.
    """
    col_i = np.asarray(col_i).astype(int)
    col_j = np.asarray(col_j).astype(int)
    ki, kj = int(card_i), int(card_j)
    flat = col_i * kj + col_j
    if count_weights is None:
        counts = np.bincount(flat, minlength=ki * kj)[: ki * kj].astype(float)
    else:
        w = np.asarray(count_weights, dtype=float)
        counts = np.bincount(flat, weights=w, minlength=ki * kj)[: ki * kj].astype(float)
    return counts.reshape(ki, kj)
