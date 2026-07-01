"""
Selection-probability weighting for customized selection.

In a standard EDA, the model is learned from the selected population giving
every selected individual the same importance (uniform weight ``1/N``).
*Customized selection* (Santana, Mendiburu & Lozano, 2014) instead assigns
each selected individual ``i`` a probability ``p_i`` (with ``sum_i p_i == 1``)
that the learning method uses to weight its counts / probability tables.

This module turns a fitness vector over the selected population into such a
probability vector ``p`` according to a chosen *weighting* scheme:

* ``"uniform"``       -> ``p_i = 1/N`` (returned as ``None``, the
                         unweighted fast path).
* ``"proportional"``  -> ``p_i ∝ f_i`` after shifting fitness to be
                         non-negative (fitness-proportional / roulette).
* ``"boltzmann"``     -> ``p_i ∝ exp(beta * z_i)`` where ``z_i`` is the
                         standardised fitness, ``beta`` an inverse-temperature
                         controlling the sharpness of the distribution.

Fitness is assumed to be *maximised* (consistent with the rest of pateda),
so higher fitness always yields higher probability.
"""

from typing import Optional

import numpy as np

# Recognised weighting-scheme aliases.
UNIFORM_ALIASES = {None, "uniform", "none", "equal"}
PROPORTIONAL_ALIASES = {"proportional", "fitness_proportional", "fp", "roulette"}
BOLTZMANN_ALIASES = {"boltzmann", "boltzman", "exponential", "exp"}


def _to_1d_fitness(fitness: np.ndarray) -> np.ndarray:
    """Collapse a (N,) / (N, 1) / (N, m) fitness array to a 1-D vector.

    For multi-objective fitness the mean across objectives is used, mirroring
    :class:`~pateda.selection.boltzmann.BoltzmannSelection`.
    """
    f = np.asarray(fitness, dtype=float)
    if f.ndim == 2:
        if f.shape[1] == 1:
            f = f[:, 0]
        else:
            f = np.mean(f, axis=1)
    return f


def compute_selection_probabilities(
    fitness: np.ndarray,
    mode: Optional[str] = "uniform",
    beta: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Compute a per-individual probability vector ``p`` from fitness.

    Args:
        fitness: Fitness of the selected individuals.  Shape ``(N,)``,
                 ``(N, 1)`` or ``(N, m)`` (multi-objective uses the mean).
        mode: Weighting scheme — ``"uniform"``, ``"proportional"`` or
              ``"boltzmann"`` (aliases accepted, case-insensitive).
        beta: Inverse-temperature for the Boltzmann scheme (ignored
              otherwise).  Larger ``beta`` concentrates probability on the
              fittest individuals; ``beta = 0`` recovers the uniform vector.

    Returns:
        A probability vector of length ``N`` summing to 1, or ``None`` when
        the scheme is uniform or degenerate (e.g. all fitness values equal).
        Returning ``None`` lets the learning methods take their original
        unweighted code path.

    Raises:
        ValueError: If ``mode`` is not a recognised weighting scheme.
    """
    key = mode.lower() if isinstance(mode, str) else mode

    if key in UNIFORM_ALIASES:
        return None

    f = _to_1d_fitness(fitness)
    n = f.shape[0]
    if n == 0:
        return None

    if key in PROPORTIONAL_ALIASES:
        shifted = f - np.min(f)
        total = shifted.sum()
        if total <= 0:
            # All fitness values equal -> uniform.
            return None
        return shifted / total

    if key in BOLTZMANN_ALIASES:
        # Standardise fitness so that ``beta`` has a scale-independent meaning
        # across problems with very different fitness magnitudes.
        mean = np.mean(f)
        std = np.std(f)
        if std <= 0:
            return None
        z = (f - mean) / std
        log_p = beta * z
        log_p -= np.max(log_p)  # log-sum-exp stability
        w = np.exp(log_p)
        total = w.sum()
        if total <= 0:
            return None
        return w / total

    raise ValueError(
        f"Unknown weighting mode {mode!r}. Choose 'uniform', "
        f"'proportional' or 'boltzmann'."
    )
