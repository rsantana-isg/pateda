"""
Multi-objective discrete benchmark functions.

These functions return an objective *vector* (one entry per objective) and are
meant to be maximised, matching pateda's default ``maximize=True`` convention.
They cover three difficulty regimes used to exercise the multi-objective
paradigms:

* :func:`mo_onemax_zeromax` -- trivial-convergence, diversity-only problem whose
  entire search space is Pareto optimal (the front is the line
  ``f1 + f2 = n``).  Ideal for checking spread.
* :func:`make_mo_deceptive` -- two conflicting deceptive (trap) objectives that
  reward all-ones vs all-zeros; requires a model that captures building blocks.
* :func:`make_mubqp` -- multi-objective Unconstrained Binary Quadratic
  Programming (mUBQP), the standard hard combinatorial MO benchmark.  Objective
  correlation (``rho``) controls instance hardness, following the design ideas
  of Liefooghe et al. ("On the Design of Hard mUBQP Instances").

Each ``make_*`` factory returns a plain callable ``f(x) -> np.ndarray`` so it
can be dropped straight into :class:`pateda.core.eda.EDA` or
:class:`pateda.multiobjective.MOEAD` as the ``fitness_func``.
"""

from typing import Callable, Optional, Tuple
import numpy as np

from pateda.functions.discrete_binary.toy_functions.trap import trap_n
from pateda.functions.discrete_binary.problems.ubqp import UBQPInstance

__all__ = [
    "mo_onemax_zeromax",
    "make_mo_onemax_zeromax",
    "make_mo_deceptive",
    "make_mubqp",
    "mo_pareto_front_onemax_zeromax",
]


def mo_onemax_zeromax(x: np.ndarray) -> np.ndarray:
    """Two-objective OneMax/ZeroMax (maximise both).

    ``f1`` counts the ones, ``f2`` counts the zeros.  Since ``f1 + f2 == n`` for
    every solution, all solutions are mutually non-dominated -- the problem
    measures only how well an algorithm spreads along the front.
    """
    x = np.asarray(x)
    ones = int(np.sum(x))
    return np.array([float(ones), float(len(x) - ones)])


def make_mo_onemax_zeromax(n_vars: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return :func:`mo_onemax_zeromax` (factory kept for API symmetry)."""
    return mo_onemax_zeromax


def mo_pareto_front_onemax_zeromax(n_vars: int) -> np.ndarray:
    """True Pareto front of :func:`mo_onemax_zeromax` for ``n_vars`` variables."""
    k = np.arange(n_vars + 1, dtype=float)
    return np.column_stack([k, n_vars - k])


def make_mo_deceptive(
    n_vars: int,
    block_size: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Two conflicting deceptive (trap) objectives.

    The genome is split into consecutive blocks of ``block_size`` bits.  The
    first objective applies a ``block_size``-trap to each block (rewarding
    all-ones blocks), the second applies the same trap to the *complemented*
    bits (rewarding all-zeros blocks).  Both are deceptive and conflict, so the
    Pareto front trades all-ones blocks against all-zeros blocks.

    Args:
        n_vars: Number of binary variables (should be a multiple of
            ``block_size``).
        block_size: Trap order ``k``.

    Returns:
        ``f(x) -> np.ndarray`` returning a 2-vector (maximise).
    """
    if n_vars % block_size != 0:
        raise ValueError("n_vars must be a multiple of block_size")

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        obj1 = trap_n(x, block_size)
        obj2 = trap_n(1 - x, block_size)
        return np.array([float(obj1), float(obj2)])

    return f


def make_mubqp(
    n_vars: int,
    n_objectives: int = 2,
    density: float = 0.4,
    rho: float = -0.2,
    weight_range: Tuple[float, float] = (-100.0, 100.0),
    seed: Optional[int] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], UBQPInstance]:
    """Build a multi-objective UBQP (mUBQP) instance and its fitness function.

    Each objective is a quadratic form ``sum_{i<=j} Q^o_{ij} x_i x_j`` over a
    shared sparsity pattern.  Objective weights are drawn with a target
    pairwise correlation ``rho`` (negative -> conflicting -> harder; positive ->
    aligned -> easier), reproducing the controllable-difficulty design used for
    hard mUBQP benchmarks.

    Args:
        n_vars: Number of binary variables.
        n_objectives: Number of objectives.
        density: Fraction of variable pairs (including the diagonal) that carry
            a non-zero interaction.
        rho: Target correlation between objective weight matrices
            (``-1 < rho < 1``); only the bi-objective case enforces it exactly.
        weight_range: ``(min, max)`` range of the underlying weights.
        seed: Random seed.

    Returns:
        Tuple ``(fitness_func, instance)`` where ``fitness_func(x)`` returns an
        ``n_objectives``-vector to be maximised.
    """
    rng = np.random.default_rng(seed)
    instance = UBQPInstance(n_vars, n_objectives)

    lo, hi = weight_range
    scale = (hi - lo) / 2.0
    shift = (hi + lo) / 2.0

    # Shared sparsity mask over the upper triangle (incl. diagonal).
    iu, ju = np.triu_indices(n_vars)
    mask = rng.random(len(iu)) < density
    iu, ju = iu[mask], ju[mask]
    n_terms = len(iu)

    # Correlated standard-normal weights per term, then map to weight_range.
    base = rng.standard_normal(n_terms)
    for o in range(n_objectives):
        indep = rng.standard_normal(n_terms)
        if o == 0:
            z = base
        else:
            z = rho * base + np.sqrt(max(0.0, 1.0 - rho * rho)) * indep
        # Map roughly into the requested range.
        weights = np.clip(z, -3.0, 3.0) / 3.0 * scale + shift
        for i, j, w in zip(iu, ju, weights):
            # UBQPInstance uses 1-based indexing.
            instance.add_interaction(o, int(i) + 1, int(j) + 1, float(w))

    def f(x: np.ndarray) -> np.ndarray:
        return instance.evaluate(np.asarray(x))

    return f, instance
