"""
Off-lattice AB protein folding model (Stillinger 1993).

The AB model is a coarse-grained protein folding benchmark in which each
residue is classified either as Hydrophobic (A) or Polar (B).  A chain of
``N`` residues is parameterised by ``N - 2`` consecutive bond angles, and
the energy combines a bond-bending term and a Lennard-Jones-style
non-bonded interaction whose attraction strength depends on the pair of
residue types.

The model is widely used as a hard continuous benchmark for EDAs, in
particular for vine copula EDAs, because it exhibits a rugged landscape
with many local minima.

References
----------
- Stillinger, F. H., Head-Gordon, T., & Hirshfeld, C. L. (1993).
  Toy model for protein folding.  *Physical Review E*, 48(2), 1469.
- Bachmann, M., Arkin, H., & Janke, W. (2005).  Multicanonical study of
  coarse-grained off-lattice models for folding heteropolymers.
  *Physical Review E*, 71(3), 031906.
"""

from typing import Callable, Sequence, Union
import numpy as np


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def fibonacci_ab_sequence(n: int) -> np.ndarray:
    """
    Build a Fibonacci-style AB sequence of length ``n``.

    The standard Stillinger sequences are defined by the recursion
    ``S_{k+1} = S_{k-1} S_k`` starting from ``S_1 = 'A'`` and ``S_2 = 'B'``.
    Chains of length 13, 21, 34, 55, ... fall on this recursion.  For an
    arbitrary requested length the Fibonacci string of equal or larger
    length is built and truncated.

    Returns
    -------
    np.ndarray of dtype int, shape (n,)
        Sequence with 0 = A (hydrophobic), 1 = B (polar).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    a = np.array([0], dtype=int)     # 'A'
    b = np.array([0, 1], dtype=int)  # 'AB' = S_2 in the Stillinger convention
    while len(b) < n:
        a, b = b, np.concatenate([a, b])
    return b[:n]


def parse_ab_sequence(sequence: Union[str, Sequence[int]]) -> np.ndarray:
    """
    Coerce a sequence specification to a 0/1 integer array.

    Accepts a string of 'A'/'B' characters or any iterable of 0/1 values.
    """
    if isinstance(sequence, str):
        return np.array([0 if ch.upper() == 'A' else 1 for ch in sequence], dtype=int)
    arr = np.asarray(sequence, dtype=int)
    if arr.ndim != 1 or arr.min() < 0 or arr.max() > 1:
        raise ValueError("AB sequence must be a 1-D array of 0/1 entries.")
    return arr


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def _interaction_coefficient(s_i: int, s_j: int) -> float:
    """
    Pairwise interaction strength C(s_i, s_j) of the AB model.

    A-A pairs are strongly attractive, A-B pairs are slightly repulsive,
    and B-B pairs are weakly attractive.
    """
    if s_i == 0 and s_j == 0:        # A-A
        return 1.0
    if s_i == 1 and s_j == 1:        # B-B
        return 0.5
    return -0.5                       # A-B or B-A


def ab_energy_2d(angles: np.ndarray, sequence: np.ndarray) -> float:
    """
    Compute the 2-D AB-model energy for a given set of bond angles.

    Parameters
    ----------
    angles : np.ndarray of shape (N - 2,)
        Bond angles (radians) between consecutive bonds.
    sequence : np.ndarray of shape (N,)
        AB sequence (0 = A, 1 = B).  ``len(sequence) == len(angles) + 2``.

    Returns
    -------
    float
        Total energy (bending + non-bonded).  Lower is better.
    """
    angles = np.asarray(angles, dtype=float)
    sequence = np.asarray(sequence, dtype=int)
    n_residues = len(sequence)
    if len(angles) != n_residues - 2:
        raise ValueError(
            f"Expected {n_residues - 2} angles for sequence length {n_residues}, "
            f"got {len(angles)}."
        )

    # --- Positions ------------------------------------------------------
    # Convention: place residue 1 at origin and the first bond along +x.
    # Bond ``i+1`` (between residues i+1 and i+2) is rotated by the
    # cumulative angle sum_{k=2}^{i+1} theta_k relative to bond 1.
    positions = np.zeros((n_residues, 2))
    positions[1, 0] = 1.0  # bond length = 1

    cumulative = 0.0
    for i in range(2, n_residues):
        cumulative += angles[i - 2]
        positions[i, 0] = positions[i - 1, 0] + np.cos(cumulative)
        positions[i, 1] = positions[i - 1, 1] + np.sin(cumulative)

    # --- Bending energy V1 ---------------------------------------------
    v1 = float(np.sum((1.0 - np.cos(angles)) / 4.0))

    # --- Non-bonded Lennard-Jones-style energy V2 ----------------------
    v2 = 0.0
    for i in range(n_residues - 2):
        diff = positions[i + 2:] - positions[i]
        r = np.sqrt(np.sum(diff * diff, axis=1))
        # Skip tiny distances to avoid blow-up if a configuration collapses.
        r = np.maximum(r, 1e-6)
        inv_r6 = 1.0 / r ** 6
        inv_r12 = inv_r6 * inv_r6
        for offset, r6, r12 in zip(range(i + 2, n_residues), inv_r6, inv_r12):
            c = _interaction_coefficient(int(sequence[i]), int(sequence[offset]))
            v2 += 4.0 * (r12 - c * r6)

    return v1 + v2


# ---------------------------------------------------------------------------
# Factory for EDA-style fitness functions (maximize ``-energy``)
# ---------------------------------------------------------------------------

def make_ab_fitness(
    sequence: Union[str, Sequence[int], np.ndarray, None] = None,
    n_residues: int = 13,
) -> Callable[[np.ndarray], float]:
    """
    Build a fitness function ``f(angles) -> -energy`` ready for an EDA.

    Either pass ``sequence`` explicitly or let the helper construct a
    Fibonacci AB sequence of ``n_residues`` residues.

    Returns
    -------
    Callable[[np.ndarray], float]
        Function suitable for ``fitness_func`` in any continuous EDA wrapper
        (since pateda EDAs maximise, the energy is negated).
    """
    if sequence is None:
        seq = fibonacci_ab_sequence(n_residues)
    else:
        seq = parse_ab_sequence(sequence)

    def _fitness(angles: np.ndarray) -> float:
        return -ab_energy_2d(angles, seq)

    _fitness.sequence = seq
    _fitness.n_residues = len(seq)
    _fitness.n_angles = len(seq) - 2
    return _fitness


# Convenience: standard Fibonacci-13 and Fibonacci-21 benchmark sequences
F13_AB_SEQUENCE = fibonacci_ab_sequence(13)
F21_AB_SEQUENCE = fibonacci_ab_sequence(21)
F34_AB_SEQUENCE = fibonacci_ab_sequence(34)
