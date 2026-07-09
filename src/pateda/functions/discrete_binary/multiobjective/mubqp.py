"""
Multi-objective Unconstrained Binary Quadratic Programming (mUBQP).

Each objective is a quadratic pseudo-boolean function

    f_k(x) = sum_{i,j} Q^k_{ij} x_i x_j ,   x in {0, 1}^n

(diagonal entries ``Q^k_{ii}`` are the linear terms, since ``x_i^2 = x_i``).
All objectives are **maximised**, matching pateda's multi-objective convention.

This module implements the instance families analysed in the reference MATLAB
code (``Multi_Objective_Code/mUBQP``) and the paper *"On the Design of Hard
mUBQP Instances"* (Liefooghe, Verel, Santana et al.):

* :func:`generate_mubqp` -- the standard Liefooghe generator parameterised by
  matrix *density* ``d`` and inter-objective *correlation* ``rho``.
* :func:`create_artificial_mubqp` -- the five hand-designed structured instance
  types (``CreateUBQInstance.m``).
* Hard-instance construction from order-5 building blocks
  (``SearchHarduBQPInstances.m``): enumerate all order-5 UBQP "chunks", score
  each pair with difficulty metrics (Pareto-set size, Boltzmann deception,
  fitness-distance correlation) and compose large instances by tiling
  (:func:`create_mubqp_from_chunk`) or random overlapping placement
  (:func:`create_heavy_mubqp_from_chunks`) of the hard chunks.

The instance file format (``.dat``) matches the MATLAB code::

    <seed>
    <n> <n_edges_1>
    <i> <j> <w>            # n_edges_1 lines, 1-based indices, objective 1
    <n> <n_edges_2>
    <i> <j> <w>            # objective 2
    ...
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from itertools import combinations
import numpy as np


# --------------------------------------------------------------------------- #
# Instance representation and evaluation
# --------------------------------------------------------------------------- #

class MUBQPInstance:
    """A multi-objective UBQP instance: one weight matrix per objective.

    Parameters
    ----------
    n_vars : int
        Number of binary variables.
    matrices : sequence of (n, n) ndarray
        One weight matrix ``Q^k`` per objective.  Entries may be stored in any
        (e.g. upper-triangular) layout; evaluation uses ``x^T Q^k x`` directly.
    """

    def __init__(self, n_vars: int, matrices: Sequence[np.ndarray], seed: int = 0):
        self.n_vars = int(n_vars)
        self.matrices = [np.asarray(Q, dtype=float) for Q in matrices]
        self.n_objectives = len(self.matrices)
        self.seed = int(seed)
        for Q in self.matrices:
            if Q.shape != (self.n_vars, self.n_vars):
                raise ValueError(f"matrix shape {Q.shape} != ({n_vars}, {n_vars})")

    # -- evaluation --------------------------------------------------------- #
    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        """Objective vector ``(f_1, ..., f_m)`` of one solution (maximised)."""
        x = np.asarray(x, dtype=float).ravel()
        return np.array([float(x @ Q @ x) for Q in self.matrices])

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Objective values for a 1-D individual or a 2-D population."""
        population = np.asarray(population, dtype=float)
        if population.ndim == 1:
            return self.evaluate_single(population)
        # F[p, k] = x_p^T Q^k x_p
        out = np.empty((population.shape[0], self.n_objectives))
        for k, Q in enumerate(self.matrices):
            out[:, k] = np.einsum("pi,ij,pj->p", population, Q, population)
        return out

    # -- I/O ---------------------------------------------------------------- #
    def edges(self, k: int) -> List[Tuple[int, int, float]]:
        """Non-zero ``(i, j, w)`` edges of objective ``k`` (0-based indices)."""
        Q = self.matrices[k]
        ii, jj = np.nonzero(Q)
        return [(int(i), int(j), float(Q[i, j])) for i, j in zip(ii, jj)]

    def save(self, filepath: str) -> None:
        """Write the instance in the packaged ``.dat`` format."""
        with open(filepath, "w") as fh:
            fh.write(f"{self.seed}\n")
            for k in range(self.n_objectives):
                edges = self.edges(k)
                fh.write(f"{self.n_vars} {len(edges)}\n")
                for (i, j, w) in edges:
                    # store 1-based indices; integer weights printed without decimals
                    wv = int(w) if float(w).is_integer() else w
                    fh.write(f"{i + 1} {j + 1} {wv}\n")

    @classmethod
    def load(cls, filepath: str) -> "MUBQPInstance":
        """Read an instance written by :meth:`save` (or the MATLAB code)."""
        with open(filepath) as fh:
            tokens = fh.read().split()
        p = 0
        seed = int(float(tokens[p])); p += 1
        matrices = []
        n_vars = None
        # keep reading objective blocks until tokens are exhausted
        while p < len(tokens):
            n = int(tokens[p]); n_edges = int(tokens[p + 1]); p += 2
            n_vars = n
            Q = np.zeros((n, n))
            for _ in range(n_edges):
                i = int(tokens[p]) - 1
                j = int(tokens[p + 1]) - 1
                w = float(tokens[p + 2])
                p += 3
                Q[i, j] += w
            matrices.append(Q)
        return cls(n_vars, matrices, seed=seed)


def create_mubqp_objective_function(instance: MUBQPInstance):
    """Return an objective ``f(pop) -> (pop, n_obj)`` maximising all objectives."""
    def objective(population: np.ndarray) -> np.ndarray:
        return instance.evaluate(population)
    return objective


# --------------------------------------------------------------------------- #
# Standard Liefooghe generator: density + objective correlation
# --------------------------------------------------------------------------- #

def generate_mubqp(n_vars: int, n_objectives: int = 2, density: float = 0.4,
                   rho: float = 0.0, weight_range: Tuple[float, float] = (-100.0, 100.0),
                   seed: Optional[int] = None) -> MUBQPInstance:
    """Standard mUBQP instance (Liefooghe et al.).

    A shared sparsity mask (fraction ``density`` of the upper-triangular cells,
    including the diagonal) carries non-zero weights.  Weights across objectives
    are drawn with target pairwise correlation ``rho`` (negative -> conflicting
    -> harder; positive -> aligned -> easier), then mapped to ``weight_range``.

    Args:
        n_vars: Number of variables.
        n_objectives: Number of objectives.
        density: Fraction of variable pairs (incl. diagonal) with a non-zero weight.
        rho: Target correlation between objective weight matrices (``-1 < rho < 1``);
            exact only for the bi-objective case.
        weight_range: ``(min, max)`` weight range.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n_vars)
    mask = rng.random(len(iu)) < density
    iu, ju = iu[mask], ju[mask]
    n_terms = len(iu)

    lo, hi = weight_range
    scale = (hi - lo) / 2.0
    shift = (hi + lo) / 2.0

    base = rng.standard_normal(n_terms)
    matrices = []
    for o in range(n_objectives):
        indep = rng.standard_normal(n_terms)
        z = base if o == 0 else rho * base + np.sqrt(max(0.0, 1.0 - rho * rho)) * indep
        weights = np.clip(z, -3.0, 3.0) / 3.0 * scale + shift
        Q = np.zeros((n_vars, n_vars))
        Q[iu, ju] = weights
        matrices.append(Q)
    return MUBQPInstance(n_vars, matrices, seed=seed if seed is not None else 0)


# --------------------------------------------------------------------------- #
# Artificial structured instance types (CreateUBQInstance.m, itype 1..5)
# --------------------------------------------------------------------------- #

def create_artificial_mubqp(n_vars: int, itype: int, seed: int = 10) -> MUBQPInstance:
    """Build one of the five hand-designed bi-objective UBQP structures.

    Ports ``CreateUBQInstance.m`` (itype in 1..5).  Objective 2 is (broadly) the
    sign-negated counterpart of objective 1, producing conflicting objectives
    with a controlled interaction structure.
    """
    n = n_vars
    Q1 = np.zeros((n, n))
    Q2 = np.zeros((n, n))

    def add(Q, i, j, w):  # 1-based (i, j) as in the MATLAB source
        Q[i - 1, j - 1] += w

    if itype == 1:
        for i in range(1, n + 1):
            add(Q1, i, i, 1); add(Q2, i, i, -1)
    elif itype == 2:
        for i in range(1, n + 1):
            add(Q1, i, i, i); add(Q2, i, i, -i)
    elif itype == 3:
        add(Q1, 1, 1, -1); add(Q2, 1, 1, 1)
        for i in range(2, n + 1):
            add(Q1, i - 1, i, 3); add(Q2, i - 1, i, -3)
            add(Q1, i, i, -1); add(Q2, i, i, 1)
    elif itype == 4:
        add(Q1, 1, 1, -1); add(Q2, 1, 1, 1)
        for i in range(2, n + 1):
            add(Q1, i - 1, i, 3 * i - 1); add(Q2, i - 1, i, -3 * i + 1)
            add(Q1, i, i, -i); add(Q2, i, i, i)
    elif itype == 5:
        if n % 4 != 0:
            raise ValueError("itype 5 requires n divisible by 4")
        base = [(1, 2, 1), (1, 4, 1), (2, 3, 1), (2, 4, -5), (3, 4, 1)]
        for block in range(n // 4):
            off = 4 * block
            for (i, j, w) in base:
                add(Q1, i + off, j + off, w)
                add(Q2, i + off, j + off, -w)
    else:
        raise ValueError(f"itype must be in 1..5, got {itype}")
    return MUBQPInstance(n, [Q1, Q2], seed=seed)


# --------------------------------------------------------------------------- #
# Hard-instance construction from order-5 building blocks
# --------------------------------------------------------------------------- #

_ORDER5_EDGES = list(combinations(range(5), 2))  # 10 pairwise edges over 5 vars


def enumerate_order5_chunks() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Enumerate all ``2^10`` order-5 UBQP chunk configurations.

    Each chunk is a UBQP over 5 variables whose 10 pairwise interaction weights
    take values in ``{-1, +1}`` (``SearchHarduBQPInstances.m``).

    Returns ``(params, edges)`` where ``params`` has shape ``(1024, 10)`` (weight
    per edge, in ``{-1, +1}``) and ``edges`` is the list of 10 ``(i, j)`` pairs.
    """
    n_edges = len(_ORDER5_EDGES)          # 10
    n_conf = 1 << n_edges                 # 1024
    bits = ((np.arange(n_conf)[:, None] >> np.arange(n_edges)[None, :]) & 1)
    params = 2 * bits - 1                 # {0,1} -> {-1,+1}
    return params.astype(float), _ORDER5_EDGES


def _chunk_matrix(weights: Sequence[float]) -> np.ndarray:
    """5x5 upper-triangular weight matrix from a length-10 edge-weight vector."""
    Q = np.zeros((5, 5))
    for (i, j), w in zip(_ORDER5_EDGES, weights):
        Q[i, j] = w
    return Q


def _all_binary(n: int) -> np.ndarray:
    """All ``2^n`` binary vectors, shape ``(2^n, n)``."""
    idx = np.arange(1 << n)
    return ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(float)


def _pareto_set_size(F: np.ndarray) -> int:
    """Number of non-dominated points (maximisation) in objective array ``F``."""
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(F >= F[i], axis=1)
        gt = np.any(F > F[i], axis=1)
        if np.any(ge & gt):
            dominated[i] = True
    # count distinct non-dominated objective vectors
    nd = F[~dominated]
    return int(len(np.unique(nd, axis=0)))


def chunk_deception(weights: Sequence[float], temperature: float = 1.0) -> float:
    """Degree of deception of an order-5 chunk (``SearchHarduBQPInstances.m``).

    Ranks all solutions by the univariate Boltzmann probability of the function
    and returns the mean univariate rank of the global optimum: a high rank
    means the univariate model is misled away from the optimum (deceptive).
    """
    pop = _all_binary(5)
    Q = _chunk_matrix(weights)
    fit = np.einsum("pi,ij,pj->p", pop, Q, pop)
    # Boltzmann distribution and univariate marginals P(x_i = 1)
    bz = np.power(2.0, fit / temperature)
    bz /= bz.sum()
    univ = np.array([bz[pop[:, i] == 1].sum() for i in range(5)])
    # univariate probability of each full solution under the product model
    prob = np.prod(np.where(pop == 1, univ, 1.0 - univ), axis=1)
    order = np.argsort(prob)             # ascending; rank 1 = least probable
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    best = np.flatnonzero(fit == fit.max())
    return float(np.mean(rank[best]))


def chunk_pair_metrics(w1: Sequence[float], w2: Sequence[float],
                       temperature: float = 1.0) -> dict:
    """Difficulty metrics for a bi-objective order-5 chunk pair.

    Returns a dict with ``pareto_size`` (number of non-dominated points),
    ``deception_1`` / ``deception_2`` (per-objective Boltzmann deception) and
    ``fdc`` (fitness-distance correlation of objective 1 -- lower/negative is
    harder).
    """
    pop = _all_binary(5)
    F = np.column_stack([
        np.einsum("pi,ij,pj->p", pop, _chunk_matrix(w1), pop),
        np.einsum("pi,ij,pj->p", pop, _chunk_matrix(w2), pop),
    ])
    return {
        "pareto_size": _pareto_set_size(F),
        "deception_1": chunk_deception(w1, temperature),
        "deception_2": chunk_deception(w2, temperature),
        "fdc": _fitness_distance_correlation(pop, F[:, 0]),
    }


def _fitness_distance_correlation(pop: np.ndarray, fit: np.ndarray) -> float:
    """Correlation between fitness and Hamming distance to the global optimum."""
    best = pop[int(np.argmax(fit))]
    dist = np.sum(pop != best, axis=1)
    if np.std(fit) == 0 or np.std(dist) == 0:
        return 0.0
    return float(np.corrcoef(fit, dist)[0, 1])


def chunk_pair_hardness(metrics: dict) -> float:
    """Scalar hardness score for a chunk pair (higher = harder).

    Combines the paper's difficulty signals: a large Pareto set (many
    non-dominated points), strong single-objective deception (the global optimum
    is improbable under the univariate Boltzmann model, i.e. *small* mean rank),
    and a negative fitness-distance correlation.  Deception is folded in as
    ``(2^5 - deception)`` so that smaller mean ranks increase the score.
    """
    n = 1 << 5
    decept = (n - metrics["deception_1"]) + (n - metrics["deception_2"])
    return metrics["pareto_size"] + 0.05 * decept - metrics["fdc"]


def select_hard_chunk_pairs(max_pairs: int = 20, n_candidates: int = 4000,
                            min_pareto: int = 1, seed: Optional[int] = None
                            ) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Search order-5 chunk pairs and return the hardest ones found.

    Follows ``SearchHarduBQPInstances.m``: score bi-objective order-5 chunk
    pairs by their difficulty (Pareto-set size, Boltzmann deception,
    fitness-distance correlation) and keep the hardest.  Because the full
    ``1024 x 1024`` scan is expensive, ``n_candidates`` random pairs are
    evaluated and ranked by :func:`chunk_pair_hardness`; pairs with a Pareto set
    smaller than ``min_pareto`` are discarded.

    Returns a list of ``(weights_1, weights_2, metrics)`` tuples, hardest first.
    """
    params, _ = enumerate_order5_chunks()
    n_conf = params.shape[0]
    rng = np.random.default_rng(seed)
    scored: List[Tuple[float, np.ndarray, np.ndarray, dict]] = []
    seen = set()
    for _ in range(n_candidates):
        i, j = int(rng.integers(n_conf)), int(rng.integers(n_conf))
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        m = chunk_pair_metrics(params[i], params[j])
        if m["pareto_size"] < min_pareto:
            continue
        scored.append((chunk_pair_hardness(m), params[i].copy(), params[j].copy(), m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(w1, w2, m) for (_, w1, w2, m) in scored[:max_pairs]]


def create_mubqp_from_chunk(chunk_w1: Sequence[float], chunk_w2: Sequence[float],
                            n_vars: int, k: int = 5, seed: int = 10) -> MUBQPInstance:
    """Tile a single order-``k`` chunk into a block-separable mUBQP instance.

    Ports ``CreateUBQInstanceFromChunk.m``: the chunk is replicated on
    ``n_vars / k`` consecutive, non-overlapping blocks of variables.
    """
    if n_vars % k != 0:
        raise ValueError(f"n_vars ({n_vars}) must be a multiple of k ({k})")
    edges = list(combinations(range(k), 2))
    Q1 = np.zeros((n_vars, n_vars))
    Q2 = np.zeros((n_vars, n_vars))
    for block in range(n_vars // k):
        off = block * k
        for (i, j), w1, w2 in zip(edges, chunk_w1, chunk_w2):
            Q1[i + off, j + off] += w1
            Q2[i + off, j + off] += w2
    return MUBQPInstance(n_vars, [Q1, Q2], seed=seed)


def create_heavy_mubqp_from_chunks(chunk_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                                   n_vars: int, k: int = 5, n_chunks: Optional[int] = None,
                                   seed: int = 101) -> MUBQPInstance:
    """Compose a dense ("heavy") mUBQP by random overlapping chunk placement.

    Ports ``CreateHeavyUBQInstanceFromListChunks.m``: repeatedly pick a hard
    chunk pair and lay it on a random size-``k`` subset of the ``n_vars``
    variables, accumulating weights.  Overlapping placements create rich,
    non-separable interaction structure.

    Args:
        chunk_pairs: pool of ``(weights_1, weights_2)`` hard chunk pairs.
        n_vars: number of variables of the composed instance.
        k: chunk order (number of variables per placement).
        n_chunks: number of placements (default ``n_vars``).
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    edges = list(combinations(range(k), 2))
    Q1 = np.zeros((n_vars, n_vars))
    Q2 = np.zeros((n_vars, n_vars))
    n_place = n_chunks if n_chunks is not None else n_vars
    pool = list(chunk_pairs)
    for _ in range(n_place):
        w1, w2 = pool[int(rng.integers(len(pool)))]
        order = rng.permutation(n_vars)[:k]
        for (a, b), c1, c2 in zip(edges, w1, w2):
            i, j = int(order[a]), int(order[b])
            lo, hi = min(i, j), max(i, j)  # keep upper-triangular
            Q1[lo, hi] += c1
            Q2[lo, hi] += c2
    return MUBQPInstance(n_vars, [Q1, Q2], seed=seed)
