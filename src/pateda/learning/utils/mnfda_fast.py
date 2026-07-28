"""
Vectorized, numerically-exact kernels for MN-FDA (proposals A, B, C, E of
``MN-FDA_analysis.md``).

These functions reproduce *exactly* (up to floating-point summation order) the
output of the reference implementations in ``mutual_information.py``,
``probability_tables.py`` and ``markov_network.py``, but replace the Python
per-sample / per-pair loops that dominate the MN-FDA runtime with numpy /
BLAS-backed operations.

They are used **only** by ``LearnMNFDA`` (and the new ``LearnMNFDASparse``); the
G-test variants ``LearnMNFDAG`` / ``LearnMNEDAG`` deliberately keep the original
reference kernels so their behaviour is untouched while the optimisations are
being validated.

Heterogeneous cardinalities
---------------------------
* Proposal B (mutual information): a fully-vectorized fast path is used when all
  variables share the same cardinality (the common binary-PBO case).  When the
  cardinalities are heterogeneous we fall back to a per-pair path that is still
  vectorized over samples (no Python per-sample loop), so MN-FDA keeps working —
  just with a smaller speed-up — for mixed-cardinality problems.
* Proposal C (probability tables): the bincount formulation uses mixed-radix
  indices and is therefore fully general (identical for homogeneous and
  heterogeneous cardinalities).
* Proposal E (clique ordering): purely combinatorial, independent of
  cardinality.
"""

from typing import List, Optional
import numpy as np
from scipy import stats as scipy_stats

from pateda.learning.utils.conversions import find_acc_card
from pateda.learning.utils.weights import weighted_bivariate_counts

_LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Proposal B: vectorized mutual-information matrix
# ---------------------------------------------------------------------------
def _mi_from_joint(joint: np.ndarray) -> float:
    """Mutual information (in bits) of a 2-D joint *count* table."""
    total = joint.sum()
    if total <= 0:
        return 0.0
    p = joint / total
    pi = p.sum(axis=1)
    pj = p.sum(axis=0)
    outer = np.outer(pi, pj)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / outer[mask])))


def _mi_matrix_hetero(population, cardinality, weights):
    """Per-pair MI for heterogeneous cardinalities (vectorized over samples)."""
    n_vars = population.shape[1]
    mi = np.zeros((n_vars, n_vars))
    card = cardinality.astype(int)
    cw = None if weights is None else np.asarray(weights, dtype=float)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            joint = weighted_bivariate_counts(
                population[:, i], population[:, j],
                int(card[i]), int(card[j]), cw,
            )
            m = _mi_from_joint(joint)
            mi[i, j] = m
            mi[j, i] = m
    return mi


def _mi_matrix_homogeneous(population, c, weights):
    """Fully-vectorized MI matrix when every variable has cardinality ``c``.

    Builds a one-hot design matrix ``D`` (N x n*c); the weighted co-occurrence
    matrix ``G = (w*D)^T D`` (n*c x n*c) holds every pairwise joint count as a
    ``c x c`` block, from which all mutual informations are computed with numpy
    broadcasting.
    """
    N, n = population.shape
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)

    # One-hot: OH[s, i*c + v] = 1 iff population[s, i] == v.
    oh = np.zeros((N, n * c))
    cols = population.astype(int) + (np.arange(n) * c)[None, :]
    oh[np.arange(N)[:, None], cols] = 1.0

    ohw = oh * w[:, None]
    G = ohw.T @ oh                       # (n*c, n*c) weighted joint counts
    counts = G.reshape(n, c, n, c)       # counts[i, a, j, b]
    total = w.sum()
    if total <= 0:
        return np.zeros((n, n))

    p = counts / total                   # p(x_i=a, x_j=b)
    # marginals from the diagonal blocks: p(x_i=a) == sum_b counts[i,a,i,b]
    pmarg = np.einsum("iaib->ia", counts) / total          # (n, c)

    denom = pmarg[:, :, None, None] * pmarg[None, None, :, :]   # (n,c,n,c)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, p / denom, 1.0)
        terms = np.where(p > 0, p * np.log2(ratio), 0.0)
    mi = terms.sum(axis=(1, 3))          # (n, n)
    np.fill_diagonal(mi, 0.0)
    # symmetrize defensively against float asymmetry
    mi = 0.5 * (mi + mi.T)
    np.fill_diagonal(mi, 0.0)
    return mi


def compute_mi_matrix_fast(population, cardinality, weights=None):
    """Vectorized pairwise mutual-information matrix (bits).

    Numerically identical to
    ``mutual_information.compute_mutual_information_matrix`` (proposal B).
    """
    card = np.asarray(cardinality).astype(int)
    if card.min() == card.max():
        return _mi_matrix_homogeneous(population, int(card[0]), weights)
    return _mi_matrix_hetero(population, card, weights)


# ---------------------------------------------------------------------------
# Proposal A: vectorized chi-square dependency graph
# ---------------------------------------------------------------------------
def chi2_adjacency(mi_matrix, n_samples, threshold=0.05):
    """Dependency graph from the chi-square test, vectorized (proposal A).

    Reproduces ``_build_dependency_graph`` / ``chi_square_test`` (df = 1) but
    evaluates the critical value ``chi2.ppf(1 - threshold, 1)`` **once** instead
    of once per variable pair.
    """
    critical_value = scipy_stats.chi2.ppf(1 - threshold, 1)
    chi2_stat = 2.0 * n_samples * mi_matrix * _LN2
    adjacency = (chi2_stat > critical_value).astype(int)
    np.fill_diagonal(adjacency, 1)
    return adjacency


# ---------------------------------------------------------------------------
# MN-FDA-S: per-variable cliques instead of full maximal-clique enumeration
# ---------------------------------------------------------------------------
def build_per_variable_cliques(mi_matrix, adjacency, max_clique_size):
    """One clique per variable, avoiding maximal-clique enumeration.

    For each variable ``x_i`` the clique is ``x_i`` together with the
    ``max_clique_size - 1`` variables of strongest mutual information *among
    those that passed the chi-square test with* ``x_i`` (i.e. ``adjacency[i]``).
    If no variable passed the test the clique is the singleton ``[i]``.

    Returns a list of ``n`` cliques, each a sorted 1-D int array of size at most
    ``max_clique_size``.
    """
    n = adjacency.shape[0]
    k = int(max_clique_size)
    cliques = []
    for i in range(n):
        nb = np.where(adjacency[i] > 0)[0]
        nb = nb[nb != i]
        if len(nb) == 0 or k <= 1:
            cliques.append(np.array([i], dtype=int))
            continue
        if len(nb) > k - 1:
            nb = nb[np.argsort(-mi_matrix[i, nb])[:k - 1]]
        clique = np.concatenate([[i], nb]).astype(int)
        cliques.append(np.sort(clique))
    return cliques


def remove_subsumed_cliques(cliques):
    """Remove duplicate and subsumed cliques.

    A clique is dropped when it is a subset of (or identical to) another clique;
    the result is the set of maximal cliques among the inputs.  Every variable
    that appeared in some input clique still appears in a kept clique, so
    coverage is preserved.
    """
    sets = [frozenset(int(v) for v in c) for c in cliques]
    order = sorted(range(len(cliques)), key=lambda idx: -len(sets[idx]))
    kept, kept_sets = [], []
    for idx in order:
        s = sets[idx]
        if any(s <= ks for ks in kept_sets):
            continue
        kept_sets.append(s)
        kept.append(np.asarray(cliques[idx], dtype=int))
    return kept


# ---------------------------------------------------------------------------
# Prune redundant cliques for PLS sampling
# ---------------------------------------------------------------------------
def prune_empty_cliques(structure: np.ndarray) -> np.ndarray:
    """Drop factorized-structure rows that introduce no new variable.

    In a PLS factorization every variable is *new* in exactly one clique, so at
    most ``n`` cliques actually contribute to sampling; a clique whose variables
    were all already sampled by earlier cliques has ``n_new == 0`` and samples
    nothing.  On a dense (uncorrected) dependency graph
    ``find_maximal_cliques_greedy`` can emit thousands of such redundant rows,
    which ``SampleFDA`` would then loop over needlessly.

    Removing the ``n_new == 0`` rows is numerically exact — those rows assign no
    variable — and the ordering of the remaining (productive) cliques is
    preserved, so every clique's overlap variables are still sampled by an
    earlier kept clique.  The result has at most ``n`` rows.
    """
    if structure.shape[0] == 0:
        return structure
    keep = structure[:, 1] > 0
    return structure[keep]


# ---------------------------------------------------------------------------
# Proposal C: vectorized clique probability tables
# ---------------------------------------------------------------------------
def _marginal_table_fast(population, new_vars, cardinality, weights, prior):
    new_cards = cardinality[new_vars].astype(int)
    acc = find_acc_card(len(new_vars), new_cards)
    n_configs = int(np.prod(new_cards))
    idx = (population[:, new_vars].astype(int) @ acc).astype(int)
    if weights is None:
        counts = np.bincount(idx, minlength=n_configs)[:n_configs].astype(float)
        w_total = population.shape[0]
    else:
        w = np.asarray(weights, dtype=float)
        counts = np.bincount(idx, weights=w, minlength=n_configs)[:n_configs]
        w_total = float(w.sum())
    if prior:
        freq = 1.0 + counts
        total = w_total + n_configs
    else:
        freq = counts
        total = w_total
    return freq / total


def _conditional_table_fast(population, overlap_vars, new_vars, cardinality,
                            weights, prior):
    overlap_cards = cardinality[overlap_vars].astype(int)
    new_cards = cardinality[new_vars].astype(int)
    overlap_acc = find_acc_card(len(overlap_vars), overlap_cards)
    new_acc = find_acc_card(len(new_vars), new_cards)
    n_overlap_configs = int(np.prod(overlap_cards))
    n_new_configs = int(np.prod(new_cards))

    o_idx = (population[:, overlap_vars].astype(int) @ overlap_acc).astype(int)
    n_idx = (population[:, new_vars].astype(int) @ new_acc).astype(int)
    flat = o_idx * n_new_configs + n_idx
    size = n_overlap_configs * n_new_configs
    if weights is None:
        counts = np.bincount(flat, minlength=size)[:size].astype(float)
    else:
        w = np.asarray(weights, dtype=float)
        counts = np.bincount(flat, weights=w, minlength=size)[:size]
    freq = counts.reshape(n_overlap_configs, n_new_configs)
    if prior:
        freq = 1.0 + freq
    row_sum = freq.sum(axis=1)
    prob = np.zeros((n_overlap_configs, n_new_configs))
    nz = row_sum > 0
    prob[nz] = freq[nz] / row_sum[nz, None]
    prob[~nz] = 1.0 / n_new_configs      # uniform fallback (only if no prior)
    return prob


def compute_clique_tables_fast(population, structure, cardinality,
                               weights=None, prior=True):
    """Vectorized version of ``probability_tables.compute_clique_tables``
    (proposal C).  Fully general for heterogeneous cardinalities."""
    cardinality = np.asarray(cardinality)
    n_cliques = structure.shape[0]
    tables = []
    for c in range(n_cliques):
        n_overlap = int(structure[c, 0])
        n_new = int(structure[c, 1])
        if n_overlap == 0:
            new_vars = structure[c, 2:2 + n_new].astype(int)
            tables.append(_marginal_table_fast(
                population, new_vars, cardinality, weights, prior))
        else:
            overlap_vars = structure[c, 2:2 + n_overlap].astype(int)
            new_vars = structure[c, 2 + n_overlap:2 + n_overlap + n_new].astype(int)
            tables.append(_conditional_table_fast(
                population, overlap_vars, new_vars, cardinality, weights, prior))
    return tables


# ---------------------------------------------------------------------------
# Proposal E: incremental clique ordering (identical output, faster build)
# ---------------------------------------------------------------------------
def order_cliques_for_sampling_fast(cliques: List[np.ndarray]) -> np.ndarray:
    """Drop-in replacement for ``order_cliques_for_sampling`` producing the
    **identical** ordering (proposal E).

    The reference builds the O(n_cliques^2) dependency relation by intersecting
    every pair of cliques.  Here we build the *same* relation using an inverted
    ``variable -> cliques`` index, so only clique pairs that actually share a
    variable are examined, and then run the identical Kahn (FIFO) topological
    sort.  The resulting order — and therefore the factorization — is unchanged.
    """
    n_cliques = len(cliques)
    var_sets = [set(int(v) for v in c) for c in cliques]

    # inverted index: variable -> cliques containing it
    var_to_cliques = {}
    for idx, s in enumerate(var_sets):
        for v in s:
            var_to_cliques.setdefault(v, []).append(idx)

    # depends_on[i] = { j != i : share a variable and vars_i is NOT subset of vars_j }
    depends_on = [set() for _ in range(n_cliques)]
    for i in range(n_cliques):
        si = var_sets[i]
        candidates = set()
        for v in si:
            candidates.update(var_to_cliques[v])
        candidates.discard(i)
        for j in candidates:
            if not si.issubset(var_sets[j]):
                depends_on[i].add(j)

    in_degree = [len(d) for d in depends_on]
    # dependents[j] = sorted list of i that depend on j (ascending i, as the
    # reference's `for i in range(n_cliques)` removal scan implies)
    dependents = [[] for _ in range(n_cliques)]
    for i in range(n_cliques):
        for j in depends_on[i]:
            dependents[j].append(i)
    for j in range(n_cliques):
        dependents[j].sort()

    queue = [i for i in range(n_cliques) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for i in dependents[node]:
            in_degree[i] -= 1
            if in_degree[i] == 0:
                queue.append(i)

    if len(order) != n_cliques:          # cycle -> reference falls back
        order = list(range(n_cliques))
    return np.array(order, dtype=int)
