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
from collections import defaultdict
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
def _pair_dof(cardinality, i_idx, j_idx):
    """Degrees of freedom ``(r_i - 1)(c_j - 1)`` for each variable pair
    (i_idx, j_idx), clipped to >= 1 (guards constant variables / self-pairs)."""
    card = np.asarray(cardinality, dtype=int)
    return np.maximum((card[i_idx] - 1) * (card[j_idx] - 1), 1).astype(float)


def chi2_adjacency(mi_matrix, n_samples, cardinality=None, threshold=0.05):
    """Dependency graph from the chi-square / G-test, vectorized (proposal A).

    The statistic is the likelihood-ratio (G) statistic ``2 * n_samples * MI *
    ln2``, asymptotically chi-square with **df = (r_i - 1)(c_j - 1)** for a pair
    of variables with cardinalities ``r_i``, ``c_j``.

    Pass ``cardinality`` (1-D array) to use the correct per-pair degrees of
    freedom -- required for **heterogeneous / non-binary** variables, where the
    old fixed ``df = 1`` (only correct for binary 2x2 tables) over-detects edges
    on the higher-cardinality variables.  ``cardinality=None`` keeps the legacy
    ``df = 1`` behaviour (binary).

    Pass the *effective* sample size (see :func:`effective_sample_size`) as
    ``n_samples`` to correct for weighted (customized-selection) counts.
    """
    chi2_stat = 2.0 * n_samples * mi_matrix * _LN2
    if cardinality is None:
        adjacency = (chi2_stat > scipy_stats.chi2.ppf(1 - threshold, 1)).astype(int)
    else:
        n = mi_matrix.shape[0]
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        df = _pair_dof(cardinality, rows, cols)
        adjacency = (chi2_stat > scipy_stats.chi2.ppf(1 - threshold, df)).astype(int)
    np.fill_diagonal(adjacency, 1)
    return adjacency


def effective_sample_size(weights, n_samples):
    """Effective sample size (ESS) of a count-scale weight vector.

    ``ESS = (sum w)^2 / sum(w^2)`` -- Kish's effective sample size.  For the
    uniform case (``weights is None``, or all weights equal) this returns
    ``n_samples`` exactly, so ``chi2 = 2 * ESS * MI * ln2`` reduces to the raw
    ``2 * N * MI * ln2``.  Under concentrated (e.g. Boltzmann) weights the ESS is
    smaller, which shrinks the chi-square statistic and therefore removes the
    spurious dependencies that would otherwise densify the graph.

    Args:
        weights: Count-scale weights (``N * p``, summing to ``N``) or ``None``.
        n_samples: Raw number of samples ``N`` (returned when ``weights`` is
            ``None``).

    Returns:
        Effective sample size (float), in ``[1, n_samples]``.
    """
    if weights is None:
        return float(n_samples)
    w = np.asarray(weights, dtype=float)
    s1 = w.sum()
    s2 = np.square(w).sum()
    if s2 <= 0.0:
        return float(n_samples)
    return float(s1 * s1 / s2)


# ---------------------------------------------------------------------------
# Sparse dependency graph (multiple-testing correction + bounded neighbourhood)
# Relocated here from the (now-excluded) mnfda_sparse module so the current
# MN-FDA learner can reuse it.
# ---------------------------------------------------------------------------
def corrected_adjacency(mi_matrix, n_samples, alpha, correction, cardinality=None):
    """Symmetric binary adjacency from chi-square p-values with a
    multiple-testing correction over the ``n(n-1)/2`` pairwise tests.

    ``correction`` in ``{"none", "bonferroni", "holm", "fdr_bh"}``.  As in
    :func:`chi2_adjacency`, pass the effective sample size as ``n_samples`` to
    account for weighted counts, and ``cardinality`` (1-D array) to use the
    correct per-pair degrees of freedom ``(r_i - 1)(c_j - 1)`` for
    heterogeneous / non-binary variables (``None`` -> legacy ``df = 1``).
    """
    n = mi_matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    chi2_stat = 2.0 * n_samples * mi_matrix[iu, ju] * _LN2
    df = 1 if cardinality is None else _pair_dof(cardinality, iu, ju)
    pvals = scipy_stats.chi2.sf(chi2_stat, df)          # 1 - cdf
    m = pvals.size

    correction = (correction or "none").lower()
    if correction in ("none", "None"):
        reject = pvals < alpha
    elif correction == "bonferroni":
        reject = pvals < (alpha / m)
    elif correction == "holm":
        order = np.argsort(pvals)
        thresh = alpha / (m - np.arange(m))
        sorted_reject = pvals[order] < thresh
        if not sorted_reject.all():
            first_fail = np.argmin(sorted_reject)      # first False
            sorted_reject[first_fail:] = False
        reject = np.zeros(m, dtype=bool)
        reject[order] = sorted_reject
    elif correction in ("fdr_bh", "bh", "fdr"):
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, m + 1) / m)
        below = ranked <= thresh
        if below.any():
            kmax = int(np.max(np.where(below)[0]))
            reject_sorted = np.zeros(m, dtype=bool)
            reject_sorted[: kmax + 1] = True
        else:
            reject_sorted = np.zeros(m, dtype=bool)
        reject = np.zeros(m, dtype=bool)
        reject[order] = reject_sorted
    else:
        raise ValueError(f"Unknown correction {correction!r}")

    adjacency = np.zeros((n, n), dtype=int)
    adjacency[iu[reject], ju[reject]] = 1
    adjacency[ju[reject], iu[reject]] = 1
    return adjacency


def neff_for_target_degree(mi_matrix, alpha, correction, cardinality,
                           target_degree, n_max):
    """Effective sample size in ``[1, n_max]`` whose FDR-corrected dependency
    graph has mean degree closest to ``target_degree`` (scale-aware tuning).

    The mean degree is monotone increasing in the effective sample size, so a
    bisection suffices.  This picks the *aggressiveness* of the significance test
    to hit a target density directly, which interpolates automatically between
    the raw ``N`` (small ``n``, where the raw test is already about right) and a
    smaller effective size (large ``n``, where the raw test over-connects) --
    with the weighting already folded into ``mi_matrix``.  If even ``n_eff =
    n_max`` does not reach the target the raw ``n_max`` is returned; if ``n_eff =
    1`` is already too dense, ``1`` is returned.
    """
    n = mi_matrix.shape[0]

    def mean_deg(neff):
        adj = corrected_adjacency(mi_matrix, neff, alpha, correction, cardinality)
        np.fill_diagonal(adj, 0)
        return adj.sum() / n

    if mean_deg(float(n_max)) <= target_degree:
        return float(n_max)
    if mean_deg(1.0) >= target_degree:
        return 1.0
    lo, hi = 1.0, float(n_max)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if mean_deg(mid) > target_degree:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bound_neighborhood(adjacency, mi_matrix, max_neighborhood):
    """Keep, per variable, only the ``max_neighborhood`` highest-MI neighbours;
    symmetrize by union.  Returns a new adjacency with self-loops on the
    diagonal."""
    n = adjacency.shape[0]
    if max_neighborhood is None or max_neighborhood <= 0:
        out = adjacency.copy()
        np.fill_diagonal(out, 1)
        return out
    kept = np.zeros((n, n), dtype=int)
    for v in range(n):
        nbrs = np.where(adjacency[v] > 0)[0]
        nbrs = nbrs[nbrs != v]
        if len(nbrs) > max_neighborhood:
            top = nbrs[np.argsort(-mi_matrix[v, nbrs])[:max_neighborhood]]
        else:
            top = nbrs
        kept[v, top] = 1
    sym = ((kept + kept.T) > 0).astype(int)
    np.fill_diagonal(sym, 1)
    return sym


# ---------------------------------------------------------------------------
# MN-FDA-S: per-variable cliques instead of full maximal-clique enumeration
# ---------------------------------------------------------------------------
def build_per_variable_cliques(mi_matrix, adjacency, max_clique_size,
                               cardinality=None, max_table_size=None):
    """One clique per variable, avoiding maximal-clique enumeration.

    For each variable ``x_i`` the clique is ``x_i`` together with the strongest
    mutual-information neighbours that passed the dependency test
    (``adjacency[i]``), up to ``max_clique_size`` variables.  If no variable
    passed the test the clique is the singleton ``[i]``.

    When ``cardinality`` and ``max_table_size`` are given, a neighbour is added
    only while the clique's **joint table size** ``prod(card over clique)`` stays
    ``<= max_table_size`` (strongest-MI neighbours are tried first, weaker ones
    skipped if they would overflow).  This bounds the clique *table* size -- not
    just the clique *size* -- which matters on high-cardinality problems where a
    size-``k`` clique table is ``prod(card)``-large.  For binary variables with
    the default ``max_table_size >= 2**max_clique_size`` the cap never binds, so
    the behaviour is unchanged.

    Returns a list of ``n`` cliques, each a sorted 1-D int array of size at most
    ``max_clique_size``.
    """
    n = adjacency.shape[0]
    k = int(max_clique_size)
    card = None if cardinality is None else np.asarray(cardinality, dtype=int)
    cap = card is not None and max_table_size is not None
    cliques = []
    for i in range(n):
        nb = np.where(adjacency[i] > 0)[0]
        nb = nb[nb != i]
        if len(nb) == 0 or k <= 1:
            cliques.append(np.array([i], dtype=int))
            continue
        # strongest-MI neighbours first
        nb = nb[np.argsort(-mi_matrix[i, nb])]
        clique = [i]
        tbl = int(card[i]) if card is not None else 1
        for j in nb:
            if len(clique) >= k:
                break
            if cap and tbl * int(card[j]) > max_table_size:
                continue                      # would overflow the table cap; try next
            clique.append(int(j))
            if card is not None:
                tbl *= int(card[j])
        cliques.append(np.sort(np.array(clique, dtype=int)))
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
# Running-intersection forest (MN-FDA-F / MN-FDA-P): a junction forest with
# bounded treewidth, built from the clique set.
# ---------------------------------------------------------------------------
def build_running_intersection_forest(cliques, rng=None):
    """Build a factorized structure that is a **junction forest** satisfying the
    running-intersection property (scenario 2).

    Each clique is added to the growing forest attached to the SINGLE already-in
    clique with which it has maximum variable overlap (ties broken randomly).
    Its separator (``overlap``) is the intersection with *that one parent*; its
    ``new`` variables are those not yet introduced by any clique.  Variables the
    clique shares with *other* (non-parent) cliques are dropped from it, so every
    variable's occurrences form a connected subtree -> running intersection holds
    and the induced treewidth never exceeds ``max_clique_size - 1``.  A clique
    that overlaps nothing starts a new tree of the forest.

    Returns a FactorizedModel ``structure`` array (rows
    ``[n_overlap, n_new, overlap..., new...]``); every row has ``n_new >= 1``.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Largest cliques first (introduce more variables early); stable by index.
    order = sorted(range(len(cliques)), key=lambda i: (-len(cliques[i]), i))

    sampled = set()
    var_to_tree = defaultdict(list)      # variable -> indices of kept cliques
    kept_varsets = []                    # variable set actually used per kept clique
    rows = []                            # (overlap_sorted, new_sorted)
    for idx in order:
        c_set = set(int(v) for v in cliques[idx])
        new = c_set - sampled
        if not new:                      # introduces no new variable -> skip
            continue
        already = c_set & sampled
        if not already:
            overlap = set()              # root of a new tree in the forest
        else:
            candidates = set()
            for v in already:
                candidates.update(var_to_tree[v])
            candidates = list(candidates)
            rng.shuffle(candidates)      # random tie-break
            best_p, best_ov = None, -1
            for p in candidates:
                ov = len(c_set & kept_varsets[p])
                if ov > best_ov:
                    best_ov, best_p = ov, p
            overlap = c_set & kept_varsets[best_p]   # separator with ONE parent

        kept = overlap | new             # clique as used (<= original clique)
        pos = len(kept_varsets)
        kept_varsets.append(kept)
        for v in kept:
            var_to_tree[v].append(pos)
        rows.append((sorted(overlap), sorted(new)))
        sampled |= new

    if not rows:
        return np.zeros((0, 2), dtype=int)
    max_w = max(len(o) + len(n) for o, n in rows)
    structure = np.zeros((len(rows), 2 + max_w), dtype=int)
    for i, (o, n) in enumerate(rows):
        structure[i, 0] = len(o)
        structure[i, 1] = len(n)
        structure[i, 2:2 + len(o)] = o
        structure[i, 2 + len(o):2 + len(o) + len(n)] = n
    return structure


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
