"""
MN-FDA-sparse: MN-FDA with a sparsified dependency graph (proposal D)

MN-FDA at large ``n`` spends most of its time building and sampling a dependency
graph that, with ``n(n-1)/2`` simultaneous chi-square tests and no
multiple-testing correction, contains many false-positive edges.  The denser the
graph, the more (and larger) the maximal cliques, and the super-quadratic
clique-ordering / sampling costs explode — which is what makes MN-FDA
intractable at ``n = 625`` (see ``MN-FDA_analysis.md``).

MN-FDA-sparse keeps MN-FDA's factorization machinery unchanged (maximal cliques,
splitting at ``max_clique_size``, PLS sampling) but sparsifies the graph in two
complementary ways:

1. **Multiple-testing correction** over the ``n(n-1)/2`` chi-square p-values
   (Benjamini-Hochberg FDR by default; Holm or Bonferroni also available).  This
   removes spurious edges the naive ``alpha = 0.05`` rule keeps.
2. **Bounded neighbourhood**: each variable keeps only its ``max_neighborhood``
   highest-mutual-information neighbours (the graph is then symmetrized).  This
   caps the graph degree, so the clique count grows ~linearly in ``n`` instead
   of exploding.

Both steps tend to *sharpen* the true block structure of additively decomposable
problems, so the effect on model quality is small and usually beneficial.

The learner uses the vectorized MN-FDA kernels (proposals A-C, E), so together
with the sparsification it is intended to scale to ``n >= 625``.
"""

from typing import Any, Optional
import numpy as np
from scipy import stats as scipy_stats

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.markov_network import (
    find_maximal_cliques_greedy,
    convert_cliques_to_factorized_structure,
)
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast,
    compute_clique_tables_fast,
    order_cliques_for_sampling_fast,
    prune_empty_cliques,
)

_LN2 = np.log(2.0)


def _corrected_adjacency(mi_matrix, n_samples, alpha, correction):
    """Upper-triangular chi-square p-values with a multiple-testing correction.

    Returns a symmetric binary adjacency (no self loops set here).
    """
    n = mi_matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    chi2_stat = 2.0 * n_samples * mi_matrix[iu, ju] * _LN2
    pvals = scipy_stats.chi2.sf(chi2_stat, 1)          # 1 - cdf, df = 1
    m = pvals.size

    correction = (correction or "none").lower()
    if correction in ("none", None):
        reject = pvals < alpha
    elif correction == "bonferroni":
        reject = pvals < (alpha / m)
    elif correction == "holm":
        order = np.argsort(pvals)
        thresh = alpha / (m - np.arange(m))
        sorted_reject = pvals[order] < thresh
        # Holm: stop at first non-rejection.
        if not sorted_reject.all():
            first_fail = np.argmin(sorted_reject)  # first False
            sorted_reject[first_fail:] = False
        reject = np.zeros(m, dtype=bool)
        reject[order] = sorted_reject
    elif correction in ("fdr_bh", "bh", "fdr"):
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, m + 1) / m)
        below = ranked <= thresh
        if below.any():
            kmax = np.max(np.where(below)[0])
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


def _bound_neighborhood(adjacency, mi_matrix, max_neighborhood):
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
    # symmetrize (union): edge if kept in either direction
    sym = ((kept + kept.T) > 0).astype(int)
    np.fill_diagonal(sym, 1)
    return sym


class LearnMNFDASparse(LearningMethod):
    """
    Learn a sparsified Markov-network factorization for MN-FDA-sparse.

    Same output type and sampling path as :class:`LearnMNFDA` (FactorizedModel +
    PLS); the difference is a multiple-testing-corrected, degree-bounded
    dependency graph.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        max_n_cliques: Optional[int] = None,
        threshold: float = 0.05,
        correction: str = "fdr_bh",
        max_neighborhood: Optional[int] = 6,
        prior: bool = True,
        return_factorized: bool = True,
    ):
        """
        Args:
            max_clique_size: Maximum clique size (default 3, as MN-FDA).
            max_n_cliques: Maximum number of cliques (None = unlimited).
            threshold: Significance level ``alpha`` for the chi-square tests
                       (default 0.05) — applied *after* the correction.
            correction: Multiple-testing correction over the n(n-1)/2 tests:
                        ``"fdr_bh"`` (default), ``"holm"``, ``"bonferroni"`` or
                        ``"none"``.
            max_neighborhood: Per-variable cap on the number of highest-MI
                        neighbours kept (default 6).  ``None`` disables the cap.
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True (default), return a FactorizedModel for
                        PLS sampling.
        """
        self.max_clique_size = max_clique_size
        self.max_n_cliques = max_n_cliques
        self.threshold = threshold
        self.correction = correction
        self.max_neighborhood = max_neighborhood
        self.prior = prior
        self.return_factorized = return_factorized

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        weights = params.get("weights", None)
        if weights is None:
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Step 1: MI matrix (vectorized).
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)

        # Step 2a: chi-square dependency graph with multiple-testing correction.
        adjacency = _corrected_adjacency(
            mi_matrix, population.shape[0], self.threshold, self.correction
        )
        # Step 2b: bound the per-variable neighbourhood by highest MI.
        adjacency = _bound_neighborhood(
            adjacency, mi_matrix, self.max_neighborhood
        )

        # Step 3: maximal cliques (same splitting as MN-FDA).
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Step 4-6: order, structure (pruned to <= n productive cliques), tables.
        clique_order = order_cliques_for_sampling_fast(cliques)
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)
        structure = prune_empty_cliques(structure)
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior
        )

        metadata = {
            "generation": generation,
            "model_type": "MN-FDA-sparse",
            "n_cliques": int(structure.shape[0]),
            "n_cliques_raw": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "threshold": self.threshold,
            "correction": self.correction,
            "max_neighborhood": self.max_neighborhood,
        }

        if self.return_factorized:
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )
        return MarkovNetworkModel(
            structure=np.array(cliques, dtype=object),
            parameters=tables,
            metadata=metadata,
        )
