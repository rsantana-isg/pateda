"""
AffEDA-sparse: affinity-propagation factorization on a significance-gated
mutual-information matrix (approach C).

Plain AffEDA feeds the *full, dense* mutual-information matrix to affinity
propagation, so variables are clustered together even when their dependency is
not statistically significant.  AffEDA-sparse first gates the similarity matrix
with a G-test of independence and a multiple-testing correction (Benjamini
--Hochberg FDR by default): pairs whose mutual information is not significant are
pushed to a floor similarity so affinity propagation will not group them.  A
variable with no significant partner therefore becomes a singleton factor.

Motivation (transferred from MN-FDA-sparse): removing spurious edges (i) yields a
cleaner factorization that better reflects the true block structure and (ii)
gives affinity propagation a better-conditioned similarity matrix, reducing the
degenerate all-equal / non-convergent cases that trigger its retry/noise paths.

The G-test uses the exact degrees of freedom ``(r_i - 1)(r_j - 1)``, so the
significance gating is correct for **heterogeneous, non-binary cardinalities**.
The learner inherits AffEDA's vectorized MI + clique-table kernels (approaches
A/B), so it is also faster than the original AffEDA.
"""

from typing import Any, Optional
import numpy as np
from scipy import stats as scipy_stats

from pateda.core.models import FactorizedModel
from pateda.learning.affinity import LearnAffinityFactorization, _fast_clique_tables
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.mnfda_fast import compute_mi_matrix_fast

_LN2 = np.log(2.0)


def _multiple_testing_reject(pvals, alpha, correction):
    """Boolean reject vector for a set of p-values under a correction."""
    m = pvals.size
    correction = (correction or "none").lower()
    if correction in ("none", None):
        return pvals < alpha
    if correction == "bonferroni":
        return pvals < (alpha / m)
    if correction == "holm":
        order = np.argsort(pvals)
        thresh = alpha / (m - np.arange(m))
        sorted_reject = pvals[order] < thresh
        if not sorted_reject.all():
            sorted_reject[np.argmin(sorted_reject):] = False
        reject = np.zeros(m, dtype=bool)
        reject[order] = sorted_reject
        return reject
    if correction in ("fdr_bh", "bh", "fdr"):
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, m + 1) / m)
        below = ranked <= thresh
        reject_sorted = np.zeros(m, dtype=bool)
        if below.any():
            reject_sorted[: np.max(np.where(below)[0]) + 1] = True
        reject = np.zeros(m, dtype=bool)
        reject[order] = reject_sorted
        return reject
    raise ValueError(f"Unknown correction {correction!r}")


def gtest_fdr_adjacency(mi_matrix, n_samples, cardinality, alpha=0.05,
                        correction="fdr_bh"):
    """Significant-dependency adjacency from a corrected G-test.

    G(i,j) = 2 N MI(i,j) ln2 ~ chi-square with df = (r_i-1)(r_j-1); p-values are
    corrected across the n(n-1)/2 tests.  Correct for heterogeneous cardinality.
    """
    n = mi_matrix.shape[0]
    card = np.asarray(cardinality).astype(int)
    iu, ju = np.triu_indices(n, k=1)
    g = 2.0 * n_samples * mi_matrix[iu, ju] * _LN2
    df = np.maximum((card[iu] - 1) * (card[ju] - 1), 1)
    pvals = scipy_stats.chi2.sf(g, df)
    reject = _multiple_testing_reject(pvals, alpha, correction)
    adj = np.zeros((n, n), dtype=bool)
    adj[iu[reject], ju[reject]] = True
    adj[ju[reject], iu[reject]] = True
    return adj


class LearnAffinitySparse(LearnAffinityFactorization):
    """
    AffEDA-sparse learner: affinity propagation on a significance-gated MI matrix.

    Inherits the vectorized MI + clique-table kernels and the (recursive)
    affinity-propagation factorization from :class:`LearnAffinityFactorization`;
    the only change is that the similarity fed to affinity propagation keeps only
    statistically significant mutual-information entries.
    """

    def __init__(
        self,
        max_clique_size: int = 5,
        threshold: float = 0.05,
        correction: str = "fdr_bh",
        preference: Optional[float] = None,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        alpha: float = 1.0,
        recursive: bool = True,
        max_recursion_depth: int = 10,
    ):
        super().__init__(
            max_clique_size=max_clique_size, preference=preference,
            damping=damping, max_iter=max_iter,
            convergence_iter=convergence_iter, alpha=alpha, recursive=recursive,
            max_recursion_depth=max_recursion_depth,
        )
        self.threshold = threshold
        self.correction = correction

    def _gated_similarity(self, mi_matrix, n_samples, cardinality):
        """Push non-significant off-diagonal MI entries to a floor value so
        affinity propagation will not cluster those pairs.  Returns
        ``(gated_matrix, preference, n_sig_edges)``."""
        sig = gtest_fdr_adjacency(mi_matrix, n_samples, cardinality,
                                  self.threshold, self.correction)
        gated = mi_matrix.astype(float).copy()
        lo, hi = float(mi_matrix.min()), float(mi_matrix.max())
        floor = lo - (hi - lo) - 1e-9          # strictly below every real MI
        off_diag_non_sig = ~sig
        np.fill_diagonal(off_diag_non_sig, False)
        gated[off_diag_non_sig] = floor
        # Preference on the scale of the significant dependencies, so variables
        # with no significant partner fall out as singletons.
        sig_vals = mi_matrix[np.triu(sig, k=1)]
        if self.preference is not None:
            pref = self.preference
        elif sig_vals.size > 0:
            pref = float(np.median(sig_vals))
        else:
            pref = 0.0
        return gated, pref, int(np.triu(sig, k=1).sum())

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Approach A: vectorized MI.
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)

        # Approach C: gate the similarity by G-test + multiple-testing correction.
        gated, pref, n_sig = self._gated_similarity(
            mi_matrix, population.shape[0], cardinality)

        # Affinity-propagation factorization on the gated similarity.
        var_indices = np.arange(n_vars)
        if self.recursive:
            cliques_list = self._recursive_factorization(gated, var_indices, pref)
        else:
            labels, _ = self._affinity_clustering(gated, pref)
            cliques_list = []
            for label in np.unique(labels):
                cluster_vars = var_indices[labels == label]
                if len(cluster_vars) <= self.max_clique_size:
                    cliques_list.append(cluster_vars)
                else:
                    for i in range(0, len(cluster_vars), self.max_clique_size):
                        cliques_list.append(cluster_vars[i:i + self.max_clique_size])

        cliques = self._create_clique_structure(cliques_list, n_vars)

        # Approach B: vectorized clique tables.
        tables = _fast_clique_tables(cliques, population, cardinality, weights,
                                     self.alpha)

        return FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "AffinitySparse",
                "max_clique_size": self.max_clique_size,
                "n_cliques": len(cliques_list),
                "n_significant_edges": n_sig,
                "correction": self.correction,
                "threshold": self.threshold,
                "preference": pref,
            },
        )
