"""
MN-FDA (current design): a scalable Markov-network Factorized Distribution
Algorithm that composes the three ideas that individually fixed the original
MN-FDA's maximal-clique blow-up:

1. **Sparse dependency graph** (as MN-FDA-sparse) -- the pairwise chi-square
   tests are corrected for multiple testing (Benjamini--Hochberg FDR by default)
   and each variable's neighbourhood is bounded to its strongest-MI partners, so
   the graph does not fill with false-positive edges as the population converges.

2. **Per-variable cliques** (as MN-FDA-S) -- instead of enumerating *all*
   maximal cliques of the graph (networkx ``find_cliques``, which is exponential
   on dense graphs and was the OOM/hang source at large ``n``), one clique per
   variable is built from its ``max_clique_size - 1`` strongest-MI significant
   neighbours; subsumed cliques are then removed.

3. **Running-intersection forest** (as MN-FDA-F) -- the cliques are assembled
   into a junction forest whose induced treewidth never exceeds
   ``max_clique_size - 1``, so the model is always tractable to sample (PLS) and,
   for MN-FDA-P, to solve exactly for its most-probable configuration.

The two public algorithms are:
  * **MN-FDA**   -- this learner + PLS sampling (``SampleFDA``);
  * **MN-FDA-P** -- this learner + PLS sampling with the junction-tree
    most-probable configuration inserted each generation (``SampleFDAWithMPC``).

The optional ``use_ess`` flag replaces the raw sample size ``N`` in the
chi-square statistic with the **effective sample size** of the (customized-
selection) weights, ``ESS = (sum w)^2 / sum(w^2)``.  Under concentrated weights
(e.g. Boltzmann selection) the ESS is smaller, which removes the extra spurious
edges that concentration would otherwise introduce.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast,
    corrected_adjacency,
    bound_neighborhood,
    effective_sample_size,
    build_per_variable_cliques,
    remove_subsumed_cliques,
    build_running_intersection_forest,
    compute_clique_tables_fast,
    neff_for_target_degree,
)


class LearnMNFDA(LearningMethod):
    """
    Learn a running-intersection-forest MN-FDA factorization.

    Pipeline: MI matrix -> multiple-testing-corrected, degree-bounded sparse
    dependency graph -> one clique per variable -> subsumption removal ->
    running-intersection forest -> clique probability tables.  Returns a
    :class:`~pateda.core.models.FactorizedModel` sampled by PLS
    (``SampleFDA``); MN-FDA-P additionally inserts the exact most-probable
    configuration (``SampleFDAWithMPC``).

    Args:
        max_clique_size: Target clique size ``k`` (default 5); also bounds the
            forest treewidth (``<= k - 1``).
        threshold: Significance level ``alpha`` for the chi-square tests, applied
            after the multiple-testing correction (default 0.05).
        correction: Multiple-testing correction over the ``n(n-1)/2`` tests:
            ``"fdr_bh"`` (default), ``"holm"``, ``"bonferroni"`` or ``"none"``.
        max_neighborhood: Per-variable cap on the number of highest-MI
            significant neighbours kept before clique construction (default 6);
            ``None`` disables the cap.
        prior: Whether to use Laplace prior smoothing (default True).
        use_ess: If True, use the effective sample size of the weights in the
            chi-square statistic (``chi2 = 2 * ESS * MI * ln2``) instead of the
            raw ``N``.  Only matters under weighted (customized) selection.
        return_factorized: If True (default), return a FactorizedModel for
            PLS / MPC sampling.
        random_state: Seed for the random tie-breaking used when several
            candidate parents share the maximum overlap during forest assembly.
    """

    def __init__(
        self,
        max_clique_size: int = 5,
        threshold: float = 0.05,
        correction: str = "fdr_bh",
        max_neighborhood: Optional[int] = 6,
        prior: bool = True,
        use_ess: bool = False,
        target_degree: Optional[float] = None,
        limit_table_size: bool = True,
        return_factorized: bool = True,
        random_state: Optional[int] = None,
    ):
        self.max_clique_size = int(max_clique_size)
        self.threshold = float(threshold)
        self.correction = correction
        self.max_neighborhood = max_neighborhood
        self.prior = bool(prior)
        self.use_ess = bool(use_ess)
        self.target_degree = target_degree
        self.limit_table_size = bool(limit_table_size)
        self.return_factorized = bool(return_factorized)
        self._rng = np.random.default_rng(random_state)

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        n_samples = population.shape[0]
        weights = params.get("weights", None)
        if weights is None:
            weights = count_weights_from_p(params.get("p"), n_samples)

        # Step 1: weighted MI matrix.
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)

        # Step 2: sparse dependency graph.  Under weighted selection, optionally
        # use the effective sample size so concentrated weights do not densify
        # the graph.
        if self.target_degree is not None:
            # Scale-aware: pick the effective size so the graph hits a target
            # mean degree (self-calibrating across n).
            n_eff = neff_for_target_degree(
                mi_matrix, self.threshold, self.correction, cardinality,
                self.target_degree, n_samples)
        elif self.use_ess:
            n_eff = effective_sample_size(weights, n_samples)
        else:
            n_eff = n_samples
        adjacency = corrected_adjacency(
            mi_matrix, n_eff, self.threshold, self.correction,
            cardinality=cardinality)
        adjacency = bound_neighborhood(adjacency, mi_matrix, self.max_neighborhood)

        # Step 3: per-variable cliques (no maximal-clique enumeration), bounding
        # the clique *table* size to n_samples on high-cardinality problems.
        max_table = n_samples if self.limit_table_size else None
        cliques = build_per_variable_cliques(
            mi_matrix, adjacency, self.max_clique_size,
            cardinality=cardinality, max_table_size=max_table)
        cliques = remove_subsumed_cliques(cliques)

        # Step 4: running-intersection forest (bounded treewidth).
        structure = build_running_intersection_forest(cliques, rng=self._rng)

        # Step 5: clique probability tables.
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior)

        metadata = {
            "generation": generation,
            "model_type": "MN-FDA",
            "n_cliques": int(structure.shape[0]) if structure.shape[0] else 0,
            "max_clique_size": self.max_clique_size,
            "threshold": self.threshold,
            "correction": self.correction,
            "max_neighborhood": self.max_neighborhood,
            "use_ess": self.use_ess,
            "effective_n": n_eff,
            "limit_table_size": self.limit_table_size,
            "running_intersection": True,
        }

        if self.return_factorized:
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata)
        return MarkovNetworkModel(
            structure=structure, parameters=tables, metadata=metadata)
