"""
MN-FDA-S-Sparse: MN-FDA-S clique construction on an MN-FDA-Sparse dependency graph

This variant combines the two ideas already implemented separately:

* the way **MN-FDA-S** learns cliques -- instead of enumerating all maximal
  cliques it builds *one clique per variable* (the variable plus its
  ``max_clique_size - 1`` strongest-mutual-information neighbours that passed the
  dependency test), then removes redundant / subsumed cliques; and

* the way **MN-FDA-sparse** filters the dependencies -- the pairwise chi-square
  tests are corrected for multiple testing (Benjamini--Hochberg FDR by default;
  Holm / Bonferroni also available) and each variable's neighbourhood is bounded
  to its strongest-MI partners.

So the "passed the test" set that MN-FDA-S uses to pick each variable's clique is
the multiple-testing-corrected, degree-bounded graph of MN-FDA-sparse rather than
the raw ``alpha = 0.05`` chi-square graph.  A variable with no *significant*
partner therefore becomes a singleton clique.  The downstream steps (greedy
junction-tree ordering, empty-clique pruning, probability tables, PLS sampling)
are exactly those of MN-FDA.

Everything uses the vectorized MN-FDA kernels, so the method is cheap and scales
to large ``n``.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.markov_network import (
    convert_cliques_to_factorized_structure,
)
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast,
    compute_clique_tables_fast,
    order_cliques_for_sampling_fast,
    prune_empty_cliques,
    build_per_variable_cliques,     # MN-FDA-S clique construction
    remove_subsumed_cliques,        # MN-FDA-S subsumption removal
)
# MN-FDA-sparse dependency filtering.
from pateda.learning.mnfda_sparse import _corrected_adjacency, _bound_neighborhood


class LearnMNFDASSparse(LearningMethod):
    """
    Learn MN-FDA-S cliques on the MN-FDA-sparse (FDR-corrected, degree-bounded)
    dependency graph.  Same FactorizedModel + PLS output as MN-FDA.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        threshold: float = 0.05,
        correction: str = "fdr_bh",
        max_neighborhood: Optional[int] = 6,
        prior: bool = True,
        return_factorized: bool = True,
    ):
        """
        Args:
            max_clique_size: Target clique size ``k`` (default 3): each
                per-variable clique is ``x_i`` plus up to ``k - 1`` neighbours.
            threshold: Significance level ``alpha`` for the chi-square tests
                (default 0.05), applied after the multiple-testing correction.
            correction: Multiple-testing correction over the n(n-1)/2 tests:
                ``"fdr_bh"`` (default), ``"holm"``, ``"bonferroni"`` or ``"none"``.
            max_neighborhood: Per-variable cap on the number of highest-MI
                significant neighbours kept before clique construction
                (default 6).  ``None`` disables the cap.
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True (default), return a FactorizedModel for
                PLS sampling.
        """
        self.max_clique_size = max_clique_size
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

        # Step 2: MN-FDA-sparse dependency filtering -- multiple-testing-corrected
        # chi-square graph, then bounded neighbourhood.
        adjacency = _corrected_adjacency(
            mi_matrix, population.shape[0], self.threshold, self.correction)
        adjacency = _bound_neighborhood(
            adjacency, mi_matrix, self.max_neighborhood)

        # Step 3: MN-FDA-S clique construction on that filtered graph -- one
        # clique per variable from its strongest-MI significant neighbours, then
        # remove subsumed cliques.
        cliques = build_per_variable_cliques(
            mi_matrix, adjacency, self.max_clique_size)
        cliques = remove_subsumed_cliques(cliques)

        # Step 4-6: greedy junction-tree ordering, factorized structure (pruned
        # to <= n productive cliques) and probability tables.
        clique_order = order_cliques_for_sampling_fast(cliques)
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)
        structure = prune_empty_cliques(structure)
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior)

        metadata = {
            "generation": generation,
            "model_type": "MN-FDA-S-Sparse",
            "n_cliques": int(structure.shape[0]),
            "n_cliques_raw": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "threshold": self.threshold,
            "correction": self.correction,
            "max_neighborhood": self.max_neighborhood,
        }

        if self.return_factorized:
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata)
        return MarkovNetworkModel(
            structure=np.array(cliques, dtype=object),
            parameters=tables,
            metadata=metadata,
        )
