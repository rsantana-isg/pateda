"""
MN-FDA-S: simplified MN-FDA that skips maximal-clique enumeration

The dominant cost that stops plain MN-FDA from scaling (see
``MN-FDA_analysis.md``) is ``find_maximal_cliques_greedy`` --- enumerating *all*
maximal cliques of the (dense, uncorrected) dependency graph, which explodes to
tens of thousands of cliques as the population converges.

MN-FDA-S avoids that enumeration entirely.  Instead of computing all cliques
simultaneously it builds **one clique per variable**:

1. Learn the same $\chi^2$ dependency graph as MN-FDA.
2. For each variable $x_i$, form a clique with $x_i$ and the ``max_clique_size
   - 1`` variables of strongest mutual information *among those that passed the
   $\chi^2$ test with* $x_i$.  If none passed, the clique is the singleton
   ``[x_i]``.
3. Remove redundant / subsumed cliques (a clique whose variables are all
   contained in another clique).
4. Build the junction-tree / factorized model with the same greedy ordering as
   MN-FDA and sample with PLS (``SampleFDA``).

This produces at most ``n`` cliques of bounded size, so learning is cheap and
scales to large ``n`` (e.g. 625) without the maximal-clique blow-up.  It uses
the vectorized MN-FDA kernels (proposals A--C, E) throughout.
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
    chi2_adjacency,
    compute_clique_tables_fast,
    order_cliques_for_sampling_fast,
    prune_empty_cliques,
    build_per_variable_cliques,
    remove_subsumed_cliques,
)


class LearnMNFDAS(LearningMethod):
    """
    Learn a simplified MN-FDA factorization (per-variable cliques).

    Same output type and sampling path as :class:`LearnMNFDA` (FactorizedModel +
    PLS); the difference is that the cliques are built one per variable from the
    strongest-MI test-passing neighbours instead of by maximal-clique
    enumeration.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        threshold: float = 0.05,
        prior: bool = True,
        return_factorized: bool = True,
    ):
        """
        Args:
            max_clique_size: Target clique size ``k`` (default 3).  Each
                per-variable clique has ``x_i`` plus up to ``k - 1`` neighbours.
            threshold: Significance level for the chi-square test (default 0.05).
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True (default), return a FactorizedModel for
                PLS sampling.
        """
        self.max_clique_size = max_clique_size
        self.threshold = threshold
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

        # Step 1: MI matrix + chi-square dependency graph (vectorized).
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)
        adjacency = chi2_adjacency(mi_matrix, population.shape[0], self.threshold)

        # Step 2: one clique per variable from the strongest-MI test-passing
        # neighbours; then drop redundant / subsumed cliques.
        cliques = build_per_variable_cliques(
            mi_matrix, adjacency, self.max_clique_size
        )
        cliques = remove_subsumed_cliques(cliques)

        # Step 3-5: greedy junction-tree ordering, factorized structure (pruned
        # to <= n productive cliques) and probability tables.
        clique_order = order_cliques_for_sampling_fast(cliques)
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)
        structure = prune_empty_cliques(structure)
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior
        )

        metadata = {
            "generation": generation,
            "model_type": "MN-FDA-S",
            "n_cliques": int(structure.shape[0]),
            "n_cliques_raw": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "threshold": self.threshold,
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
