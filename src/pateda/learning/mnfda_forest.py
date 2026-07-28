"""
MN-FDA-F: MN-FDA restricted to a running-intersection forest

Identical clique discovery to MN-FDA (chi-square dependency graph -> maximal
cliques bounded by ``max_clique_size``), but the cliques are assembled into a
**junction forest** that satisfies the running-intersection property
(``build_running_intersection_forest``): each clique is attached to the single
already-in clique of maximum overlap (ties broken randomly), its separator is the
intersection with that one parent, and variables shared with other cliques are
dropped.  Consequently the model is a forest whose induced treewidth never
exceeds ``max_clique_size - 1``.

This is the model shared by:
  * **MN-FDA-F** — sampled entirely with PLS (``SampleFDA``), like MN-FDA; and
  * **MN-FDA-P** — the same forest, but the exact most-probable configuration
    (junction-tree max-product) is inserted into each new population.  Because
    the forest has bounded treewidth, that exact MPC is always tractable (it can
    no longer blow up memory as it could on the dense, high-treewidth maximal-
    clique model).
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.markov_network import find_maximal_cliques_greedy
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast,
    chi2_adjacency,
    compute_clique_tables_fast,
    build_running_intersection_forest,
)


class LearnMNFDAForest(LearningMethod):
    """
    Learn an MN-FDA factorization constrained to a running-intersection forest.

    Same output type and PLS sampling path as :class:`LearnMNFDA`
    (FactorizedModel); the difference is that the clique structure is a junction
    forest (treewidth <= max_clique_size - 1) instead of the unconstrained,
    possibly high-treewidth clique decomposition.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        threshold: float = 0.05,
        prior: bool = True,
        return_factorized: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            max_clique_size: Maximum clique size (default 3); also bounds the
                forest treewidth (<= max_clique_size - 1).
            threshold: Significance level for the chi-square test (default 0.05).
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True (default), return a FactorizedModel for
                PLS / MPC.
            random_state: Seed for the random tie-breaking used when several
                candidate parents share the maximum overlap.
        """
        self.max_clique_size = max_clique_size
        self.threshold = threshold
        self.prior = prior
        self.return_factorized = return_factorized
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
        weights = params.get("weights", None)
        if weights is None:
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Step 1-2: MI matrix + chi-square dependency graph (vectorized).
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)
        adjacency = chi2_adjacency(mi_matrix, population.shape[0], self.threshold)

        # Step 3: maximal cliques (as MN-FDA).
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, None)

        # Step 4: assemble a running-intersection forest (bounded treewidth).
        structure = build_running_intersection_forest(cliques, rng=self._rng)

        # Step 5: probability tables (vectorized).
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior)

        max_size = int(structure[:, 0].astype(int).max()
                       + structure[:, 1].astype(int).max()) if structure.shape[0] else 0
        metadata = {
            "generation": generation,
            "model_type": "MN-FDA-F",
            "n_cliques": int(structure.shape[0]),
            "max_clique_size": self.max_clique_size,
            "threshold": self.threshold,
            "running_intersection": True,
        }

        if self.return_factorized:
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata)
        return MarkovNetworkModel(
            structure=structure, parameters=tables, metadata=metadata)
