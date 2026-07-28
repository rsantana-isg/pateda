"""
MN-EDAG: Markov-network EDA with the G-test of independence (Gibbs sampling)

This is the algorithm that learns an **undirected Markov network** from a
G-test of pairwise independence and samples new solutions with **Gibbs
sampling** (GS) over per-variable conditional tables.  Because it pairs a Markov
network model with Gibbs sampling it belongs to the MN-EDA family, hence the
name MN-EDAG (previously mislabelled MN-FDAG).

G-test: G(Xi, Xj) = 2*N*MI(Xi, Xj)*ln(2) ~ chi-square with
df = (card_i-1)*(card_j-1).

References:
- Santana, R. (2005). "Estimation of distribution algorithms with Kikuchi
  approximations." Evolutionary Computation, 13(1):67-97.
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov
  Networks." Memetic Computing, 5(1):3-17.
- C++ implementation: cpp_EDAs/FDA.cpp:1610-1632 (LearnMatrixGTest)
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.mutual_information import (
    compute_g_test_matrix,
)
from pateda.learning.utils.markov_network import (
    find_maximal_cliques_greedy,
    order_cliques_for_sampling,
    convert_cliques_to_factorized_structure,
    cliques_to_neighborhoods,
)
from pateda.learning.utils.probability_tables import (
    compute_clique_tables,
    compute_moa_tables,
)
from pateda.learning.utils.weights import count_weights_from_p


class LearnMNEDAG(LearningMethod):
    """
    Learn a Markov network with the G-test for MN-EDAG.

    Structure is obtained from the G-test of pairwise independence; the default
    model artefact is a :class:`MarkovNetworkModel` with MOA-style per-variable
    conditional tables, ready for the fast Gibbs sampling path.

    Notes
    -----
    The significance level defaults to ``alpha = 0.05`` (an edge is kept when
    the G-test p-value is below ``alpha``).  The previous MN-FDAG code shipped a
    default of ``alpha = 1.05``, which — because every p-value is below 1.05 —
    made the learner return a *complete* graph on every generation; this is
    fixed here.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        max_n_cliques: Optional[int] = None,
        alpha: float = 0.05,
        prior: bool = True,
        return_factorized: bool = False,
        max_neighborhood: Optional[int] = 8,
    ):
        """
        Initialize MN-EDAG learner.

        Args:
            max_clique_size: Maximum clique size (default 3).
            max_n_cliques: Maximum number of cliques (None = unlimited).
            alpha: Significance level for the G-test (default 0.05).  Lower alpha
                   = more conservative (fewer edges).
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True, return a FactorizedModel (PLS sampling).
                             If False (default, the MN-EDAG identity), return a
                             MarkovNetworkModel with per-variable conditional
                             tables in the MOA layout — the fast Gibbs path.
            max_neighborhood: Cap on the per-variable Markov-blanket size (only
                             used when ``return_factorized=False``).  Default 8.
        """
        self.max_clique_size = max_clique_size
        self.max_n_cliques = max_n_cliques
        self.alpha = alpha
        self.prior = prior
        self.return_factorized = return_factorized
        self.max_neighborhood = max_neighborhood

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn a Markov network with the G-test and build sampling artefacts.

        Args:
            generation: Current generation number.
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            population: Selected population (n_selected, n_vars).
            fitness: Fitness values (not used).
            **params: Additional parameters (weights / p).

        Returns:
            MarkovNetworkModel (Gibbs) or FactorizedModel (PLS).
        """
        weights = params.get("weights", None)
        if weights is None:
            # Customized selection: count-scale weights (N * p) or None (uniform).
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Step 1: Compute mutual information and G-test adjacency.
        g_matrix, adjacency = compute_g_test_matrix(
            population, cardinality, weights, self.alpha
        )

        # Step 2: Find maximal cliques.
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        metadata = {
            "generation": generation,
            "model_type": "MN-EDAG",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "alpha": self.alpha,
            "g_test": True,
        }

        if self.return_factorized:
            # FactorizedModel path: PLS sampling via SampleFDA.
            clique_order = order_cliques_for_sampling(cliques)
            structure = convert_cliques_to_factorized_structure(cliques, clique_order)
            tables = compute_clique_tables(
                population, cliques, structure, cardinality, weights, self.prior
            )
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )

        # MarkovNetworkModel path: MOA-style per-variable conditional tables so
        # SampleGibbs uses the fast direct-table lookup.
        neighbors_list = cliques_to_neighborhoods(
            cliques, n_vars,
            mi_matrix=g_matrix,
            max_neighborhood=self.max_neighborhood,
        )
        per_var_cliques = [
            np.concatenate([[v], neighbors_list[v]]).astype(int)
            if len(neighbors_list[v]) > 0
            else np.array([v], dtype=int)
            for v in range(n_vars)
        ]
        tables = compute_moa_tables(
            population, neighbors_list, cardinality, weights, self.prior
        )
        metadata["neighbors"] = neighbors_list
        return MarkovNetworkModel(
            structure=np.array(per_var_cliques, dtype=object),
            parameters=tables,
            metadata=metadata,
        )
