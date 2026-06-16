"""
MN-FDAG: MN-FDA with G-test of independence

Variant of MN-FDA that uses G-test statistics instead of chi-square for
detecting variable dependencies.

G-test: G(Xi, Xj) = 2*N*MI(Xi, Xj) ~ chi-square with df = (card_i-1)*(card_j-1)

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/FDA.cpp:1610-1632 (LearnMatrixGTest)
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.mutual_information import (
    compute_mutual_information_matrix,
    compute_g_test_matrix,
)
from pateda.learning.utils.markov_network import (
    build_dependency_graph_gtest,
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


class LearnMNFDAG(LearningMethod):
    """
    Learn Markov network using G-test of independence

    Difference from base MN-FDA:
    - Uses G-statistic: G(Xi, Xj) = 2*N*MI(Xi, Xj)
    - More statistically principled than simple chi-square threshold
    - Degrees of freedom: df = (card_i - 1) * (card_j - 1)

    The G-test is more accurate for detecting dependencies, especially
    for variables with different cardinalities.
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        max_n_cliques: Optional[int] = None,
        alpha: float = 0.05,
        prior: bool = True,
        return_factorized: bool = True,
        max_neighborhood: Optional[int] = 8,
    ):
        """
        Initialize MN-FDAG learner

        Args:
            max_clique_size: Maximum clique size (default 3)
            max_n_cliques: Maximum number of cliques (None = unlimited)
            alpha: Significance level for G-test (default 0.05)
                  Lower alpha = more conservative (fewer edges)
            prior: Whether to use Laplace prior smoothing (default True)
            return_factorized: If True, return FactorizedModel (for PLS).
                             If False, return MarkovNetworkModel with
                             per-variable conditional tables matching the
                             MOA layout — enables the fast Gibbs path.
            max_neighborhood: Cap on the per-variable Markov blanket size
                             (only used when ``return_factorized=False``).
                             Keeps conditional tables tractable; default 8.
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
        Learn factorized Markov network using G-test

        Algorithm (Algorithm 2 with G-test):
        1. Compute mutual information matrix
        2. Apply G-test for each variable pair
        3. Build dependency graph from significant pairs
        4. Find maximal cliques
        5. Compute probability tables

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population (n_selected, n_vars)
            fitness: Fitness values (not used)
            **params: Additional parameters (weights)

        Returns:
            FactorizedModel or MarkovNetworkModel
        """
        weights = params.get("weights", None)
        if weights is None:
            # Customized selection: count-scale weights (N * p) or None (uniform).
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Step 1: Compute mutual information and G-test
        g_matrix, adjacency = compute_g_test_matrix(
            population, cardinality, weights, self.alpha
        )

        # Step 2: Adjacency already computed by G-test

        # Step 3: Find maximal cliques
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Create metadata shared by both return paths
        metadata = {
            "generation": generation,
            "model_type": "MN-FDAG",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "alpha": self.alpha,
            "g_test": True,
        }

        if self.return_factorized:
            # FactorizedModel path: PLS sampling via SampleFDA
            clique_order = order_cliques_for_sampling(cliques)
            structure = convert_cliques_to_factorized_structure(cliques, clique_order)
            tables = compute_clique_tables(
                population, cliques, structure, cardinality, weights, self.prior
            )
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )

        # MarkovNetworkModel path: produce MOA-style per-variable artifacts so
        # SampleGibbs uses the fast direct-table lookup instead of the slow
        # generic clique search.
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
