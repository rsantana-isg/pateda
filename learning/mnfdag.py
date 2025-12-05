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
)
from pateda.learning.utils.probability_tables import compute_clique_tables


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
    ):
        """
        Initialize MN-FDAG learner

        Args:
            max_clique_size: Maximum clique size (default 3)
            max_n_cliques: Maximum number of cliques (None = unlimited)
            alpha: Significance level for G-test (default 0.05)
                  Lower alpha = more conservative (fewer edges)
            prior: Whether to use Laplace prior smoothing (default True)
            return_factorized: If True, return FactorizedModel (for PLS)
                             If False, return MarkovNetworkModel
        """
        self.max_clique_size = max_clique_size
        self.max_n_cliques = max_n_cliques
        self.alpha = alpha
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

        # Step 1: Compute mutual information and G-test
        g_matrix, adjacency = compute_g_test_matrix(
            population, cardinality, weights, self.alpha
        )

        # Step 2: Adjacency already computed by G-test

        # Step 3: Find maximal cliques
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Step 4: Order cliques
        clique_order = order_cliques_for_sampling(cliques)

        # Step 5: Convert to factorized structure
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)

        # Step 6: Compute probability tables
        tables = compute_clique_tables(
            population, cliques, structure, cardinality, weights, self.prior
        )

        # Create metadata
        metadata = {
            "generation": generation,
            "model_type": "MN-FDAG",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "alpha": self.alpha,
            "g_test": True,
        }

        if self.return_factorized:
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )
        else:
            return MarkovNetworkModel(
                structure=np.array(cliques, dtype=object),
                parameters=tables,
                metadata=metadata,
            )
