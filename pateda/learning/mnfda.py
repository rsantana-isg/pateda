"""
MN-FDA: Markov Network Factorized Distribution Algorithm

Implements structure and parameter learning for MN-FDA, a Markov network-based EDA.

Algorithm (from Santana 2013):
1. Learn an independence graph G using chi-square/G-test
2. If necessary, refine the graph
3. Find the set L of all maximal cliques of G
4. Construct a labeled junction graph from L
5. Find the marginal probabilities for the cliques in the JG

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/mainmoa.cpp (Markovinit), FDA.cpp (UpdateModel)
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.mutual_information import (
    compute_mutual_information_matrix,
    chi_square_test,
)
from pateda.learning.utils.markov_network import (
    build_dependency_graph_threshold,
    find_maximal_cliques_greedy,
    order_cliques_for_sampling,
    convert_cliques_to_factorized_structure,
)
from pateda.learning.utils.probability_tables import compute_clique_tables


class LearnMNFDA(LearningMethod):
    """
    Learn Markov network factorization for MN-FDA

    Uses chi-square test to detect pairwise dependencies, then builds
    a factorized model from maximal cliques.

    The learned model can be used with:
    - SampleFDA (PLS sampling) - recommended for small cliques
    - SampleGibbs (Gibbs sampling) - works for any structure
    - SampleMAP (MAP-based sampling) - for exploration
    """

    def __init__(
        self,
        max_clique_size: int = 3,
        max_n_cliques: Optional[int] = None,
        threshold: float = 0.05,
        prior: bool = True,
        return_factorized: bool = True,
    ):
        """
        Initialize MN-FDA learner

        Args:
            max_clique_size: Maximum clique size (default 3)
                           Larger cliques = more expressive but slower
            max_n_cliques: Maximum number of cliques (None = unlimited)
            threshold: Chi-square significance threshold (default 0.05)
            prior: Whether to use Laplace prior smoothing (default True)
            return_factorized: If True, return FactorizedModel (for PLS sampling)
                             If False, return MarkovNetworkModel (for other sampling)
        """
        self.max_clique_size = max_clique_size
        self.max_n_cliques = max_n_cliques
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
        """
        Learn factorized Markov network from population

        Implements Algorithm 2 from Santana (2013):
        1. Learn independence graph G using chi-square test
        2. (Optional) Refine the graph
        3. Find maximal cliques
        4. Construct junction graph
        5. Compute marginal probabilities

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population (n_selected, n_vars)
            fitness: Fitness values (not used in learning)
            **params: Additional parameters
                     - weights: Sample weights (optional)

        Returns:
            FactorizedModel or MarkovNetworkModel depending on return_factorized
        """
        # Get parameters
        weights = params.get("weights", None)

        # Step 1: Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population, cardinality, weights)

        # Step 2: Build dependency graph using chi-square test
        adjacency = self._build_dependency_graph(
            mi_matrix, population.shape[0], cardinality
        )

        # Step 3: Find maximal cliques
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Step 4: Order cliques for sampling
        clique_order = order_cliques_for_sampling(cliques)

        # Step 5: Convert to factorized structure
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)

        # Step 6: Compute probability tables
        tables = compute_clique_tables(
            population, cliques, structure, cardinality, weights, self.prior
        )

        # Create and return model
        metadata = {
            "generation": generation,
            "model_type": "MN-FDA",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "threshold": self.threshold,
        }

        if self.return_factorized:
            # Return FactorizedModel for PLS sampling
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )
        else:
            # Return MarkovNetworkModel for Gibbs/MAP sampling
            return MarkovNetworkModel(
                structure=np.array(cliques, dtype=object),
                parameters=tables,
                metadata=metadata,
            )

    def _build_dependency_graph(
        self, mi_matrix: np.ndarray, n_samples: int, cardinality: np.ndarray
    ) -> np.ndarray:
        """
        Build dependency graph using chi-square test

        For each pair (i,j), test if MI(i,j) is significant using chi-square test.

        Reference: C++ FDA.cpp:1635-1672 (LearnMatrix)

        Args:
            mi_matrix: Mutual information matrix
            n_samples: Number of samples
            cardinality: Variable cardinalities

        Returns:
            Adjacency matrix (binary)
        """
        n_vars = mi_matrix.shape[0]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mi = mi_matrix[i, j]

                # Chi-square test for independence
                _, is_dependent = chi_square_test(mi, n_samples, self.threshold)

                if is_dependent:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

        # Self-loops
        np.fill_diagonal(adjacency, 1)

        return adjacency
