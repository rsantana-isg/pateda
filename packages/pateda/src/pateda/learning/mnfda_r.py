"""
MN-FDA_r: Markov Network FDA restricted to known variable interactions

Variant of MN-FDA that accepts a binary interaction matrix R where
R[i, j] = 1 means that variables X_i and X_j interact. Mutual information
(and the subsequent chi-square test) is computed only for pairs of variables
that interact according to R.

This reduces computational cost and incorporates prior structural knowledge
about the problem into Markov network structure learning.

Algorithm:
1. For each pair (i, j) with R[i, j] = 1: compute MI(X_i, X_j) and apply
   chi-square test to decide whether to add edge (i, j) to the graph.
2. For pairs with R[i, j] = 0: no edge is added regardless of the data.
3. Find maximal cliques of the resulting restricted dependency graph.
4. Construct a labeled junction graph from the cliques.
5. Compute marginal probabilities for the cliques.

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov
  Networks." Memetic Computing, 5(1):3-17.
- Santana, R., Mendiburu, A., and Lozano, J.A. (2012). "Structural transfer
  using EDAs: An application to multi-marker tagging SNP selection."
  Proceedings of the 2012 IEEE Congress on Evolutionary Computation (CEC-2012).
- Santana, R. (2017). "Gray-box optimization and factorized distribution
  algorithms: where two worlds collide." arXiv:1707.03093.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel, FactorizedModel
from pateda.learning.utils.mutual_information import (
    compute_pairwise_mi,
    chi_square_test,
)
from pateda.learning.utils.markov_network import (
    find_maximal_cliques_greedy,
    order_cliques_for_sampling,
    convert_cliques_to_factorized_structure,
)
from pateda.learning.utils.probability_tables import compute_clique_tables
from pateda.learning.utils.table_size import split_variables_by_table_limit


class LearnMNFDAR(LearningMethod):
    """
    Learn Markov network factorization restricted to known interactions (MN-FDA_r)

    Variant of LearnMNFDA that restricts pairwise dependency testing to those
    variable pairs that are known to interact according to a binary interaction
    matrix R. For non-interacting pairs (R[i, j] = 0), no edge is added to
    the dependency graph, regardless of empirical mutual information.

    The resulting Markov network is a subgraph of what MN-FDA would produce
    on the same population, constrained to edges consistent with the known
    problem structure.

    The learned model can be used with:
    - SampleFDA (PLS sampling) - recommended for small cliques
    - SampleGibbs (Gibbs sampling) - works for any structure
    """

    def __init__(
        self,
        interaction_matrix: np.ndarray,
        max_clique_size: int = 3,
        max_n_cliques: Optional[int] = None,
        threshold: float = 0.05,
        prior: bool = True,
        return_factorized: bool = True,
    ):
        """
        Initialize MN-FDA_r learner

        Args:
            interaction_matrix: Binary matrix R of shape (n_vars, n_vars).
                R[i, j] = 1 means variables X_i and X_j interact.
                R[i, j] = 0 means the pair is excluded from testing.
                The matrix should be symmetric; only the upper triangle is used.
            max_clique_size: Maximum clique size (default 3)
            max_n_cliques: Maximum number of cliques (None = unlimited)
            threshold: Chi-square significance threshold (default 0.05)
            prior: Whether to use Laplace prior smoothing (default True)
            return_factorized: If True, return FactorizedModel (for PLS sampling)
                             If False, return MarkovNetworkModel (for other sampling)
        """
        self.interaction_matrix = np.asarray(interaction_matrix, dtype=int)
        self.max_clique_size = max_clique_size
        self.max_n_cliques = max_n_cliques
        self.threshold = threshold
        self.prior = prior
        self.return_factorized = return_factorized

    @staticmethod
    def _apply_table_size_limit(
        cliques: list, cardinality: np.ndarray, n_samples: int
    ) -> list:
        """Split oversized cliques so each resulting joint table fits in sample size."""
        limited = []
        for clique in cliques:
            limited.extend(split_variables_by_table_limit(clique, cardinality, n_samples))
        return limited

    def _build_restricted_dependency_graph(
        self,
        population: np.ndarray,
        n_samples: int,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Build dependency graph restricted to interacting variable pairs

        For each pair (i, j) with interaction_matrix[i, j] = 1:
            Compute MI(X_i, X_j) and apply chi-square test.
            Add edge if the test indicates statistical dependence.

        For pairs with interaction_matrix[i, j] = 0: no edge added.

        Args:
            population: Population array (n_samples, n_vars)
            n_samples: Number of samples
            cardinality: Variable cardinalities

        Returns:
            Adjacency matrix (binary) with self-loops on diagonal
        """
        n_vars = population.shape[1]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Only test pairs that interact
                if self.interaction_matrix[i, j] == 0:
                    continue

                mi = compute_pairwise_mi(
                    population, i, j,
                    int(cardinality[i]), int(cardinality[j])
                )

                _, is_dependent = chi_square_test(mi, n_samples, self.threshold)

                if is_dependent:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

        # Self-loops
        np.fill_diagonal(adjacency, 1)

        return adjacency

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
        Learn restricted Markov network factorization from population

        Only tests variable pairs that are known to interact (R[i, j] = 1).
        Pairs with R[i, j] = 0 cannot form edges in the dependency graph.

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population (n_selected, n_vars)
            fitness: Fitness values (not used in learning)
            **params: Additional parameters
                     - weights: Sample weights (optional, not used here)

        Returns:
            FactorizedModel or MarkovNetworkModel depending on return_factorized

        Raises:
            ValueError: If interaction_matrix shape does not match n_vars
        """
        if self.interaction_matrix.shape != (n_vars, n_vars):
            raise ValueError(
                f"interaction_matrix shape {self.interaction_matrix.shape} "
                f"does not match n_vars={n_vars}. "
                f"Expected ({n_vars}, {n_vars})."
            )

        weights = params.get("weights", None)
        n_samples = population.shape[0]

        # Step 1: Build restricted dependency graph
        adjacency = self._build_restricted_dependency_graph(
            population, n_samples, cardinality
        )

        # Step 2: Find maximal cliques
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )
        cliques = self._apply_table_size_limit(cliques, cardinality, population.shape[0])

        # Step 3: Order cliques for sampling
        clique_order = order_cliques_for_sampling(cliques)

        # Step 4: Convert to factorized structure
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)

        # Step 5: Compute probability tables
        tables = compute_clique_tables(
            population, cliques, structure, cardinality, weights, self.prior
        )

        n_interactions = int(np.sum(np.triu(self.interaction_matrix, k=1)))

        metadata = {
            "generation": generation,
            "model_type": "MN-FDA_r",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "threshold": self.threshold,
            "n_interactions": n_interactions,
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
