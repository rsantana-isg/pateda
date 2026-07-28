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
from pateda.learning.utils.markov_network import (
    find_maximal_cliques_greedy,
    convert_cliques_to_factorized_structure,
)
from pateda.learning.utils.weights import count_weights_from_p
# Vectorized, numerically-exact kernels (proposals A, B, C, E of
# MN-FDA_analysis.md).  MN-FDA uses these; the G-test variants MN-FDAG / MN-EDAG
# deliberately keep the reference kernels.
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast,               # proposal B
    chi2_adjacency,                       # proposal A
    compute_clique_tables_fast,           # proposal C
    order_cliques_for_sampling_fast,      # proposal E
    prune_empty_cliques,                  # keep PLS at <= n cliques
)


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
        # Get parameters.  Customized selection: if no explicit sample weights
        # were given, derive count-scale weights (N * p) from p (None = uniform).
        weights = params.get("weights", None)
        if weights is None:
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Step 1: Compute mutual information matrix (proposal B: vectorized).
        mi_matrix = compute_mi_matrix_fast(population, cardinality, weights)

        # Step 2: Build dependency graph using chi-square test (proposal A:
        # critical value evaluated once, comparison vectorized).
        adjacency = chi2_adjacency(
            mi_matrix, population.shape[0], self.threshold
        )

        # Step 3: Find maximal cliques
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Step 4: Order cliques for sampling (proposal E: identical order,
        # built via an inverted variable->clique index).
        clique_order = order_cliques_for_sampling_fast(cliques)

        # Step 5: Convert to factorized structure, then drop redundant cliques
        # that introduce no new variable.  PLS visits each variable exactly once,
        # so at most n cliques are productive; on a dense graph the greedy clique
        # finder can emit thousands of empty (n_new==0) rows that SampleFDA would
        # otherwise loop over.  Pruning them is exact (they sample nothing).
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)
        structure = prune_empty_cliques(structure)

        # Step 6: Compute probability tables (proposal C: bincount over
        # mixed-radix indices; general for heterogeneous cardinalities).
        tables = compute_clique_tables_fast(
            population, structure, cardinality, weights, self.prior
        )

        # Create and return model.  n_cliques counts the *productive* cliques
        # actually used by PLS (<= n) after pruning; n_cliques_raw is the count
        # before pruning (maximal cliques found on the dependency graph).
        metadata = {
            "generation": generation,
            "model_type": "MN-FDA",
            "n_cliques": int(structure.shape[0]),
            "n_cliques_raw": len(cliques),
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
        Build dependency graph using the chi-square test (df = 1).

        For each pair (i, j) an edge is added iff ``2 N MI(i,j) ln 2`` exceeds
        the chi-square critical value at significance ``self.threshold``.  This
        is proposal A: the critical value is evaluated once and the comparison
        is vectorized (identical output to the per-pair reference).

        Reference: C++ FDA.cpp:1635-1672 (LearnMatrix)

        Args:
            mi_matrix: Mutual information matrix
            n_samples: Number of samples
            cardinality: Variable cardinalities (unused; df = 1 approximation)

        Returns:
            Adjacency matrix (binary)
        """
        return chi2_adjacency(mi_matrix, n_samples, self.threshold)
