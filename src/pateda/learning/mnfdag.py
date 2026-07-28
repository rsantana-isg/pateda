"""
MN-FDAG: MN-FDA with the G-test of independence

MN-FDAG is **identical to MN-FDA** — same maximal-clique factorization, same
clique splitting at ``max_clique_size``, same clique ordering and probability
tables, and the same PLS (``SampleFDA``) sampling — except that the pairwise
dependencies are detected with the **G-test** instead of the chi-square test.

G-test: G(Xi, Xj) = 2*N*MI(Xi, Xj)*ln(2) ~ chi-square with
df = (card_i-1)*(card_j-1).  Compared with the chi-square test used by MN-FDA
(which fixes df = 1), the G-test uses the exact degrees of freedom and is
therefore more appropriate when variables have larger / heterogeneous
cardinalities.

Note
----
This learner intentionally uses the *reference* (non-vectorized) kernels
(``compute_g_test_matrix``, ``order_cliques_for_sampling``,
``compute_clique_tables``).  The vectorization proposals A-C, E of
``MN-FDA_analysis.md`` are, for now, applied only to :class:`LearnMNFDA`; once
validated there they can be ported to MN-FDAG as well.

References:
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
)
from pateda.learning.utils.probability_tables import compute_clique_tables
from pateda.learning.utils.weights import count_weights_from_p


class LearnMNFDAG(LearningMethod):
    """
    Learn a Markov-network factorization for MN-FDA using the G-test.

    The pipeline is exactly the one of :class:`LearnMNFDA` (independence graph ->
    maximal cliques bounded by ``max_clique_size`` -> clique ordering ->
    marginal/conditional tables -> PLS sampling); the only difference is that
    edges come from the G-test rather than the chi-square (df = 1) test.
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
        Initialize MN-FDAG learner.

        Args:
            max_clique_size: Maximum clique size (default 3, as MN-FDA).
            max_n_cliques: Maximum number of cliques (None = unlimited).
            alpha: Significance level for the G-test (default 0.05).  Lower alpha
                   = more conservative (fewer edges).
            prior: Whether to use Laplace prior smoothing (default True).
            return_factorized: If True (default, the MN-FDAG identity), return a
                             FactorizedModel for PLS sampling (SampleFDA).  If
                             False, return a MarkovNetworkModel over the same
                             cliques.
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
        Learn a factorized Markov network using the G-test.

        Mirrors :meth:`LearnMNFDA.learn` step by step, replacing the chi-square
        dependency graph with a G-test dependency graph.

        Args:
            generation: Current generation number.
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            population: Selected population (n_selected, n_vars).
            fitness: Fitness values (not used).
            **params: Additional parameters (weights / p).

        Returns:
            FactorizedModel (PLS) or MarkovNetworkModel.
        """
        weights = params.get("weights", None)
        if weights is None:
            # Customized selection: count-scale weights (N * p) or None (uniform).
            weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Steps 1-2: MI matrix + G-test dependency graph.
        g_matrix, adjacency = compute_g_test_matrix(
            population, cardinality, weights, self.alpha
        )

        # Step 3: Find maximal cliques (same clique splitting as MN-FDA).
        cliques = find_maximal_cliques_greedy(
            adjacency, self.max_clique_size, self.max_n_cliques
        )

        # Step 4: Order cliques for sampling.
        clique_order = order_cliques_for_sampling(cliques)

        # Step 5: Convert to factorized structure.
        structure = convert_cliques_to_factorized_structure(cliques, clique_order)

        # Step 6: Compute probability tables.
        tables = compute_clique_tables(
            population, cliques, structure, cardinality, weights, self.prior
        )

        metadata = {
            "generation": generation,
            "model_type": "MN-FDAG",
            "n_cliques": len(cliques),
            "max_clique_size": max(len(c) for c in cliques),
            "alpha": self.alpha,
            "g_test": True,
        }

        if self.return_factorized:
            # FactorizedModel for PLS sampling (SampleFDA) — the MN-FDAG default.
            return FactorizedModel(
                structure=structure, parameters=tables, metadata=metadata
            )
        # MarkovNetworkModel over the same cliques (alternative sampling paths).
        return MarkovNetworkModel(
            structure=np.array(cliques, dtype=object),
            parameters=tables,
            metadata=metadata,
        )
