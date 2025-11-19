"""
MOA: Markovianity Based Optimization Algorithm

Implements structure and parameter learning for MOA, which uses local Markov
properties to create a simpler network structure than MN-FDA.

Algorithm (from Santana 2013, Algorithm 3):
1. Estimate the structure of a Markov network (local neighborhoods)
2. Estimate local Markov conditional probabilities p(x_i | N_i) for each variable

Key difference from MN-FDA:
- MOA creates one clique per variable containing {Xi, neighbors(Xi)}
- Simpler structure, faster learning
- Designed for Gibbs sampling

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/mainmoa.cpp:587-729, FDA.cpp:1369-1443
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MarkovNetworkModel
from pateda.learning.utils.mutual_information import (
    compute_mutual_information_matrix,
)
from pateda.learning.utils.markov_network import (
    find_k_neighbors,
    neighbors_to_cliques,
)
from pateda.learning.utils.probability_tables import compute_moa_tables


class LearnMOA(LearningMethod):
    """
    Learn local Markov structure for MOA algorithm

    For each variable Xi, finds its k nearest neighbors based on mutual
    information, then learns P(Xi | neighbors).

    The structure is simpler than MN-FDA:
    - Exactly n cliques (one per variable)
    - Each clique: {Xi, neighbor_1, ..., neighbor_k}
    - Designed for efficient Gibbs sampling
    """

    def __init__(
        self,
        k_neighbors: int = 3,
        threshold_factor: float = 1.5,
        prior: bool = True,
    ):
        """
        Initialize MOA learner

        Args:
            k_neighbors: Maximum number of neighbors per variable (default 3)
                       Paper uses k=8 for some experiments
            threshold_factor: Multiplier for average MI threshold (default 1.5)
                            From paper: TR = avg(MI) * 1.5
            prior: Whether to use Laplace prior smoothing (default True)
        """
        self.k_neighbors = k_neighbors
        self.threshold_factor = threshold_factor
        self.prior = prior

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> MarkovNetworkModel:
        """
        Learn local Markov network structure for MOA

        Algorithm (from Algorithm 3, steps 5-6):
        1. Compute mutual information matrix
        2. For each variable Xi:
           - Find k nearest neighbors with MI > threshold
           - Create clique {Xi, neighbors}
        3. Compute conditional probability tables P(Xi | neighbors)

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population (n_selected, n_vars)
            fitness: Fitness values (not used)
            **params: Additional parameters (weights)

        Returns:
            MarkovNetworkModel with local neighborhood structure
        """
        weights = params.get("weights", None)

        # Step 1: Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population, cardinality, weights)

        # Step 2: Find k nearest neighbors for each variable
        neighbors_list = find_k_neighbors(
            mi_matrix, self.k_neighbors, self.threshold_factor
        )

        # Step 3: Convert to clique structure
        cliques = neighbors_to_cliques(neighbors_list)

        # Step 4: Compute conditional probability tables P(Xi | neighbors)
        tables = compute_moa_tables(
            population, neighbors_list, cardinality, weights, self.prior
        )

        # Create metadata
        metadata = {
            "generation": generation,
            "model_type": "MOA",
            "n_cliques": len(cliques),
            "k_neighbors": self.k_neighbors,
            "threshold_factor": self.threshold_factor,
            "neighbors": neighbors_list,  # Store for Gibbs sampling
        }

        return MarkovNetworkModel(
            structure=np.array(cliques, dtype=object),
            parameters=tables,
            metadata=metadata,
        )
