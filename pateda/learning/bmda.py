"""
Bivariate Marginal Distribution Algorithm (BMDA) learning

Equivalent to MATEDA's LearnBMDA.m
BMDA models pairwise dependencies between variables.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnBMDA(LearningMethod):
    """
    Learn a BMDA (Bivariate Marginal Distribution Algorithm) model

    BMDA represents the probability distribution using bivariate marginals,
    capturing pairwise dependencies between variables. It uses a tree structure
    or complete pairwise model.
    """

    def __init__(
        self,
        structure: Optional[np.ndarray] = None,
        structure_learning: str = "tree",
        alpha: float = 0.0,
    ):
        """
        Initialize BMDA learning

        Args:
            structure: Pre-defined dependency structure (adjacency matrix or edge list)
                      If None, structure is learned from data
            structure_learning: Method for learning structure ("tree", "complete", "greedy")
                              - "tree": Learn maximum spanning tree based on mutual information
                              - "complete": Use all pairwise dependencies
                              - "greedy": Greedy structure learning
            alpha: Smoothing parameter for probability estimation
        """
        self.structure = structure
        self.structure_learning = structure_learning
        self.alpha = alpha

    def _mutual_information(
        self, var1_data: np.ndarray, var2_data: np.ndarray, k1: int, k2: int
    ) -> float:
        """
        Calculate mutual information between two variables

        Args:
            var1_data: Data for first variable
            var2_data: Data for second variable
            k1: Cardinality of first variable
            k2: Cardinality of second variable

        Returns:
            Mutual information value
        """
        n = len(var1_data)

        # Calculate joint and marginal counts
        joint_counts = np.zeros((k1, k2))
        for i in range(n):
            joint_counts[int(var1_data[i]), int(var2_data[i])] += 1

        # Add small constant to avoid log(0)
        joint_counts += 1e-10
        joint_probs = joint_counts / np.sum(joint_counts)

        # Marginal probabilities
        marginal1 = np.sum(joint_probs, axis=1)
        marginal2 = np.sum(joint_probs, axis=0)

        # Calculate mutual information
        mi = 0.0
        for i in range(k1):
            for j in range(k2):
                if joint_probs[i, j] > 1e-10:
                    mi += joint_probs[i, j] * np.log(
                        joint_probs[i, j] / (marginal1[i] * marginal2[j])
                    )

        return mi

    def _learn_tree_structure(
        self, population: np.ndarray, n_vars: int, cardinality: np.ndarray
    ) -> np.ndarray:
        """
        Learn tree structure using maximum spanning tree on mutual information

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Adjacency matrix representing tree structure
        """
        # Calculate mutual information between all pairs
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mi = self._mutual_information(
                    population[:, i],
                    population[:, j],
                    int(cardinality[i]),
                    int(cardinality[j]),
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Find maximum spanning tree using Prim's algorithm
        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        visited = np.zeros(n_vars, dtype=bool)
        visited[0] = True

        for _ in range(n_vars - 1):
            max_mi = -1
            best_i, best_j = -1, -1

            for i in range(n_vars):
                if visited[i]:
                    for j in range(n_vars):
                        if not visited[j] and mi_matrix[i, j] > max_mi:
                            max_mi = mi_matrix[i, j]
                            best_i, best_j = i, j

            if best_i >= 0:
                adjacency[best_i, best_j] = 1
                adjacency[best_j, best_i] = 1
                visited[best_j] = True

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
        Learn BMDA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for BMDA learning)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel with bivariate structure
        """
        alpha = params.get("alpha", self.alpha)
        pop_size = population.shape[0]

        # Learn or use provided structure
        if self.structure is not None:
            adjacency = self.structure
        elif self.structure_learning == "tree":
            adjacency = self._learn_tree_structure(population, n_vars, cardinality)
        elif self.structure_learning == "complete":
            # Complete graph (all pairwise dependencies)
            adjacency = np.ones((n_vars, n_vars), dtype=int)
            np.fill_diagonal(adjacency, 0)
        else:
            raise ValueError(f"Unknown structure_learning method: {self.structure_learning}")

        # Convert adjacency matrix to clique representation
        # For BMDA, we need both univariate and bivariate cliques
        cliques = []
        tables = []

        # Track which variables are in bivariate cliques
        in_bivariate = np.zeros(n_vars, dtype=bool)

        # Add bivariate cliques (edges in the tree/graph)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] > 0:
                    # Bivariate clique: [0, 2, i, j]
                    cliques.append([0, 2, i, j])
                    in_bivariate[i] = True
                    in_bivariate[j] = True

                    # Learn joint probability table
                    k1, k2 = int(cardinality[i]), int(cardinality[j])
                    joint_counts = np.zeros((k1, k2))

                    for val1 in range(k1):
                        for val2 in range(k2):
                            count = np.sum(
                                (population[:, i] == val1) & (population[:, j] == val2)
                            )
                            joint_counts[val1, val2] = count

                    # Apply smoothing
                    if alpha > 0:
                        joint_counts += alpha

                    # Normalize
                    joint_probs = joint_counts / np.sum(joint_counts)
                    tables.append(joint_probs.flatten())

        # Add univariate cliques for variables not in bivariate cliques
        for i in range(n_vars):
            if not in_bivariate[i]:
                cliques.append([0, 1, i])

                # Learn marginal probability
                k = int(cardinality[i])
                counts = np.zeros(k)
                for val in range(k):
                    counts[val] = np.sum(population[:, i] == val)

                # Apply smoothing
                if alpha > 0:
                    counts += alpha

                # Normalize
                probs = counts / np.sum(counts)
                tables.append(probs)

        # Convert to numpy array
        # Pad cliques to same length
        max_len = max(len(c) for c in cliques)
        cliques_array = np.zeros((len(cliques), max_len))
        for i, clique in enumerate(cliques):
            cliques_array[i, : len(clique)] = clique

        # Create and return model
        model = FactorizedModel(
            structure=cliques_array,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "BMDA",
                "structure_learning": self.structure_learning,
                "alpha": alpha,
                "adjacency": adjacency,
            },
        )

        return model
