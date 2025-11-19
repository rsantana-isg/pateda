"""
Tree Model (Tree EDA) learning

Implements learning of tree-structured probabilistic models using maximum
spanning tree based on mutual information. Tree models are a type of Bayesian
network where the dependency graph is constrained to be a tree structure.

Tree-structured Bayesian Networks:
A tree is a special case of a Bayesian network where:
- The graph is acyclic and connected
- Each node has at most one parent (except the root which has no parent)
- The structure can be learned efficiently in O(n²) time using maximum spanning tree

The probability distribution factorizes as:
    P(X) = P(X_root) * ∏ᵢ P(Xᵢ | parent(Xᵢ))

where the root variable has no parent and all other variables have exactly one parent.

Learning Algorithm (based on Chow-Liu):
1. Compute pairwise mutual information I(Xᵢ, Xⱼ) for all variable pairs
2. Find maximum weighted spanning tree using MI as edge weights
3. Choose a root variable (typically the one involved in the highest MI edge)
4. Orient edges away from root to create directed tree
5. Learn conditional probability tables P(Xᵢ | parent(Xᵢ)) from data

Advantages over UMDA:
- Can model pairwise dependencies between variables
- Still efficient to learn and sample from (polynomial time)
- Often performs better on problems with moderate variable interactions

Relationship to other EDAs:
- COMIT: Uses tree models with incremental learning
- BMDA: Bivariate extension that models pairwise dependencies
- BOA: More general Bayesian network that can have multiple parents

When to use:
- Problems with pairwise but not higher-order variable interactions
- Need balance between model complexity and learning efficiency
- When UMDA is too simple but full Bayesian networks are too complex

Based on MATEDA-2.0 LearnTreeModel.m

References:
- Chow, C., & Liu, C. (1968). "Approximating discrete probability distributions
  with dependence trees." IEEE Transactions on Information Theory, 14(3):462-467.
- Baluja, S., & Davies, S. (1997). "Using optimal dependency-trees for
  combinatorial optimization: Learning the structure of the search space."
  ICML 1997.
- Pelikan, M., Goldberg, D.E., & Lobo, F.G. (2002). "A survey of optimization
  by building and using probabilistic models." Computational Optimization and
  Applications, 21(1):5–20.
- Santana, R. (2003). "A Markov network based factorized distribution algorithm
  for optimization." ECML 2003.
- MATEDA-2.0 User Guide, Section 4.2: "Bayesian network based factorizations"
"""

from typing import Any, Optional, List, Tuple
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import find_marginal_prob


class LearnTreeModel(LearningMethod):
    """
    Learn a Tree Model for EDAs

    A tree model captures dependencies between variables using a tree structure,
    where each variable has at most one parent. The tree is learned by finding
    the maximum weighted spanning tree on the mutual information matrix.

    The probability distribution factorizes as:
        P(X) = P(X_root) * ∏ᵢ P(Xᵢ | parent(Xᵢ))

    Mutual Information:
    The edge weights for the maximum spanning tree are computed using mutual
    information, which measures the statistical dependence between two variables:

        I(Xᵢ, Xⱼ) = ∑ₓᵢ,ₓⱼ P(Xᵢ=xᵢ, Xⱼ=xⱼ) log(P(Xᵢ=xᵢ, Xⱼ=xⱼ) / (P(Xᵢ=xᵢ)P(Xⱼ=xⱼ)))

    Normalized MI is computed by dividing by the product of cardinalities to make
    values comparable across variable pairs with different cardinalities.

    Clique Representation:
    In MATEDA's factorization framework, tree edges are represented as cliques:
    - Root nodes: [0, 1, child_idx, 0]
      → 0 parents, 1 new variable, variable index, unused
    - Non-root nodes: [1, 1, parent_idx, child_idx]
      → 1 parent, 1 new variable, parent index, child index

    Implementation details:
    - Uses Prim's algorithm variant for maximum spanning tree
    - Handles ties in MI values through random shuffling
    - Optional mi_threshold to create disconnected components (forest) when
      MI values are below threshold
    - Supports Laplace smoothing to avoid zero probabilities
    """

    def __init__(
        self,
        alpha: float = 0.0,
        mi_threshold: float = 1e-4,
    ):
        """
        Initialize Tree Model learning

        Args:
            alpha: Smoothing parameter for Laplace estimation (default: 0.0)
            mi_threshold: Minimum mutual information threshold for creating edges
        """
        self.alpha = alpha
        self.mi_threshold = mi_threshold

    def _compute_mutual_information_matrix(
        self,
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Compute normalized mutual information matrix from marginal probabilities

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities
            univ_prob: List of univariate probability distributions
            biv_prob: List of lists containing bivariate probability distributions

        Returns:
            Symmetric matrix of normalized mutual information values
        """
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars - 1):
            for j in range(i + 1, n_vars):
                # Compute mutual information from marginal probabilities
                mi = 0.0
                card_i = int(cardinality[i])
                card_j = int(cardinality[j])

                for k in range(card_i):
                    for l in range(card_j):
                        # Get bivariate probability P(X_i=k, X_j=l)
                        idx = card_j * k + l
                        p_ij = biv_prob[i][j][idx]
                        p_i = univ_prob[i][k]
                        p_j = univ_prob[j][l]

                        if p_ij > 0 and p_i > 0 and p_j > 0:
                            mi += p_ij * np.log(p_ij / (p_i * p_j))

                # Normalize by cardinalities
                mi /= (card_i * card_j)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _create_tree_structure(
        self,
        mi_matrix: np.ndarray,
        n_vars: int,
    ) -> np.ndarray:
        """
        Create maximum weighted tree structure from mutual information matrix

        Uses a variant of Prim's algorithm to find maximum spanning tree.

        Args:
            mi_matrix: Symmetric matrix of mutual information values
            n_vars: Number of variables

        Returns:
            Tree structure as cliques array where each row represents a variable:
            [n_parents, n_new_vars, parent_idx, child_idx]
            For root nodes: [0, 1, child_idx, 0]
        """
        # Initialize structure
        cliques = np.zeros((n_vars, 4), dtype=int)
        cliques[:, 1] = 1  # Each clique adds one new variable

        # Tree parent tracking
        tree = np.arange(n_vars)  # Initially each node is its own parent

        # Random shuffle to avoid bias when MI values are equal
        shuffle = np.random.permutation(n_vars)

        # Find root: node with maximum MI value
        max_pos = np.unravel_index(np.argmax(mi_matrix), mi_matrix.shape)
        root = max_pos[0]

        # Set root
        cliques[root, :] = [0, 1, root, 0]
        tree[root] = -1
        index = [root]

        # Build tree by adding variables one at a time
        for i in range(n_vars - 1):
            max_val = -10.0
            max_son_idx = -1
            max_father_idx = -1

            # Find the edge with maximum MI connecting tree to non-tree nodes
            for j in range(n_vars):
                for k in range(n_vars):
                    j_shuffled = shuffle[j]
                    k_shuffled = shuffle[k]

                    # j is not in tree, k is in tree
                    if tree[j_shuffled] == j_shuffled and tree[k_shuffled] != k_shuffled:
                        mi_val = mi_matrix[j_shuffled, k_shuffled]
                        if mi_val > max_val:
                            max_son_idx = j_shuffled
                            max_father_idx = k_shuffled
                            max_val = mi_val

            index.append(max_son_idx)

            if max_val > self.mi_threshold:
                tree[max_son_idx] = max_father_idx
                cliques[max_son_idx, :] = [1, 1, max_father_idx, max_son_idx]
            else:
                # Create isolated root if MI is below threshold
                tree[max_son_idx] = -1
                cliques[max_son_idx, :] = [0, 1, max_son_idx, 0]

        # Reorder cliques according to the order variables were added
        cliques = cliques[index, :]

        return cliques

    def _learn_parameters(
        self,
        cliques: np.ndarray,
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Learn conditional probability tables for tree structure

        Args:
            cliques: Tree structure
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities
            univ_prob: Univariate marginal probabilities
            biv_prob: Bivariate marginal probabilities

        Returns:
            List of probability tables (conditional or marginal)
        """
        n_samples = population.shape[0]
        tables = []

        for j in range(n_vars):
            n_parents = int(cliques[j, 0])

            if n_parents == 0:
                # Root node: use marginal probability
                child_idx = int(cliques[j, 2])
                card_child = int(cardinality[child_idx])

                # Apply Laplace smoothing
                table = (univ_prob[child_idx] * n_samples + 1) / (n_samples + card_child)
                tables.append(table)

            else:
                # Non-root: use conditional probability P(child | parent)
                child_idx = int(cliques[j, 3])
                parent_idx = int(cliques[j, 2])
                card_child = int(cardinality[child_idx])
                card_parent = int(cardinality[parent_idx])

                # Get bivariate probabilities
                if parent_idx < child_idx:
                    biv_probs = biv_prob[parent_idx][child_idx]
                    # Reshape to [parent, child]
                    aux_biv_prob = biv_probs.reshape(card_child, card_parent).T
                else:
                    biv_probs = biv_prob[child_idx][parent_idx]
                    aux_biv_prob = biv_probs.reshape(card_parent, card_child)

                # Compute conditional probability P(child | parent)
                parent_probs = np.tile(univ_prob[parent_idx].reshape(-1, 1), (1, card_child))

                # Apply Laplace smoothing to bivariate
                lap_biv_prob = (aux_biv_prob * n_samples + 1) / (n_samples + card_child * card_parent)
                cond_biv_prob = lap_biv_prob / parent_probs

                # Normalize to ensure valid probability distribution
                cond_biv_prob = cond_biv_prob / np.tile(
                    cond_biv_prob.sum(axis=1, keepdims=True), (1, card_child)
                )

                # Alternative: use MLE (no smoothing)
                cond_biv_prob_mle = aux_biv_prob / parent_probs

                tables.append(cond_biv_prob_mle)

        return tables

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
        Learn Tree Model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for tree learning)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel with tree structure

        Note:
            The returned model has:
            - structure: Cliques array defining tree edges
            - parameters: List of conditional/marginal probability tables
        """
        alpha = params.get("alpha", self.alpha)

        # Learn univariate and bivariate marginal probabilities
        univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality)

        # Compute mutual information matrix
        mi_matrix = self._compute_mutual_information_matrix(
            population, n_vars, cardinality, univ_prob, biv_prob
        )

        # Create tree structure
        cliques = self._create_tree_structure(mi_matrix, n_vars)

        # Learn parameters (conditional probability tables)
        tables = self._learn_parameters(
            cliques, population, n_vars, cardinality, univ_prob, biv_prob
        )

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "Tree",
                "alpha": alpha,
                "mi_threshold": self.mi_threshold,
                "mi_matrix": mi_matrix,
            },
        )

        return model
