"""
Tree-EDA_r: Tree EDA restricted to known variable interactions

Variant of Tree-EDA that accepts a binary interaction matrix R where
R[i, j] = 1 means that variables X_i and X_j interact. Mutual information
is computed only for pairs of variables that interact according to R.

This reduces computational cost and incorporates prior structural knowledge
about the problem into model learning.

Definition (from Santana et al., 2012):
    Tree-EDAr restricts the calculation of mutual information to those pairs
    of variables that have some potential interaction relationship. The
    interaction matrix is derived from problem structure knowledge (e.g.,
    variables that appear in the same clause of a SAT formula).

    Step 6 of Tree-EDA is modified so that mutual information between
    two variables Xi and Xj is computed only if R[i, j] = 1.

References:
- Santana, R., Mendiburu, A., and Lozano, J.A. (2012). "Structural transfer
  using EDAs: An application to multi-marker tagging SNP selection."
  Proceedings of the 2012 IEEE Congress on Evolutionary Computation (CEC-2012),
  pages 3484-3491, Brisbane, Australia.
- Santana, R. (2017). "Gray-box optimization and factorized distribution
  algorithms: where two worlds collide." arXiv:1707.03093.
"""

from typing import Any, Optional, List
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import find_marginal_prob


class LearnTreeModelR(LearningMethod):
    """
    Learn a Tree Model restricted to known variable interactions (Tree-EDA_r)

    Variant of LearnTreeModel that restricts mutual information computation
    to variable pairs that are known to interact, based on a binary interaction
    matrix R supplied at construction time.

    When R[i, j] = 1, the MI between X_i and X_j is computed normally.
    When R[i, j] = 0, the MI is set to 0, preventing an edge (i, j) from
    appearing in the spanning tree.

    This allows incorporating prior structural knowledge (e.g., variable
    co-occurrence in SAT clauses, adjacency in an Ising model, linkage
    disequilibrium between SNPs) to bias the tree construction toward
    known interactions, reducing both computational cost and model complexity.

    The probability distribution factorizes as:
        P(X) = P(X_root) * ∏ᵢ P(Xᵢ | parent(Xᵢ))

    where edges only connect interacting variable pairs.
    """

    def __init__(
        self,
        interaction_matrix: np.ndarray,
        alpha: float = 0.0,
        mi_threshold: float = 1e-4,
    ):
        """
        Initialize Tree-EDA_r

        Args:
            interaction_matrix: Binary matrix R of shape (n_vars, n_vars).
                R[i, j] = 1 means variables X_i and X_j interact.
                R[i, j] = 0 means MI(X_i, X_j) is not considered.
                The matrix should be symmetric; only the upper triangle is used.
            alpha: Smoothing parameter for Laplace estimation (default: 0.0)
            mi_threshold: Minimum mutual information threshold for creating edges
        """
        self.interaction_matrix = np.asarray(interaction_matrix, dtype=int)
        self.alpha = alpha
        self.mi_threshold = mi_threshold

    def _compute_restricted_mi_matrix(
        self,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Compute mutual information matrix restricted to interacting pairs

        For pairs (i, j) where interaction_matrix[i, j] = 0, MI is set to 0.
        For interacting pairs, MI is computed from marginal probabilities.

        Args:
            n_vars: Number of variables
            cardinality: Variable cardinalities
            univ_prob: List of univariate probability distributions
            biv_prob: List of lists containing bivariate probability distributions

        Returns:
            Symmetric matrix of normalized mutual information values,
            with zeros for non-interacting pairs.
        """
        mi_matrix = np.zeros((n_vars, n_vars))

        # Pre-compute univariate entropies once for all variables
        entropies = np.array([
            -sum(p * np.log(p) for p in univ_prob[i] if p > 0)
            for i in range(n_vars)
        ])

        for i in range(n_vars - 1):
            for j in range(i + 1, n_vars):
                # Skip pairs that do not interact
                if self.interaction_matrix[i, j] == 0:
                    continue

                # Compute mutual information from marginal probabilities
                mi = 0.0
                card_i = int(cardinality[i])
                card_j = int(cardinality[j])

                for k in range(card_i):
                    for l in range(card_j):
                        idx = card_j * k + l
                        p_ij = biv_prob[i][j][idx]
                        p_i = univ_prob[i][k]
                        p_j = univ_prob[j][l]

                        if p_ij > 0 and p_i > 0 and p_j > 0:
                            mi += p_ij * np.log(p_ij / (p_i * p_j))

                # Compute normalized mutual information (symmetric uncertainty)
                denom = entropies[i] + entropies[j]
                nmi = max(0.0, min(1.0, 2.0 * mi / denom)) if denom > 0 else 0.0
                mi_matrix[i, j] = nmi
                mi_matrix[j, i] = nmi

        return mi_matrix

    def _create_tree_structure(
        self,
        mi_matrix: np.ndarray,
        n_vars: int,
    ) -> np.ndarray:
        """
        Create maximum weighted tree structure from restricted MI matrix

        Uses Prim's algorithm variant. Because MI is zero for non-interacting
        pairs, spanning tree edges are restricted to interacting variable pairs.

        Args:
            mi_matrix: Symmetric MI matrix (zeros for non-interacting pairs)
            n_vars: Number of variables

        Returns:
            Tree structure as cliques array where each row represents a variable:
            [n_parents, n_new_vars, parent_idx, child_idx]
            For root nodes: [0, 1, child_idx, 0]
        """
        cliques = np.zeros((n_vars, 4), dtype=int)
        cliques[:, 1] = 1  # Each clique adds one new variable

        tree = np.arange(n_vars)  # Initially each node is its own parent

        shuffle = np.random.permutation(n_vars)

        # Find root: node with maximum MI value (restricted to interacting pairs)
        max_pos = np.unravel_index(np.argmax(mi_matrix), mi_matrix.shape)
        root = max_pos[0]

        cliques[root, :] = [0, 1, root, 0]
        tree[root] = -1
        index = [root]

        for i in range(n_vars - 1):
            max_val = -10.0
            max_son_idx = -1
            max_father_idx = -1

            for j in range(n_vars):
                for k in range(n_vars):
                    j_shuffled = shuffle[j]
                    k_shuffled = shuffle[k]

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
                # Create isolated root if MI is below threshold or no interaction
                tree[max_son_idx] = -1
                cliques[max_son_idx, :] = [0, 1, max_son_idx, 0]

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
                child_idx = int(cliques[j, 2])
                card_child = int(cardinality[child_idx])

                table = (univ_prob[child_idx] * n_samples + 1) / (n_samples + card_child)
                tables.append(table)

            else:
                child_idx = int(cliques[j, 3])
                parent_idx = int(cliques[j, 2])
                card_child = int(cardinality[child_idx])
                card_parent = int(cardinality[parent_idx])

                if parent_idx < child_idx:
                    biv_probs = biv_prob[parent_idx][child_idx]
                    # biv_prob[i][j] is stored row-major with i as the row variable.
                    # Here i=parent_idx, so reshape directly to (card_parent, card_child).
                    aux_biv_prob = biv_probs.reshape(card_parent, card_child)
                else:
                    biv_probs = biv_prob[child_idx][parent_idx]
                    # biv_prob[child][parent] has child as row variable;
                    # transpose to get (card_parent, card_child).
                    aux_biv_prob = biv_probs.reshape(card_child, card_parent).T

                lap_biv_prob = (aux_biv_prob * n_samples + 1) / (n_samples + card_child * card_parent)
                lap_parent_probs = (univ_prob[parent_idx] * n_samples + 1) / (n_samples + card_parent)

                parent_probs = np.tile(lap_parent_probs.reshape(-1, 1), (1, card_child))
                cond_biv_prob = lap_biv_prob / parent_probs

                cond_biv_prob = cond_biv_prob / np.tile(
                    cond_biv_prob.sum(axis=1, keepdims=True), (1, card_child)
                )

                tables.append(cond_biv_prob)

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
        Learn Tree-EDA_r model from population

        Only computes mutual information for variable pairs with known
        interactions (interaction_matrix[i, j] = 1).

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for tree learning)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel with restricted tree structure

        Raises:
            ValueError: If interaction_matrix shape does not match n_vars
        """
        if self.interaction_matrix.shape != (n_vars, n_vars):
            raise ValueError(
                f"interaction_matrix shape {self.interaction_matrix.shape} "
                f"does not match n_vars={n_vars}. "
                f"Expected ({n_vars}, {n_vars})."
            )

        alpha = params.get("alpha", self.alpha)

        # Learn univariate and bivariate marginal probabilities
        univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality, alpha=alpha)

        # Compute restricted mutual information matrix
        mi_matrix = self._compute_restricted_mi_matrix(
            n_vars, cardinality, univ_prob, biv_prob
        )

        # Create tree structure (restricted to interacting pairs via MI=0)
        cliques = self._create_tree_structure(mi_matrix, n_vars)

        # Learn parameters (conditional probability tables)
        tables = self._learn_parameters(
            cliques, population, n_vars, cardinality, univ_prob, biv_prob
        )

        n_interactions = int(np.sum(np.triu(self.interaction_matrix, k=1)))

        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "Tree-EDA_r",
                "alpha": alpha,
                "mi_threshold": self.mi_threshold,
                "mi_matrix": mi_matrix,
                "n_interactions": n_interactions,
            },
        )

        return model
