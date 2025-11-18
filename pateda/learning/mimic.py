"""
MIMIC (Mutual Information Maximization for Input Clustering)

MIMIC is an EDA that builds a chain-structured probabilistic model by selecting
variables based on minimum conditional entropy. It creates a specific dependency
chain where each variable depends on at most one other variable.

The algorithm:
1. Start with the variable having minimum entropy (most predictable)
2. Iteratively add variables with minimum conditional entropy given the chain head
3. Each new variable becomes the new head of the chain

This creates a chain structure: X_1 <- X_2 <- ... <- X_n where each variable
depends only on its predecessor in the chain.

References:
    - De Bonet, J. S., Isbell, C. L., & Viola, P. (1997). MIMIC: Finding optima
      by estimating probability densities. In Advances in neural information
      processing systems (pp. 424-430).
    - Pelikan, M., Goldberg, D. E., & Lobo, F. G. (2002). A survey of optimization
      by building and using probabilistic models. Computational Optimization and
      Applications, 21(1), 5-20.
"""

from typing import Any, List, Tuple
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import find_marginal_prob


class LearnMIMIC(LearningMethod):
    """
    Learn a MIMIC (Mutual Information Maximization for Input Clustering) model

    MIMIC builds a chain-structured model where each variable depends on at most
    one other variable. The chain is constructed by selecting variables based on
    minimum conditional entropy, which tends to create a dependency structure that
    captures the most important relationships in the data.

    The probability distribution factorizes as a chain:
    P(X) = P(X_root) * ∏_i P(X_i | parent(X_i))

    where each variable has exactly one parent (except the root).
    """

    def __init__(self, alpha: float = 0.0, epsilon: float = 1e-10):
        """
        Initialize MIMIC learning

        Args:
            alpha: Smoothing parameter for Laplace estimation (default: 0.0)
            epsilon: Small value to avoid log(0) in entropy calculations
        """
        self.alpha = alpha
        self.epsilon = epsilon

    def _compute_entropy(
        self,
        var_idx: int,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        """
        Compute entropy of a variable

        Args:
            var_idx: Index of the variable
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Entropy value H(X_i)
        """
        k = int(cardinality[var_idx])
        pop_size = population.shape[0]

        # Count occurrences
        counts = np.zeros(k)
        for val in range(k):
            counts[val] = np.sum(population[:, var_idx] == val)

        # Calculate probabilities with smoothing
        probs = (counts + self.alpha) / (pop_size + k * self.alpha)

        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > self.epsilon:
                entropy -= p * np.log2(p)

        return entropy

    def _compute_conditional_entropy(
        self,
        var_idx: int,
        parent_idx: int,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        """
        Compute conditional entropy H(X_i | X_parent)

        Args:
            var_idx: Index of the variable
            parent_idx: Index of the parent variable
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Conditional entropy value H(X_i | X_parent)
        """
        k_var = int(cardinality[var_idx])
        k_parent = int(cardinality[parent_idx])
        pop_size = population.shape[0]

        # Compute joint counts
        joint_counts = np.zeros((k_parent, k_var))
        for i in range(pop_size):
            parent_val = int(population[i, parent_idx])
            var_val = int(population[i, var_idx])
            joint_counts[parent_val, var_val] += 1

        # Compute conditional entropy
        cond_entropy = 0.0
        for parent_val in range(k_parent):
            # Count for this parent value
            parent_count = np.sum(joint_counts[parent_val, :])

            if parent_count == 0:
                continue

            # Probability of parent value
            p_parent = parent_count / pop_size

            # Conditional probabilities P(X_i | X_parent = parent_val)
            cond_probs = (joint_counts[parent_val, :] + self.alpha) / (
                parent_count + k_var * self.alpha
            )

            # Add contribution to conditional entropy
            for p_cond in cond_probs:
                if p_cond > self.epsilon:
                    cond_entropy -= p_parent * p_cond * np.log2(p_cond)

        return cond_entropy

    def _build_chain(
        self,
        n_vars: int,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Build the MIMIC chain structure

        Args:
            n_vars: Number of variables
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            List of (variable, parent) tuples representing the chain structure.
            The root has parent -1.
        """
        linked = [False] * n_vars
        chain = []

        # Find variable with minimum entropy (most predictable)
        min_entropy = float("inf")
        root_var = 0

        for i in range(n_vars):
            entropy_i = self._compute_entropy(i, population, cardinality)
            if entropy_i < min_entropy:
                min_entropy = entropy_i
                root_var = i

        # Root has no parent
        linked[root_var] = True
        chain.append((root_var, -1))
        current_head = root_var

        # Build chain by iteratively adding variables with minimum conditional entropy
        for _ in range(n_vars - 1):
            min_cond_entropy = float("inf")
            best_var = -1

            # Find unlinked variable with minimum conditional entropy given current head
            for j in range(n_vars):
                if not linked[j]:
                    cond_entropy = self._compute_conditional_entropy(
                        j, current_head, population, cardinality
                    )

                    if cond_entropy < min_cond_entropy:
                        min_cond_entropy = cond_entropy
                        best_var = j

            # Add best variable to chain
            linked[best_var] = True
            chain.append((best_var, current_head))
            current_head = best_var

        return chain

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
        Learn MIMIC model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for MIMIC learning)
            **params: Additional parameters
                     - alpha: Override smoothing parameter

        Returns:
            Learned FactorizedModel with chain structure
        """
        alpha = params.get("alpha", self.alpha)

        # Build chain structure
        chain = self._build_chain(n_vars, population, cardinality)

        # Get marginal probabilities
        univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality)

        # Build cliques and tables from chain
        cliques = []
        tables = []

        for var_idx, parent_idx in chain:
            k_var = int(cardinality[var_idx])

            if parent_idx == -1:
                # Root node - univariate marginal
                clique = np.zeros(3)
                clique[0] = 0  # No overlapping variables
                clique[1] = 1  # One new variable
                clique[2] = var_idx

                # Get univariate probability with smoothing
                prob = univ_prob[var_idx].copy()
                if alpha > 0:
                    counts = prob * population.shape[0]
                    counts += alpha
                    prob = counts / np.sum(counts)

                cliques.append(clique)
                tables.append(prob)
            else:
                # Non-root node - conditional on parent
                k_parent = int(cardinality[parent_idx])

                clique = np.zeros(4)
                clique[0] = 1  # One overlapping variable (parent)
                clique[1] = 1  # One new variable (current)
                clique[2] = parent_idx  # Parent index
                clique[3] = var_idx  # Current variable index

                # Build conditional probability table P(var | parent)
                # Get bivariate probabilities
                if parent_idx < var_idx:
                    biv_probs = biv_prob[parent_idx][var_idx]
                    # Reshape to [parent, child]
                    aux_biv_prob = biv_probs.reshape(k_var, k_parent).T
                else:
                    biv_probs = biv_prob[var_idx][parent_idx]
                    aux_biv_prob = biv_probs.reshape(k_parent, k_var)

                # Compute conditional probability P(child | parent)
                parent_probs = np.tile(
                    univ_prob[parent_idx].reshape(-1, 1), (1, k_var)
                )

                # Avoid division by zero
                cond_table = np.zeros((k_parent, k_var))
                for p_val in range(k_parent):
                    if parent_probs[p_val, 0] > self.epsilon:
                        cond_table[p_val, :] = aux_biv_prob[p_val, :] / parent_probs[p_val, 0]
                    else:
                        cond_table[p_val, :] = 1.0 / k_var  # Uniform

                    # Normalize to ensure valid probability distribution
                    total = np.sum(cond_table[p_val, :])
                    if total > self.epsilon:
                        cond_table[p_val, :] /= total

                cliques.append(clique)
                tables.append(cond_table)

        # Convert to numpy array
        max_clique_size = max(len(c) for c in cliques)
        cliques_array = np.zeros((len(cliques), max_clique_size))
        for i, clique in enumerate(cliques):
            cliques_array[i, : len(clique)] = clique

        # Create and return model
        model = FactorizedModel(
            structure=cliques_array,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "MIMIC",
                "alpha": alpha,
                "chain_order": [var for var, _ in chain],
            },
        )

        return model
