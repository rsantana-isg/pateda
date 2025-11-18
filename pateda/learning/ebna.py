"""
Estimation of Bayesian Networks Algorithm (EBNA) learning

Equivalent to MATEDA's LearnEBNA.m
EBNA learns a Bayesian network structure with local structures (limited parent sets).
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel


class LearnEBNA(LearningMethod):
    """
    Learn an EBNA (Estimation of Bayesian Networks Algorithm) model

    EBNA learns a Bayesian network where each variable can have multiple parents,
    but the number of parents is typically limited (local structure).
    Uses scoring metrics like BIC or AIC for structure learning.
    """

    def __init__(
        self,
        max_parents: int = 2,
        score_metric: str = "bic",
        alpha: float = 0.0,
        structure: Optional[np.ndarray] = None,
    ):
        """
        Initialize EBNA learning

        Args:
            max_parents: Maximum number of parents per variable
            score_metric: Scoring metric for structure learning ("bic", "aic", "k2")
            alpha: Smoothing parameter for probability estimation
            structure: Pre-defined structure (DAG as adjacency matrix)
                      If provided, structure learning is skipped
        """
        self.max_parents = max_parents
        self.score_metric = score_metric
        self.alpha = alpha
        self.structure = structure

    def _calculate_local_score(
        self,
        var: int,
        parents: list,
        population: np.ndarray,
        cardinality: np.ndarray,
        n_samples: int,
    ) -> float:
        """
        Calculate local score for a variable given its parents

        Args:
            var: Variable index
            parents: List of parent indices
            population: Population data
            cardinality: Variable cardinalities
            n_samples: Number of samples

        Returns:
            Score value (higher is better)
        """
        k = int(cardinality[var])  # Variable cardinality
        n_parents = len(parents)

        if n_parents == 0:
            # No parents: just marginal distribution
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var] == val) + self.alpha

            log_likelihood = 0.0
            for count in counts:
                if count > 0:
                    log_likelihood += count * np.log(count / (n_samples + k * self.alpha))

            # BIC penalty
            num_params = k - 1  # Degrees of freedom for multinomial
            if self.score_metric == "bic":
                penalty = 0.5 * num_params * np.log(n_samples)
            elif self.score_metric == "aic":
                penalty = num_params
            else:  # k2 or no penalty
                penalty = 0

            return log_likelihood - penalty

        else:
            # With parents: conditional distribution
            parent_card = [int(cardinality[p]) for p in parents]
            n_parent_configs = int(np.prod(parent_card))

            # Calculate parent configuration indices for all samples
            parent_configs = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                config = 0
                mult = 1
                for j, p in enumerate(parents):
                    config += int(population[i, p]) * mult
                    mult *= parent_card[j]
                parent_configs[i] = config

            # Calculate conditional counts
            log_likelihood = 0.0
            for parent_config in range(n_parent_configs):
                mask = parent_configs == parent_config
                n_config = np.sum(mask)

                if n_config > 0:
                    counts = np.zeros(k)
                    for val in range(k):
                        counts[val] = np.sum(population[mask, var] == val) + self.alpha

                    for count in counts:
                        if count > 0:
                            log_likelihood += count * np.log(
                                count / (n_config + k * self.alpha)
                            )

            # BIC penalty
            num_params = n_parent_configs * (k - 1)
            if self.score_metric == "bic":
                penalty = 0.5 * num_params * np.log(n_samples)
            elif self.score_metric == "aic":
                penalty = num_params
            else:
                penalty = 0

            return log_likelihood - penalty

    def _would_create_cycle(
        self, adjacency: np.ndarray, parent: int, child: int
    ) -> bool:
        """
        Check if adding an edge parent->child would create a cycle

        Args:
            adjacency: Current adjacency matrix
            parent: Parent node
            child: Child node

        Returns:
            True if adding edge would create cycle
        """
        # Check if there's already a path from child to parent
        # If so, adding parent->child would create a cycle
        n_vars = adjacency.shape[0]
        visited = np.zeros(n_vars, dtype=bool)
        stack = [child]

        while stack:
            node = stack.pop()
            if node == parent:
                return True  # Found path from child to parent

            if visited[node]:
                continue

            visited[node] = True

            # Add all children of this node to stack
            for next_node in range(n_vars):
                if adjacency[node, next_node] > 0 and not visited[next_node]:
                    stack.append(next_node)

        return False  # No cycle

    def _greedy_structure_learning(
        self, population: np.ndarray, n_vars: int, cardinality: np.ndarray
    ) -> np.ndarray:
        """
        Learn Bayesian network structure using greedy hill-climbing

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Adjacency matrix (parents -> children)
        """
        n_samples = population.shape[0]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        # For each variable, greedily select best parent set
        for var in range(n_vars):
            # Start with no parents
            current_parents = []
            current_score = self._calculate_local_score(
                var, current_parents, population, cardinality, n_samples
            )

            # Greedily add parents
            for _ in range(min(self.max_parents, n_vars - 1)):
                best_parent = -1
                best_score = current_score

                # Try adding each possible parent
                for candidate in range(n_vars):
                    if candidate == var or candidate in current_parents:
                        continue

                    # Check if adding this edge would create a cycle
                    if self._would_create_cycle(adjacency, candidate, var):
                        continue

                    test_parents = current_parents + [candidate]
                    score = self._calculate_local_score(
                        var, test_parents, population, cardinality, n_samples
                    )

                    if score > best_score:
                        best_score = score
                        best_parent = candidate

                if best_parent >= 0:
                    current_parents.append(best_parent)
                    current_score = best_score
                    # Add edge to adjacency matrix
                    adjacency[best_parent, var] = 1
                else:
                    break  # No improvement

        return adjacency

    def _learn_cpd(
        self,
        var: int,
        parents: list,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Learn conditional probability distribution for a variable

        Args:
            var: Variable index
            parents: Parent indices
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            CPD table (conditioned on parent configurations)
        """
        k = int(cardinality[var])
        n_samples = population.shape[0]

        if len(parents) == 0:
            # Marginal distribution
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var] == val) + self.alpha

            return counts / np.sum(counts)

        else:
            # Conditional distribution
            parent_card = [int(cardinality[p]) for p in parents]
            n_parent_configs = int(np.prod(parent_card))

            # Calculate parent configuration indices
            parent_configs = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                config = 0
                mult = 1
                for j, p in enumerate(parents):
                    config += int(population[i, p]) * mult
                    mult *= parent_card[j]
                parent_configs[i] = config

            # CPD table: [n_parent_configs, k]
            cpd = np.zeros((n_parent_configs, k))

            for parent_config in range(n_parent_configs):
                mask = parent_configs == parent_config
                counts = np.zeros(k)

                for val in range(k):
                    counts[val] = np.sum(population[mask, var] == val) + self.alpha

                cpd[parent_config, :] = counts / np.sum(counts)

            return cpd

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> BayesianNetworkModel:
        """
        Learn EBNA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for EBNA learning)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        # Learn or use provided structure
        if self.structure is not None:
            adjacency = self.structure
        else:
            adjacency = self._greedy_structure_learning(population, n_vars, cardinality)

        # Learn CPDs for each variable
        cpds = {}
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            cpd = self._learn_cpd(var, parents, population, cardinality)
            cpds[var] = {"parents": parents, "cpd": cpd}

        # Create and return model
        model = BayesianNetworkModel(
            structure=adjacency,
            parameters=cpds,
            metadata={
                "generation": generation,
                "model_type": "EBNA",
                "max_parents": self.max_parents,
                "score_metric": self.score_metric,
            },
        )

        return model
