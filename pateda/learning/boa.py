"""
Bayesian Optimization Algorithm (BOA) learning

Equivalent to MATEDA's LearnBOA.m
BOA learns a Bayesian network using K2 or BD metrics with decision trees/graphs.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel


class LearnBOA(LearningMethod):
    """
    Learn a BOA (Bayesian Optimization Algorithm) model

    BOA learns a Bayesian network using sophisticated scoring metrics and
    structure learning algorithms. Can use decision trees or decision graphs
    for compact representation of CPDs.
    """

    def __init__(
        self,
        max_parents: int = 3,
        score_metric: str = "k2",
        metric_alpha: float = 1.0,
        use_decision_graphs: bool = False,
        ordering: Optional[np.ndarray] = None,
    ):
        """
        Initialize BOA learning

        Args:
            max_parents: Maximum number of parents per variable
            score_metric: Scoring metric ("k2", "bd", "bic")
            metric_alpha: Alpha parameter for BD metric (prior strength)
            use_decision_graphs: Use decision graphs for compact CPD representation
            ordering: Variable ordering for K2 algorithm (if None, use natural order)
        """
        self.max_parents = max_parents
        self.score_metric = score_metric
        self.metric_alpha = metric_alpha
        self.use_decision_graphs = use_decision_graphs
        self.ordering = ordering

    def _k2_score(
        self,
        var: int,
        parents: list,
        population: np.ndarray,
        cardinality: np.ndarray,
        alpha: float = 1.0,
    ) -> float:
        """
        Calculate K2 score for a variable given its parents

        The K2 score is based on the Bayesian-Dirichlet metric with uniform prior.

        Args:
            var: Variable index
            parents: List of parent indices
            population: Population data
            cardinality: Variable cardinalities
            alpha: Prior parameter (equivalent sample size)

        Returns:
            K2 score (log probability)
        """
        from scipy.special import gammaln

        k = int(cardinality[var])  # Variable cardinality
        n_samples = population.shape[0]

        if len(parents) == 0:
            # No parents: just marginal distribution
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var] == val)

            # K2 score
            score = gammaln(alpha) - gammaln(n_samples + alpha)
            for count in counts:
                score += gammaln(count + alpha / k) - gammaln(alpha / k)

            return score

        else:
            # With parents: conditional distribution
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

            # K2 score
            score = 0.0
            for parent_config in range(n_parent_configs):
                mask = parent_configs == parent_config
                n_config = np.sum(mask)

                if n_config > 0 or alpha > 0:
                    counts = np.zeros(k)
                    for val in range(k):
                        counts[val] = np.sum(population[mask, var] == val)

                    # Add contribution from this parent configuration
                    score += gammaln(alpha) - gammaln(n_config + alpha)
                    for count in counts:
                        score += gammaln(count + alpha / k) - gammaln(alpha / k)

            return score

    def _bd_score(
        self,
        var: int,
        parents: list,
        population: np.ndarray,
        cardinality: np.ndarray,
        alpha: float = 1.0,
    ) -> float:
        """
        Calculate Bayesian-Dirichlet (BD) score

        Similar to K2 but with explicit prior strength parameter.

        Args:
            var: Variable index
            parents: List of parent indices
            population: Population data
            cardinality: Variable cardinalities
            alpha: Prior parameter strength

        Returns:
            BD score
        """
        # BD is essentially the same as K2 with adjustable alpha
        return self._k2_score(var, parents, population, cardinality, alpha)

    def _k2_algorithm(
        self, population: np.ndarray, n_vars: int, cardinality: np.ndarray, ordering: np.ndarray
    ) -> np.ndarray:
        """
        K2 algorithm for Bayesian network structure learning

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities
            ordering: Variable ordering

        Returns:
            Adjacency matrix (parents -> children)
        """
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        # Process variables in given order
        for i, var in enumerate(ordering):
            # Can only have parents from earlier in ordering
            possible_parents = ordering[:i]

            current_parents = []
            current_score = self._k2_score(
                var, current_parents, population, cardinality, self.metric_alpha
            )

            improved = True
            while improved and len(current_parents) < self.max_parents:
                improved = False
                best_parent = -1
                best_score = current_score

                # Try adding each possible parent
                for candidate in possible_parents:
                    if candidate in current_parents:
                        continue

                    test_parents = current_parents + [candidate]
                    score = self._k2_score(
                        var, test_parents, population, cardinality, self.metric_alpha
                    )

                    if score > best_score:
                        best_score = score
                        best_parent = candidate
                        improved = True

                if improved:
                    current_parents.append(best_parent)
                    current_score = best_score

            # Set adjacency matrix
            for parent in current_parents:
                adjacency[parent, var] = 1

        return adjacency

    def _learn_cpd(
        self,
        var: int,
        parents: list,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Learn conditional probability distribution

        Args:
            var: Variable index
            parents: Parent indices
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            CPD table
        """
        k = int(cardinality[var])
        n_samples = population.shape[0]
        alpha_smooth = self.metric_alpha / k  # Smoothing

        if len(parents) == 0:
            # Marginal distribution
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var] == val) + alpha_smooth

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

            # CPD table
            cpd = np.zeros((n_parent_configs, k))

            for parent_config in range(n_parent_configs):
                mask = parent_configs == parent_config
                counts = np.zeros(k)

                for val in range(k):
                    counts[val] = np.sum(population[mask, var] == val) + alpha_smooth

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
        Learn BOA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        # Get variable ordering
        if self.ordering is not None:
            ordering = self.ordering
        else:
            ordering = np.arange(n_vars)

        # Learn structure using K2 algorithm
        adjacency = self._k2_algorithm(population, n_vars, cardinality, ordering)

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
                "model_type": "BOA",
                "max_parents": self.max_parents,
                "score_metric": self.score_metric,
                "metric_alpha": self.metric_alpha,
                "ordering": ordering,
            },
        )

        return model
