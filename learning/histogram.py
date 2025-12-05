"""
Histogram Model Learning for Permutation-based EDAs

This module implements learning methods for histogram-based models:
- Edge Histogram Model (EHM): Models transitions between consecutive positions
- Node Histogram Model (NHM): Models the probability of items at each position

References:
    [1] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: A review of distances
        for the Mallows and Generalized Mallows estimation of distribution
        algorithms. Computational Optimization and Applications, 2015
"""

import numpy as np
from typing import Dict, Any


class LearnEHM:
    """Learn Edge Histogram Model for permutations"""

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Learn method to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_pop=population,
            selected_fitness=fitness,
            **kwargs
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        symmetric: bool = True,
        beta_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Learn Edge Histogram Model from selected population.

        The EHM counts edges (transitions) between consecutive positions
        in permutations and uses these counts as probabilities.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used)
            symmetric: If True, use symmetric EHM (count both directions)
            beta_ratio: Prior parameter for mutation-like effect (>= 0)

        Returns:
            Model dictionary containing:
                - ehm_matrix: Edge histogram matrix (n_vars x n_vars)
                - symmetric: Whether the model is symmetric
        """
        n_selected = selected_pop.shape[0]

        # Initialize edge histogram matrix
        ehm = np.zeros((n_vars, n_vars))

        # Count edges in all permutations
        # Edge from position i to position i+1 (including wrap-around)
        for perm in selected_pop:
            for j in range(n_vars - 1):
                # Convert to 0-indexed if needed
                if np.min(selected_pop) == 1:
                    from_item = perm[j] - 1
                    to_item = perm[j + 1] - 1
                else:
                    from_item = perm[j]
                    to_item = perm[j + 1]

                ehm[from_item, to_item] += 1

        # Calculate epsilon (prior/smoothing parameter)
        if symmetric:
            epsilon = beta_ratio * ((2 * n_selected) / (n_vars - 1))
            # For symmetric: add both directions and epsilon
            ehm_model = ehm + ehm.T + epsilon * np.ones((n_vars, n_vars))
        else:
            epsilon = beta_ratio * (n_selected / (n_vars - 1))
            # For asymmetric: add only epsilon
            ehm_model = ehm + epsilon * np.ones((n_vars, n_vars))

        return {
            "ehm_matrix": ehm_model,
            "symmetric": symmetric,
            "model_type": "ehm",
        }


class LearnNHM:
    """Learn Node Histogram Model for permutations"""

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        beta_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Learn Node Histogram Model from selected population.

        The NHM counts how many times each item appears at each position
        and uses these counts as probabilities.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used)
            beta_ratio: Prior parameter for smoothing (>= 0)

        Returns:
            Model dictionary containing:
                - nhm_matrix: Node histogram matrix (n_vars x n_vars)
                              nhm[position, item] = count of item at position
        """
        n_selected = selected_pop.shape[0]

        # Initialize node histogram matrix
        # nhm[i, j] = probability of item j at position i
        nhm = np.zeros((n_vars, n_vars))

        # Count item occurrences at each position
        for perm in selected_pop:
            for pos in range(n_vars):
                # Convert to 0-indexed if needed
                if np.min(selected_pop) == 1:
                    item = perm[pos] - 1
                else:
                    item = perm[pos]

                nhm[pos, item] += 1

        # Add prior/smoothing
        epsilon = beta_ratio * n_selected
        nhm_model = nhm + epsilon * np.ones((n_vars, n_vars))

        # Normalize to get probabilities
        for pos in range(n_vars):
            row_sum = np.sum(nhm_model[pos, :])
            if row_sum > 0:
                nhm_model[pos, :] /= row_sum

        return {
            "nhm_matrix": nhm_model,
            "model_type": "nhm",
        }


def learn_ehm(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Edge Histogram Model.

    See LearnEHM for parameter details.
    """
    learner = LearnEHM()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )


def learn_nhm(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Node Histogram Model.

    See LearnNHM for parameter details.
    """
    learner = LearnNHM()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )
