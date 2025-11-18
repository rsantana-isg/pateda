"""
Mallows Model Learning for Permutation-based EDAs

This module implements learning methods for Mallows models with different
distance metrics (Kendall, Cayley, Ulam).

References:
    [1] C. L. Mallows: Non-null ranking models. Biometrika, 1957
    [2] J. Ceberio, A. Mendiburu, J.A Lozano: Introducing the Mallows Model
        on Estimation of Distribution Algorithms. ICONIP 2011
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from scipy.optimize import fminbound
from pateda.permutation.distances import kendall_distance, cayley_distance, ulam_distance
from pateda.permutation.consensus import find_consensus_borda, compose_permutations


class LearnMallowsKendall:
    """Learn Mallows model with Kendall distance"""

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        """
        Learn Mallows model with Kendall distance.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used in learning)
            initial_theta: Initial theta parameter value
            upper_theta: Upper bound for theta
            max_iter: Maximum iterations for optimization
            consensus_method: Method to find consensus ("borda" or "median")

        Returns:
            Model dictionary containing:
                - v_probs: Probability matrix for v-vector
                - consensus: Consensus ranking
                - theta: Learned theta parameter
                - psis: Normalization constants
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        if consensus_method == "borda":
            consensus = find_consensus_borda(selected_pop)
        else:
            consensus = find_consensus_median(kendall_distance, selected_pop) # type: ignore

        # 2. Calculate theta parameter
        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(theta, n_vars)

        # 4. Calculate v-vector probability matrix
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, theta)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": theta,
            "psis": psis,
            "model_type": "mallows_kendall",
        }

    def _calculate_theta(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
    ) -> float:
        """Calculate theta parameter using maximum likelihood estimation."""
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus
        n_pop, n_vars = population.shape
        v_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            v_vec = self._v_vector(composition)
            v_vectors.append(v_vec)

        v_vectors_array = np.array(v_vectors)
        v_mean = np.mean(v_vectors_array, axis=0)

        # Use optimization to find theta that matches expected v-vector mean
        def objective(theta):
            expected_v = self._expected_v_vector(theta, n_vars)
            return np.sum((expected_v - v_mean) ** 2)

        # Use bounded optimization
        theta_opt = fminbound(objective, 0.001, upper_theta, xtol=1e-6, maxfun=max_iter)

        return float(theta_opt)

    def _v_vector(self, perm: np.ndarray) -> np.ndarray:
        """Calculate v-vector (Lehmer code) for a permutation."""
        n = len(perm)
        v = np.zeros(n, dtype=int)

        for i in range(n):
            v[i] = np.sum(perm[i] > perm[i + 1 :])

        return v

    def _expected_v_vector(self, theta: float, n: int) -> np.ndarray:
        """Calculate expected v-vector under Mallows model with given theta."""
        expected_v = np.zeros(n)

        for j in range(n - 1):
            # Expected value for position j
            # E[v_j] = sum_{r=0}^{n-j} r * P(v_j = r)
            psi_j = (1 - np.exp(-(n - j + 1) * theta)) / (1 - np.exp(-theta))

            expected_val = 0.0
            for r in range(n - j + 1):
                prob_r = np.exp(-r * theta) / psi_j
                expected_val += r * prob_r

            expected_v[j] = expected_val

        return expected_v

    def _calculate_psi_constants(self, theta: float, n: int) -> np.ndarray:
        """Calculate psi normalization constants."""
        j = np.arange(1, n)  # j from 1 to n-1
        psis = (1 - np.exp(-(n - j + 1) * theta)) / (1 - np.exp(-theta))
        return psis

    def _calculate_v_prob_matrix(
        self, n_vars: int, psis: np.ndarray, theta: float
    ) -> np.ndarray:
        """Calculate probability matrix for v-vector values."""
        v_probs = np.zeros((n_vars - 1, n_vars))

        for j in range(n_vars - 1):
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta) / psis[j]

        return v_probs


# For consistency with other learning methods
def learn_mallows_kendall(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Mallows model with Kendall distance.

    See LearnMallowsKendall for parameter details.
    """
    learner = LearnMallowsKendall()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )
