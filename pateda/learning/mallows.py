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
from scipy.optimize import fminbound, newton
from pateda.permutation.distances import (
    kendall_distance,
    cayley_distance,
    ulam_distance,
    _x_vector_cycles,
)
from pateda.permutation.consensus import find_consensus_borda, find_consensus_median, compose_permutations


class LearnMallowsKendall:
    """Learn Mallows model with Kendall distance"""

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


class LearnMallowsCayley:
    """Learn Mallows model with Cayley distance"""

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
        Learn Mallows model with Cayley distance.

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
                - x_probs: Probability vector for x-vector
                - consensus: Consensus ranking
                - theta: Learned theta parameter
                - psis: Normalization constants
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        if consensus_method == "borda":
            consensus = find_consensus_borda(selected_pop)
        else:
            consensus = find_consensus_median(cayley_distance, selected_pop)  # type: ignore

        # 2. Calculate theta parameter
        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(theta, n_vars)

        # 4. Calculate x-vector probability vector
        x_probs = self._calculate_x_prob_vector(psis)

        return {
            "x_probs": x_probs,
            "consensus": consensus,
            "theta": theta,
            "psis": psis,
            "model_type": "mallows_cayley",
        }

    def _calculate_theta(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> float:
        """
        Calculate theta parameter using Newton-Raphson method.

        References:
            [1] E. Irurozki, B. Calvo, J.A Lozano: Sampling and learning mallows
                and generalized mallows models under the cayley distance. Tech. Rep., 2013
        """
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus and calculate x-vectors
        n_pop, _ = population.shape
        x_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            x_vec = _x_vector_cycles(composition)
            x_vectors.append(x_vec)

        x_vectors_array = np.array(x_vectors)
        x_mean = np.mean(x_vectors_array, axis=0)

        # Define the theta function and its derivative for Newton-Raphson
        def theta_function(theta):
            """Function to find root: f(theta) = 0"""
            j = np.arange(1, n_vars)  # j from 1 to n-1
            return np.sum(j / (j + np.exp(theta))) - np.sum(x_mean)

        def theta_derivative(theta):
            """Derivative of theta function"""
            j = np.arange(1, n_vars)  # j from 1 to n-1
            return np.sum((-j * np.exp(theta)) / ((np.exp(theta) + j) ** 2))

        # Use Newton-Raphson to find theta
        try:
            theta_opt = newton(
                theta_function,
                initial_theta,
                fprime=theta_derivative,
                maxiter=max_iter,
                tol=1e-6,
            )
            # Clip to valid range
            theta_opt = np.clip(theta_opt, 0.001, upper_theta)
        except:
            # Fallback to simple search if Newton-Raphson fails
            theta_opt = fminbound(
                lambda t: abs(theta_function(t)), 0.001, upper_theta, xtol=1e-6
            )

        return float(theta_opt)

    def _calculate_psi_constants(self, theta: float, n: int) -> np.ndarray:
        """
        Calculate psi normalization constants for Cayley distance.

        Psi_j = (n-j) * exp(-theta) + 1
        """
        j = np.arange(1, n)  # j from 1 to n-1
        psis = (n - j) * np.exp(-theta) + 1
        return psis

    def _calculate_x_prob_vector(self, psis: np.ndarray) -> np.ndarray:
        """
        Calculate probability vector for x-vector values.

        For Cayley distance: P(x_j = 1) = 1 / Psi_j
        """
        x_probs = 1.0 / psis
        return x_probs


def learn_mallows_cayley(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Mallows model with Cayley distance.

    See LearnMallowsCayley for parameter details.
    """
    learner = LearnMallowsCayley()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )


class LearnGeneralizedMallowsKendall:
    """Learn Generalized Mallows model with Kendall distance

    The Generalized Mallows model uses a position-dependent spread parameter theta,
    where theta is a vector of length n-1 instead of a single value.

    References:
        [1] M.A. Fligner, J.S. Verducci: Distance based ranking models. JRSS, 1986
        [2] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: A Distance-based
            Ranking Model Estimation of Distribution Algorithm for the Flowshop
            Scheduling Problem. IEEE TEVC, 2014
    """

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
        Learn Generalized Mallows model with Kendall distance.

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
                - theta: Learned theta parameter vector (length n-1)
                - psis: Normalization constants (length n-1)
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        if consensus_method == "borda":
            consensus = find_consensus_borda(selected_pop)
        else:
            consensus = find_consensus_median(kendall_distance, selected_pop)  # type: ignore

        # 2. Calculate theta parameters (vector of length n-1)
        thetas = self._calculate_thetas(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(thetas, n_vars)

        # 4. Calculate v-vector probability matrix
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, thetas)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": thetas,
            "psis": psis,
            "model_type": "generalized_mallows_kendall",
        }

    def _calculate_thetas(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> np.ndarray:
        """Calculate theta parameters (one for each position) using MLE."""
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus
        n_pop = population.shape[0]
        v_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            v_vec = self._v_vector(composition)
            v_vectors.append(v_vec)

        v_vectors_array = np.array(v_vectors)
        v_mean = np.mean(v_vectors_array, axis=0)

        # Calculate theta_j for each position j independently
        thetas = np.zeros(n_vars - 1)

        for j in range(n_vars - 1):
            # For position j, we need to find theta_j that matches E[v_j]
            def theta_function(theta):
                """Function f(theta) whose root gives the MLE"""
                n_j = n_vars - j  # Number of items from position j onward
                if abs(theta) < 1e-10:
                    return n_j / 2.0 - v_mean[j]

                # E[V_j] = (n_j * exp(-theta) - exp(-(n_j+1)*theta)) / (1 - exp(-(n_j+1)*theta))
                exp_theta = np.exp(-theta)
                exp_nj1_theta = np.exp(-(n_j + 1) * theta)

                if abs(1 - exp_nj1_theta) < 1e-10:
                    expected_vj = n_j / 2.0
                else:
                    numerator = n_j * exp_theta - exp_nj1_theta
                    denominator = 1 - exp_nj1_theta
                    expected_vj = numerator / denominator - 1.0

                return expected_vj - v_mean[j]

            def theta_derivative(theta):
                """Derivative of theta function"""
                n_j = n_vars - j
                if abs(theta) < 1e-10:
                    return -(n_j * (n_j + 1)) / 12.0

                exp_theta = np.exp(-theta)
                exp_nj1_theta = np.exp(-(n_j + 1) * theta)

                if abs(1 - exp_nj1_theta) < 1e-10:
                    return -(n_j * (n_j + 1)) / 12.0

                # Numerical derivative
                delta = 1e-8
                return (theta_function(theta + delta) - theta_function(theta)) / delta

            # Use Newton-Raphson for this position
            try:
                theta_j = newton(
                    theta_function,
                    initial_theta,
                    fprime=theta_derivative,
                    maxiter=max_iter,
                    tol=1e-6,
                )
                theta_j = np.clip(theta_j, 0.001, upper_theta)
            except:
                # Fallback to bounded optimization
                theta_j = fminbound(
                    lambda t: abs(theta_function(t)),
                    0.001,
                    upper_theta,
                    xtol=1e-6,
                    maxfun=max_iter,
                )

            thetas[j] = theta_j

        return thetas

    def _v_vector(self, perm: np.ndarray) -> np.ndarray:
        """Calculate v-vector (Lehmer code) for a permutation."""
        n = len(perm)
        v = np.zeros(n, dtype=int)

        for i in range(n):
            v[i] = np.sum(perm[i] > perm[i + 1 :])

        return v

    def _calculate_psi_constants(self, thetas: np.ndarray, n: int) -> np.ndarray:
        """Calculate psi normalization constants for each position."""
        psis = np.zeros(n - 1)

        for j in range(n - 1):
            # Psi_j = sum_{r=0}^{n-j} exp(-r * theta_j)
            theta_j = thetas[j]
            n_j = n - j
            psi_j = np.sum(np.exp(-np.arange(n_j + 1) * theta_j))
            psis[j] = psi_j

        return psis

    def _calculate_v_prob_matrix(
        self, n_vars: int, psis: np.ndarray, thetas: np.ndarray
    ) -> np.ndarray:
        """Calculate probability matrix for v-vector values."""
        v_probs = np.zeros((n_vars - 1, n_vars))

        for j in range(n_vars - 1):
            theta_j = thetas[j]
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta_j) / psis[j]

        return v_probs


class LearnGeneralizedMallowsCayley:
    """Learn Generalized Mallows model with Cayley distance

    The Generalized Mallows model uses a position-dependent spread parameter theta,
    where theta is a vector of length n-1 instead of a single value.

    References:
        [1] M.A. Fligner, J.S. Verducci: Distance based ranking models. JRSS, 1986
        [2] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: Extending Distance-based
            Ranking Models in EDAs. CEC 2014
    """

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
        Learn Generalized Mallows model with Cayley distance.

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
                - x_probs: Probability matrix for x-vector (n-1 x 2)
                - consensus: Consensus ranking
                - theta: Learned theta parameter vector (length n-1)
                - psis: Normalization constants (length n-1)
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        if consensus_method == "borda":
            consensus = find_consensus_borda(selected_pop)
        else:
            consensus = find_consensus_median(cayley_distance, selected_pop)  # type: ignore

        # 2. Calculate theta parameters (vector of length n-1)
        thetas = self._calculate_thetas(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(thetas, n_vars)

        # 4. Calculate x-vector probability matrix
        x_probs = self._calculate_x_prob_matrix(psis)

        return {
            "x_probs": x_probs,
            "consensus": consensus,
            "theta": thetas,
            "psis": psis,
            "model_type": "generalized_mallows_cayley",
        }

    def _calculate_thetas(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> np.ndarray:
        """Calculate theta parameters (one for each position) using MLE."""
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus
        n_pop = population.shape[0]
        x_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            x_vec = _x_vector_cycles(composition)
            x_vectors.append(x_vec)

        x_vectors_array = np.array(x_vectors)
        x_mean = np.mean(x_vectors_array, axis=0)

        # Calculate theta_j for each position j independently
        thetas = np.zeros(n_vars - 1)

        for j in range(n_vars - 1):
            # For Cayley distance with Generalized Mallows:
            # P(X_j = 1) = 1 / Psi_j = 1 / ((n-j)*exp(-theta_j) + 1)
            # E[X_j] = P(X_j = 1)
            # So: x_mean[j] = 1 / ((n-j)*exp(-theta_j) + 1)

            # Solving for theta_j:
            # theta_j = -log((1/x_mean[j] - 1) / (n-j))

            n_j = n_vars - j - 1  # Note: j ranges from 0 to n-2

            if x_mean[j] > 0 and x_mean[j] < 1:
                # Direct analytical solution
                inner_val = (1.0 / x_mean[j] - 1.0) / (n_j + 1)
                if inner_val > 0:
                    theta_j = -np.log(inner_val)
                    theta_j = np.clip(theta_j, 0.001, upper_theta)
                else:
                    theta_j = upper_theta
            elif x_mean[j] >= 1:
                theta_j = upper_theta
            else:
                theta_j = 0.001

            thetas[j] = theta_j

        return thetas

    def _calculate_psi_constants(self, thetas: np.ndarray, n: int) -> np.ndarray:
        """Calculate psi normalization constants for each position."""
        psis = np.zeros(n - 1)

        for j in range(n - 1):
            # Psi_j = (n-j-1)*exp(-theta_j) + 1
            n_j = n - j - 1
            psis[j] = (n_j + 1) * np.exp(-thetas[j]) + 1

        return psis

    def _calculate_x_prob_matrix(self, psis: np.ndarray) -> np.ndarray:
        """Calculate probability matrix for x-vector values.

        Returns a matrix of shape (n-1, 2) where:
        - Column 0 is P(X_j = 0)
        - Column 1 is P(X_j = 1)
        """
        n = len(psis)
        x_probs = np.zeros((n, 2))

        for j in range(n):
            # P(X_j = 1) = 1 / Psi_j
            prob_1 = 1.0 / psis[j]
            prob_0 = 1.0 - prob_1
            x_probs[j, 0] = prob_0
            x_probs[j, 1] = prob_1

        return x_probs
