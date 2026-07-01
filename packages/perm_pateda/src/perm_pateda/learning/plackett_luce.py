"""
Plackett-Luce Model Learning for Permutation-based EDAs

This module implements the learning method for the Plackett-Luce model
using the MM (Minorization-Maximization) algorithm.

References:
    [1] R.D. Luce: Individual Choice Behavior: A Theoretical Analysis. Wiley, 1959
    [2] R.L. Plackett: The analysis of permutations. Applied Statistics, 1975
    [3] J. Ceberio, A. Mendiburu, J.A. Lozano: The Plackett-Luce ranking model
        on permutation-based optimization problems. CEC 2013
"""

import numpy as np
from typing import Dict, Any


class LearnPlackettLuce:
    """Learn Plackett-Luce model from a population of permutations.

    The model is parameterized by a weight vector w of length n, where w[i]
    represents the relative preference for item i. Weights are estimated via
    the MM (Minorization-Maximization) algorithm.
    """

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Entry point to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            **kwargs,
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Learn Plackett-Luce model weights from a population of permutations.

        Uses the MM (Minorization-Maximization) algorithm to find the MLE
        of the weight vector.

        Args:
            generation:  Current generation number
            n_vars:      Number of variables (permutation length)
            cardinality: Not used, kept for interface compatibility
            population:  Selected population of permutations, shape (K, n)
            fitness:     Not used, kept for interface compatibility
            max_iter:    Maximum iterations for the MM algorithm
            tol:         Convergence tolerance (L1 norm of weight update)

        Returns:
            Model dictionary containing:
                - weights:     Learned weight vector of length n_vars (normalized)
                - model_type:  "plackett_luce"
        """
        data = np.array(population, dtype=int)
        K, n = data.shape

        weights = np.ones(n, dtype=float) / n

        # Win counts: item i wins once per position it occupies except last
        wins = np.zeros(n, dtype=float)
        for k in range(K):
            for p in range(n - 1):
                wins[data[k, p]] += 1

        for _ in range(max_iter):
            old_weights = weights.copy()
            denominators = np.zeros(n, dtype=float)

            for k in range(K):
                perm = data[k]

                # suffix_sums[p] = sum of weights of items at positions p..n-1
                suffix_sums = np.cumsum(weights[perm][::-1])[::-1]

                # inv_suffix_sums[p] = 1 / suffix_sums[p] for p in 0..n-2
                inv_suffix_sums = np.zeros(n, dtype=float)
                valid = suffix_sums[:-1] > 0
                inv_suffix_sums[:-1][valid] = 1.0 / suffix_sums[:-1][valid]

                # cum_inv[p] = sum_{q=0}^{p} inv_suffix_sums[q]
                cum_inv = np.cumsum(inv_suffix_sums)

                # Vectorized denominator accumulation
                denominators[perm] += cum_inv

            weights = wins / np.maximum(denominators, 1e-12)
            weights = np.maximum(weights, 1e-9)
            weights /= np.sum(weights)

            if np.linalg.norm(weights - old_weights, ord=1) < tol:
                break

        return {
            "weights": weights,
            "model_type": "plackett_luce",
        }