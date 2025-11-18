"""
Mallows Model Sampling for Permutation-based EDAs

This module implements sampling methods for Mallows models with different
distance metrics (Kendall, Cayley, Ulam).

References:
    [1] C. L. Mallows: Non-null ranking models. Biometrika, 1957
    [2] J. Ceberio, A. Mendiburu, J.A Lozano: Introducing the Mallows Model
        on Estimation of Distribution Algorithms. ICONIP 2011
"""

import numpy as np
from typing import Dict, Any
from pateda.permutation.consensus import compose_permutations


class SampleMallowsKendall:
    """Sample from Mallows model with Kendall distance"""

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """
        Sample permutations from Mallows model with Kendall distance.

        Args:
            n_vars: Number of variables (permutation length)
            model: Model dictionary from learning phase containing:
                   - v_probs: Probability matrix for v-vector
                   - consensus: Consensus ranking
                   - theta: Theta parameter
                   - psis: Normalization constants
            cardinality: Not used for permutations
            population: Current population (not used)
            fitness: Fitness values (not used)
            sample_size: Number of permutations to sample

        Returns:
            Array of sampled permutations, shape (sample_size, n_vars)
        """
        v_probs = model["v_probs"]
        consensus = model["consensus"]

        new_pop = np.zeros((sample_size, n_vars), dtype=int)

        # Generate random values for all samples at once
        rand_values = np.random.rand(sample_size, n_vars - 1)

        for i in range(sample_size):
            # Sample v-vector
            v_vector = self._sample_v_vector(v_probs, rand_values[i], n_vars)

            # Generate permutation from v-vector
            perm = self._generate_perm_from_v(v_vector, n_vars)

            # Compose with consensus
            new_perm = compose_permutations(perm, consensus)

            new_pop[i] = new_perm

        return new_pop

    def _sample_v_vector(
        self, v_probs: np.ndarray, rand_values: np.ndarray, n_vars: int
    ) -> np.ndarray:
        """Sample a v-vector from the probability matrix."""
        v_vec = np.zeros(n_vars, dtype=int)

        for j in range(n_vars - 1):
            # Sample v[j] from categorical distribution
            cumsum = np.cumsum(v_probs[j, : n_vars - j])
            rand_val = rand_values[j]

            # Find index where cumsum >= rand_val
            index = np.searchsorted(cumsum, rand_val)

            v_vec[j] = index

        v_vec[n_vars - 1] = 0  # Last position is always 0

        return v_vec

    def _generate_perm_from_v(self, v: np.ndarray, n_vars: int) -> np.ndarray:
        """
        Generate permutation from v-vector (Lehmer code).

        The v-vector represents the permutation in a canonical way.
        v[i] indicates how many available positions to skip.
        """
        available = list(range(n_vars))
        perm = np.zeros(n_vars, dtype=int)

        for i in range(n_vars - 1):
            # Find the v[i]-th available position
            val = int(v[i])

            # Count non-removed positions
            index = 0
            count = 0

            while count <= val:
                if available[index] != -1:
                    if count == val:
                        break
                    count += 1
                index += 1

            perm[i] = available[index]
            available[index] = -1  # Mark as used

        # Last position gets the remaining element
        for idx, val in enumerate(available):
            if val != -1:
                perm[n_vars - 1] = val
                break

        return perm


def sample_mallows_kendall(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Convenience function to sample from Mallows model with Kendall distance.

    See SampleMallowsKendall for parameter details.
    """
    sampler = SampleMallowsKendall()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size)
