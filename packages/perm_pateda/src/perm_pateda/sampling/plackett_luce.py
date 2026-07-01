"""
Plackett-Luce Model Sampling for Permutation-based EDAs

This module implements the sampling method for the Plackett-Luce model.
Permutations are generated sequentially: at each step, the next item is
drawn from the remaining items with probability proportional to its weight.

References:
    [1] R.D. Luce: Individual Choice Behavior: A Theoretical Analysis. Wiley, 1959
    [2] R.L. Plackett: The analysis of permutations. Applied Statistics, 1975
    [3] J. Ceberio, A. Mendiburu, J.A. Lozano: The Plackett-Luce ranking model
        on permutation-based optimization problems. CEC 2013
"""

import numpy as np
from typing import Dict, Any


class SamplePlackettLuce:
    """Sample permutations from a Plackett-Luce model.

    Uses the Gumbel-max trick for efficient vectorized sampling: adding
    Gumbel noise to log-weights and argsort-ing produces draws equivalent
    to the sequential proportional selection process.
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
        **kwargs,
    ) -> np.ndarray:
        """Entry point to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
        )

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
        **kwargs,
    ) -> np.ndarray:
        """
        Sample permutations from the Plackett-Luce model.

        Uses the Gumbel-max trick: drawing Gumbel noise and adding it to
        log-weights, then argsort-ing, is equivalent to sequentially sampling
        each position proportionally to remaining weights — but fully
        vectorized over the entire population at once.

        Args:
            n_vars:      Number of variables (permutation length)
            model:       Model dictionary from learning phase containing:
                         - weights: Weight vector of length n_vars
            cardinality: Not used, kept for interface compatibility
            population:  Not used, kept for interface compatibility
            fitness:     Not used, kept for interface compatibility
            sample_size: Number of permutations to generate
            rng:         Random generator passed by _PermEDA

        Returns:
            Array of sampled permutations, shape (sample_size, n_vars)
        """
        weights = model["weights"]

        log_w = np.log(np.maximum(weights, 1e-12))  # (n_vars,)

        # Gumbel noise: shape (sample_size, n_vars)
        u = rng.uniform(0.0, 1.0, size=(sample_size, n_vars))
        gumbel_noise = -np.log(-np.log(u + 1e-12))

        scores = log_w + gumbel_noise  # (sample_size, n_vars)

        # argsort descending → permutation for each row
        samples = np.argsort(scores, axis=1)[:, ::-1].astype(int)

        return samples