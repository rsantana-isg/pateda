"""
Histogram Model Sampling for Permutation-based EDAs

This module implements sampling methods for histogram-based models:
- Edge Histogram Model (EHM)
- Node Histogram Model (NHM)
"""

import numpy as np
from typing import Dict, Any, Optional


class SampleEHM:
    """Sample from Edge Histogram Model"""

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample permutations from Edge Histogram Model.

        The EHM builds permutations sequentially, choosing the next item
        based on edge probabilities from the current item.

        Args:
            n_vars: Number of variables (permutation length)
            model: Model dictionary containing:
                   - ehm_matrix: Edge histogram matrix
            cardinality: Not used for permutations
            population: Current population (not used)
            fitness: Fitness values (not used)
            sample_size: Number of permutations to sample
            rng: Random number generator (optional)

        Returns:
            Array of sampled permutations, shape (sample_size, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        ehm_matrix = model["ehm_matrix"]

        # Determine if 0-indexed or 1-indexed
        min_val = 0 if np.min(population) == 0 else 1

        new_pop = np.zeros((sample_size, n_vars), dtype=int)

        for i in range(sample_size):
            # Start with a random city/item
            current_perm = [rng.integers(0, n_vars)]

            for j in range(1, n_vars - 1):
                # Get remaining items
                remaining = [x for x in range(n_vars) if x not in current_perm]

                # Get probabilities for transitions from current item
                current_item = current_perm[-1]
                probs = ehm_matrix[current_item, remaining].copy()

                # Normalize probabilities
                probs = probs / np.sum(probs)

                # Sample next item using stochastic universal sampling
                next_item_idx = self._sample_categorical(probs, rng)
                next_item = remaining[next_item_idx]

                current_perm.append(next_item)

            # Add last remaining item
            remaining = [x for x in range(n_vars) if x not in current_perm]
            current_perm.append(remaining[0])

            # Convert to appropriate indexing
            new_pop[i] = np.array(current_perm) + min_val

        return new_pop

    def _sample_categorical(self, probs: np.ndarray, rng: np.random.Generator) -> int:
        """Sample from categorical distribution."""
        cumsum = np.cumsum(probs)
        rand_val = rng.random()
        return int(np.searchsorted(cumsum, rand_val))


class SampleNHM:
    """Sample from Node Histogram Model"""

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample permutations from Node Histogram Model.

        The NHM samples each position independently based on the probability
        of items at that position, then repairs to ensure valid permutations.

        Args:
            n_vars: Number of variables (permutation length)
            model: Model dictionary containing:
                   - nhm_matrix: Node histogram matrix
            cardinality: Not used for permutations
            population: Current population (not used)
            fitness: Fitness values (not used)
            sample_size: Number of permutations to sample
            rng: Random number generator (optional)

        Returns:
            Array of sampled permutations, shape (sample_size, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        nhm_matrix = model["nhm_matrix"]

        # Determine if 0-indexed or 1-indexed
        min_val = 0 if np.min(population) == 0 else 1

        new_pop = np.zeros((sample_size, n_vars), dtype=int)

        for i in range(sample_size):
            perm = []
            used = set()

            for pos in range(n_vars):
                # Get probabilities for this position
                probs = nhm_matrix[pos, :].copy()

                # Zero out probabilities for already used items
                for used_item in used:
                    probs[used_item] = 0

                # Normalize
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    # If all items used, uniform over remaining
                    remaining = [x for x in range(n_vars) if x not in used]
                    probs = np.zeros(n_vars)
                    probs[remaining] = 1.0 / len(remaining)

                # Sample item for this position
                item = self._sample_categorical(probs, rng)
                perm.append(item)
                used.add(item)

            # Convert to appropriate indexing
            new_pop[i] = np.array(perm) + min_val

        return new_pop

    def _sample_categorical(self, probs: np.ndarray, rng: np.random.Generator) -> int:
        """Sample from categorical distribution."""
        cumsum = np.cumsum(probs)
        rand_val = rng.random()
        return int(np.searchsorted(cumsum, rand_val))


def sample_ehm(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Convenience function to sample from Edge Histogram Model.

    See SampleEHM for parameter details.
    """
    sampler = SampleEHM()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size)


def sample_nhm(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Convenience function to sample from Node Histogram Model.

    See SampleNHM for parameter details.
    """
    sampler = SampleNHM()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size)
