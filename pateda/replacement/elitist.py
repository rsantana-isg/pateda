"""
Elitist replacement

Equivalent to MATEDA's elitist replacement strategies
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import ReplacementMethod


class ElitistReplacement(ReplacementMethod):
    """
    Elitist replacement: keep best individuals from both populations

    The best n_elite individuals from the old population are preserved
    and the rest is replaced by the new population.
    """

    def __init__(self, n_elite: Optional[int] = None, elite_ratio: float = 0.1):
        """
        Initialize elitist replacement

        Args:
            n_elite: Number of elite individuals to preserve (None = use ratio)
            elite_ratio: Fraction of population to preserve (used if n_elite is None)
        """
        self.n_elite = n_elite
        self.elite_ratio = elite_ratio

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replace population while preserving elite individuals

        Args:
            old_pop: Previous generation population
            old_fitness: Previous generation fitness
            new_pop: Newly sampled population
            new_fitness: New population fitness
            **params: Additional parameters
                     - n_elite: Override instance n_elite
                     - elite_ratio: Override instance elite_ratio

        Returns:
            Tuple of (combined_population, combined_fitness)
        """
        old_pop_size = old_pop.shape[0]
        new_pop_size = new_pop.shape[0]

        # Determine number of elite individuals
        n_elite = params.get("n_elite", self.n_elite)
        if n_elite is None:
            elite_ratio = params.get("elite_ratio", self.elite_ratio)
            n_elite = max(1, int(old_pop_size * elite_ratio))

        # Ensure we don't select more elite than available
        n_elite = min(n_elite, old_pop_size)

        # Handle multi-objective fitness by using mean fitness for selection
        # For single-objective, this just flattens if needed
        if old_fitness.ndim == 2 and old_fitness.shape[1] > 1:
            old_fitness_flat = np.mean(old_fitness, axis=1)
            new_fitness_flat = np.mean(new_fitness, axis=1)
        else:
            old_fitness_flat = old_fitness.flatten()
            new_fitness_flat = new_fitness.flatten()

        # Get best individuals from old population
        elite_indices = np.argsort(old_fitness_flat)[-n_elite:]
        elite_pop = old_pop[elite_indices]
        elite_fitness = old_fitness[elite_indices]

        # Take best individuals from new population to fill the rest
        n_from_new = new_pop_size - n_elite

        if n_from_new > 0:
            # Sort new population and take best n_from_new
            new_sorted_indices = np.argsort(new_fitness_flat)[-n_from_new:]
            selected_new_pop = new_pop[new_sorted_indices]
            selected_new_fitness = new_fitness[new_sorted_indices]

            # Combine elite and new
            combined_pop = np.vstack([elite_pop, selected_new_pop])
            # Use vstack for fitness to handle both 1D and 2D cases
            if old_fitness.ndim == 1:
                combined_fitness = np.hstack([elite_fitness, selected_new_fitness])
            else:
                combined_fitness = np.vstack([elite_fitness, selected_new_fitness])
        else:
            # Only keep elite (shouldn't normally happen)
            combined_pop = elite_pop
            combined_fitness = elite_fitness

        return combined_pop, combined_fitness


class RTRReplacement(ReplacementMethod):
    """
    Restricted Tournament Replacement (RTR)

    Each new individual competes with similar individuals from old population
    for survival. Helps maintain diversity.
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize RTR replacement

        Args:
            window_size: Number of random individuals to compare for similarity
        """
        self.window_size = window_size

    def _hamming_distance(self, ind1: np.ndarray, ind2: np.ndarray) -> int:
        """Calculate Hamming distance between two individuals"""
        return np.sum(ind1 != ind2)

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replace using restricted tournament replacement

        Args:
            old_pop: Previous generation population
            old_fitness: Previous generation fitness
            new_pop: Newly sampled population
            new_fitness: New population fitness
            **params: Additional parameters
                     - window_size: Override instance window_size

        Returns:
            Tuple of (combined_population, combined_fitness)
        """
        old_pop_size = old_pop.shape[0]
        new_pop_size = new_pop.shape[0]
        window_size = params.get("window_size", self.window_size)

        # Start with old population
        combined_pop = old_pop.copy()
        combined_fitness = old_fitness.copy()

        # For each new individual
        for i in range(new_pop_size):
            new_ind = new_pop[i]
            new_fit = new_fitness[i]

            # Randomly select window_size individuals from current population
            window_size_actual = min(window_size, old_pop_size)
            random_indices = np.random.choice(old_pop_size, size=window_size_actual, replace=False)

            # Find most similar individual in window
            min_distance = float("inf")
            most_similar_idx = random_indices[0]

            for idx in random_indices:
                distance = self._hamming_distance(new_ind, combined_pop[idx])
                if distance < min_distance:
                    min_distance = distance
                    most_similar_idx = idx

            # Replace if new individual is better
            if new_fit > combined_fitness[most_similar_idx]:
                combined_pop[most_similar_idx] = new_ind
                combined_fitness[most_similar_idx] = new_fit

        return combined_pop, combined_fitness
