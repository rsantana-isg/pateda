"""
Maximum generations or optimum reached stopping condition

Stops when either maximum generations is reached OR the optimal fitness is reached.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import StopCondition

# Tolerance for checking if optimum is reached (absolute difference)
# A value of 1e-6 is used to account for floating-point arithmetic precision
# while being strict enough to ensure the optimum is truly reached
OPTIMUM_TOLERANCE = 1e-6


class MaxGenerationsOrOptimum(StopCondition):
    """
    Stop after a maximum number of generations OR when optimal fitness is reached

    This stopping condition combines two criteria:
    1. Maximum number of generations (like MaxGenerations)
    2. Optimal fitness value reached (within tolerance)

    The algorithm stops when EITHER condition is met.
    """

    def __init__(self, max_gen: int, optimal_fitness: Optional[float] = None):
        """
        Initialize stopping condition

        Args:
            max_gen: Maximum number of generations to run
            optimal_fitness: Optimal fitness value (if known). If None, only max_gen is used.
        """
        self.max_gen = max_gen
        self.optimal_fitness = optimal_fitness

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        """
        Check if maximum generations reached OR optimal fitness reached

        Args:
            generation: Current generation number
            population: Current population (unused)
            fitness: Current fitness values (2D array: n_individuals x n_objectives)
            **params: Additional parameters (unused)

        Returns:
            True if generation >= max_gen OR best fitness >= optimal_fitness, False otherwise
        """
        # Check max generations
        if generation >= self.max_gen:
            return True

        # Check optimal fitness if specified
        if self.optimal_fitness is not None:
            # Handle both single and multi-objective fitness
            if fitness.ndim == 1:
                best_fitness = np.max(fitness)
            elif fitness.shape[1] == 1:
                best_fitness = np.max(fitness[:, 0])
            else:
                # Multi-objective: use mean fitness (simple aggregation)
                mean_fitness = np.mean(fitness, axis=1)
                best_fitness = np.max(mean_fitness)

            # Check if optimum reached (within tolerance)
            if abs(best_fitness - self.optimal_fitness) < OPTIMUM_TOLERANCE:
                return True

        return False

    def reset(self) -> None:
        """Reset stopping condition (nothing to reset)"""
        pass
