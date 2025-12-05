"""
Greedy local search optimization

Equivalent to MATEDA's Greedy_search_OffHP.m
Uses a simple greedy hill-climbing approach.
"""

from typing import Any, Callable, Tuple
import numpy as np

from pateda.core.components import LocalOptMethod
from pateda.repairing.trigonometric import trigonometric_repairing


class GreedySearch(LocalOptMethod):
    """
    Greedy local search optimization

    This method applies a simple greedy hill-climbing strategy where random
    positions in the solution are perturbed and accepted if they improve
    (or approximately maintain) the fitness value.

    Originally designed for continuous optimization problems like the
    off-lattice HP protein model where variables represent angles.

    Attributes:
        trials: Number of local optimization moves per solution (default: 100)
        apply_repairing: Whether to apply trigonometric repairing (default: True)
        tolerance: Acceptance tolerance for new solutions (default: 0.01)
    """

    def __init__(
        self,
        trials: int = 100,
        apply_repairing: bool = True,
        tolerance: float = 0.01,
    ):
        """
        Initialize the greedy search method

        Args:
            trials: Number of local optimization moves per solution
            apply_repairing: Whether to apply trigonometric repairing
            tolerance: Acceptance tolerance (accept if new_fitness >= old_fitness - tolerance)
        """
        self.trials = trials
        self.apply_repairing = apply_repairing
        self.tolerance = tolerance

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Apply greedy local search to population

        Args:
            population: Population to optimize (n_individuals, n_vars)
            fitness: Current fitness values (n_individuals,)
            fitness_func: Fitness evaluation function
            cardinality: Variable cardinalities or ranges (2, n_vars) for continuous
            **params: Additional parameters
                trials (int): Override default number of trials
                range_values (np.ndarray): Range for repairing (default: [0, 2π])

        Returns:
            Tuple of:
                - Optimized population (n_individuals, n_vars)
                - Optimized fitness values (n_individuals,)
                - Number of function evaluations used
        """
        n_individuals, n_vars = population.shape
        trials = params.get('trials', self.trials)

        new_pop = population.copy()
        new_fitness = fitness.copy()
        total_evals = 0

        for i in range(n_individuals):
            best_x = new_pop[i, :].copy()
            best_fitness = new_fitness[i]

            for _ in range(trials):
                # Create new candidate by perturbing two random positions
                candidate_x = best_x.copy()

                # Select two random positions to modify
                pos_a = np.random.randint(0, n_vars)
                pos_b = np.random.randint(0, n_vars)

                # Set random values (for angle-based: random angle in [0, 2π])
                candidate_x[pos_a] = 2 * np.pi * np.random.rand()
                candidate_x[pos_b] = 2 * np.pi * np.random.rand()

                # Evaluate candidate
                candidate_fitness = fitness_func(candidate_x)
                total_evals += 1

                # Accept if improvement (or within tolerance)
                # Note: tolerance allows escaping local optima slightly
                if (candidate_fitness - best_fitness) > -self.tolerance:
                    best_fitness = candidate_fitness
                    best_x = candidate_x.copy()

            new_pop[i, :] = best_x
            new_fitness[i] = best_fitness

        # Apply repairing if needed (e.g., for angle-based representations)
        if self.apply_repairing:
            range_values = params.get(
                'range_values',
                np.array([np.zeros(n_vars), 2 * np.pi * np.ones(n_vars)])
            )
            new_pop = trigonometric_repairing(new_pop, range_values)

        return new_pop, new_fitness, total_evals
