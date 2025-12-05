"""
SciPy-based local search optimization

Equivalent to MATEDA's local_search_OffHP.m
Uses SciPy's optimization methods to refine solutions.
"""

from typing import Any, Callable, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from pateda.core.components import LocalOptMethod
from pateda.repairing.trigonometric import trigonometric_repairing


class ScipyLocalSearch(LocalOptMethod):
    """
    Local search using SciPy's optimization methods

    This method applies gradient-based local optimization to each individual
    in the population using SciPy's minimize function. Originally designed for
    continuous optimization problems like the off-lattice HP protein model.

    By default, uses the BFGS quasi-Newton method with optional trigonometric
    repairing for angle-based representations.

    Attributes:
        method: SciPy optimization method (default: 'BFGS')
        max_iter: Maximum iterations per individual (default: 10000)
        max_fun_evals: Maximum function evaluations per individual (default: 20000)
        apply_repairing: Whether to apply trigonometric repairing (default: True)
    """

    def __init__(
        self,
        method: str = 'BFGS',
        max_iter: int = 10000,
        max_fun_evals: int = 20000,
        apply_repairing: bool = True,
    ):
        """
        Initialize the local search method

        Args:
            method: SciPy optimization method ('BFGS', 'L-BFGS-B', 'CG', etc.)
            max_iter: Maximum iterations per optimization
            max_fun_evals: Maximum function evaluations per optimization
            apply_repairing: Whether to apply trigonometric repairing after optimization
        """
        self.method = method
        self.max_iter = max_iter
        self.max_fun_evals = max_fun_evals
        self.apply_repairing = apply_repairing

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Apply local optimization to population using SciPy

        Args:
            population: Population to optimize (n_individuals, n_vars)
            fitness: Current fitness values (n_individuals,)
            fitness_func: Fitness evaluation function (maximization assumed)
            cardinality: Variable cardinalities or ranges (2, n_vars) for continuous
            **params: Additional parameters
                range_values (np.ndarray): Range for repairing (default: [0, 2π])

        Returns:
            Tuple of:
                - Optimized population (n_individuals, n_vars)
                - Optimized fitness values (n_individuals,)
                - Number of function evaluations used
        """
        n_individuals, n_vars = population.shape
        new_pop = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        total_evals = 0

        # Set up optimization options
        options = {
            'maxiter': self.max_iter,
            'disp': False,
        }

        for i in range(n_individuals):
            x0 = population[i, :]

            # SciPy minimizes, but fitness_func typically maximizes
            # So we negate the function
            def objective(x):
                return -fitness_func(x)

            # Perform optimization
            result = minimize(
                objective,
                x0,
                method=self.method,
                options=options,
            )

            new_pop[i, :] = result.x
            new_fitness[i] = -result.fun  # Convert back to maximization
            total_evals += result.nfev

        # Apply repairing if needed (e.g., for angle-based representations)
        if self.apply_repairing:
            range_values = params.get(
                'range_values',
                np.array([np.zeros(n_vars), 2 * np.pi * np.ones(n_vars)])
            )
            new_pop = trigonometric_repairing(new_pop, range_values)

        return new_pop, new_fitness, total_evals
