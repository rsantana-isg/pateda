"""
Simulated Annealing for discrete optimization problems

Implements a simple variant of simulated annealing with exponential
cooling schedule for discrete combinatorial optimization.
"""

from typing import Any, Callable, Tuple
import numpy as np

from pateda.core.components import LocalOptMethod


class DiscreteSimulatedAnnealing(LocalOptMethod):
    """
    Simulated Annealing for discrete optimization problems

    Simulated Annealing (SA) is a probabilistic metaheuristic that accepts
    worse solutions with a temperature-dependent probability, allowing escape
    from local optima. The temperature decreases over time according to a
    cooling schedule, gradually transitioning from exploration to exploitation.

    The algorithm follows the classic Metropolis criterion:
    - Accept improvements immediately
    - Accept worse solutions with probability exp(ΔE / T)
    - Decrease temperature T over time

    This implementation uses:
    - Exponential cooling schedule: T(t) = T0 * alpha^t
    - Random neighborhood exploration (flip random variables)
    - Configurable number of iterations and cooling parameters

    Attributes:
        initial_temp: Initial temperature T0 (default: 1.0)
        final_temp: Final temperature, stopping criterion (default: 0.01)
        alpha: Cooling rate, must be in (0, 1) (default: 0.95)
        iterations_per_temp: Number of iterations at each temperature (default: 100)
        num_flips: Number of variables to flip simultaneously (default: 1)

    Examples:
        >>> # Optimize a population for a discrete problem
        >>> sa = DiscreteSimulatedAnnealing(
        ...     initial_temp=2.0,
        ...     alpha=0.9,
        ...     iterations_per_temp=50
        ... )
        >>> cardinality = np.array([10, 10, 10])  # 3 variables, 10 values each
        >>> pop = np.random.randint(0, 10, size=(5, 3))
        >>> fitness = np.array([fitness_func(ind) for ind in pop])
        >>> opt_pop, opt_fitness = sa.optimize(pop, fitness, fitness_func, cardinality)
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        alpha: float = 0.95,
        iterations_per_temp: int = 100,
        num_flips: int = 1,
    ):
        """
        Initialize the Simulated Annealing method

        Args:
            initial_temp: Initial temperature T0. Higher values allow more
                exploration at the beginning. Should be chosen based on
                typical fitness differences in the problem.
            final_temp: Final temperature, algorithm stops when T < final_temp.
                Smaller values allow more thorough exploitation.
            alpha: Cooling rate (geometric cooling). Must be in (0, 1).
                Values closer to 1 result in slower cooling.
                Typical range: [0.8, 0.99]
            iterations_per_temp: Number of iterations at each temperature level.
                More iterations improve solution quality but increase runtime.
            num_flips: Number of variables to flip simultaneously in each move.
                Larger values increase neighborhood size but may slow convergence.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if initial_temp <= final_temp:
            raise ValueError(
                f"initial_temp ({initial_temp}) must be > final_temp ({final_temp})"
            )

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.num_flips = num_flips

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Simulated Annealing to discrete population

        Args:
            population: Population to optimize (n_individuals, n_vars).
                Each element should be an integer in [0, cardinality[i]-1].
            fitness: Current fitness values (n_individuals,)
            fitness_func: Fitness evaluation function that takes a 1D array
                and returns a scalar fitness value
            cardinality: Variable cardinalities (n_vars,).
                cardinality[i] is the number of possible values for variable i.
            **params: Additional parameters
                initial_temp (float): Override default initial temperature
                final_temp (float): Override default final temperature
                alpha (float): Override default cooling rate
                iterations_per_temp (int): Override iterations per temperature
                num_flips (int): Override number of simultaneous flips

        Returns:
            Tuple of:
                - Optimized population (n_individuals, n_vars)
                - Optimized fitness values (n_individuals,)

        Notes:
            Each individual is optimized independently using SA.
            The best solution found during the search is returned,
            not necessarily the final solution.
        """
        n_individuals, n_vars = population.shape

        # Get parameters (allow override)
        initial_temp = params.get('initial_temp', self.initial_temp)
        final_temp = params.get('final_temp', self.final_temp)
        alpha = params.get('alpha', self.alpha)
        iterations_per_temp = params.get('iterations_per_temp', self.iterations_per_temp)
        num_flips = params.get('num_flips', self.num_flips)

        # Ensure cardinality is 1D
        if cardinality.ndim == 2:
            # If continuous format (2, n_vars), convert
            cardinality = cardinality[1, :] - cardinality[0, :]

        new_pop = population.copy()
        new_fitness = fitness.copy()

        # Optimize each individual independently
        for i in range(n_individuals):
            current_x = new_pop[i, :].copy().astype(int)
            current_fitness = new_fitness[i]

            # Track best solution found
            best_x = current_x.copy()
            best_fitness = current_fitness

            # Initialize temperature
            temp = initial_temp

            # Main SA loop
            while temp > final_temp:
                for _ in range(iterations_per_temp):
                    # Generate neighbor by flipping random variables
                    neighbor_x = current_x.copy()

                    # Select random variables to flip
                    flip_indices = np.random.choice(
                        n_vars,
                        size=min(num_flips, n_vars),
                        replace=False
                    )

                    # Flip selected variables to random values
                    for idx in flip_indices:
                        if cardinality[idx] > 1:
                            # Generate random value (potentially different from current)
                            neighbor_x[idx] = np.random.randint(0, int(cardinality[idx]))

                    # Evaluate neighbor
                    neighbor_fitness = fitness_func(neighbor_x)

                    # Compute fitness difference (for maximization)
                    delta_fitness = neighbor_fitness - current_fitness

                    # Acceptance criterion
                    if delta_fitness > 0:
                        # Accept improvement
                        accept = True
                    else:
                        # Accept worse solution with probability exp(ΔE/T)
                        # For maximization: ΔE = new - old (negative for worse)
                        # Probability = exp(ΔE/T) decreases as T decreases
                        acceptance_prob = np.exp(delta_fitness / temp)
                        accept = np.random.rand() < acceptance_prob

                    if accept:
                        current_x = neighbor_x.copy()
                        current_fitness = neighbor_fitness

                        # Update best if improved
                        if current_fitness > best_fitness:
                            best_x = current_x.copy()
                            best_fitness = current_fitness

                # Cool down
                temp *= alpha

            # Store best solution found
            new_pop[i, :] = best_x
            new_fitness[i] = best_fitness

        return new_pop, new_fitness


class DiscreteSimulatedAnnealingLinear(LocalOptMethod):
    """
    Simulated Annealing with linear cooling schedule

    An alternative SA implementation using linear cooling: T(t) = T0 - t * (T0 - Tf) / max_iter
    This provides a simpler, more predictable cooling schedule.

    Attributes:
        initial_temp: Initial temperature T0 (default: 1.0)
        final_temp: Final temperature Tf (default: 0.01)
        max_iterations: Total number of iterations (default: 1000)
        num_flips: Number of variables to flip simultaneously (default: 1)

    Examples:
        >>> # Optimize with linear cooling
        >>> sa = DiscreteSimulatedAnnealingLinear(
        ...     initial_temp=5.0,
        ...     max_iterations=2000
        ... )
        >>> cardinality = np.array([5, 5, 5])
        >>> pop = np.random.randint(0, 5, size=(10, 3))
        >>> fitness = np.array([fitness_func(ind) for ind in pop])
        >>> opt_pop, opt_fitness = sa.optimize(pop, fitness, fitness_func, cardinality)
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        max_iterations: int = 1000,
        num_flips: int = 1,
    ):
        """
        Initialize the Simulated Annealing method with linear cooling

        Args:
            initial_temp: Initial temperature T0
            final_temp: Final temperature Tf
            max_iterations: Total number of iterations
            num_flips: Number of variables to flip simultaneously
        """
        if initial_temp <= final_temp:
            raise ValueError(
                f"initial_temp ({initial_temp}) must be > final_temp ({final_temp})"
            )

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.num_flips = num_flips

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Simulated Annealing with linear cooling to discrete population

        Args:
            population: Population to optimize (n_individuals, n_vars)
            fitness: Current fitness values (n_individuals,)
            fitness_func: Fitness evaluation function
            cardinality: Variable cardinalities (n_vars,)
            **params: Additional parameters
                initial_temp (float): Override default initial temperature
                final_temp (float): Override default final temperature
                max_iterations (int): Override default max iterations
                num_flips (int): Override number of simultaneous flips

        Returns:
            Tuple of (optimized_population, optimized_fitness)
        """
        n_individuals, n_vars = population.shape

        # Get parameters
        initial_temp = params.get('initial_temp', self.initial_temp)
        final_temp = params.get('final_temp', self.final_temp)
        max_iterations = params.get('max_iterations', self.max_iterations)
        num_flips = params.get('num_flips', self.num_flips)

        # Ensure cardinality is 1D
        if cardinality.ndim == 2:
            cardinality = cardinality[1, :] - cardinality[0, :]

        new_pop = population.copy()
        new_fitness = fitness.copy()

        # Optimize each individual
        for i in range(n_individuals):
            current_x = new_pop[i, :].copy().astype(int)
            current_fitness = new_fitness[i]

            best_x = current_x.copy()
            best_fitness = current_fitness

            # Linear cooling: T(t) = T0 - t * (T0 - Tf) / max_iter
            temp_step = (initial_temp - final_temp) / max_iterations

            for iteration in range(max_iterations):
                # Update temperature linearly
                temp = initial_temp - iteration * temp_step

                # Generate neighbor
                neighbor_x = current_x.copy()
                flip_indices = np.random.choice(
                    n_vars,
                    size=min(num_flips, n_vars),
                    replace=False
                )

                for idx in flip_indices:
                    if cardinality[idx] > 1:
                        neighbor_x[idx] = np.random.randint(0, int(cardinality[idx]))

                # Evaluate
                neighbor_fitness = fitness_func(neighbor_x)
                delta_fitness = neighbor_fitness - current_fitness

                # Accept or reject
                if delta_fitness > 0 or np.random.rand() < np.exp(delta_fitness / max(temp, 1e-10)):
                    current_x = neighbor_x.copy()
                    current_fitness = neighbor_fitness

                    if current_fitness > best_fitness:
                        best_x = current_x.copy()
                        best_fitness = current_fitness

            new_pop[i, :] = best_x
            new_fitness[i] = best_fitness

        return new_pop, new_fitness
