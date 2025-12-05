"""
Greedy local search optimization for discrete problems

A simple greedy hill-climbing approach for discrete optimization problems.
Uses random neighborhood exploration with immediate acceptance of improvements.
"""

from typing import Any, Callable, Tuple
import numpy as np

from pateda.core.components import LocalOptMethod


class DiscreteGreedySearch(LocalOptMethod):
    """
    Greedy local search optimization for discrete problems

    This method applies a simple greedy hill-climbing strategy where random
    variables are perturbed to neighboring values. Improvements are accepted
    immediately, creating a deterministic local search that terminates when
    no improvement is found within a trial budget.

    The algorithm explores the discrete neighborhood by:
    1. Selecting a random variable
    2. Trying a random value from its domain
    3. Accepting if the fitness improves (or is within tolerance)
    4. Repeating until the trial budget is exhausted

    This is suitable for discrete combinatorial optimization problems where
    variables take integer values from finite domains.

    Attributes:
        trials: Maximum number of local optimization moves per solution (default: 100)
        tolerance: Acceptance tolerance for new solutions (default: 0.0)
        num_flips: Number of variables to flip simultaneously (default: 1)

    Examples:
        >>> # Optimize a population for a discrete problem
        >>> greedy = DiscreteGreedySearch(trials=200, num_flips=2)
        >>> cardinality = np.array([5, 5, 5])  # 3 variables with 5 values each
        >>> pop = np.random.randint(0, 5, size=(10, 3))
        >>> fitness = np.array([fitness_func(ind) for ind in pop])
        >>> opt_pop, opt_fitness = greedy.optimize(pop, fitness, fitness_func, cardinality)
    """

    def __init__(
        self,
        trials: int = 100,
        tolerance: float = 0.0,
        num_flips: int = 1,
    ):
        """
        Initialize the discrete greedy search method

        Args:
            trials: Maximum number of local optimization moves per solution.
                Higher values allow more thorough exploration but increase computation.
            tolerance: Acceptance tolerance (accept if new_fitness >= old_fitness - tolerance).
                A small positive value allows escaping shallow local optima.
            num_flips: Number of variables to flip simultaneously in each move.
                Higher values increase neighborhood size but may reduce search efficiency.
        """
        self.trials = trials
        self.tolerance = tolerance
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
        Apply greedy local search to discrete population

        Args:
            population: Population to optimize (n_individuals, n_vars).
                Each element should be an integer in [0, cardinality[i]-1].
            fitness: Current fitness values (n_individuals,)
            fitness_func: Fitness evaluation function that takes a 1D array
                and returns a scalar fitness value
            cardinality: Variable cardinalities (n_vars,).
                cardinality[i] is the number of possible values for variable i.
            **params: Additional parameters
                trials (int): Override default number of trials
                num_flips (int): Override default number of simultaneous flips
                tolerance (float): Override default tolerance

        Returns:
            Tuple of:
                - Optimized population (n_individuals, n_vars)
                - Optimized fitness values (n_individuals,)

        Notes:
            The algorithm terminates early if no improvement is found,
            making it efficient for problems with clear local optima.
        """
        n_individuals, n_vars = population.shape
        trials = params.get('trials', self.trials)
        num_flips = params.get('num_flips', self.num_flips)
        tolerance = params.get('tolerance', self.tolerance)

        # Ensure cardinality is 1D
        if cardinality.ndim == 2:
            # If continuous format (2, n_vars), just use as is
            # This shouldn't happen for discrete but handle gracefully
            cardinality = cardinality[1, :] - cardinality[0, :]

        new_pop = population.copy()
        new_fitness = fitness.copy()

        for i in range(n_individuals):
            best_x = new_pop[i, :].copy().astype(int)
            best_fitness = new_fitness[i]

            no_improvement_count = 0
            max_no_improvement = min(trials // 10, 20)  # Early stopping criterion

            for trial in range(trials):
                # Create new candidate by flipping random variables
                candidate_x = best_x.copy()

                # Select random variables to flip
                flip_indices = np.random.choice(
                    n_vars,
                    size=min(num_flips, n_vars),
                    replace=False
                )

                # Assign random values from each variable's domain
                for idx in flip_indices:
                    # Generate random value different from current (if possible)
                    if cardinality[idx] > 1:
                        # Generate new value different from current
                        new_val = np.random.randint(0, int(cardinality[idx]))
                        # Optionally ensure it's different (helps exploration)
                        while new_val == candidate_x[idx] and cardinality[idx] > 1 and np.random.rand() < 0.7:
                            new_val = np.random.randint(0, int(cardinality[idx]))
                        candidate_x[idx] = new_val
                    else:
                        # Only one possible value, no change
                        pass

                # Evaluate candidate
                candidate_fitness = fitness_func(candidate_x)

                # Accept if improvement (or within tolerance)
                if (candidate_fitness - best_fitness) > -tolerance:
                    if candidate_fitness > best_fitness:
                        no_improvement_count = 0  # Reset counter on improvement
                    else:
                        no_improvement_count += 1

                    best_fitness = candidate_fitness
                    best_x = candidate_x.copy()
                else:
                    no_improvement_count += 1

                # Early stopping if no improvement for a while
                if no_improvement_count >= max_no_improvement:
                    break

            new_pop[i, :] = best_x
            new_fitness[i] = best_fitness

        return new_pop, new_fitness
