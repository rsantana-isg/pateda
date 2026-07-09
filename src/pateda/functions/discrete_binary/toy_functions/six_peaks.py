"""
Six-Peaks function for optimization benchmarking

Based on Section 3.3 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997), and originally proposed by De Bonet et al. (1997).
"""

import numpy as np
from pateda.functions.discrete_binary.toy_functions.four_peaks import head, tail


def six_peaks(x: np.ndarray, t: int, reward: float = 100.0) -> float:
    """
    Evaluate the Six-Peaks function for a binary vector.

    SixPeaks(T, X) = max(head(X0, X), tail(not X0, X)) + Reward(T, X)
    where Reward(T, X) = reward if (head(X0, X) > T) and (tail(not X0, X) > T), else 0.
    Here, X0 is the first bit of X, and not X0 is the opposite binary value.

    Args:
        x: Binary vector representing a solution.
        t: Threshold parameter T.
        reward: The reward value added when both head and tail conditions exceed T (default: 100.0).

    Returns:
        Fitness value.
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) == 0:
        return 0.0

    x0 = int(x[0])
    not_x0 = 1 - x0

    h_x0 = head(x0, x)
    t_not_x0 = tail(not_x0, x)

    bonus = reward if (h_x0 > t and t_not_x0 > t) else 0.0
    return float(max(h_x0, t_not_x0) + bonus)


def create_six_peaks_objective_function(t: int, reward: float = 100.0):
    """
    Create a Six-Peaks objective function for use with EDAs

    Args:
        t: Threshold parameter T.
        reward: The reward value added when conditions are met (default: 100.0).

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Six-Peaks function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([six_peaks(population, t, reward)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = six_peaks(population[i], t, reward)

        return fitness

    return objective
