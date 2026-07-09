"""
Four-Peaks function for optimization benchmarking

Based on Section 3.3 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997), and originally proposed by Baluja & Caruana (1995).
"""

import numpy as np


def head(val: int, x: np.ndarray) -> int:
    """Count contiguous leading elements set to val."""
    count = 0
    for bit in x:
        if bit == val:
            count += 1
        else:
            break
    return count


def tail(val: int, x: np.ndarray) -> int:
    """Count contiguous trailing elements set to val."""
    count = 0
    for bit in reversed(x):
        if bit == val:
            count += 1
        else:
            break
    return count


def four_peaks(x: np.ndarray, t: int, reward: float = 100.0) -> float:
    """
    Evaluate the Four-Peaks function for a binary vector.

    FourPeaks(T, X) = max(head(1, X), tail(0, X)) + Reward(T, X)
    where Reward(T, X) = reward if (head(1, X) > T) and (tail(0, X) > T), else 0.

    Args:
        x: Binary vector representing a solution.
        t: Threshold parameter T.
        reward: The reward value added when both head and tail exceed T (default: 100.0).

    Returns:
        Fitness value.
    """
    if x.ndim == 2:
        x = x.flatten()

    h1 = head(1, x)
    t0 = tail(0, x)

    bonus = reward if (h1 > t and t0 > t) else 0.0
    return float(max(h1, t0) + bonus)


def create_four_peaks_objective_function(t: int, reward: float = 100.0):
    """
    Create a Four-Peaks objective function for use with EDAs

    Args:
        t: Threshold parameter T.
        reward: The reward value added when both head and tail exceed T (default: 100.0).

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Four-Peaks function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([four_peaks(population, t, reward)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = four_peaks(population[i], t, reward)

        return fitness

    return objective
