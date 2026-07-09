"""
Continuous-Peaks function for optimization benchmarking

Based on Section 3.3 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997).
In this variant, sequences of contiguous 0s and 1s can occur anywhere in the string.
"""

import numpy as np


def max_run(val: int, x: np.ndarray) -> int:
    """Find the length of the longest contiguous run of val in x."""
    max_len = 0
    current_len = 0
    for bit in x:
        if bit == val:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 0
    return max_len


def continuous_peaks(x: np.ndarray, t: int, reward: float = 100.0) -> float:
    """
    Evaluate the Continuous-Peaks function for a binary vector.

    ContinuousPeaks(T, X) = max(max_run(0, X), max_run(1, X)) + Reward(T, X)
    where Reward(T, X) = reward if (max_run(0, X) > T) and (max_run(1, X) > T), else 0.

    Args:
        x: Binary vector representing a solution.
        t: Threshold parameter T.
        reward: The reward value added when both runs of 0s and 1s exceed T (default: 100.0).

    Returns:
        Fitness value.
    """
    if x.ndim == 2:
        x = x.flatten()

    r0 = max_run(0, x)
    r1 = max_run(1, x)

    bonus = reward if (r0 > t and r1 > t) else 0.0
    return float(max(r0, r1) + bonus)


def create_continuous_peaks_objective_function(t: int, reward: float = 100.0):
    """
    Create a Continuous-Peaks objective function for use with EDAs

    Args:
        t: Threshold parameter T.
        reward: The reward value added when conditions are met (default: 100.0).

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Continuous-Peaks function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([continuous_peaks(population, t, reward)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = continuous_peaks(population[i], t, reward)

        return fitness

    return objective
