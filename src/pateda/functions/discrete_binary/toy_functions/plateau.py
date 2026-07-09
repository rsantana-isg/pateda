"""
Plateau function for optimization benchmarking

Originally proposed by Mühlenbein & Schlierkamp-Voosen (1993).
The binary vector of length n is divided into non-overlapping blocks of 3 bits.
Each block has a value of 1 if all its bits are 1, and 0 otherwise.
The goal is to maximize the sum of these block values.
"""

import numpy as np


def plateau(x: np.ndarray) -> float:
    """
    Evaluate the Plateau function for a binary vector.

    The input vector length must be a multiple of 3.

    Args:
        x: Binary vector representing a solution.

    Returns:
        Fitness value (number of blocks of 3 that are all 1s).

    Raises:
        ValueError: If length of x is not a multiple of 3.
    """
    if x.ndim == 2:
        x = x.flatten()

    n_vars = len(x)
    if n_vars % 3 != 0:
        raise ValueError(f"Plateau function requires vector length to be a multiple of 3, got {n_vars}")

    fitness = 0.0
    for i in range(0, n_vars, 3):
        block = x[i : i + 3]
        if np.sum(block) == 3:
            fitness += 1.0

    return float(fitness)


def create_plateau_objective_function():
    """
    Create a Plateau objective function for use with EDAs

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Plateau function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([plateau(population)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = plateau(population[i])

        return fitness

    return objective
