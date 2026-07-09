"""
Checkerboard function for optimization benchmarking

Based on Section 3.1 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997), and originally proposed by Boyan (1993).
"""

import numpy as np


def checkerboard(x: np.ndarray, n: int = None) -> float:
    """
    Evaluate the Checkerboard function for a binary vector.

    The object is to create a checkerboard pattern of 0's and 1's in an NxN grid.
    For each bit in an (N-2)x(N-2) grid centered in the NxN grid, +1 is added
    to the fitness for each of the four primary neighbors that have the opposite value.

    Args:
        x: Binary vector representing the flattened NxN grid (length must be a perfect square)
        n: Grid dimension. If None, it will be inferred as sqrt(len(x))

    Returns:
        Fitness value (sum of opposite neighbors for the center grid)

    Raises:
        ValueError: If len(x) is not a perfect square or doesn't match n*n
    """
    if x.ndim == 2:
        x = x.flatten()

    n_vars = len(x)
    if n is None:
        n = int(np.round(np.sqrt(n_vars)))

    if n * n != n_vars:
        raise ValueError(f"Checkerboard requires vector length to be a perfect square N^2, got {n_vars} (inferred N={n})")

    grid = x.reshape((n, n))
    fitness = 0.0

    # Evaluate each bit in the centered (N-2)x(N-2) grid
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            val = grid[i, j]
            # Check 4 primary neighbors
            if grid[i - 1, j] != val:
                fitness += 1.0
            if grid[i + 1, j] != val:
                fitness += 1.0
            if grid[i, j - 1] != val:
                fitness += 1.0
            if grid[i, j + 1] != val:
                fitness += 1.0

    return float(fitness)


def create_checkerboard_objective_function(n: int = None):
    """
    Create a Checkerboard objective function for use with EDAs

    Args:
        n: Grid dimension NxN. If None, it will be inferred during evaluation.

    Returns:
        Objective function that takes a population array and returns fitness values
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Checkerboard function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([checkerboard(population, n)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = checkerboard(population[i], n)

        return fitness

    return objective
