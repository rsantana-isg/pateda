"""
Trap function for optimization benchmarking

The trap function is a common benchmark in evolutionary algorithms.
It is deceptive, leading search towards local optima.

Based on MATEDA-2.0 evalfunctrapn.m and Trapn.m
"""

import numpy as np
from typing import Union


def trap_partition(vector: np.ndarray, n_trap: int) -> float:
    """
    Evaluate a single partition of the trap function

    Args:
        vector: Binary vector for this partition
        n_trap: Number of variables in the trap partition

    Returns:
        Fitness value for this partition

    Note:
        - If all bits are 1, returns n_trap (global optimum)
        - Otherwise, returns n_trap - 1 - sum(vector) (deceptive local optima)
    """
    s = np.sum(vector)
    if s == n_trap:
        return float(n_trap)
    else:
        return float(n_trap - 1 - s)


def trap_n(vector: np.ndarray, n_trap: int = 5) -> float:
    """
    Evaluate the non-overlapping trap function

    The function is decomposed into non-overlapping partitions of size n_trap.
    f(x) = f_d(x_0,...,x_{k-1}) + ... + f_d(x_{n-k},...,x_{n-1})

    where f_d is the deceptive trap function for each partition.

    Args:
        vector: Binary solution vector (1D array)
        n_trap: Number of variables in each trap partition (default: 5)

    Returns:
        Sum of trap values over all partitions

    Example:
        >>> x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        >>> trap_n(x, n_trap=5)
        9.0  # First partition: 5, second partition: 4
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n_vars = len(vector)
    val = 0.0

    # Evaluate each partition
    for i in range(0, n_vars, n_trap):
        partition = vector[i:i + n_trap]
        val += trap_partition(partition, n_trap)

    return val


def create_trap_objective_function(n_trap: int = 5):
    """
    Create a trap function objective for use with EDA

    Args:
        n_trap: Number of variables in each trap partition

    Returns:
        Objective function that takes a 2D population array and returns fitness values

    Example:
        >>> trap_func = create_trap_objective_function(n_trap=5)
        >>> population = np.array([[1,1,1,1,1], [0,0,0,0,0]])
        >>> fitness = trap_func(population)
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate trap function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars)

        Returns:
            1D array of fitness values of shape (pop_size,)
        """
        if population.ndim == 1:
            return np.array([trap_n(population, n_trap)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = trap_n(population[i], n_trap)

        return fitness

    return objective
