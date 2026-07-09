"""
Equal Products function for optimization benchmarking

Based on Section 3.5 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997).
The goal is to partition a set of N randomly chosen numbers in (0..5) into two subsets
such that the absolute difference between their products is minimized.
"""

import numpy as np


class EqualProductsInstance:
    """
    Represents an instance of the Equal Products problem, containing the set of numbers
    to be partitioned.
    """
    def __init__(self, numbers: np.ndarray):
        """
        Args:
            numbers: 1D array of real numbers to partition.
        """
        self.numbers = numbers
        self.n_vars = len(numbers)


def generate_random_equal_products(n_vars: int = 40) -> EqualProductsInstance:
    """
    Generate a random Equal Products instance.

    Args:
        n_vars: Number of variables (default: 40).

    Returns:
        An EqualProductsInstance object.
    """
    # Numbers are uniformly distributed in (0, 5]
    numbers = np.random.uniform(1e-4, 5.0, size=n_vars)
    return EqualProductsInstance(numbers)


def eval_equal_products(x: np.ndarray, instance: EqualProductsInstance) -> float:
    """
    Evaluate the Equal Products function for a binary vector.
    The goal is to minimize this value.

    Args:
        x: Binary vector.
        instance: The EqualProductsInstance containing the numbers.

    Returns:
        Absolute difference between the products of selected and unselected numbers.
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) != instance.n_vars:
        raise ValueError(f"Solution length ({len(x)}) does not match instance variable count ({instance.n_vars})")

    prod_selected = 1.0
    prod_not_selected = 1.0

    for i in range(len(x)):
        if x[i] == 1:
            prod_selected *= instance.numbers[i]
        else:
            prod_not_selected *= instance.numbers[i]

    return float(np.abs(prod_selected - prod_not_selected))


def create_equal_products_objective_function(instance: EqualProductsInstance):
    """
    Create an Equal Products objective function for use with EDAs.
    Note: The goal is to minimize this difference.

    Args:
        instance: The EqualProductsInstance containing the numbers.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Equal Products for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([eval_equal_products(population, instance)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_equal_products(population[i], instance)

        return fitness

    return objective
