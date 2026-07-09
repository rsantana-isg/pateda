"""
Max-Cut problem for graph optimization benchmarking

Codification: Binary vector, one variable for each node.
1 means the node is in partition S, 0 means it is in partition V\\S.
Goal: Maximize the sum of weights of edges crossing the cut.
"""

import numpy as np
from pateda.functions.graph_utils import read_max_cut_graph


class MaxCutInstance:
    """
    Represents an instance of the Max-Cut problem.
    """
    def __init__(self, n_nodes: int, adj_matrix: np.ndarray, weights: np.ndarray):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix
        self.weights = weights

    @classmethod
    def from_file(cls, filepath: str):
        """Load instance from a Max-Cut file."""
        n_nodes, adj_matrix, weights = read_max_cut_graph(filepath)
        return cls(n_nodes, adj_matrix, weights)


def eval_max_cut(x: np.ndarray, instance: MaxCutInstance) -> float:
    """
    Evaluate the Max-Cut objective function for a binary vector.

    Args:
        x: Binary vector representing node partition.
        instance: The MaxCutInstance.

    Returns:
        Fitness value (sum of weights of edges crossing the cut).
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) != instance.n_nodes:
        raise ValueError(f"Solution length ({len(x)}) does not match graph size ({instance.n_nodes})")

    # Fast vectorized cut calculation:
    # outer_diff[u, v] is 1 if x[u] != x[v], 0 otherwise
    outer_diff = np.abs(np.subtract.outer(x, x))

    # Sum of weights of crossing edges (each crossing edge is counted twice: u->v and v->u)
    total_cut_weight = 0.5 * np.sum(instance.weights * outer_diff)

    return float(total_cut_weight)


def create_max_cut_objective_function(instance: MaxCutInstance):
    """
    Create a Max-Cut objective function for use with EDAs.

    Args:
        instance: The MaxCutInstance.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Max-Cut for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([eval_max_cut(population, instance)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_max_cut(population[i], instance)

        return fitness

    return objective
