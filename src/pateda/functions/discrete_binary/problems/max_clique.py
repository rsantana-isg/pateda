"""
Maximum Clique problem for graph optimization benchmarking

Codification: Binary vector, one variable for each node.
1 means the node is in the maximum clique, 0 otherwise.
Goal: Maximize clique size subject to the constraint that all selected nodes are pairwise adjacent.
"""

import numpy as np
from pateda.functions.graph_utils import read_dimacs_graph


class MaxCliqueInstance:
    """
    Represents an instance of the Maximum Clique problem.
    """
    def __init__(self, n_nodes: int, adj_matrix: np.ndarray):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix

    @classmethod
    def from_file(cls, filepath: str):
        """Load instance from a DIMACS .clq file."""
        n_nodes, adj_matrix = read_dimacs_graph(filepath)
        return cls(n_nodes, adj_matrix)


def eval_max_clique(x: np.ndarray, instance: MaxCliqueInstance, penalty: float = None) -> float:
    """
    Evaluate the Maximum Clique objective function for a binary vector.

    Args:
        x: Binary vector representing node selection.
        instance: The MaxCliqueInstance.
        penalty: Penalty coefficient for edge violations. Defaults to n_nodes + 1.

    Returns:
        Fitness value (clique size minus penalty for non-adjacent pairs).
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) != instance.n_nodes:
        raise ValueError(f"Solution length ({len(x)}) does not match graph size ({instance.n_nodes})")

    if penalty is None:
        penalty = float(instance.n_nodes + 1)

    size = np.sum(x)
    selected = np.where(x == 1)[0]
    k = len(selected)

    if k <= 1:
        return float(size)

    # Sub-adjacency matrix of selected vertices
    sub_adj = instance.adj_matrix[selected][:, selected]
    connected_pairs = np.sum(sub_adj) // 2
    total_pairs = k * (k - 1) // 2
    violations = total_pairs - connected_pairs

    fitness = size - penalty * violations
    return float(fitness)


def create_max_clique_objective_function(instance: MaxCliqueInstance, penalty: float = None):
    """
    Create a Maximum Clique objective function for use with EDAs.

    Args:
        instance: The MaxCliqueInstance.
        penalty: Penalty coefficient.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([eval_max_clique(population, instance, penalty)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_max_clique(population[i], instance, penalty)

        return fitness

    return objective
