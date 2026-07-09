"""
Dominating Set problem for graph optimization benchmarking

Codification: Binary vector, one variable for each node.
1 means the node is in the dominating set, 0 otherwise.
Goal: Minimize dominating set size subject to the constraint that every node is dominated.
"""

import numpy as np
from pateda.functions.graph_utils import read_dimacs_graph


class DominatingSetInstance:
    """
    Represents an instance of the Dominating Set problem.
    """
    def __init__(self, n_nodes: int, adj_matrix: np.ndarray):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix

    @classmethod
    def from_file(cls, filepath: str):
        """Load instance from a DIMACS file."""
        n_nodes, adj_matrix = read_dimacs_graph(filepath)
        return cls(n_nodes, adj_matrix)


def eval_dominating_set(x: np.ndarray, instance: DominatingSetInstance, penalty: float = None) -> float:
    """
    Evaluate the Dominating Set objective function for a binary vector.

    Args:
        x: Binary vector representing node selection.
        instance: The DominatingSetInstance.
        penalty: Penalty coefficient for undominated nodes. Defaults to n_nodes + 1.

    Returns:
        Fitness value (n_nodes - set_size - penalty * undominated_nodes).
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) != instance.n_nodes:
        raise ValueError(f"Solution length ({len(x)}) does not match graph size ({instance.n_nodes})")

    if penalty is None:
        penalty = float(instance.n_nodes + 1)

    set_size = np.sum(x)

    # A node is dominated if it is in the dominating set or has an active neighbor
    # We can compute this using matrix-vector multiplication
    dominated = (x == 1) | (instance.adj_matrix @ x > 0)
    undominated_count = instance.n_nodes - np.sum(dominated)

    # Maximize fitness: start with n_nodes, subtract set size and penalty for violations
    fitness = instance.n_nodes - set_size - penalty * undominated_count
    return float(fitness)


def create_dominating_set_objective_function(instance: DominatingSetInstance, penalty: float = None):
    """
    Create a Dominating Set objective function for use with EDAs.

    Args:
        instance: The DominatingSetInstance.
        penalty: Penalty coefficient.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([eval_dominating_set(population, instance, penalty)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_dominating_set(population[i], instance, penalty)

        return fitness

    return objective
