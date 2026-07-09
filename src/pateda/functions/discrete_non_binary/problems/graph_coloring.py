"""
Graph Coloring problem for graph optimization benchmarking

Codification: Discrete vector (maximum k values), one variable for each node.
The value represents the color assigned to the node.
Goal: Minimize the number of colors used, subject to the constraint that adjacent nodes have different colors.
"""

import numpy as np
from pateda.functions.graph_utils import read_dimacs_graph


class GraphColoringInstance:
    """
    Represents an instance of the Graph Coloring problem.
    """
    def __init__(self, n_nodes: int, adj_matrix: np.ndarray):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix
        # Precompute unique undirected edges for fast violation checking
        u, v = np.where(self.adj_matrix)
        self.edges = [(int(ui), int(vi)) for ui, vi in zip(u, v) if ui < vi]

    @classmethod
    def from_file(cls, filepath: str):
        """Load instance from a DIMACS .col file."""
        n_nodes, adj_matrix = read_dimacs_graph(filepath)
        return cls(n_nodes, adj_matrix)


def eval_graph_coloring(c: np.ndarray, instance: GraphColoringInstance, penalty: float = None) -> float:
    """
    Evaluate the Graph Coloring objective function for an integer vector.

    Args:
        c: Integer vector of color assignments (0-indexed).
        instance: The GraphColoringInstance.
        penalty: Penalty coefficient for color conflicts. Defaults to n_nodes + 1.

    Returns:
        Fitness value (n_nodes - unique_colors - penalty * conflicts).
    """
    if c.ndim == 2:
        c = c.flatten()

    if len(c) != instance.n_nodes:
        raise ValueError(f"Solution length ({len(c)}) does not match graph size ({instance.n_nodes})")

    if penalty is None:
        # A conflict should be penalized heavily, more than any possible savings in colors
        penalty = float(instance.n_nodes + 1)

    # Number of unique colors used
    unique_colors = len(np.unique(c))

    # Number of edge conflicts (adjacent nodes with same color)
    conflicts = 0
    for u, v in instance.edges:
        if c[u] == c[v]:
            conflicts += 1

    # Maximize fitness
    fitness = instance.n_nodes - unique_colors - penalty * conflicts
    return float(fitness)


def create_graph_coloring_objective_function(instance: GraphColoringInstance, penalty: float = None):
    """
    Create a Graph Coloring objective function for use with EDAs.

    Args:
        instance: The GraphColoringInstance.
        penalty: Penalty coefficient.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([eval_graph_coloring(population, instance, penalty)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_graph_coloring(population[i], instance, penalty)

        return fitness

    return objective
