"""
Clique Covering problem for graph optimization benchmarking

Codification: Discrete vector (maximum k values), one variable for each node.
The value represents the clique index assigned to the node.
Goal: Minimize the number of cliques used to cover all nodes, subject to the constraint
that all nodes in the same clique partition must be adjacent in the graph.
"""

import numpy as np
from pateda.functions.graph_utils import read_dimacs_graph


class CliqueCoveringInstance:
    """
    Represents an instance of the Clique Covering problem.
    """
    def __init__(self, n_nodes: int, adj_matrix: np.ndarray):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix

    @classmethod
    def from_file(cls, filepath: str):
        """Load instance from a DIMACS .clq file."""
        n_nodes, adj_matrix = read_dimacs_graph(filepath)
        return cls(n_nodes, adj_matrix)


def eval_clique_covering(y: np.ndarray, instance: CliqueCoveringInstance, penalty: float = None) -> float:
    """
    Evaluate the Clique Covering objective function for an integer vector.

    Args:
        y: Integer vector of clique partition assignments (0-indexed).
        instance: The CliqueCoveringInstance.
        penalty: Penalty coefficient for non-adjacent nodes in the same partition.
                 Defaults to n_nodes + 1.

    Returns:
        Fitness value (n_nodes - unique_cliques - penalty * violations).
    """
    if y.ndim == 2:
        y = y.flatten()

    if len(y) != instance.n_nodes:
        raise ValueError(f"Solution length ({len(y)}) does not match graph size ({instance.n_nodes})")

    if penalty is None:
        penalty = float(instance.n_nodes + 1)

    # Number of unique cliques used
    unique_cliques = len(np.unique(y))

    # Calculate violations: count pairs of nodes with the same clique index that are NOT adjacent
    violations = 0
    for clique_idx in np.unique(y):
        members = np.where(y == clique_idx)[0]
        k = len(members)
        if k <= 1:
            continue

        # Count connections within the members of this clique index
        sub_adj = instance.adj_matrix[members][:, members]
        connected_pairs = np.sum(sub_adj) // 2
        total_pairs = k * (k - 1) // 2
        violations += (total_pairs - connected_pairs)

    # Maximize fitness
    fitness = instance.n_nodes - unique_cliques - penalty * violations
    return float(fitness)


def create_clique_covering_objective_function(instance: CliqueCoveringInstance, penalty: float = None):
    """
    Create a Clique Covering objective function for use with EDAs.

    Args:
        instance: The CliqueCoveringInstance.
        penalty: Penalty coefficient.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([eval_clique_covering(population, instance, penalty)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_clique_covering(population[i], instance, penalty)

        return fitness

    return objective
