"""
Unconstrained Binary Quadratic Programming (uBQP) Functions

This module provides functions for evaluating and loading uBQP problems,
including multi-objective variants.
"""

import numpy as np
from typing import List, Tuple, Optional


class UBQPInstance:
    """
    Represents an unconstrained Binary Quadratic Programming problem.

    A uBQP problem has the form:
    maximize sum_{(i,j)} w_{ij} * x_i * x_j

    For multi-objective problems, we have multiple objective functions,
    each with its own set of interaction weights.
    """

    def __init__(self, n_vars: int, n_objectives: int = 1):
        """
        Initialize a uBQP instance.

        Parameters
        ----------
        n_vars : int
            Number of binary variables
        n_objectives : int
            Number of objectives
        """
        self.n_vars = n_vars
        self.n_objectives = n_objectives
        # Each objective stores interactions as list of (i, j, weight) tuples
        self.objectives = [[] for _ in range(n_objectives)]

    def add_interaction(self, obj_idx: int, i: int, j: int, weight: float):
        """
        Add an interaction term to an objective.

        Parameters
        ----------
        obj_idx : int
            Objective index (0-based)
        i : int
            First variable index (1-based)
        j : int
            Second variable index (1-based)
        weight : float
            Interaction weight
        """
        self.objectives[obj_idx].append((i, j, weight))

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate a solution or population on all objectives.

        Parameters
        ----------
        solution : np.ndarray
            Binary vector or matrix (pop_size, n_vars)

        Returns
        -------
        np.ndarray
            Objective values (single value or vector for each solution)
        """
        if solution.ndim == 1:
            return self._evaluate_single(solution)
        else:
            return np.array([self._evaluate_single(sol) for sol in solution])

    def _evaluate_single(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate a single solution.

        Parameters
        ----------
        solution : np.ndarray
            Binary vector

        Returns
        -------
        np.ndarray
            Vector of objective values
        """
        results = np.zeros(self.n_objectives)

        for obj_idx, interactions in enumerate(self.objectives):
            value = 0.0
            for i, j, weight in interactions:
                # Convert from 1-based to 0-based indexing
                value += solution[i - 1] * solution[j - 1] * weight
            results[obj_idx] = value

        return results


def evaluate_ubqp(solution: np.ndarray, ubqp_instance: UBQPInstance) -> np.ndarray:
    """
    Evaluate a solution or population on a uBQP instance.

    This is a wrapper function that can be used as a fitness function.

    Parameters
    ----------
    solution : np.ndarray
        Binary vector or matrix (pop_size, n_vars)
    ubqp_instance : UBQPInstance
        The uBQP problem instance

    Returns
    -------
    np.ndarray
        Objective values
    """
    return ubqp_instance.evaluate(solution)


def load_ubqp_instance(filename: str, n_vars: int, n_objectives: int) -> UBQPInstance:
    """
    Load a uBQP instance from a file.

    Expected file format:
    - First line: seed (integer)
    - For each objective:
      - Line: objective_index n_edges
      - Following n_edges lines: i j weight

    Parameters
    ----------
    filename : str
        Path to the file
    n_vars : int
        Number of variables
    n_objectives : int
        Number of objectives

    Returns
    -------
    UBQPInstance
        The loaded uBQP instance
    """
    instance = UBQPInstance(n_vars, n_objectives)

    with open(filename, 'r') as f:
        # Read seed (not used)
        seed = int(f.readline().strip())

        for obj_idx in range(n_objectives):
            # Read objective header
            line = f.readline().strip().split()
            obj_id, n_edges = int(line[0]), int(line[1])

            # Read edges
            for _ in range(n_edges):
                parts = f.readline().strip().split()
                i, j, weight = int(parts[0]), int(parts[1]), float(parts[2])
                instance.add_interaction(obj_idx, i, j, weight)

    return instance


def generate_random_ubqp(
    n_vars: int,
    density: float = 0.1,
    n_objectives: int = 1,
    weight_range: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = None
) -> UBQPInstance:
    """
    Generate a random uBQP instance.

    Parameters
    ----------
    n_vars : int
        Number of variables
    density : float
        Density of non-zero interactions (0 to 1)
    n_objectives : int
        Number of objectives
    weight_range : tuple
        (min, max) range for interaction weights
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    UBQPInstance
        The generated uBQP instance
    """
    if seed is not None:
        np.random.seed(seed)

    instance = UBQPInstance(n_vars, n_objectives)

    # Total possible interactions (including diagonal)
    n_possible = n_vars * n_vars
    n_interactions = int(n_possible * density)

    for obj_idx in range(n_objectives):
        # Generate random interactions
        for _ in range(n_interactions):
            i = np.random.randint(1, n_vars + 1)
            j = np.random.randint(1, n_vars + 1)
            weight = np.random.uniform(weight_range[0], weight_range[1])
            instance.add_interaction(obj_idx, i, j, weight)

    return instance


def save_ubqp_instance(instance: UBQPInstance, filename: str, seed: int = 0):
    """
    Save a uBQP instance to a file.

    Parameters
    ----------
    instance : UBQPInstance
        The uBQP instance to save
    filename : str
        Path to the output file
    seed : int
        Seed value to write (for compatibility)
    """
    with open(filename, 'w') as f:
        # Write seed
        f.write(f"{seed}\n")

        # Write each objective
        for obj_idx, interactions in enumerate(instance.objectives):
            f.write(f"{obj_idx + 1} {len(interactions)}\n")
            for i, j, weight in interactions:
                f.write(f"{i} {j} {weight}\n")


def create_max_cut_ubqp(adjacency_matrix: np.ndarray) -> UBQPInstance:
    """
    Create a uBQP instance from a Max-Cut problem.

    In Max-Cut, we want to partition graph vertices to maximize
    the number of edges between partitions.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix of the graph (symmetric)

    Returns
    -------
    UBQPInstance
        uBQP instance representing the Max-Cut problem
    """
    n_vars = adjacency_matrix.shape[0]
    instance = UBQPInstance(n_vars, n_objectives=1)

    # Add interactions for each edge
    # Max-Cut: maximize sum of w_ij * (x_i * (1-x_j) + (1-x_i) * x_j)
    # This can be reformulated as a uBQP problem
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adjacency_matrix[i, j] != 0:
                weight = adjacency_matrix[i, j]
                # The reformulation leads to: w_ij - 2*w_ij*x_i*x_j
                # We only model the quadratic part here
                instance.add_interaction(0, i + 1, j + 1, -2 * weight)

    return instance


def create_set_packing_ubqp(sets: List[List[int]], weights: Optional[np.ndarray] = None) -> UBQPInstance:
    """
    Create a uBQP instance from a Set Packing problem.

    In Set Packing, we want to select a maximum weight collection
    of disjoint sets.

    Parameters
    ----------
    sets : list of lists
        Each inner list contains element indices
    weights : np.ndarray, optional
        Weight for each set (default: all 1.0)

    Returns
    -------
    UBQPInstance
        uBQP instance representing the Set Packing problem
    """
    n_sets = len(sets)
    instance = UBQPInstance(n_sets, n_objectives=1)

    if weights is None:
        weights = np.ones(n_sets)

    # Add linear terms (weights for selecting sets)
    for i in range(n_sets):
        instance.add_interaction(0, i + 1, i + 1, weights[i])

    # Add penalty terms for overlapping sets
    # Large negative weight for selecting both sets that share elements
    penalty = -2 * np.max(weights)
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            # Check if sets i and j overlap
            if set(sets[i]).intersection(set(sets[j])):
                instance.add_interaction(0, i + 1, j + 1, penalty)

    return instance
