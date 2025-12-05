"""
Linear Ordering Problem (LOP) for permutation-based EDAs

The LOP is a combinatorial optimization problem where, given a matrix of weights,
the goal is to find a permutation that maximizes (or minimizes) the sum of weights
above the diagonal.

The problem has applications in:
- Ranking and ordering
- Input-output economics
- Archaeology (seriation problem)
- Graph theory (feedback arc set problem)
- Machine learning (ranking SVMs)

References:
    [1] M. Grötschel, M. Jünger, G. Reinelt: A cutting plane algorithm for
        the linear ordering problem. Operations Research, 1984
    [2] R. Martí, G. Reinelt: The Linear Ordering Problem: Exact and Heuristic
        Methods in Combinatorial Optimization. Springer, 2011
    [3] LOLIB: http://grafo.etsii.urjc.es/optsicom/lolib/
"""

import numpy as np
from typing import Optional


class LOP:
    """
    Linear Ordering Problem

    Given an n x n matrix B of weights, find a permutation π that maximizes:
        sum_{i<j} B[π(i), π(j)]

    Equivalently, this maximizes the sum of weights where i appears before j
    in the permutation.
    """

    def __init__(self, weight_matrix: np.ndarray, maximize: bool = True):
        """
        Initialize LOP with a weight matrix.

        Args:
            weight_matrix: Matrix of weights (n x n)
                          B[i,j] = benefit of placing i before j
            maximize: If True, maximize the objective (default);
                     if False, minimize

        Raises:
            ValueError: If matrix is not square
        """
        if weight_matrix.shape[0] != weight_matrix.shape[1]:
            raise ValueError("Weight matrix must be square")

        self.weight_matrix = weight_matrix
        self.n = weight_matrix.shape[0]
        self.maximize = maximize

    def __call__(self, permutation: np.ndarray) -> float:
        """
        Evaluate a permutation.

        Args:
            permutation: A permutation (0-indexed or 1-indexed)

        Returns:
            Fitness value (sum of weights for maximize=True)
        """
        perm = np.array(permutation, dtype=int)

        # Convert to 0-indexed if needed
        if np.min(perm) == 1:
            perm = perm - 1

        # Calculate objective: sum of weights B[π(i), π(j)] for i < j
        objective = 0.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Element perm[i] appears before element perm[j]
                # Add weight B[perm[i], perm[j]]
                objective += self.weight_matrix[perm[i], perm[j]]

        return objective if self.maximize else -objective

    def evaluate_objective(self, permutation: np.ndarray) -> float:
        """
        Evaluate the actual objective value (for maximize=True).

        Args:
            permutation: A permutation

        Returns:
            Objective value
        """
        fitness = self.__call__(permutation)
        return fitness if self.maximize else -fitness


def create_random_lop(n: int, seed: Optional[int] = None, symmetric: bool = False) -> LOP:
    """
    Create a random LOP instance.

    Args:
        n: Size of the problem
        seed: Random seed for reproducibility
        symmetric: If True, create a symmetric weight matrix (tournament problem)

    Returns:
        LOP instance with random weights

    Example:
        >>> lop = create_random_lop(10, seed=42)
        >>> perm = np.arange(10)
        >>> fitness = lop(perm)
    """
    if seed is not None:
        np.random.seed(seed)

    # Create random weight matrix
    weight_matrix = np.random.randint(0, 101, size=(n, n))

    if symmetric:
        # For tournament problems: B[i,j] + B[j,i] = constant
        # This represents wins/losses in round-robin tournaments
        for i in range(n):
            for j in range(i + 1, n):
                total = 100  # Total points per match
                weight_matrix[i, j] = np.random.randint(0, total + 1)
                weight_matrix[j, i] = total - weight_matrix[i, j]

    # Zero diagonal (no self-ordering)
    np.fill_diagonal(weight_matrix, 0)

    return LOP(weight_matrix)


def create_tournament_lop(n: int, seed: Optional[int] = None) -> LOP:
    """
    Create a LOP instance representing a round-robin tournament.

    In a tournament, each pair of teams plays once, and B[i,j] represents
    the points team i scored against team j.

    Args:
        n: Number of teams
        seed: Random seed for reproducibility

    Returns:
        LOP instance representing a tournament

    Example:
        >>> lop = create_tournament_lop(8, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    weight_matrix = np.zeros((n, n))

    # Simulate tournament results
    for i in range(n):
        for j in range(i + 1, n):
            # Simulate a match between teams i and j
            # Winner gets more points
            if np.random.rand() < 0.5:
                # Team i wins
                weight_matrix[i, j] = np.random.randint(51, 101)
                weight_matrix[j, i] = 100 - weight_matrix[i, j]
            else:
                # Team j wins
                weight_matrix[j, i] = np.random.randint(51, 101)
                weight_matrix[i, j] = 100 - weight_matrix[j, i]

    return LOP(weight_matrix)


def create_triangular_lop(n: int, seed: Optional[int] = None) -> LOP:
    """
    Create a LOP instance with triangular structure.

    This type of problem has a known optimal or near-optimal solution
    based on the triangular inequality structure.

    Args:
        n: Size of the problem
        seed: Random seed

    Returns:
        LOP instance with triangular structure
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base values for each item
    base_values = np.random.rand(n) * 100

    # Weight matrix based on differences in base values
    weight_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Higher weight if base_values[i] > base_values[j]
                diff = base_values[i] - base_values[j]
                weight_matrix[i, j] = 50 + diff + np.random.randn() * 5

    # Ensure non-negative weights
    weight_matrix = np.maximum(weight_matrix, 0)
    np.fill_diagonal(weight_matrix, 0)

    return LOP(weight_matrix)


def create_sparse_lop(n: int, density: float = 0.3, seed: Optional[int] = None) -> LOP:
    """
    Create a sparse LOP instance.

    Args:
        n: Size of the problem
        density: Proportion of non-zero elements (0 to 1)
        seed: Random seed

    Returns:
        Sparse LOP instance

    Example:
        >>> lop = create_sparse_lop(20, density=0.2, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    weight_matrix = np.zeros((n, n))

    # Number of non-zero elements
    num_nonzero = int(n * n * density)

    # Randomly place non-zero weights
    for _ in range(num_nonzero):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            weight_matrix[i, j] = np.random.randint(1, 101)

    return LOP(weight_matrix)


def load_lolib_instance(weight_matrix: np.ndarray) -> LOP:
    """
    Load a LOP instance in LOLIB format.

    LOLIB is a library of LOP instances available at:
    http://grafo.etsii.urjc.es/optsicom/lolib/

    Args:
        weight_matrix: Weight matrix from LOLIB file

    Returns:
        LOP instance

    Example:
        >>> # After loading matrix from file
        >>> lop = load_lolib_instance(weight_mat)
    """
    return LOP(weight_matrix)


def feedback_arc_set_to_lop(adjacency_matrix: np.ndarray) -> LOP:
    """
    Convert a Feedback Arc Set problem to a LOP.

    The Feedback Arc Set problem asks for the minimum number of arcs
    to remove from a directed graph to make it acyclic. This is equivalent
    to a LOP where the weight matrix is the adjacency matrix.

    Args:
        adjacency_matrix: Adjacency matrix of directed graph
                         A[i,j] = 1 if there's an arc from i to j

    Returns:
        LOP instance (minimize to find minimum feedback arc set)
    """
    return LOP(adjacency_matrix, maximize=False)
