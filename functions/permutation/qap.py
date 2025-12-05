"""
Quadratic Assignment Problem (QAP) for permutation-based EDAs

The QAP is a fundamental combinatorial optimization problem where the goal
is to assign a set of facilities to a set of locations in such a way that
minimizes the total cost, which is the sum of the products of flow and distance
between facilities.

The QAP is NP-hard and has applications in facility location, keyboard design,
hospital layout, and scheduling.

References:
    [1] T. C. Koopmans, M. J. Beckmann: Assignment Problems and the Location
        of Economic Activities. Econometrica, 1957
    [2] E. Cela: The Quadratic Assignment Problem: Theory and Algorithms.
        Kluwer Academic Publishers, 1998
    [3] QAPLIB: http://coral.ise.lehigh.edu/data-sets/qaplib/
"""

import numpy as np
from typing import Optional, Tuple


class QAP:
    """
    Quadratic Assignment Problem

    Given two n x n matrices:
    - Flow matrix F: F[i,j] = flow between facilities i and j
    - Distance matrix D: D[i,j] = distance between locations i and j

    The objective is to find a permutation π that minimizes:
        sum_{i,j} F[i,j] * D[π(i), π(j)]
    """

    def __init__(
        self,
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        minimize: bool = True
    ):
        """
        Initialize QAP with flow and distance matrices.

        Args:
            flow_matrix: Matrix of flows between facilities (n x n)
            distance_matrix: Matrix of distances between locations (n x n)
            minimize: If True, minimize the cost (default); if False, maximize

        Raises:
            ValueError: If matrices are not square or have different sizes
        """
        if flow_matrix.shape[0] != flow_matrix.shape[1]:
            raise ValueError("Flow matrix must be square")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        if flow_matrix.shape[0] != distance_matrix.shape[0]:
            raise ValueError("Flow and distance matrices must have the same size")

        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.n = flow_matrix.shape[0]
        self.minimize = minimize

    def __call__(self, permutation: np.ndarray) -> float:
        """
        Evaluate a permutation (assignment of facilities to locations).

        Args:
            permutation: A permutation of facilities (0-indexed or 1-indexed)

        Returns:
            Fitness value (negative cost for minimization, positive for maximization)
        """
        perm = np.array(permutation, dtype=int)

        # Convert to 0-indexed if needed
        if np.min(perm) == 1:
            perm = perm - 1

        # Calculate total cost
        cost = 0.0
        for i in range(self.n):
            for j in range(self.n):
                # Facility i is assigned to location perm[i]
                # Facility j is assigned to location perm[j]
                # Cost is flow[i,j] * distance[perm[i], perm[j]]
                cost += self.flow_matrix[i, j] * self.distance_matrix[perm[i], perm[j]]

        # Return negative cost for minimization (EDAs maximize)
        return -cost if self.minimize else cost

    def evaluate_cost(self, permutation: np.ndarray) -> float:
        """
        Evaluate the actual cost of a permutation (positive value).

        Args:
            permutation: A permutation of facilities

        Returns:
            Total cost of the assignment
        """
        fitness = self.__call__(permutation)
        return -fitness if self.minimize else fitness


def create_random_qap(n: int, seed: Optional[int] = None, sparse: bool = False) -> QAP:
    """
    Create a random QAP instance.

    Args:
        n: Size of the problem (number of facilities/locations)
        seed: Random seed for reproducibility
        sparse: If True, create sparse matrices (many zero flows)

    Returns:
        QAP instance with random flow and distance matrices

    Example:
        >>> qap = create_random_qap(10, seed=42)
        >>> perm = np.arange(10)
        >>> fitness = qap(perm)
    """
    if seed is not None:
        np.random.seed(seed)

    # Create flow matrix
    if sparse:
        # Sparse flow matrix (many zeros)
        flow_matrix = np.zeros((n, n))
        num_flows = n * (n - 1) // 4  # About 25% filled
        for _ in range(num_flows):
            i, j = np.random.randint(0, n, size=2)
            if i != j:
                flow_matrix[i, j] = np.random.randint(1, 101)
    else:
        # Dense flow matrix
        flow_matrix = np.random.randint(0, 101, size=(n, n))
        np.fill_diagonal(flow_matrix, 0)  # No self-flow

    # Make flow matrix symmetric (optional, but common in QAP)
    flow_matrix = (flow_matrix + flow_matrix.T) / 2

    # Create distance matrix (Euclidean distances)
    locations = np.random.rand(n, 2) * 100
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(locations[i] - locations[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return QAP(flow_matrix, distance_matrix)


def create_uniform_qap(n: int, seed: Optional[int] = None) -> QAP:
    """
    Create a QAP instance with uniformly distributed values.

    This is a standard test problem type from the literature.

    Args:
        n: Size of the problem
        seed: Random seed for reproducibility

    Returns:
        QAP instance with uniform random matrices
    """
    if seed is not None:
        np.random.seed(seed)

    # Uniform random flow and distance matrices
    flow_matrix = np.random.uniform(0, 100, size=(n, n))
    distance_matrix = np.random.uniform(0, 100, size=(n, n))

    # Zero diagonal
    np.fill_diagonal(flow_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)

    # Make symmetric
    flow_matrix = (flow_matrix + flow_matrix.T) / 2
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    return QAP(flow_matrix, distance_matrix)


def create_grid_qap(grid_size: int, seed: Optional[int] = None) -> QAP:
    """
    Create a QAP instance based on a grid layout.

    Locations are arranged in a grid pattern, and distances are Manhattan distances.
    This is useful for facility layout problems.

    Args:
        grid_size: Size of the grid (e.g., 3 for 3x3=9 locations)
        seed: Random seed for flow matrix generation

    Returns:
        QAP instance with grid-based distances

    Example:
        >>> qap = create_grid_qap(3, seed=42)  # 3x3 grid = 9 facilities
    """
    if seed is not None:
        np.random.seed(seed)

    n = grid_size * grid_size

    # Create grid coordinates
    locations = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

    # Calculate Manhattan distances
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.abs(locations[i] - locations[j]).sum()

    # Random flow matrix
    flow_matrix = np.random.randint(0, 101, size=(n, n))
    np.fill_diagonal(flow_matrix, 0)
    flow_matrix = (flow_matrix + flow_matrix.T) / 2

    return QAP(flow_matrix, distance_matrix)


def load_qaplib_instance(flow_matrix: np.ndarray, distance_matrix: np.ndarray) -> QAP:
    """
    Load a QAP instance in QAPLIB format.

    QAPLIB is a library of QAP instances available at:
    http://coral.ise.lehigh.edu/data-sets/qaplib/

    Args:
        flow_matrix: Flow matrix from QAPLIB file
        distance_matrix: Distance matrix from QAPLIB file

    Returns:
        QAP instance

    Example:
        >>> # After loading matrices from file
        >>> qap = load_qaplib_instance(flow_mat, dist_mat)
    """
    return QAP(flow_matrix, distance_matrix)
