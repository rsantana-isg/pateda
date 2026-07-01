"""
Maximum Independent Set (MIS) for permutation-based EDAs
 
The MIS problem asks for the largest subset of vertices in an undirected graph
such that no two vertices in the subset are adjacent.
 
Under the permutation picture, the problem is reformulated as finding a
permutation of vertices π and a value k such that the first k vertices in π
form an independent set, maximizing k.
 
The permutation-based objective is:
    Maximize k  subject to  Tr(P A Pᵀ C(k)) = 0
 
where P is the permutation matrix of π, A is the adjacency matrix, and C(k)
is the k×k upper-left block of ones (truncation matrix).
 
References:
    [1] Y. Min: "Permutation Picture of Graph Combinatorial Optimization Problems"
        arXiv:2410.17111v1 [cs.AI], 2024. Section 4.2.
"""
 
import numpy as np
from typing import Optional
 
 
class MIS:
    """
    Maximum Independent Set Problem
 
    Given an undirected graph G = (V, E), find the largest subset S ⊆ V
    such that no two vertices in S are adjacent.
 
    Under the permutation picture, a permutation π of the vertices is
    evaluated by finding the largest prefix of π that forms an independent set.
    """
 
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize MIS with an adjacency matrix.
 
        Args:
            adjacency_matrix: Binary symmetric matrix of shape (n, n).
                              adjacency_matrix[i, j] = 1 if there is an edge
                              between vertices i and j, 0 otherwise.
 
        Raises:
            ValueError: If the matrix is not square or not symmetric.
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if not np.allclose(adjacency_matrix, adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetric (undirected graph)")
 
        self.adjacency_matrix = adjacency_matrix.astype(float)
        self.n = adjacency_matrix.shape[0]
 
    def __call__(self, permutation: np.ndarray) -> float:
        """
        Evaluate a permutation by finding the size of the independent set
        formed by the largest valid prefix.
 
        Args:
            permutation: A permutation of vertex indices (0-indexed or 1-indexed).
 
        Returns:
            k: Size of the independent set found (positive, higher is better).
        """
        perm = np.array(permutation, dtype=int)
 
        # Convert to 0-indexed if needed
        if np.min(perm) == 1:
            perm = perm - 1
 
        # Find the largest k such that perm[:k] is an independent set.
        # Equivalent to checking Tr(P A Pᵀ C(k)) == 0 for increasing k.
        independent_set = []
        for vertex in perm:
            # Check if vertex is adjacent to any already-selected vertex
            if all(self.adjacency_matrix[vertex, v] == 0 for v in independent_set):
                independent_set.append(vertex)
 
        return float(len(independent_set))
 
    def evaluate_independent_set(self, permutation: np.ndarray) -> list:
        """
        Return the actual independent set (list of vertices) found by the
        permutation.
 
        Args:
            permutation: A permutation of vertex indices.
 
        Returns:
            List of vertex indices forming the independent set.
        """
        perm = np.array(permutation, dtype=int)
        if np.min(perm) == 1:
            perm = perm - 1
 
        independent_set = []
        for vertex in perm:
            if all(self.adjacency_matrix[vertex, v] == 0 for v in independent_set):
                independent_set.append(vertex)
 
        return independent_set
 
    def is_valid_independent_set(self, vertices: list) -> bool:
        """
        Check whether a given set of vertices is a valid independent set.
 
        Args:
            vertices: List of vertex indices (0-indexed).
 
        Returns:
            True if no two vertices in the set are adjacent.
        """
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self.adjacency_matrix[vertices[i], vertices[j]] == 1:
                    return False
        return True
 
 
def create_random_mis(n: int, edge_probability: float = 0.3,
                      seed: Optional[int] = None) -> MIS:
    """
    Create a random MIS instance using an Erdős–Rényi random graph G(n, p).
 
    Args:
        n: Number of vertices.
        edge_probability: Probability of each edge existing (default 0.3).
        seed: Random seed for reproducibility.
 
    Returns:
        MIS instance.
 
    Example:
        >>> mis = create_random_mis(10, edge_probability=0.3, seed=42)
        >>> perm = np.arange(10)
        >>> fitness = mis(perm)
    """
    if seed is not None:
        np.random.seed(seed)
 
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
 
    return MIS(adjacency_matrix)
 
 
def create_mis_from_edges(n: int, edges: list) -> MIS:
    """
    Create a MIS instance from an explicit list of edges.
 
    Args:
        n: Number of vertices (0-indexed: 0 to n-1).
        edges: List of (i, j) tuples representing edges.
 
    Returns:
        MIS instance.
 
    Example:
        >>> mis = create_mis_from_edges(5, [(0,1), (1,2), (2,3), (3,4)])
        >>> perm = np.array([0, 2, 4, 1, 3])
        >>> fitness = mis(perm)  # should return 3.0
    """
    adjacency_matrix = np.zeros((n, n))
    for i, j in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
 
    return MIS(adjacency_matrix)
