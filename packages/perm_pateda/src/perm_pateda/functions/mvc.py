"""
Minimum Vertex Cover (MVC) for permutation-based EDAs

The MVC problem asks for the smallest subset S of vertices in an undirected
graph G = (V, E) such that every edge in E has at least one endpoint in S.

Under the permutation picture, the problem is reformulated as finding a
permutation of vertices π and a value k such that the first k vertices in π
form a vertex cover, minimizing k:

    Minimize k  subject to  Tr(P A Pᵀ C(k)) = 0

where P is the permutation matrix of π, A is the adjacency matrix, and C(k)
is defined as C(k)_ij = 1 if i > k and j > k, else 0.

The constraint Tr(P A Pᵀ C(k)) = 0 ensures there are no edges among the
last n-k vertices in π, meaning every edge has at least one endpoint in the
first k vertices (the vertex cover).

Since EDAs in pateda maximise fitness, we return -k so that minimising k
corresponds to maximising the fitness value.

References:
    [1] Y. Min: "Permutation Picture of Graph Combinatorial Optimization Problems"
        arXiv:2410.17111v1 [cs.AI], 2024. Section 4.5.
"""

import numpy as np
from typing import Optional, Tuple


class MVC:
    """
    Minimum Vertex Cover Problem

    Given an undirected graph G = (V, E), find the smallest subset S ⊆ V
    such that every edge (i, j) ∈ E has at least one endpoint in S.

    Under the permutation picture, a permutation π is evaluated by finding
    the smallest prefix of length k such that every edge has at least one
    endpoint among {π(0), ..., π(k-1)}.

    Fitness returned is -k so that EDAs (which maximise) are driven towards
    smaller vertex covers.
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize MVC with an adjacency matrix.

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
        Evaluate a permutation by finding the smallest valid vertex cover prefix.

        Given π, finds the smallest k such that {π(0), ..., π(k-1)} covers
        all edges. Equivalent to finding the smallest k such that no edge
        exists entirely within {π(k), ..., π(n-1)}, i.e. Tr(P A Pᵀ C(k)) = 0.

        Args:
            permutation: A permutation of vertex indices (0-indexed or 1-indexed).

        Returns:
            -k where k is the size of the smallest vertex cover prefix found.
            Higher (less negative) fitness = smaller cover = better solution.
        """
        perm = np.array(permutation, dtype=int)

        if np.min(perm) == 1:
            perm = perm - 1

        # Reorder adjacency matrix according to permutation
        A_perm = self.adjacency_matrix[np.ix_(perm, perm)]

        # Find smallest k such that A_perm[k:, k:] has no edges (all zeros)
        # i.e. Tr(P A Pᵀ C(k)) = 0 where C(k)_ij = 1 if i > k and j > k
        for k in range(self.n):
            # Check if the subgraph induced by vertices k..n-1 has no edges
            if A_perm[k:, k:].sum() == 0:
                return float(-k)

        # Fallback: full set covers everything (always valid)
        return float(-self.n)

    def evaluate_vertex_cover(self, permutation: np.ndarray) -> Tuple[float, list]:
        """
        Return the vertex cover found by the permutation, with full details.

        Args:
            permutation: A permutation of vertex indices.

        Returns:
            Tuple of (cover_size, cover_vertices) where:
                cover_size:    Number of vertices in the cover.
                cover_vertices: List of vertex indices forming the cover.
        """
        perm = np.array(permutation, dtype=int)
        if np.min(perm) == 1:
            perm = perm - 1

        A_perm = self.adjacency_matrix[np.ix_(perm, perm)]

        for k in range(self.n):
            if A_perm[k:, k:].sum() == 0:
                return k, perm[:k].tolist()

        return self.n, perm.tolist()

    def is_valid_vertex_cover(self, vertices: list) -> bool:
        """
        Check whether a given set of vertices is a valid vertex cover.

        Args:
            vertices: List of vertex indices (0-indexed).

        Returns:
            True if every edge has at least one endpoint in the set.
        """
        vertex_set = set(vertices)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adjacency_matrix[i, j] == 1:
                    if i not in vertex_set and j not in vertex_set:
                        return False
        return True


def create_random_mvc(n: int, edge_probability: float = 0.4,
                      seed: Optional[int] = None) -> "MVC":
    """
    Create a random MVC instance using an Erdős–Rényi random graph G(n, p).

    Args:
        n: Number of vertices.
        edge_probability: Probability of each edge existing (default 0.4).
        seed: Random seed for reproducibility.

    Returns:
        MVC instance.

    Example:
        >>> mvc = create_random_mvc(10, edge_probability=0.4, seed=42)
        >>> perm = np.arange(10)
        >>> fitness = mvc(perm)
    """
    if seed is not None:
        np.random.seed(seed)

    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return MVC(adjacency_matrix)


def create_mvc_from_edges(n: int, edges: list) -> "MVC":
    """
    Create a MVC instance from an explicit list of edges.

    Args:
        n: Number of vertices (0-indexed: 0 to n-1).
        edges: List of (i, j) tuples representing edges.

    Returns:
        MVC instance.

    Example:
        >>> mvc = create_mvc_from_edges(4, [(0,1), (1,2), (2,3), (3,0)])
        >>> perm = np.array([0, 2, 1, 3])
        >>> fitness = mvc(perm)  # cover of size 2: vertices 0 and 2
    """
    adjacency_matrix = np.zeros((n, n))
    for i, j in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    return MVC(adjacency_matrix)