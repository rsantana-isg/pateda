"""
Maximum Cut (Max Cut) for permutation-based EDAs

The Max Cut problem asks for a partition of the vertices of an undirected graph
G = (V, E) into two disjoint subsets S and V\S such that the number of edges
between the two subsets (the cut size) is maximized.

Under the permutation picture, the problem is reformulated as finding a
permutation of vertices π and a cut point k such that the edges crossing
between {π(0), ..., π(k-1)} and {π(k), ..., π(n-1)} are maximized:

    Maximize  (1/2) * Tr(P A Pᵀ C(k))

where P is the permutation matrix of π, A is the adjacency matrix, and C(k)
is defined as C(k)_ij = 1 if (i <= k < j) or (j <= k < i), else 0.

References:
    [1] Y. Min: "Permutation Picture of Graph Combinatorial Optimization Problems"
        arXiv:2410.17111v1 [cs.AI], 2024. Section 4.3.
"""

import numpy as np
from typing import Optional, Tuple


class MaxCut:
    """
    Maximum Cut Problem

    Given an undirected graph G = (V, E), find a partition (S, V\S) of the
    vertices that maximizes the number of edges crossing between S and V\S.

    Under the permutation picture, a permutation π and a cut point k define
    the partition S = {π(0), ..., π(k-1)} and V\S = {π(k), ..., π(n-1)}.
    The best (π, k) pair is found by evaluating all n-1 cut points for each
    permutation and returning the maximum cut size found.
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize MaxCut with an adjacency matrix.

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
        Evaluate a permutation by finding the best cut point k.

        For a given permutation π, evaluates all n-1 possible cut points and
        returns the maximum cut size found. This corresponds to:

            max_k  (1/2) * Tr(P A Pᵀ C(k))

        Args:
            permutation: A permutation of vertex indices (0-indexed or 1-indexed).

        Returns:
            Maximum cut size across all cut points k in {1, ..., n-1}.
            Higher is better (EDAs in pateda maximise fitness).
        """
        perm = np.array(permutation, dtype=int)

        if np.min(perm) == 1:
            perm = perm - 1

        # Reorder adjacency matrix according to permutation:
        # (P A Pᵀ)_ij = A[perm[i], perm[j]]
        A_perm = self.adjacency_matrix[np.ix_(perm, perm)]

        best_cut = 0.0
        for k in range(1, self.n):
            # Count edges between {0..k-1} and {k..n-1} in permuted graph
            # = sum of A_perm[i, j] for i < k <= j  (each edge counted twice)
            cut_size = A_perm[:k, k:].sum()  # already counts each edge once
            if cut_size > best_cut:
                best_cut = cut_size

        return best_cut

    def evaluate_cut(self, permutation: np.ndarray) -> Tuple[float, int, list, list]:
        """
        Return the best cut found by the permutation, with full details.

        Args:
            permutation: A permutation of vertex indices.

        Returns:
            Tuple of (cut_size, best_k, set_S, set_T) where:
                cut_size: Number of edges in the best cut.
                best_k:   Cut point that achieves the best cut.
                set_S:    Vertices in S = {π(0), ..., π(best_k - 1)}.
                set_T:    Vertices in T = {π(best_k), ..., π(n-1)}.
        """
        perm = np.array(permutation, dtype=int)
        if np.min(perm) == 1:
            perm = perm - 1

        A_perm = self.adjacency_matrix[np.ix_(perm, perm)]

        best_cut = 0.0
        best_k = 1
        for k in range(1, self.n):
            cut_size = A_perm[:k, k:].sum()
            if cut_size > best_cut:
                best_cut = cut_size
                best_k = k

        set_S = perm[:best_k].tolist()
        set_T = perm[best_k:].tolist()
        return best_cut, best_k, set_S, set_T

    def evaluate_partition(self, set_S: list) -> float:
        """
        Evaluate the cut size for an explicit partition S, V\S.

        Args:
            set_S: List of vertex indices (0-indexed) in subset S.

        Returns:
            Number of edges between S and V\S.
        """
        set_T = [v for v in range(self.n) if v not in set_S]
        cut_size = 0.0
        for i in set_S:
            for j in set_T:
                cut_size += self.adjacency_matrix[i, j]
        return cut_size


def create_random_max_cut(n: int, edge_probability: float = 0.5,
                          seed: Optional[int] = None) -> MaxCut:
    """
    Create a random Max Cut instance using an Erdős–Rényi random graph G(n, p).

    Args:
        n: Number of vertices.
        edge_probability: Probability of each edge existing (default 0.5).
        seed: Random seed for reproducibility.

    Returns:
        MaxCut instance.

    Example:
        >>> mc = create_random_max_cut(10, edge_probability=0.5, seed=42)
        >>> perm = np.arange(10)
        >>> fitness = mc(perm)
    """
    if seed is not None:
        np.random.seed(seed)

    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return MaxCut(adjacency_matrix)


def create_max_cut_from_edges(n: int, edges: list) -> MaxCut:
    """
    Create a Max Cut instance from an explicit list of edges.

    Args:
        n: Number of vertices (0-indexed: 0 to n-1).
        edges: List of (i, j) tuples representing edges.

    Returns:
        MaxCut instance.

    Example:
        >>> mc = create_max_cut_from_edges(4, [(0,1), (1,2), (2,3), (3,0)])
        >>> perm = np.array([0, 2, 1, 3])
        >>> fitness = mc(perm)  # should return 4.0 (all edges in cut)
    """
    adjacency_matrix = np.zeros((n, n))
    for i, j in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    return MaxCut(adjacency_matrix)