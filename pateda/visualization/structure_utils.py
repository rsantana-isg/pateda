"""
Utility functions for manipulating and extracting graph structures.

This module provides helper functions for working with learned structures
from EDAs, including conversion between representations and extraction of
substructures.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def convert_from_index(
    index_matrix: np.ndarray,
    edge_vector: np.ndarray
) -> np.ndarray:
    """
    Convert a vector of edge indices to an adjacency matrix.

    This function transforms a compact edge representation (using indices)
    back into a full adjacency matrix representation.

    Parameters
    ----------
    index_matrix : np.ndarray
        Matrix where cell (i, j) contains the index of the edge between
        variables i and j. Shape: (n_variables, n_variables).
    edge_vector : np.ndarray
        Vector where position i indicates the presence (>0) or absence (0)
        of the edge with index i.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (n_variables, n_variables) where non-zero
        values indicate edges between variables.

    Examples
    --------
    >>> import numpy as np
    >>> n = 4
    >>> index_matrix = np.arange(n*n).reshape(n, n)
    >>> edge_vector = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    >>> adj_matrix = convert_from_index(index_matrix, edge_vector)

    Notes
    -----
    - Only edges present in edge_vector (value > 0) are added to the matrix
    - The value from edge_vector is transferred to the adjacency matrix
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    n = index_matrix.shape[0]

    # Find indices of present edges
    present_edges = np.where(edge_vector > 0)[0]

    # Initialize empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Fill in the edges
    for edge_idx in present_edges:
        # Find position in index_matrix that contains this edge index
        positions = np.where(index_matrix == edge_idx)

        if len(positions[0]) > 0:
            row, col = positions[0][0], positions[1][0]
            adjacency_matrix[row, col] = edge_vector[edge_idx]

    return adjacency_matrix


def select_edges_to_show(
    run_structures: Dict,
    min_occurrences: int
) -> np.ndarray:
    """
    Select edges that appear frequently across learned structures.

    From all structures learned across all runs and generations, extract
    the indices of edges that were learned at least a specified number of times.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data with the following keys:
        - 'all_sum_matrices': Matrix where each element counts edge occurrences
        - Other structure-related data
    min_occurrences : int
        Minimum number of times an edge must appear to be selected.

    Returns
    -------
    np.ndarray
        Indices of edges that appear at least min_occurrences times.

    Examples
    --------
    >>> run_structures = {'all_sum_matrices': np.random.randint(0, 10, (100,))}
    >>> selected = select_edges_to_show(run_structures, min_occurrences=5)
    >>> # selected contains indices of edges appearing >= 5 times

    Notes
    -----
    - Useful for filtering out infrequent edges before visualization
    - Helps reduce clutter in complex structure visualizations
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    all_sum_matrices = run_structures['all_sum_matrices']

    # Sum across all structures and find edges exceeding threshold
    total_occurrences = np.sum(all_sum_matrices, axis=1) if all_sum_matrices.ndim > 1 else all_sum_matrices

    selected_edges = np.where(total_occurrences > min_occurrences)[0]

    return selected_edges


def extract_substructures(
    run_structures: Dict,
    edge_indices: np.ndarray,
    min_edges: int = 0
) -> np.ndarray:
    """
    Extract substructures containing specified edges from learned structures.

    From all structures learned across runs and generations, extract those
    substructures that contain at least a minimum number of the specified edges.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data with keys:
        - 'all_big_matrices': List of matrices for each run, where each matrix
          has edges as rows and generations as columns
    edge_indices : np.ndarray
        Indices of edges to extract.
    min_edges : int, default=0
        Minimum number of edges (from edge_indices) that must be present
        in a structure for it to be extracted.

    Returns
    -------
    np.ndarray
        Matrix where each row is a selected substructure. Each element contains
        the generation number multiplied by the edge presence (0 if absent).
        Shape: (n_selected_structures, n_edge_indices).

    Examples
    --------
    >>> run_structures = {
    ...     'all_big_matrices': [np.random.randint(0, 2, (50, 10))]
    ... }
    >>> edge_indices = np.array([0, 5, 10, 15, 20])
    >>> substructures = extract_substructures(run_structures, edge_indices, min_edges=2)

    Notes
    -----
    - Each row in the output represents one structure from one generation
    - The generation number is encoded in the values (generation * presence)
    - Only structures with more than min_edges present edges are included
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    all_big_matrices = run_structures['all_big_matrices']

    n_runs = len(all_big_matrices)
    max_gen = all_big_matrices[0].shape[1] if n_runs > 0 else 0

    all_rep_vectors = []

    for run_idx in range(n_runs):
        run_data = all_big_matrices[run_idx]

        for gen_idx in range(max_gen):
            # Extract edges for this generation
            edge_vector = run_data[edge_indices, gen_idx]

            # Check if enough edges are present
            if np.sum(edge_vector) > min_edges:
                # Multiply by generation number (1-indexed)
                gen_vector = (gen_idx + 1) * edge_vector
                all_rep_vectors.append(gen_vector)

    if len(all_rep_vectors) > 0:
        return np.array(all_rep_vectors)
    else:
        return np.array([]).reshape(0, len(edge_indices))


def find_index_matrix(n_variables: int) -> np.ndarray:
    """
    Create an index matrix for edges in a graph.

    Creates a matrix where each cell (i, j) contains a unique index for
    the edge between variables i and j. Useful for compact edge representation.

    Parameters
    ----------
    n_variables : int
        Number of variables/nodes in the graph.

    Returns
    -------
    np.ndarray
        Index matrix of shape (n_variables, n_variables) where each unique
        edge has a unique index.

    Examples
    --------
    >>> index_matrix = find_index_matrix(5)
    >>> # index_matrix[i, j] gives the index for edge (i, j)

    Notes
    -----
    - Indices are assigned sequentially
    - For undirected graphs, index_matrix[i, j] == index_matrix[j, i]
    - For directed graphs, they may differ
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    # Create a matrix with unique indices for each edge
    # Using upper triangular indexing for undirected graphs
    index_matrix = np.zeros((n_variables, n_variables), dtype=int)

    idx = 0
    for i in range(n_variables):
        for j in range(n_variables):
            if i != j:  # Exclude self-loops
                index_matrix[i, j] = idx
                idx += 1

    return index_matrix
