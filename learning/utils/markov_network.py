"""
Markov Network structure learning utilities

Implements graph construction, clique finding, and neighborhood detection
for MN-FDA and MOA algorithms.

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/FDA.cpp, cpp_EDAs/cliques.cpp
"""

from typing import List, Optional, Tuple
import numpy as np
import networkx as nx


def build_dependency_graph_threshold(
    mi_matrix: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Build dependency graph using threshold on mutual information

    Creates adjacency matrix where Matrix[i,j] = 1 if MI(i,j) > threshold

    Reference: C++ FDA.cpp:1635-1672 (LearnMatrix)

    Args:
        mi_matrix: Mutual information matrix (n_vars, n_vars)
        threshold: MI threshold for creating edge

    Returns:
        Adjacency matrix (n_vars, n_vars), binary
    """
    n_vars = mi_matrix.shape[0]
    adjacency = np.zeros((n_vars, n_vars), dtype=int)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if mi_matrix[i, j] > threshold:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    # Diagonal = 1 (self-loops)
    np.fill_diagonal(adjacency, 1)

    return adjacency


def build_dependency_graph_gtest(
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Use pre-computed adjacency from G-test

    The adjacency matrix is already computed by compute_g_test_matrix()
    in mutual_information.py, this is just a pass-through for consistency.

    Args:
        adjacency: Pre-computed adjacency matrix from G-test

    Returns:
        Adjacency matrix
    """
    return adjacency


def find_maximal_cliques_greedy(
    adjacency: np.ndarray,
    max_clique_size: int = 3,
    max_n_cliques: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Find maximal cliques in dependency graph using greedy algorithm

    This is a simplified version suitable for bounded clique sizes.
    For small max_clique_size (2-4), this is efficient.

    Reference: C++ implementation in cpp_EDAs/cliques.cpp

    Args:
        adjacency: Adjacency matrix (n_vars, n_vars)
        max_clique_size: Maximum clique size allowed
        max_n_cliques: Maximum number of cliques (None = unlimited)

    Returns:
        List of cliques, where each clique is a numpy array of variable indices
    """
    n_vars = adjacency.shape[0]

    # Use NetworkX for clique finding
    G = nx.from_numpy_array(adjacency)

    # Find all maximal cliques
    all_cliques = list(nx.find_cliques(G))

    # Filter by size and convert to numpy arrays
    valid_cliques = []
    for clique in all_cliques:
        if len(clique) <= max_clique_size:
            valid_cliques.append(np.array(clique, dtype=int))

    # If clique is too large, split it
    for clique in all_cliques:
        if len(clique) > max_clique_size:
            # Greedily split large clique into smaller ones
            clique_list = list(clique)
            for i in range(0, len(clique_list), max_clique_size):
                subclique = clique_list[i : i + max_clique_size]
                if len(subclique) > 1:  # Skip single-variable cliques
                    valid_cliques.append(np.array(subclique, dtype=int))

    # Limit number of cliques if requested
    if max_n_cliques is not None and len(valid_cliques) > max_n_cliques:
        # Sort by size (larger cliques first) and take top N
        valid_cliques.sort(key=lambda c: len(c), reverse=True)
        valid_cliques = valid_cliques[:max_n_cliques]

    # Add singleton cliques for variables not covered
    covered_vars = set()
    for clique in valid_cliques:
        covered_vars.update(clique)

    for var in range(n_vars):
        if var not in covered_vars:
            valid_cliques.append(np.array([var], dtype=int))

    return valid_cliques


def cliques_to_junction_graph(
    cliques: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Convert cliques to junction graph structure

    Junction graph requirements:
    - Nodes are cliques
    - Edges connect cliques with non-empty intersection
    - Running intersection property (for exact inference)

    Reference: Algorithm 2, step 4 (Construct labeled JG from L)

    Args:
        cliques: List of cliques

    Returns:
        Tuple of (cliques, edges) where edges are (i, j) clique index pairs
    """
    n_cliques = len(cliques)
    edges = []

    # Find edges: connect cliques with overlapping variables
    for i in range(n_cliques):
        for j in range(i + 1, n_cliques):
            intersection = np.intersect1d(cliques[i], cliques[j])
            if len(intersection) > 0:
                edges.append((i, j))

    return cliques, edges


def find_k_neighbors(
    mi_matrix: np.ndarray,
    k_neighbors: int,
    threshold_factor: float = 1.5,
) -> List[np.ndarray]:
    """
    Find k-nearest neighbors for each variable based on mutual information

    Used by MOA algorithm. For each variable Xi, find the k variables
    with highest MI(Xi, Xj) that exceed the threshold.

    Threshold: TR = avg(MI) * threshold_factor

    Reference:
    - Algorithm 3 (MOA)
    - C++ FDA.cpp:1369-1443 (FindNeighbors)
    - mainmoa.cpp:656 (threshold = 1.5)

    Args:
        mi_matrix: Mutual information matrix (n_vars, n_vars)
        k_neighbors: Maximum number of neighbors per variable
        threshold_factor: Multiplier for average MI threshold (default 1.5)

    Returns:
        List of neighbor arrays, where neighbors[i] contains indices of
        neighbors of variable i (excluding i itself)
    """
    n_vars = mi_matrix.shape[0]

    # Compute threshold: average of all positive MI values * threshold_factor
    positive_mi = mi_matrix[np.triu_indices(n_vars, k=1)]
    positive_mi = positive_mi[positive_mi > 0]

    if len(positive_mi) > 0:
        avg_mi = np.mean(positive_mi)
        threshold = avg_mi * threshold_factor
    else:
        threshold = 0.0

    neighbors_list = []

    for var in range(n_vars):
        # Get MI values for this variable (excluding self)
        mi_values = mi_matrix[var, :].copy()
        mi_values[var] = -np.inf  # Exclude self

        # Find variables above threshold
        candidates = np.where(mi_values > threshold)[0]

        if len(candidates) > k_neighbors:
            # Keep top k by MI value
            top_k_indices = np.argsort(mi_values[candidates])[-k_neighbors:]
            neighbors = candidates[top_k_indices]
        else:
            neighbors = candidates

        # Sort by MI value (descending)
        if len(neighbors) > 0:
            neighbors = neighbors[np.argsort(-mi_values[neighbors])]

        neighbors_list.append(neighbors)

    return neighbors_list


def neighbors_to_cliques(neighbors_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert neighbor lists to clique structure for MOA

    For MOA, each clique contains:
    - The variable Xi (first position)
    - Its neighbors (remaining positions)

    Reference: mainmoa.cpp:669, UpdateModelProteinMPM

    Args:
        neighbors_list: List of neighbor arrays from find_k_neighbors()

    Returns:
        List of cliques suitable for MarkovNetworkModel
    """
    cliques = []

    for var_i, neighbors in enumerate(neighbors_list):
        # Clique = [var_i, neighbor_1, neighbor_2, ...]
        if len(neighbors) > 0:
            clique = np.concatenate([[var_i], neighbors])
        else:
            # Singleton clique if no neighbors
            clique = np.array([var_i])

        cliques.append(clique)

    return cliques


def order_cliques_for_sampling(cliques: List[np.ndarray]) -> np.ndarray:
    """
    Order cliques for probabilistic logic sampling (PLS)

    Ordering requirement:
    - A clique can only be sampled after all its "overlap" variables
      have been assigned values
    - Root cliques (no overlap) come first
    - Uses topological ordering

    Reference:
    - C++ FDA.cpp: SetOrderofCliques, SimpleOrdering
    - pateda/sampling/fda.py: SampleFDA (assumes cliques are ordered)

    Args:
        cliques: List of cliques (unordered)

    Returns:
        Ordered indices: order[i] is the index of the i-th clique to sample
    """
    n_cliques = len(cliques)

    # Build dependency graph
    # Clique B depends on clique A if B has variables not in A, but some overlap
    depends_on = [set() for _ in range(n_cliques)]

    for i in range(n_cliques):
        vars_i = set(cliques[i])

        for j in range(n_cliques):
            if i == j:
                continue

            vars_j = set(cliques[j])
            intersection = vars_i & vars_j

            # If clique i has variables that are in clique j,
            # and clique i has variables not in j,
            # then i depends on j for those overlapping variables
            if len(intersection) > 0 and not vars_i.issubset(vars_j):
                depends_on[i].add(j)

    # Topological sort using Kahn's algorithm
    in_degree = [len(deps) for deps in depends_on]
    queue = [i for i in range(n_cliques) if in_degree[i] == 0]
    order = []

    while queue:
        # Process node with no dependencies
        node = queue.pop(0)
        order.append(node)

        # Remove this node from dependencies
        for i in range(n_cliques):
            if node in depends_on[i]:
                depends_on[i].remove(node)
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    # If not all cliques were ordered, there's a cycle
    # Fall back to original order
    if len(order) != n_cliques:
        order = list(range(n_cliques))

    return np.array(order, dtype=int)


def convert_cliques_to_factorized_structure(
    cliques: List[np.ndarray], order: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert cliques to FactorizedModel structure format

    FactorizedModel structure format (from pateda/core/models.py):
    Each row: [n_overlap, n_new, overlap_var_1, ..., overlap_var_k,
               new_var_1, ..., new_var_m]

    This is the format expected by SampleFDA.

    Args:
        cliques: List of cliques
        order: Clique ordering (if None, use given order)

    Returns:
        Structure array for FactorizedModel
    """
    if order is None:
        order = np.arange(len(cliques))

    n_cliques = len(cliques)

    # Determine overlap and new variables for each clique
    max_clique_size = max(len(c) for c in cliques)
    structure = np.zeros((n_cliques, 2 + 2 * max_clique_size), dtype=int)

    sampled_vars = set()

    for idx, clique_idx in enumerate(order):
        clique = cliques[clique_idx]
        clique_vars = set(clique)

        # Overlap = variables already sampled
        overlap = clique_vars & sampled_vars
        # New = variables not yet sampled
        new = clique_vars - sampled_vars

        n_overlap = len(overlap)
        n_new = len(new)

        overlap_list = sorted(list(overlap))
        new_list = sorted(list(new))

        # Fill structure row
        structure[idx, 0] = n_overlap
        structure[idx, 1] = n_new
        structure[idx, 2 : 2 + n_overlap] = overlap_list
        structure[idx, 2 + n_overlap : 2 + n_overlap + n_new] = new_list

        # Update sampled variables
        sampled_vars.update(new_list)

    return structure
