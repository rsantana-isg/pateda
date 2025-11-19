"""
A posteriori dependency analysis for EDAs.

This module provides tools for analyzing dependencies between variables
discovered during EDA evolution. It includes correlation analysis and
probabilistic graphical model learning from population data.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Optional, Tuple, Any
import warnings


def compute_correlation_matrix(
    population: np.ndarray,
    method: str = 'pearson',
    abs_values: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute correlation matrix for population variables.

    Analyzes pairwise correlations between variables in the population
    to identify statistical dependencies.

    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_variables).
    method : str, default='pearson'
        Correlation method:
        - 'pearson': Pearson correlation (linear relationships)
        - 'spearman': Spearman correlation (monotonic relationships)
        - 'kendall': Kendall tau (rank correlation)
    abs_values : bool, default=False
        If True, return absolute values of correlations.

    Returns
    -------
    dict
        Dictionary containing:
        - 'correlation_matrix': Correlation matrix (n_variables, n_variables)
        - 'pvalue_matrix': P-values for correlations
        - 'significant_pairs': List of (i, j, correlation) for significant pairs
        - 'method': Correlation method used

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.random.rand(100, 10)
    >>> result = compute_correlation_matrix(pop)
    >>> corr_matrix = result['correlation_matrix']
    >>> print(f"Max correlation: {np.max(np.abs(corr_matrix - np.eye(10))):.4f}")

    Notes
    -----
    - Pearson assumes linear relationships and normally distributed data
    - Spearman is robust to outliers and non-linear monotonic relationships
    - Kendall is more robust but computationally intensive
    - High correlations suggest potential dependencies to model
    - Original concept from MATEDA 2.0 (Section 8.2)
    """
    n_samples, n_vars = population.shape

    correlation_matrix = np.zeros((n_vars, n_vars))
    pvalue_matrix = np.ones((n_vars, n_vars))

    # Compute pairwise correlations
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                correlation_matrix[i, j] = 1.0
                pvalue_matrix[i, j] = 0.0
            else:
                x = population[:, i]
                y = population[:, j]

                if method == 'pearson':
                    corr, pval = sp_stats.pearsonr(x, y)
                elif method == 'spearman':
                    corr, pval = sp_stats.spearmanr(x, y)
                elif method == 'kendall':
                    corr, pval = sp_stats.kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown correlation method: {method}")

                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
                pvalue_matrix[i, j] = pval
                pvalue_matrix[j, i] = pval

    if abs_values:
        correlation_matrix = np.abs(correlation_matrix)

    # Find significant pairs (p < 0.05, excluding diagonal)
    significant_pairs = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if pvalue_matrix[i, j] < 0.05:
                significant_pairs.append((i, j, correlation_matrix[i, j]))

    return {
        'correlation_matrix': correlation_matrix,
        'pvalue_matrix': pvalue_matrix,
        'significant_pairs': significant_pairs,
        'method': method,
        'n_significant': len(significant_pairs)
    }


def learn_bayesian_network(
    population: np.ndarray,
    method: str = 'k2',
    max_parents: int = 3,
    score_function: str = 'bic'
) -> Dict[str, Any]:
    """
    Learn Bayesian network structure from discrete population data.

    Uses structure learning algorithms to discover a Bayesian network
    that represents dependencies in the population.

    Parameters
    ----------
    population : np.ndarray
        Discrete population matrix of shape (pop_size, n_variables).
        Values should be integers representing categorical states.
    method : str, default='k2'
        Structure learning method:
        - 'k2': K2 algorithm (requires variable ordering)
        - 'greedy': Greedy hill-climbing search
        - 'mi': Mutual information based (Chow-Liu tree)
    max_parents : int, default=3
        Maximum number of parents per node.
    score_function : str, default='bic'
        Scoring function: 'bic', 'aic', or 'll' (log-likelihood).

    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacency_matrix': Binary adjacency matrix (i->j if [i,j]=1)
        - 'edge_list': List of (parent, child) tuples
        - 'score': Structure score
        - 'method': Method used
        - 'n_edges': Number of edges learned

    Examples
    --------
    >>> import numpy as np
    >>> # Binary population
    >>> pop = np.random.randint(0, 2, (100, 10))
    >>> result = learn_bayesian_network(pop, method='mi')
    >>> print(f"Learned {result['n_edges']} dependencies")

    Notes
    -----
    - For continuous data, discretize first or use learn_gaussian_network
    - K2 algorithm is fast but ordering-dependent
    - Mutual information method learns tree structures (Chow-Liu)
    - Greedy search explores more complex structures but is slower
    - Original concept from MATEDA 2.0 (Section 8.2)
    """
    n_samples, n_vars = population.shape

    # Check if data is discrete
    if not np.all(population == population.astype(int)):
        warnings.warn(
            "Population contains non-integer values. "
            "Bayesian network learning expects discrete data. "
            "Consider discretization or use learn_gaussian_network."
        )

    adjacency_matrix = np.zeros((n_vars, n_vars), dtype=int)

    if method == 'mi':
        # Chow-Liu tree: learn maximum spanning tree based on mutual information
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mi = compute_mutual_information(population[:, i], population[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Find maximum spanning tree using Kruskal's algorithm
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                edges.append((mi_matrix[i, j], i, j))

        edges.sort(reverse=True)  # Sort by MI (descending)

        # Union-find for cycle detection
        parent = list(range(n_vars))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Add edges without creating cycles
        for mi, i, j in edges:
            if union(i, j):
                adjacency_matrix[i, j] = 1  # Directed edge i -> j

        score = sum(mi_matrix[i, j] for i in range(n_vars) for j in range(i + 1, n_vars)
                   if adjacency_matrix[i, j] == 1)

    elif method == 'greedy':
        # Greedy hill-climbing
        current_score = compute_structure_score(
            population, adjacency_matrix, score_function
        )

        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_adjacency = adjacency_matrix.copy()

            # Try adding edges
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adjacency_matrix[i, j] == 0:
                        # Check max_parents constraint
                        if np.sum(adjacency_matrix[:, j]) >= max_parents:
                            continue

                        # Try adding edge i -> j
                        test_adjacency = adjacency_matrix.copy()
                        test_adjacency[i, j] = 1

                        # Check for cycles
                        if has_cycle(test_adjacency):
                            continue

                        test_score = compute_structure_score(
                            population, test_adjacency, score_function
                        )

                        if test_score > best_score:
                            best_score = test_score
                            best_adjacency = test_adjacency
                            improved = True

            if improved:
                adjacency_matrix = best_adjacency
                current_score = best_score

        score = current_score

    elif method == 'k2':
        # K2 algorithm (simplified version)
        # Use natural ordering 0, 1, 2, ..., n_vars-1
        for j in range(n_vars):
            current_parents = []
            current_score = compute_local_score(
                population, j, current_parents, score_function
            )

            improved = True
            while improved and len(current_parents) < max_parents:
                improved = False
                best_score = current_score
                best_parent = None

                # Try adding each potential parent
                for i in range(j):  # K2 respects ordering
                    if i not in current_parents:
                        test_parents = current_parents + [i]
                        test_score = compute_local_score(
                            population, j, test_parents, score_function
                        )

                        if test_score > best_score:
                            best_score = test_score
                            best_parent = i
                            improved = True

                if improved:
                    current_parents.append(best_parent)
                    current_score = best_score

            # Set edges for learned parents
            for parent in current_parents:
                adjacency_matrix[parent, j] = 1

        score = compute_structure_score(population, adjacency_matrix, score_function)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract edge list
    edge_list = []
    for i in range(n_vars):
        for j in range(n_vars):
            if adjacency_matrix[i, j] == 1:
                edge_list.append((i, j))

    return {
        'adjacency_matrix': adjacency_matrix,
        'edge_list': edge_list,
        'score': float(score),
        'method': method,
        'n_edges': len(edge_list)
    }


def learn_gaussian_network(
    population: np.ndarray,
    method: str = 'correlation',
    threshold: float = 0.3,
    max_edges: Optional[int] = None
) -> Dict[str, Any]:
    """
    Learn Gaussian network (undirected graphical model) from continuous data.

    Discovers dependencies in continuous variables using correlation-based
    or partial correlation-based methods.

    Parameters
    ----------
    population : np.ndarray
        Continuous population matrix of shape (pop_size, n_variables).
    method : str, default='correlation'
        Learning method:
        - 'correlation': Thresholded correlation
        - 'partial_correlation': Partial correlation (more precise)
        - 'glasso': Graphical Lasso (sparse inverse covariance)
    threshold : float, default=0.3
        Correlation threshold for edge inclusion.
    max_edges : int, optional
        Maximum number of edges to include. If None, uses threshold only.

    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacency_matrix': Symmetric adjacency matrix
        - 'edge_list': List of (i, j) tuples
        - 'edge_weights': Correlation/partial correlation values for edges
        - 'method': Method used
        - 'n_edges': Number of edges learned

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.random.randn(100, 10)
    >>> result = learn_gaussian_network(pop, threshold=0.4)
    >>> print(f"Learned network with {result['n_edges']} edges")

    Notes
    -----
    - Correlation captures marginal dependencies
    - Partial correlation captures conditional dependencies
    - Graphical Lasso provides sparse precision matrix estimation
    - For discrete data, use learn_bayesian_network instead
    - Original concept from MATEDA 2.0 (Section 8.2)
    """
    n_samples, n_vars = population.shape

    adjacency_matrix = np.zeros((n_vars, n_vars))
    edge_weights = {}

    if method == 'correlation':
        # Simple correlation thresholding
        corr_result = compute_correlation_matrix(population, method='pearson')
        corr_matrix = corr_result['correlation_matrix']

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr_matrix[i, j]) >= threshold:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                    edge_weights[(i, j)] = corr_matrix[i, j]

    elif method == 'partial_correlation':
        # Partial correlation using precision matrix
        cov_matrix = np.cov(population.T)

        # Add small regularization for numerical stability
        cov_matrix += np.eye(n_vars) * 1e-6

        try:
            precision_matrix = np.linalg.inv(cov_matrix)

            # Partial correlation from precision matrix
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    partial_corr = -precision_matrix[i, j] / np.sqrt(
                        precision_matrix[i, i] * precision_matrix[j, j]
                    )

                    if abs(partial_corr) >= threshold:
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1
                        edge_weights[(i, j)] = partial_corr

        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix is singular. Using correlation method instead.")
            return learn_gaussian_network(population, method='correlation', threshold=threshold)

    elif method == 'glasso':
        # Graphical Lasso (simplified version without sklearn)
        # Fall back to partial correlation for this implementation
        warnings.warn("Graphical Lasso not fully implemented. Using partial correlation.")
        return learn_gaussian_network(population, method='partial_correlation', threshold=threshold)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply max_edges constraint if specified
    if max_edges is not None and len(edge_weights) > max_edges:
        # Keep only top-k edges by weight
        sorted_edges = sorted(edge_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top_edges = sorted_edges[:max_edges]

        # Rebuild adjacency matrix
        adjacency_matrix = np.zeros((n_vars, n_vars))
        edge_weights = {}
        for (i, j), weight in top_edges:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
            edge_weights[(i, j)] = weight

    # Extract edge list
    edge_list = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)
                 if adjacency_matrix[i, j] == 1]

    return {
        'adjacency_matrix': adjacency_matrix,
        'edge_list': edge_list,
        'edge_weights': edge_weights,
        'method': method,
        'n_edges': len(edge_list)
    }


def analyze_variable_dependencies(
    all_statistics: Dict[int, Dict[str, Any]],
    generation: int,
    method: str = 'auto'
) -> Dict[str, Any]:
    """
    Analyze dependencies between variables at a specific generation.

    Automatically detects whether data is discrete or continuous and
    applies appropriate dependency analysis.

    Parameters
    ----------
    all_statistics : dict
        Dictionary of statistics per generation.
    generation : int
        Generation to analyze.
    method : str, default='auto'
        Analysis method: 'auto', 'discrete', 'continuous'.

    Returns
    -------
    dict
        Dependency analysis results including learned structure.

    Examples
    --------
    >>> # Assume all_statistics collected during EDA run
    >>> deps = analyze_variable_dependencies(all_statistics, generation=10)
    >>> print(f"Found {deps['n_edges']} dependencies")
    """
    if generation not in all_statistics:
        raise ValueError(f"Generation {generation} not found in statistics")

    # For this function to work, we'd need the population data stored
    # This is a placeholder for the interface
    raise NotImplementedError(
        "This function requires population data to be stored in all_statistics. "
        "Use learn_bayesian_network or learn_gaussian_network directly with population data."
    )


# Helper functions

def compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two discrete variables."""
    # Create contingency table
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    contingency = np.zeros((len(unique_x), len(unique_y)))

    for i, val_x in enumerate(unique_x):
        for j, val_y in enumerate(unique_y):
            contingency[i, j] = np.sum((x == val_x) & (y == val_y))

    # Normalize to get probabilities
    contingency = contingency / np.sum(contingency)

    # Marginal probabilities
    p_x = np.sum(contingency, axis=1)
    p_y = np.sum(contingency, axis=0)

    # Compute MI
    mi = 0.0
    for i in range(len(unique_x)):
        for j in range(len(unique_y)):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log(contingency[i, j] / (p_x[i] * p_y[j]))

    return mi


def compute_structure_score(
    population: np.ndarray,
    adjacency_matrix: np.ndarray,
    score_function: str = 'bic'
) -> float:
    """Compute score for a Bayesian network structure."""
    n_samples, n_vars = population.shape
    score = 0.0

    for j in range(n_vars):
        parents = np.where(adjacency_matrix[:, j] == 1)[0]
        score += compute_local_score(population, j, parents.tolist(), score_function)

    return score


def compute_local_score(
    population: np.ndarray,
    node: int,
    parents: List[int],
    score_function: str = 'bic'
) -> float:
    """Compute local score for a node given its parents."""
    n_samples = population.shape[0]

    # Count configurations
    node_values = population[:, node]
    unique_node = np.unique(node_values)
    n_node_states = len(unique_node)

    if len(parents) == 0:
        # No parents: just count node states
        counts = np.array([np.sum(node_values == val) for val in unique_node])
        log_likelihood = np.sum(counts * np.log(counts / n_samples + 1e-10))

        n_params = n_node_states - 1

    else:
        # With parents: count joint configurations
        parent_data = population[:, parents]

        # Find unique parent configurations
        unique_configs = np.unique(parent_data, axis=0)
        n_configs = len(unique_configs)

        log_likelihood = 0.0
        for config in unique_configs:
            # Find samples matching this parent configuration
            mask = np.all(parent_data == config, axis=1)
            n_config = np.sum(mask)

            if n_config > 0:
                node_given_config = node_values[mask]
                for val in unique_node:
                    count = np.sum(node_given_config == val)
                    if count > 0:
                        log_likelihood += count * np.log(count / n_config + 1e-10)

        n_params = n_configs * (n_node_states - 1)

    # Apply penalty based on score function
    if score_function == 'bic':
        penalty = 0.5 * n_params * np.log(n_samples)
        score = log_likelihood - penalty
    elif score_function == 'aic':
        penalty = n_params
        score = log_likelihood - penalty
    elif score_function == 'll':
        score = log_likelihood
    else:
        raise ValueError(f"Unknown score function: {score_function}")

    return score


def has_cycle(adjacency_matrix: np.ndarray) -> bool:
    """Check if directed graph has a cycle using DFS."""
    n_vars = adjacency_matrix.shape[0]
    visited = np.zeros(n_vars, dtype=bool)
    rec_stack = np.zeros(n_vars, dtype=bool)

    def dfs(node):
        visited[node] = True
        rec_stack[node] = True

        # Check all neighbors
        for neighbor in range(n_vars):
            if adjacency_matrix[node, neighbor] == 1:
                if not visited[neighbor]:
                    if dfs(neighbor):
                        return True
                elif rec_stack[neighbor]:
                    return True

        rec_stack[node] = False
        return False

    for node in range(n_vars):
        if not visited[node]:
            if dfs(node):
                return True

    return False
