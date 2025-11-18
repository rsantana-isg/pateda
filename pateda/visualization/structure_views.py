"""
High-level visualization methods for EDA structures.

This module provides comprehensive visualization functions for analyzing
structures learned by Estimation of Distribution Algorithms (EDAs). These
include summary views, edge dependency analysis, and parallel coordinate plots.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

from pateda.visualization.basic_plots import show_image, show_parallel_coordinates
from pateda.visualization.structure_utils import (
    convert_from_index,
    select_edges_to_show,
    extract_substructures
)
from pateda.visualization.clustering import cluster_using_correlation, cluster_using_distance


def view_summary_structures(
    run_structures: Dict,
    pcolors: int = 150,
    fontsize: int = 14,
    title: str = "Summary of All Learned Structures",
    **kwargs
) -> plt.Figure:
    """
    Show summary visualization of all learned structures.

    Creates a heatmap where each edge has a color proportional to the number
    of times it appeared in structures learned across all generations and runs.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data with key:
        - 'sum_all_contact_matrix': Matrix with aggregated edge counts
    pcolors : int, default=150
        Range of colors for the image.
    fontsize : int, default=14
        Font size for labels.
    title : str, default="Summary of All Learned Structures"
        Title for the plot.
    **kwargs
        Additional keyword arguments passed to show_image.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the summary visualization.

    Examples
    --------
    >>> run_structures = {
    ...     'sum_all_contact_matrix': np.random.randint(0, 100, (20, 20))
    ... }
    >>> fig = view_summary_structures(run_structures)
    >>> plt.show()

    Notes
    -----
    - Higher color intensity indicates more frequent edges
    - Useful for identifying consistently learned dependencies
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    sum_all_contact_matrix = run_structures['sum_all_contact_matrix']

    fig = show_image(
        sum_all_contact_matrix,
        pcolors=pcolors,
        fontsize=fontsize,
        title=title,
        **kwargs
    )

    return fig


def view_edge_dependency_structures(
    run_structures: Dict,
    substructure_spec: np.ndarray,
    selected_runs: List[int],
    selected_generations: List[int],
    display_type: str = 'one_graph',
    pcolors: int = 100,
    fontsize: int = 14
) -> Tuple[Optional[plt.Figure], np.ndarray]:
    """
    Search and visualize substructures with specific edge patterns.

    Searches for structures that match a specified pattern (e.g., edges that
    must be present or absent) and visualizes the matching structures.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing:
        - 'index_matrix': Matrix mapping edge pairs to indices
        - 'all_big_matrices': List of edge matrices per run
    substructure_spec : np.ndarray
        Specification of substructure as (n_conditions, 3) array where each row
        is [var_i, var_j, presence], with presence=1 for required edges and
        presence=0 for forbidden edges.
    selected_runs : list of int
        Indices of runs to inspect.
    selected_generations : list of int
        Indices of generations to inspect.
    display_type : str, default='one_graph'
        Type of display:
        - 'all_graphs': Show each matching structure separately
        - 'one_graph': Show aggregated matching structures
        - 'no_graph': Don't show graphs, only return matches
    pcolors : int, default=100
        Range of colors for visualization.
    fontsize : int, default=14
        Font size for labels.

    Returns
    -------
    fig : plt.Figure or None
        Matplotlib figure (None if display_type='no_graph').
    matches : np.ndarray
        Array of shape (n_matches, 2) where each row is [run_idx, generation_idx]
        indicating where the substructure was found.

    Examples
    --------
    >>> # Find structures with edges (3,4) and (4,5) present, but not (3,5)
    >>> substructure = np.array([[3, 4, 1], [4, 5, 1], [3, 5, 0]])
    >>> fig, matches = view_edge_dependency_structures(
    ...     run_structures, substructure, [0, 1, 2], list(range(10))
    ... )

    Notes
    -----
    - Useful for hypothesis testing about structure patterns
    - Can identify when specific dependencies are learned together
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    index_matrix = run_structures['index_matrix']
    all_big_matrices = run_structures['all_big_matrices']

    n = index_matrix.shape[0]
    sum_contact_matrix = np.zeros((n, n))
    matches = []

    # Convert substructure specification to index conditions
    n_conditions = substructure_spec.shape[0]
    index_conditions = []
    value_conditions = []

    for i in range(n_conditions):
        var_i, var_j, presence = substructure_spec[i]
        edge_idx = index_matrix[int(var_i), int(var_j)]
        index_conditions.append(edge_idx)
        value_conditions.append(presence)

    index_conditions = np.array(index_conditions)
    value_conditions = np.array(value_conditions)

    # Search through selected runs and generations
    for run_idx in selected_runs:
        run_data = all_big_matrices[run_idx]

        for gen_idx in selected_generations:
            edge_vector = run_data[:, gen_idx]

            # Check if all conditions are satisfied
            conditions_met = np.all(edge_vector[index_conditions] == value_conditions)

            if conditions_met:
                contact_matrix = convert_from_index(index_matrix, edge_vector)

                if display_type == 'one_graph':
                    sum_contact_matrix += contact_matrix
                elif display_type == 'all_graphs':
                    show_image(
                        contact_matrix,
                        pcolors=pcolors,
                        fontsize=fontsize,
                        title=f"Run {run_idx}, Generation {gen_idx}"
                    )

                matches.append([run_idx, gen_idx])

    matches = np.array(matches) if matches else np.array([]).reshape(0, 2)

    # Create final visualization
    fig = None
    if display_type == 'one_graph' and len(matches) > 0:
        fig = show_image(
            sum_contact_matrix,
            pcolors=pcolors,
            fontsize=fontsize,
            title="Aggregated Matching Structures"
        )

    return fig, matches


def view_parallel_coordinate_structures(
    run_structures: Dict,
    edge_indices: Optional[np.ndarray] = None,
    min_occurrences: int = 60,
    min_edges: int = 2,
    ordering_method: str = 'none',
    distance: str = 'correlation',
    fontsize: int = 14,
    title: str = "Parallel Coordinates of Structure Evolution"
) -> Tuple[plt.Figure, Dict]:
    """
    Visualize structure evolution using parallel coordinates.

    Shows when frequent edges appear across generations. Each vertical axis
    represents an edge, and the vertical position indicates the generation
    when that edge was learned. Lines connecting points show edges that
    appeared together in the same structure.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data.
    edge_indices : np.ndarray or None, default=None
        Specific edges to visualize. If None, edges are selected based on
        min_occurrences.
    min_occurrences : int, default=60
        Minimum times an edge must appear to be visualized (if edge_indices=None).
    min_edges : int, default=2
        Minimum number of edges a structure must have to be included.
    ordering_method : str, default='none'
        Method to order edges:
        - 'none': Use given order
        - 'random': Random order
        - 'correlation': Order by correlation clustering
        - 'distance': Order by distance-based clustering
    distance : str, default='correlation'
        Distance metric for clustering (if ordering_method uses clustering).
    fontsize : int, default=14
        Font size for labels.
    title : str, default="Parallel Coordinates of Structure Evolution"
        Plot title.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure with parallel coordinates plot.
    results : dict
        Dictionary containing:
        - 'ordering': Final edge ordering used
        - 'edge_list': List of edges shown (as [var_i, var_j] pairs)

    Examples
    --------
    >>> fig, results = view_parallel_coordinate_structures(
    ...     run_structures,
    ...     min_occurrences=50,
    ...     ordering_method='correlation'
    ... )
    >>> plt.show()

    Notes
    -----
    - Helps identify when specific edges emerge during evolution
    - Clustering can reduce visual clutter
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    index_matrix = run_structures.get('index_matrix')

    # Select edges if not provided
    if edge_indices is None:
        edge_indices = select_edges_to_show(run_structures, min_occurrences)

        if len(edge_indices) == 0:
            raise ValueError(
                "No edges selected. Reduce min_occurrences parameter."
            )

    # Extract substructures
    all_rep_vectors = extract_substructures(run_structures, edge_indices, min_edges)

    if len(all_rep_vectors) == 0:
        raise ValueError("No structures found matching criteria.")

    n_edges = len(edge_indices)

    # Determine ordering
    if ordering_method == 'random':
        ordering = np.random.permutation(n_edges)
    elif ordering_method == 'correlation':
        result = cluster_using_correlation(all_rep_vectors, distance)
        ordering = result['ordering']
    elif ordering_method == 'distance':
        result = cluster_using_distance(all_rep_vectors, distance)
        ordering = result['ordering']
    else:  # 'none' or any other value
        ordering = np.arange(n_edges)

    # Reorder data
    reordered_vectors = all_rep_vectors[:, ordering]

    # Create parallel coordinates plot
    fig = show_parallel_coordinates(
        reordered_vectors,
        fontsize=fontsize,
        title=title
    )

    # Build edge list
    edge_list = []
    if index_matrix is not None:
        for i in range(n_edges):
            edge_idx = edge_indices[ordering[i]]
            positions = np.where(index_matrix == edge_idx)
            if len(positions[0]) > 0:
                row, col = positions[0][0], positions[1][0]
                edge_list.append([row, col])

    results = {
        'ordering': ordering,
        'edge_list': np.array(edge_list) if edge_list else None
    }

    return fig, results
