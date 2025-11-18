"""
Visualization module for PATEDA.

This module provides comprehensive visualization tools for analyzing
structures learned by Estimation of Distribution Algorithms (EDAs).
It includes basic plotting functions, clustering methods, and high-level
visualization methods for structure analysis.

Main Components:
- Basic plots: Heatmaps, parallel coordinates
- Clustering: Methods for ordering variables based on relationships
- Structure utilities: Conversion and extraction functions
- Structure views: High-level visualization methods

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

# Basic plotting functions
from pateda.visualization.basic_plots import (
    show_image,
    show_parallel_coordinates,
)

# Clustering functions
from pateda.visualization.clustering import (
    cluster_using_correlation,
    cluster_using_distance,
)

# Structure utility functions
from pateda.visualization.structure_utils import (
    convert_from_index,
    select_edges_to_show,
    extract_substructures,
    find_index_matrix,
)

# High-level structure visualization methods
from pateda.visualization.structure_views import (
    view_summary_structures,
    view_edge_dependency_structures,
    view_parallel_coordinate_structures,
)

__all__ = [
    # Basic plots
    "show_image",
    "show_parallel_coordinates",
    # Clustering
    "cluster_using_correlation",
    "cluster_using_distance",
    # Structure utilities
    "convert_from_index",
    "select_edges_to_show",
    "extract_substructures",
    "find_index_matrix",
    # Structure views
    "view_summary_structures",
    "view_edge_dependency_structures",
    "view_parallel_coordinate_structures",
]
