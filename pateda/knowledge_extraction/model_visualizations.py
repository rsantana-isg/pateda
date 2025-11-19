"""
Advanced visualizations for learned EDA structures.

This module provides sophisticated visualization methods for analyzing
structures learned by EDAs, including dendrogram representations and
glyph-based visualizations for structure comparison.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Optional, Tuple, Any, Union


def view_dendrogram_structure(
    run_structures: Dict,
    method: str = 'average',
    metric: str = 'hamming',
    selected_runs: Optional[List[int]] = None,
    selected_generations: Optional[List[int]] = None,
    fontsize: int = 12,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Dendrogram of Learned Structures"
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create dendrogram visualization of learned structure similarity.

    Performs hierarchical clustering on learned structures to show which
    structures are similar and how they group together. Useful for understanding
    structure evolution and identifying structure families.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data with keys:
        - 'all_big_matrices': List of edge matrices per run
        - 'index_matrix': Matrix mapping edges to indices
    method : str, default='average'
        Linkage method for hierarchical clustering:
        - 'single': Single linkage (minimum distance)
        - 'complete': Complete linkage (maximum distance)
        - 'average': Average linkage (UPGMA)
        - 'ward': Ward's method (minimizes variance)
    metric : str, default='hamming'
        Distance metric:
        - 'hamming': Hamming distance (proportion of different edges)
        - 'jaccard': Jaccard distance
        - 'euclidean': Euclidean distance
    selected_runs : list of int, optional
        Runs to include. If None, uses all runs.
    selected_generations : list of int, optional
        Generations to include. If None, uses all generations.
    fontsize : int, default=12
        Font size for labels.
    figsize : tuple, default=(12, 8)
        Figure size (width, height).
    title : str, default="Dendrogram of Learned Structures"
        Plot title.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure with dendrogram.
    results : dict
        Dictionary containing:
        - 'linkage_matrix': Linkage matrix from hierarchical clustering
        - 'labels': Labels for each structure (run-generation pairs)
        - 'distance_matrix': Pairwise distance matrix
        - 'n_structures': Number of structures analyzed

    Examples
    --------
    >>> fig, results = view_dendrogram_structure(
    ...     run_structures,
    ...     method='ward',
    ...     metric='hamming'
    ... )
    >>> plt.show()
    >>> print(f"Analyzed {results['n_structures']} structures")

    Notes
    -----
    - Dendrogram height indicates dissimilarity between structures
    - Structures that merge early are more similar
    - Useful for identifying phases of structure learning
    - Can reveal whether algorithm converges to similar structures
    - Original concept from MATEDA 2.0 (Section 9, ViewDenDroStruct)
    """
    all_big_matrices = run_structures['all_big_matrices']
    n_runs = len(all_big_matrices)

    if n_runs == 0:
        raise ValueError("No structures found in run_structures")

    # Determine which runs and generations to use
    if selected_runs is None:
        selected_runs = list(range(n_runs))

    if selected_generations is None:
        # Use all generations from first run
        n_gens = all_big_matrices[0].shape[1]
        selected_generations = list(range(n_gens))

    # Collect structure vectors
    structure_vectors = []
    labels = []

    for run_idx in selected_runs:
        if run_idx >= n_runs:
            continue

        run_data = all_big_matrices[run_idx]
        n_gens = min(run_data.shape[1], max(selected_generations) + 1)

        for gen_idx in selected_generations:
            if gen_idx >= n_gens:
                continue

            edge_vector = run_data[:, gen_idx]
            structure_vectors.append(edge_vector)
            labels.append(f"R{run_idx}G{gen_idx}")

    if len(structure_vectors) == 0:
        raise ValueError("No structures selected")

    structure_matrix = np.array(structure_vectors)

    # Compute pairwise distances
    if metric == 'hamming':
        # Hamming distance: proportion of different edges
        distances = pdist(structure_matrix, metric='hamming')
    elif metric == 'jaccard':
        distances = pdist(structure_matrix, metric='jaccard')
    elif metric == 'euclidean':
        distances = pdist(structure_matrix, metric='euclidean')
    else:
        distances = pdist(structure_matrix, metric=metric)

    distance_matrix = squareform(distances)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distances, method=method)

    # Create dendrogram plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        orientation='top',
        distance_sort='ascending',
        show_leaf_counts=False
    )

    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlabel('Structure (Run-Generation)', fontsize=fontsize)
    ax.set_ylabel('Distance', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize - 2, rotation=90)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()

    results = {
        'linkage_matrix': linkage_matrix,
        'labels': labels,
        'distance_matrix': distance_matrix,
        'n_structures': len(structure_vectors),
        'dendrogram': dendro
    }

    return fig, results


def view_glyph_structure(
    run_structures: Dict,
    selected_runs: Optional[List[int]] = None,
    selected_generations: Optional[List[int]] = None,
    glyph_type: str = 'star',
    layout: str = 'grid',
    max_glyphs: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Glyph Visualization of Learned Structures"
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create glyph-based visualization of learned structures.

    Represents each learned structure as a glyph (visual icon) where glyph
    properties encode structure characteristics. Allows visual comparison of
    many structures simultaneously.

    Parameters
    ----------
    run_structures : dict
        Dictionary containing structure data.
    selected_runs : list of int, optional
        Runs to visualize. If None, uses all runs.
    selected_generations : list of int, optional
        Generations to visualize. If None, samples generations.
    glyph_type : str, default='star'
        Type of glyph representation:
        - 'star': Star glyph (radial segments)
        - 'chernoff': Chernoff face (anthropomorphic)
        - 'box': Box glyph (rectangular segments)
        - 'circle': Circle with sectors
    layout : str, default='grid'
        Layout arrangement:
        - 'grid': Regular grid
        - 'temporal': Arranged by time/generation
        - 'similarity': Arranged by structure similarity (MDS)
    max_glyphs : int, default=50
        Maximum number of glyphs to display.
    figsize : tuple, default=(14, 10)
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure with glyph visualization.
    results : dict
        Dictionary containing:
        - 'n_glyphs': Number of glyphs displayed
        - 'positions': Glyph positions
        - 'labels': Structure labels

    Examples
    --------
    >>> fig, results = view_glyph_structure(
    ...     run_structures,
    ...     glyph_type='star',
    ...     max_glyphs=30
    ... )
    >>> plt.show()

    Notes
    -----
    - Glyphs provide intuitive visual comparison
    - Different glyph types emphasize different structure aspects
    - Useful for spotting patterns across many structures
    - Star glyphs show edge frequencies as radial segments
    - Original concept from MATEDA 2.0 (Section 9, ViewGlyphStruct)
    """
    all_big_matrices = run_structures.get('all_big_matrices', [])
    index_matrix = run_structures.get('index_matrix')

    n_runs = len(all_big_matrices)
    if n_runs == 0:
        raise ValueError("No structures found")

    # Determine which runs and generations to use
    if selected_runs is None:
        selected_runs = list(range(n_runs))

    if selected_generations is None:
        # Sample generations evenly
        n_gens = all_big_matrices[0].shape[1]
        step = max(1, n_gens // 10)
        selected_generations = list(range(0, n_gens, step))

    # Collect structures
    structures = []
    labels = []

    for run_idx in selected_runs:
        if run_idx >= n_runs:
            continue

        run_data = all_big_matrices[run_idx]
        n_gens = run_data.shape[1]

        for gen_idx in selected_generations:
            if gen_idx >= n_gens:
                continue

            if len(structures) >= max_glyphs:
                break

            edge_vector = run_data[:, gen_idx]
            structures.append(edge_vector)
            labels.append(f"R{run_idx}G{gen_idx}")

        if len(structures) >= max_glyphs:
            break

    if len(structures) == 0:
        raise ValueError("No structures selected")

    structure_matrix = np.array(structures)
    n_glyphs = len(structures)
    n_features = structure_matrix.shape[1]

    # Determine glyph positions
    if layout == 'grid':
        # Arrange in grid
        n_cols = int(np.ceil(np.sqrt(n_glyphs)))
        n_rows = int(np.ceil(n_glyphs / n_cols))

        positions = []
        for i in range(n_glyphs):
            row = i // n_cols
            col = i % n_cols
            positions.append((col, n_rows - row - 1))

        positions = np.array(positions)

    elif layout == 'temporal':
        # Arrange chronologically
        positions = np.array([[i, 0] for i in range(n_glyphs)])

    elif layout == 'similarity':
        # Use MDS to arrange by similarity
        from sklearn.manifold import MDS

        if n_glyphs > 1:
            mds = MDS(n_components=2, random_state=42)
            positions = mds.fit_transform(structure_matrix)
        else:
            positions = np.array([[0, 0]])

    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw glyphs
    for i, (pos, struct, label) in enumerate(zip(positions, structures, labels)):
        if glyph_type == 'star':
            draw_star_glyph(ax, pos, struct, size=0.4)
        elif glyph_type == 'circle':
            draw_circle_glyph(ax, pos, struct, size=0.4)
        elif glyph_type == 'box':
            draw_box_glyph(ax, pos, struct, size=0.4)
        elif glyph_type == 'chernoff':
            draw_chernoff_glyph(ax, pos, struct, size=0.4)
        else:
            raise ValueError(f"Unknown glyph_type: {glyph_type}")

        # Add label if not too many glyphs
        if n_glyphs <= 20:
            ax.text(pos[0], pos[1] - 0.6, label, ha='center', fontsize=8)

    # Set axis properties
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    # Set reasonable limits
    x_margin = (positions[:, 0].max() - positions[:, 0].min()) * 0.1 + 1
    y_margin = (positions[:, 1].max() - positions[:, 1].min()) * 0.1 + 1

    ax.set_xlim(positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin)
    ax.set_ylim(positions[:, 1].min() - y_margin, positions[:, 1].max() + y_margin)

    plt.tight_layout()

    results = {
        'n_glyphs': n_glyphs,
        'positions': positions,
        'labels': labels,
        'glyph_type': glyph_type,
        'layout': layout
    }

    return fig, results


# Glyph drawing functions

def draw_star_glyph(ax: plt.Axes, center: np.ndarray, values: np.ndarray, size: float = 0.4):
    """Draw star glyph where each ray length represents a feature value."""
    n_features = min(len(values), 20)  # Limit to 20 rays for clarity
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)

    # Sample features if too many
    if len(values) > n_features:
        indices = np.linspace(0, len(values) - 1, n_features, dtype=int)
        values = values[indices]

    # Normalize values to [0, 1]
    if np.max(values) > 0:
        values_norm = values / np.max(values)
    else:
        values_norm = values

    # Create polygon points
    points = []
    for angle, val in zip(angles, values_norm):
        length = size * (0.2 + 0.8 * val)  # Min length 0.2*size
        x = center[0] + length * np.cos(angle)
        y = center[1] + length * np.sin(angle)
        points.append([x, y])

    if len(points) > 2:
        polygon = Polygon(points, closed=True, facecolor='skyblue',
                         edgecolor='navy', alpha=0.7, linewidth=1.5)
        ax.add_patch(polygon)


def draw_circle_glyph(ax: plt.Axes, center: np.ndarray, values: np.ndarray, size: float = 0.4):
    """Draw circle glyph with colored sectors representing features."""
    n_features = min(len(values), 16)

    if len(values) > n_features:
        indices = np.linspace(0, len(values) - 1, n_features, dtype=int)
        values = values[indices]

    # Normalize
    if np.max(values) > 0:
        values_norm = values / np.max(values)
    else:
        values_norm = values

    # Draw background circle
    circle = Circle(center, size, facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(circle)

    # Draw sectors
    angles = np.linspace(0, 360, n_features + 1)
    for i, val in enumerate(values_norm):
        if val > 0:
            color_intensity = val
            wedge = plt.matplotlib.patches.Wedge(
                center, size, angles[i], angles[i + 1],
                facecolor=plt.cm.Blues(color_intensity),
                edgecolor='navy',
                linewidth=0.5
            )
            ax.add_patch(wedge)


def draw_box_glyph(ax: plt.Axes, center: np.ndarray, values: np.ndarray, size: float = 0.4):
    """Draw box glyph with bar segments around perimeter."""
    n_features = min(len(values), 12)

    if len(values) > n_features:
        indices = np.linspace(0, len(values) - 1, n_features, dtype=int)
        values = values[indices]

    # Normalize
    if np.max(values) > 0:
        values_norm = values / np.max(values)
    else:
        values_norm = values

    # Draw background box
    box = FancyBboxPatch(
        (center[0] - size/2, center[1] - size/2), size, size,
        boxstyle="round,pad=0.05",
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(box)

    # Draw bars around perimeter
    bars_per_side = (n_features + 3) // 4
    for i, val in enumerate(values_norm):
        side = i // bars_per_side
        pos_in_side = i % bars_per_side

        bar_length = size * val * 0.3

        if side == 0:  # Top
            x = center[0] - size/2 + (pos_in_side + 0.5) * size / bars_per_side
            y = center[1] + size/2
            rect = Rectangle((x - 0.02, y), 0.04, bar_length,
                           facecolor='steelblue', edgecolor='navy')
        elif side == 1:  # Right
            x = center[0] + size/2
            y = center[1] + size/2 - (pos_in_side + 0.5) * size / bars_per_side
            rect = Rectangle((x, y - 0.02), bar_length, 0.04,
                           facecolor='steelblue', edgecolor='navy')
        elif side == 2:  # Bottom
            x = center[0] + size/2 - (pos_in_side + 0.5) * size / bars_per_side
            y = center[1] - size/2
            rect = Rectangle((x - 0.02, y - bar_length), 0.04, bar_length,
                           facecolor='steelblue', edgecolor='navy')
        else:  # Left
            x = center[0] - size/2
            y = center[1] - size/2 + (pos_in_side + 0.5) * size / bars_per_side
            rect = Rectangle((x - bar_length, y - 0.02), bar_length, 0.04,
                           facecolor='steelblue', edgecolor='navy')

        ax.add_patch(rect)


def draw_chernoff_glyph(ax: plt.Axes, center: np.ndarray, values: np.ndarray, size: float = 0.4):
    """Draw simplified Chernoff face (uses first few features)."""
    # Normalize
    if np.max(values) > 0:
        values_norm = values / np.max(values)
    else:
        values_norm = np.zeros_like(values)

    # Extract features (use first 6 for face properties)
    features = np.pad(values_norm, (0, max(0, 6 - len(values_norm))), constant_values=0.5)[:6]

    face_width = size * (0.8 + 0.4 * features[0])
    face_height = size * (0.8 + 0.4 * features[1])
    eye_size = size * 0.1 * (0.5 + features[2])
    mouth_curve = features[3] - 0.5  # -0.5 to 0.5
    eye_separation = size * 0.3 * (0.8 + 0.4 * features[4])

    # Draw face
    face = plt.matplotlib.patches.Ellipse(
        center, face_width, face_height,
        facecolor='peachpuff',
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(face)

    # Draw eyes
    left_eye = Circle((center[0] - eye_separation/2, center[1] + size*0.1),
                     eye_size, facecolor='black')
    right_eye = Circle((center[0] + eye_separation/2, center[1] + size*0.1),
                      eye_size, facecolor='black')
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    # Draw mouth (simplified curve)
    mouth_y = center[1] - size * 0.15
    mouth_points = []
    for i in range(5):
        x = center[0] - size*0.2 + i * size*0.1
        y = mouth_y + mouth_curve * size * 0.15 * np.sin(i * np.pi / 4)
        mouth_points.append([x, y])

    mouth_line = LineCollection([mouth_points], colors='black', linewidths=2)
    ax.add_collection(mouth_line)
