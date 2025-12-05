"""
Basic plotting functions for PATEDA visualizations.

This module provides fundamental visualization functions for displaying
matrices, parallel coordinates, and other data structures used in
Estimation of Distribution Algorithms (EDAs).

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pandas.plotting import parallel_coordinates
import pandas as pd


def show_image(
    matrix: np.ndarray,
    pcolors: int = 150,
    fontsize: int = 14,
    title: Optional[str] = None,
    xlabel: str = "Variable I",
    ylabel: str = "Variable J",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Display a matrix as a heatmap with colors proportional to cell values.

    This function visualizes an incident/adjacency matrix by associating colors
    with the magnitude of cell values. Useful for displaying learned structures
    from EDAs.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to display, typically an adjacency or contact matrix.
    pcolors : int, default=150
        Range of colors for the image. Higher values provide more color gradation.
        This parameter affects the normalization of the colormap.
    fontsize : int, default=14
        Font size for axis labels and other text elements.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    xlabel : str, default="Variable I"
        Label for the x-axis.
    ylabel : str, default="Variable J"
        Label for the y-axis.
    cmap : str, default="viridis"
        Matplotlib colormap name. Options include 'viridis', 'plasma', 'hot',
        'coolwarm', etc.
    ax : Optional[plt.Axes], default=None
        Matplotlib axes object. If None, a new figure is created.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.random.rand(10, 10)
    >>> fig = show_image(matrix, title="Random Matrix")
    >>> plt.show()

    Notes
    -----
    - The matrix values are normalized based on the maximum value
    - The y-axis origin is set to 'lower' to match MATLAB's convention
    - A colorbar is automatically added to show the value scale
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    max_val = np.max(matrix)

    # Normalize matrix similar to MATLAB's approach
    if max_val > 0:
        normalized_matrix = pcolors * matrix / max_val
    else:
        normalized_matrix = pcolors * matrix

    # Display the image with origin at bottom-left (like MATLAB's 'axis xy')
    im = ax.imshow(normalized_matrix, cmap=cmap, origin='lower', aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize)

    # Set tick label font size
    ax.tick_params(labelsize=fontsize - 2)

    plt.tight_layout()

    return fig


def show_parallel_coordinates(
    vectors: np.ndarray,
    fontsize: int = 14,
    title: Optional[str] = None,
    labels: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.7,
    colormap: str = "viridis"
) -> plt.Figure:
    """
    Display parallel coordinates plot of vectors.

    Parallel coordinates are useful for visualizing high-dimensional data
    where each observation is represented as a polyline connecting points
    across parallel vertical axes (one for each dimension).

    Parameters
    ----------
    vectors : np.ndarray
        Data matrix where each row is an observation and each column
        corresponds to a variable/dimension.
    fontsize : int, default=14
        Font size for axis labels and tick labels.
    title : Optional[str], default=None
        Title for the plot.
    labels : Optional[list], default=None
        Labels for each variable axis. If None, default labels are used.
    ax : Optional[plt.Axes], default=None
        Matplotlib axes object. If None, a new figure is created.
    alpha : float, default=0.7
        Transparency level for the lines (0=transparent, 1=opaque).
    colormap : str, default="viridis"
        Name of matplotlib colormap for coloring the lines.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the parallel coordinates plot.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(50, 5)
    >>> fig = show_parallel_coordinates(data, title="Parallel Coordinates Example")
    >>> plt.show()

    Notes
    -----
    - Uses pandas parallel_coordinates for implementation
    - Each line represents one observation (row) in the data
    - Useful for identifying patterns and clusters in high-dimensional data
    - Original version: MATEDA 2.0 (8/26/2008)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    n_samples, n_vars = vectors.shape

    # Create column names
    if labels is None:
        labels = [f"Var{i+1}" for i in range(n_vars)]

    # Convert to DataFrame for pandas parallel_coordinates
    df = pd.DataFrame(vectors, columns=labels)

    # Add a dummy class column for coloring (using row index)
    df['class'] = range(n_samples)

    # Create parallel coordinates plot
    parallel_coordinates(
        df,
        'class',
        ax=ax,
        alpha=alpha,
        colormap=colormap
    )

    # Remove the legend (it's just indices, not meaningful)
    if ax.get_legend():
        ax.get_legend().remove()

    if title:
        ax.set_title(title, fontsize=fontsize)

    ax.tick_params(labelsize=fontsize - 2)
    ax.set_ylabel("Value", fontsize=fontsize)

    plt.tight_layout()

    return fig
