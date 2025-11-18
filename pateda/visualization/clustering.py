"""
Clustering functions for ordering variables in visualizations.

This module provides clustering methods to order features/variables
based on their relationships. This is particularly useful for reducing
clutter in parallel coordinate visualizations.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
from typing import Tuple, Dict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation


def cluster_using_correlation(
    data: np.ndarray,
    distance: str = "correlation"
) -> Dict[str, np.ndarray]:
    """
    Order features based on their correlation using affinity propagation clustering.

    This function clusters variables that are strongly correlated together,
    which helps reduce cluttering in parallel coordinate plots and improves
    visualization clarity.

    Parameters
    ----------
    data : np.ndarray
        Data matrix where rows are observations and columns are features/variables.
    distance : str, default="correlation"
        Distance metric to use. Can be any metric supported by scipy.spatial.distance.pdist
        such as 'correlation', 'euclidean', 'cosine', etc.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'ordering': New ordering of features where subsequent features are more related
        - 'clusters': Original clustering labels from affinity propagation
        - 'cluster_centers': Indices of cluster exemplars

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 10)
    >>> result = cluster_using_correlation(data)
    >>> new_order = result['ordering']
    >>> # Use new_order to rearrange columns for better visualization

    Notes
    -----
    - Uses affinity propagation algorithm for clustering
    - The self-similarity measure is set to the mean of pairwise similarities
    - Variables are ordered by their cluster membership
    - Original version: MATEDA 2.0 (8/26/2008)

    See Also
    --------
    cluster_using_distance : Similar function with explicit distance parameter
    """
    n_edges = data.shape[1]

    # Compute pairwise distances between features (transpose to compare columns)
    y = pdist(data.T, distance)

    # Convert distances to similarities (affinity propagation maximizes similarity)
    rho = np.max(y) - squareform(y)

    # Set preference (self-similarity) to the mean similarity
    preference = np.mean(rho)

    # Perform affinity propagation clustering
    af = AffinityPropagation(
        affinity='precomputed',
        preference=preference,
        random_state=None
    )

    # Fit and predict clusters
    cluster_labels = af.fit_predict(rho)

    # Get cluster centers (exemplars)
    cluster_centers = af.cluster_centers_indices_

    # Sort features by cluster labels to group similar variables
    ordering = np.argsort(cluster_labels)

    return {
        'ordering': ordering,
        'clusters': cluster_labels,
        'cluster_centers': cluster_centers
    }


def cluster_using_distance(
    data: np.ndarray,
    distance: str = "euclidean"
) -> Dict[str, np.ndarray]:
    """
    Order features based on a distance metric using affinity propagation clustering.

    This function clusters features with strong similarity together based on
    the specified distance metric. Ordering helps reduce cluttering in parallel
    coordinate displays, improving visualization.

    Parameters
    ----------
    data : np.ndarray
        Data matrix where rows are observations and columns are features/variables.
    distance : str, default="euclidean"
        Distance metric to use. Can be any metric supported by scipy.spatial.distance.pdist
        Examples: 'euclidean', 'correlation', 'cosine', 'cityblock', 'hamming', etc.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'ordering': New ordering of features where subsequent features are more related
        - 'clusters': Original clustering labels from affinity propagation
        - 'cluster_centers': Indices of cluster exemplars

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 10)
    >>> result = cluster_using_distance(data, distance='euclidean')
    >>> reordered_data = data[:, result['ordering']]

    Notes
    -----
    - Uses affinity propagation algorithm for clustering
    - The preference parameter (self-similarity) is set to the mean of all similarities
    - Features are ordered by their assigned cluster
    - This function is equivalent to cluster_using_correlation but emphasizes
      the distance parameter
    - Original version: MATEDA 2.0 (8/26/2008)

    See Also
    --------
    cluster_using_correlation : Similar function with correlation emphasis
    """
    n_edges = data.shape[1]

    # Compute pairwise distances between features (transpose to compare columns)
    y = pdist(data.T, distance)

    # Convert distances to similarities (affinity propagation maximizes similarity)
    # Subtract from max to convert distance to similarity
    rho = np.max(y) - squareform(y)

    # Set preference (self-similarity) to the mean similarity
    preference = np.mean(rho)

    # Perform affinity propagation clustering
    af = AffinityPropagation(
        affinity='precomputed',
        preference=preference,
        random_state=None
    )

    # Fit and predict clusters
    cluster_labels = af.fit_predict(rho)

    # Get cluster centers (exemplars)
    cluster_centers = af.cluster_centers_indices_

    # Sort features by cluster labels to group similar variables
    ordering = np.argsort(cluster_labels)

    return {
        'ordering': ordering,
        'clusters': cluster_labels,
        'cluster_centers': cluster_centers
    }
