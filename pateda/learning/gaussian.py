"""
Gaussian Model Learning for Continuous EDAs

This module provides learning algorithms for various Gaussian-based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def learn_gaussian_univariate(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a univariate Gaussian model (independent variables).

    This corresponds to Gaussian UMDA, where each variable is modeled
    independently with its own mean and standard deviation.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in this learning method)
    params : dict, optional
        Additional parameters (not used for this method)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'means': array of means for each variable
        - 'stds': array of standard deviations for each variable
        - 'type': 'gaussian_univariate'
    """
    means = np.mean(population, axis=0)
    stds = np.std(population, axis=0)

    # Prevent zero standard deviation
    stds = np.maximum(stds, 1e-10)

    return {
        'means': means,
        'stds': stds,
        'type': 'gaussian_univariate'
    }


def learn_gaussian_full(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a full multivariate Gaussian model.

    Models dependencies between all variables using a full covariance matrix.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in this learning method)
    params : dict, optional
        Additional parameters (not used for this method)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'mean': mean vector
        - 'cov': covariance matrix
        - 'type': 'gaussian_full'
    """
    mean = np.mean(population, axis=0)
    cov = np.cov(population, rowvar=False)

    # Ensure positive definiteness by adding small regularization
    n_vars = population.shape[1]
    cov += np.eye(n_vars) * 1e-6

    return {
        'mean': mean,
        'cov': cov,
        'type': 'gaussian_full'
    }


def learn_mixture_gaussian_univariate(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Learn a mixture of univariate Gaussian models using k-means clustering.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values for clustering
    params : dict
        Parameters containing:
        - 'n_clusters': number of mixture components
        - 'what_to_cluster': 'vars', 'objs', or 'vars_and_objs'
        - 'normalize': whether to normalize before clustering
        - 'distance': distance metric for clustering (default: 'euclidean')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'components': list of dicts with 'means', 'stds', 'weight' for each component
        - 'n_clusters': number of components
        - 'type': 'mixture_gaussian_univariate'
    """
    n_clusters = params.get('n_clusters', 3)
    what_to_cluster = params.get('what_to_cluster', 'vars')
    normalize = params.get('normalize', True)

    # Prepare data for clustering
    if what_to_cluster == 'vars':
        cluster_data = population.copy()
    elif what_to_cluster == 'objs':
        cluster_data = fitness.reshape(-1, 1) if fitness.ndim == 1 else fitness
    elif what_to_cluster == 'vars_and_objs':
        fitness_2d = fitness.reshape(-1, 1) if fitness.ndim == 1 else fitness
        cluster_data = np.hstack([population, fitness_2d])
    else:
        raise ValueError(f"Unknown clustering target: {what_to_cluster}")

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(cluster_data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_data)

    # Learn Gaussian model for each cluster
    components = []
    pop_size = len(population)

    for i in range(n_clusters):
        mask = labels == i
        cluster_pop = population[mask]

        if len(cluster_pop) > 1:
            means = np.mean(cluster_pop, axis=0)
            stds = np.std(cluster_pop, axis=0)
        else:
            # Fallback to overall statistics for small clusters
            means = np.mean(population, axis=0)
            stds = np.std(population, axis=0)

        # Prevent zero standard deviation
        stds = np.maximum(stds, 1e-10)

        weight = np.sum(mask) / pop_size

        components.append({
            'means': means,
            'stds': stds,
            'weight': weight
        })

    return {
        'components': components,
        'n_clusters': n_clusters,
        'type': 'mixture_gaussian_univariate'
    }


def learn_mixture_gaussian_full(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Learn a mixture of full multivariate Gaussian models using k-means clustering.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values for clustering
    params : dict
        Parameters containing:
        - 'n_clusters': number of mixture components
        - 'what_to_cluster': 'vars', 'objs', or 'vars_and_objs'
        - 'normalize': whether to normalize before clustering
        - 'distance': distance metric for clustering (default: 'euclidean')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'components': list of dicts with 'mean', 'cov', 'weight' for each component
        - 'n_clusters': number of components
        - 'type': 'mixture_gaussian_full'
    """
    n_clusters = params.get('n_clusters', 3)
    what_to_cluster = params.get('what_to_cluster', 'vars')
    normalize = params.get('normalize', True)

    # Prepare data for clustering
    if what_to_cluster == 'vars':
        cluster_data = population.copy()
    elif what_to_cluster == 'objs':
        cluster_data = fitness.reshape(-1, 1) if fitness.ndim == 1 else fitness
    elif what_to_cluster == 'vars_and_objs':
        fitness_2d = fitness.reshape(-1, 1) if fitness.ndim == 1 else fitness
        cluster_data = np.hstack([population, fitness_2d])
    else:
        raise ValueError(f"Unknown clustering target: {what_to_cluster}")

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(cluster_data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_data)

    # Learn Gaussian model for each cluster
    components = []
    pop_size = len(population)
    n_vars = population.shape[1]

    for i in range(n_clusters):
        mask = labels == i
        cluster_pop = population[mask]

        if len(cluster_pop) > 1:
            mean = np.mean(cluster_pop, axis=0)
            cov = np.cov(cluster_pop, rowvar=False)
        else:
            # Fallback to overall statistics for small clusters
            mean = np.mean(population, axis=0)
            cov = np.cov(population, rowvar=False)

        # Ensure positive definiteness
        cov += np.eye(n_vars) * 1e-6

        weight = np.sum(mask) / pop_size

        components.append({
            'mean': mean,
            'cov': cov,
            'weight': weight
        })

    return {
        'components': components,
        'n_clusters': n_clusters,
        'type': 'mixture_gaussian_full'
    }
