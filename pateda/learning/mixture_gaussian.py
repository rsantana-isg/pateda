"""
Mixture Gaussian Model Learning for Continuous EDAs

This module provides learning algorithms for mixture of Gaussian models,
which can model multimodal distributions useful for problems with multiple
optima or clusters.

Mixture of Gaussians:
- Weighted sum of multiple Gaussian components
- Can model multimodal distributions
- Each component can be univariate or multivariate
- Useful for problems with multiple optima or clusters

Learning Methods:
1. K-means based clustering: Fast, good for well-separated clusters
2. EM algorithm: More principled, handles overlapping clusters better

References:
- Bosman, P.A.N., & Thierens, D. (2000). "Expanding from discrete to continuous
  estimation of distribution algorithms: The IDEA." Parallel Problem Solving from
  Nature PPSN VI, pp. 767-776.
"""

import numpy as np
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


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


def learn_mixture_gaussian_em(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Learn a Gaussian mixture model using Expectation-Maximization algorithm.

    Uses sklearn's GaussianMixture which implements EM algorithm for
    more principled mixture learning compared to k-means clustering.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used but kept for API consistency)
    params : dict
        Parameters containing:
        - 'n_components': number of mixture components
        - 'covariance_type': 'full', 'tied', 'diag', or 'spherical' (default: 'full')
        - 'max_iter': maximum EM iterations (default: 100)
        - 'random_state': random seed (default: 42)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'gm_model': trained sklearn GaussianMixture object
        - 'n_components': number of components
        - 'type': 'mixture_gaussian_em'
    """
    n_components = params.get('n_components', 3)
    covariance_type = params.get('covariance_type', 'full')
    max_iter = params.get('max_iter', 100)
    random_state = params.get('random_state', 42)

    # Fit Gaussian Mixture using EM
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state
    )
    gm.fit(population)

    return {
        'gm_model': gm,
        'n_components': n_components,
        'type': 'mixture_gaussian_em'
    }
