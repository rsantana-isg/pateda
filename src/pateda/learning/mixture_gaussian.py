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

from pateda.core.components import LearningMethod
from pateda.core.models import MixtureModel
from pateda.learning.utils.weights import normalize_probabilities


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
    p = normalize_probabilities(params.get('p'), len(population))

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

    # Perform k-means clustering (sample_weight scales centroid contributions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if p is not None:
        labels = kmeans.fit_predict(cluster_data, sample_weight=p)
    else:
        labels = kmeans.fit_predict(cluster_data)

    # Learn Gaussian model for each cluster
    components = []
    pop_size = len(population)

    for i in range(n_clusters):
        mask = labels == i
        cluster_pop = population[mask]

        if len(cluster_pop) > 1:
            if p is not None:
                cluster_p = p[mask] / p[mask].sum()
                means = np.average(cluster_pop, weights=cluster_p, axis=0)
                stds = np.sqrt(np.average((cluster_pop - means) ** 2, weights=cluster_p, axis=0))
            else:
                means = np.mean(cluster_pop, axis=0)
                stds = np.std(cluster_pop, axis=0)
        else:
            # Fallback to overall statistics for small clusters
            if p is not None:
                means = np.average(population, weights=p, axis=0)
                stds = np.sqrt(np.average((population - means) ** 2, weights=p, axis=0))
            else:
                means = np.mean(population, axis=0)
                stds = np.std(population, axis=0)

        # Prevent zero standard deviation
        stds = np.maximum(stds, 1e-10)

        weight = float(p[mask].sum()) if p is not None else np.sum(mask) / pop_size

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
    p = normalize_probabilities(params.get('p'), len(population))

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

    # Perform k-means clustering (sample_weight scales centroid contributions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if p is not None:
        labels = kmeans.fit_predict(cluster_data, sample_weight=p)
    else:
        labels = kmeans.fit_predict(cluster_data)

    # Learn Gaussian model for each cluster
    components = []
    pop_size = len(population)
    n_vars = population.shape[1]

    for i in range(n_clusters):
        mask = labels == i
        cluster_pop = population[mask]

        if len(cluster_pop) > 1:
            if p is not None:
                cluster_p = p[mask] / p[mask].sum()
                mean = np.average(cluster_pop, weights=cluster_p, axis=0)
                cov = np.cov(cluster_pop, rowvar=False, aweights=cluster_p)
            else:
                mean = np.mean(cluster_pop, axis=0)
                cov = np.cov(cluster_pop, rowvar=False)
        else:
            # Fallback to overall statistics for small clusters
            if p is not None:
                mean = np.average(population, weights=p, axis=0)
                cov = np.cov(population, rowvar=False, aweights=p)
            else:
                mean = np.mean(population, axis=0)
                cov = np.cov(population, rowvar=False)

        # Ensure positive definiteness
        cov += np.eye(n_vars) * 1e-6

        weight = float(p[mask].sum()) if p is not None else np.sum(mask) / pop_size

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
    p = normalize_probabilities(params.get('p'), len(population))

    # Fit Gaussian Mixture using EM
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state
    )
    if p is not None:
        # sklearn GaussianMixture does not accept sample_weight; use largest-remainder
        # integer resampling so each individual appears proportional to its weight.
        # For uniform p this reproduces the original population exactly.
        n = len(population)
        quotients, remainders = np.divmod(p * n, 1)
        counts = quotients.astype(int)
        deficit = n - int(counts.sum())
        if deficit > 0:
            top_k = np.argsort(remainders)[::-1][:deficit]
            counts[top_k] += 1
        resampled = np.repeat(population, counts, axis=0)
        gm.fit(resampled)
    else:
        gm.fit(population)

    return {
        'gm_model': gm,
        'n_components': n_components,
        'type': 'mixture_gaussian_em'
    }


# ===================================================================
# Class-based wrappers for component architecture
# ===================================================================


class LearnMixtureGaussian(LearningMethod):
    """
    Class-based wrapper for mixture of Gaussians learning.

    Uses k-means clustering to partition the population and learns
    a Gaussian model for each cluster.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components (default: 3)
    what_to_cluster : str
        What to use for clustering: 'vars', 'objs', or 'vars_and_objs' (default: 'vars')
    normalize : bool
        Whether to normalize data before clustering (default: True)
    covariance_type : str
        'univariate' for independent variables or 'full' for multivariate (default: 'univariate')
    """

    def __init__(
        self,
        n_clusters: int = 3,
        what_to_cluster: str = 'vars',
        normalize: bool = True,
        covariance_type: str = 'univariate'
    ):
        """
        Initialize mixture Gaussian learning

        Args:
            n_clusters: Number of mixture components
            what_to_cluster: Clustering target ('vars', 'objs', or 'vars_and_objs')
            normalize: Whether to normalize before clustering
            covariance_type: 'univariate' or 'full'
        """
        self.n_clusters = n_clusters
        self.what_to_cluster = what_to_cluster
        self.normalize = normalize
        self.covariance_type = covariance_type

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> MixtureModel:
        """
        Learn mixture of Gaussians model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable bounds (2, n_vars) array with [lower, upper] bounds
            population: Selected population to learn from
            fitness: Fitness values (used for clustering if what_to_cluster != 'vars')
            **params: Additional parameters

        Returns:
            Learned MixtureModel with Gaussian components
        """
        learning_params = {
            'n_clusters': self.n_clusters,
            'what_to_cluster': self.what_to_cluster,
            'normalize': self.normalize,
            'p': params.get('p', None),
        }

        # Use the appropriate functional learning method
        if self.covariance_type == 'univariate':
            model_dict = learn_mixture_gaussian_univariate(population, fitness, learning_params)
        else:
            model_dict = learn_mixture_gaussian_full(population, fitness, learning_params)

        # Convert to MixtureModel
        components = model_dict['components']
        component_structures = [None] * len(components)  # No structure for Gaussian components

        return MixtureModel(
            structure=component_structures,
            parameters={
                'components': components,
                'n_clusters': model_dict['n_clusters'],
                'type': model_dict['type']
            },
            metadata={
                'generation': generation,
                'model_type': f'Mixture Gaussian ({self.covariance_type})',
                'n_clusters': self.n_clusters,
            }
        )
