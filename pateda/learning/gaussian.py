"""
Gaussian Model Learning for Continuous EDAs

This module provides learning algorithms for various Gaussian-based probabilistic
models used in continuous optimization. Gaussian networks extend the concepts of
discrete EDAs (like Bayesian networks) to continuous domains by using Gaussian
distributions instead of discrete probability tables.

Gaussian Networks:
A Gaussian network is a Bayesian network where all variables are continuous and
the conditional distributions are Gaussian (normal). The joint distribution of
a Gaussian network is a multivariate Gaussian (or Gaussian distribution).

Types of Gaussian Models:
1. Univariate Gaussian (Gaussian UMDA):
   - Each variable is independent with its own mean and variance
   - Joint distribution: p(x) = ∏ᵢ N(xᵢ | μᵢ, σᵢ²)
   - Simplest model, assumes no variable dependencies
   - Equivalent to continuous UMDA

2. Full Multivariate Gaussian:
   - Models all pairwise correlations via full covariance matrix
   - Joint distribution: p(x) = N(x | μ, Σ)
   - Can capture all linear dependencies
   - Requires O(n²) parameters for n variables

3. Structured Gaussian Networks:
   - Use Bayesian network structure to specify conditional independencies
   - More parameter-efficient than full covariance
   - Each variable has Gaussian conditional distribution given parents
   - Can model sparse dependency structures

4. Mixture of Gaussians:
   - Weighted sum of multiple Gaussian components
   - Can model multimodal distributions
   - Each component can be univariate or multivariate
   - Useful for problems with multiple optima or clusters

Mathematical Foundations:
- Univariate: X ~ N(μ, σ²) with density f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
- Multivariate: X ~ N(μ, Σ) with density f(x) = (1/√((2π)ⁿ|Σ|)) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
- Conditional: If X|Y ~ N(μ + Σ_XY Σ_YY⁻¹(y-μ_Y), Σ_XX - Σ_XY Σ_YY⁻¹ Σ_YX)

Parameter Estimation:
- Maximum Likelihood Estimation (MLE):
  * Mean: μ̂ = (1/N) ∑ᵢ xᵢ
  * Covariance: Σ̂ = (1/N) ∑ᵢ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ
- Regularization often needed to prevent singular covariance matrices
- Can add small diagonal term (ridge regularization) for numerical stability

Advantages for Continuous Optimization:
- Natural representation for continuous variables
- Efficient parameter estimation (closed-form MLE)
- Sampling is straightforward (multivariate normal sampling)
- Can capture correlations and rotations in search space
- Well-studied theoretical properties

Challenges:
- Limited to modeling linear dependencies (correlations)
- Full covariance requires many samples to estimate reliably (O(n²) parameters)
- May struggle with highly non-linear or non-convex landscapes
- Gaussian assumption may not fit actual distribution of good solutions

Related Algorithms:
- EMNA (Estimation of Multivariate Normal Algorithm): Uses full Gaussian
- IDEA (Iterated Density-Estimation Evolutionary Algorithm): Gaussian mixtures
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy (related approach)

Equivalent to MATEDA's Gaussian learning functions (Section 4.2.3)

References:
- Larrañaga, P., & Lozano, J.A. (Eds.). (2002). "Estimation of Distribution
  Algorithms: A New Tool for Evolutionary Computation." Kluwer Academic Publishers.
- Bosman, P.A.N., & Thierens, D. (2000). "Expanding from discrete to continuous
  estimation of distribution algorithms: The IDEA." Parallel Problem Solving from
  Nature PPSN VI, pp. 767-776.
- MATEDA-2.0 User Guide, Section 4.2.3: "Gaussian network based factorizations"
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
