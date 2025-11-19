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
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso, ElasticNet, Lars, LassoLars
from scipy import linalg


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


def learn_weighted_gaussian_univariate(
def _compute_regularized_weights(
    population: np.ndarray,
    regularization: str = 'lasso',
    alpha: float = 0.01,
    l1_ratio: float = 0.5
) -> np.ndarray:
    """
    Compute regularized dependency weights between variables.

    For each variable Xi, learns a regularized linear regression model
    predicting Xi from all other variables. The regression coefficients
    represent the strength of dependencies.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars)
    regularization : str
        Type of regularization: 'lasso', 'elasticnet', 'lars', or 'lassolars'
    alpha : float
        Regularization parameter (strength of penalty)
    l1_ratio : float
        For elasticnet: mixing parameter between L1 and L2 (default: 0.5)

    Returns
    -------
    weights : np.ndarray
        Matrix of shape (n_vars, n_vars) where weights[i, j] is the
        influence of variable j on variable i
    """
    n_samples, n_vars = population.shape
    weights = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        # Target variable
        y = population[:, i]

        # Predictor variables (all except i)
        X = np.delete(population, i, axis=1)

        # Learn regularized regression model
        if regularization == 'lasso':
            model = Lasso(alpha=alpha, max_iter=2000, tol=1e-4)
        elif regularization == 'elasticnet':
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, tol=1e-4)
        elif regularization == 'lars':
            model = Lars(n_nonzero_coefs=min(n_vars, n_samples // 2))
        elif regularization == 'lassolars':
            model = LassoLars(alpha=alpha, max_iter=2000)
        else:
            raise ValueError(f"Unknown regularization method: {regularization}")

        # Fit model
        try:
            model.fit(X, y)
            coefs = model.coef_
        except Exception:
            # If fitting fails, use zero weights
            coefs = np.zeros(n_vars - 1)

        # Insert coefficients into weight matrix
        # (accounting for the removed column)
        weights[i, :i] = coefs[:i]
        if i < n_vars - 1:
            weights[i, (i + 1):] = coefs[i:]

    return weights


def _cluster_variables_affinity_propagation(
    weights: np.ndarray,
    preference: Optional[float] = None,
    damping: float = 0.5,
    max_iter: int = 200
) -> List[List[int]]:
    """
    Cluster variables into disjoint groups using affinity propagation.

    Parameters
    ----------
    weights : np.ndarray
        Dependency weight matrix of shape (n_vars, n_vars)
    preference : float, optional
        Preference for each point to become an exemplar
        (default: median of similarity values)
    damping : float
        Damping factor in [0.5, 1) to avoid oscillations
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    cliques : list of lists
        List of variable index groups (cliques)
    """
    n_vars = weights.shape[0]

    # Compute similarity matrix from weights
    # Use absolute values and symmetrize
    similarity = np.abs(weights) + np.abs(weights.T)

    # Convert to negative distances (affinity propagation expects similarities)
    # We negate because AP maximizes similarities
    similarity = -similarity

    # Set diagonal to preference (self-similarity)
    if preference is None:
        preference = np.median(similarity[similarity != 0])

    np.fill_diagonal(similarity, preference)

    # Apply affinity propagation
    try:
        ap = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            preference=preference,
            affinity='precomputed',
            random_state=42
        )
        labels = ap.fit_predict(similarity)

        # Group variables by cluster labels
        n_clusters = len(set(labels))
        cliques = [[] for _ in range(n_clusters)]
        for var_idx, cluster_idx in enumerate(labels):
            cliques[cluster_idx].append(var_idx)

    except Exception:
        # If AP fails, fall back to treating each variable independently
        cliques = [[i] for i in range(n_vars)]

    # Remove empty cliques
    cliques = [c for c in cliques if len(c) > 0]

    return cliques


def learn_gmrf_eda(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a weighted univariate Gaussian model with proportional selection.

    Uses fitness-based exponential weighting instead of truncation selection.
    Solutions with better fitness contribute more to the learned model.
    Learn a Gaussian Markov Random Field (GMRF) using regularized estimation.

    Implements the hybrid GMRF-EDA algorithm from Karshenas et al. (2012):
    1. Compute regularized regression weights for variable dependencies
    2. Cluster variables into disjoint cliques using affinity propagation
    3. Estimate a multivariate Gaussian distribution for each clique

    This creates a Marginal Product Model (MPM) where cliques don't overlap
    and each clique models a subset of interacting variables.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values used for weighting (minimization assumed)
    params : dict, optional
        Additional parameters:
        - 'alpha': regularization for weight computation (default: 1e-10)
        - 'beta': scaling factor for exponential weights (default: 0.1)
        Fitness values (not used in this learning method)
    params : dict, optional
        Additional parameters:
        - 'regularization': Type of regularization ('lasso', 'elasticnet',
          'lars', 'lassolars') (default: 'lasso')
        - 'alpha': Regularization strength (default: 0.01)
        - 'l1_ratio': For elasticnet, mixing between L1 and L2 (default: 0.5)
        - 'preference': Affinity propagation preference (default: median)
        - 'damping': Affinity propagation damping factor (default: 0.5)
        - 'max_iter': Maximum iterations for AP (default: 200)
        - 'min_clique_size': Minimum variables per clique (default: 1)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'means': weighted mean for each variable
        - 'stds': weighted standard deviation for each variable
        - 'type': 'weighted_gaussian_univariate'
        - 'cliques': List of variable index lists for each clique
        - 'clique_models': List of dicts with 'mean' and 'cov' for each clique
        - 'weights': Dependency weight matrix (for analysis)
        - 'type': 'gmrf_eda'

    References
    ----------
    Karshenas, H., Santana, R., Bielza, C., & Larrañaga, P. (2012).
    Continuous Estimation of Distribution Algorithms Based on Factorized
    Gaussian Markov Networks. In Markov Networks in Evolutionary
    Computation (pp. 157-173). Springer.
    """
    if params is None:
        params = {}

    alpha = params.get('alpha', 1e-10)
    beta = params.get('beta', 0.1)

    # Compute exponentially scaled fitness weights
    # Normalize fitness to [0, 1] range
    weights = (fitness - np.min(fitness) + alpha) / (np.max(fitness) - np.min(fitness) + alpha)
    # Apply exponential scaling
    weights = np.exp(beta * weights)
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    # Compute weighted mean
    weighted_means = np.sum(weights.reshape(-1, 1) * population, axis=0)

    # Compute weighted variance
    weighted_vars = np.sum(weights.reshape(-1, 1) * (population - weighted_means)**2, axis=0)
    weighted_stds = np.sqrt(weighted_vars)

    # Prevent zero standard deviation
    weighted_stds = np.maximum(weighted_stds, 1e-10)

    return {
        'means': weighted_means,
        'stds': weighted_stds,
        'type': 'weighted_gaussian_univariate'
    }


def learn_weighted_gaussian_full(
    n_samples, n_vars = population.shape

    # Extract parameters
    regularization = params.get('regularization', 'lasso')
    alpha = params.get('alpha', 0.01)
    l1_ratio = params.get('l1_ratio', 0.5)
    preference = params.get('preference', None)
    damping = params.get('damping', 0.5)
    max_iter = params.get('max_iter', 200)
    min_clique_size = params.get('min_clique_size', 1)

    # Step 1: Compute regularized dependency weights
    weights = _compute_regularized_weights(
        population,
        regularization=regularization,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    # Step 2: Cluster variables using affinity propagation
    cliques = _cluster_variables_affinity_propagation(
        weights,
        preference=preference,
        damping=damping,
        max_iter=max_iter
    )

    # Filter out small cliques if requested
    if min_clique_size > 1:
        cliques = [c for c in cliques if len(c) >= min_clique_size]

    # Step 3: Estimate MGD for each clique
    clique_models = []

    for clique in cliques:
        clique_pop = population[:, clique]

        if len(clique) == 1:
            # Univariate case
            mean = np.mean(clique_pop, axis=0)
            var = np.var(clique_pop, axis=0)
            # Ensure positive variance
            var = max(var, 1e-6)
            cov = np.array([[var]])
        else:
            # Multivariate case
            mean = np.mean(clique_pop, axis=0)
            cov = np.cov(clique_pop, rowvar=False)

            # Ensure positive definiteness
            cov += np.eye(len(clique)) * 1e-6

        clique_models.append({
            'mean': mean,
            'cov': cov
        })

    return {
        'cliques': cliques,
        'clique_models': clique_models,
        'weights': weights,
        'type': 'gmrf_eda'
    }


def learn_gmrf_eda_lasso(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a weighted full multivariate Gaussian model with proportional selection.

    Uses fitness-based exponential weighting instead of truncation selection.
    Solutions with better fitness contribute more to the learned model.
    Learn GMRF-EDA using LASSO (L1) regularization.

    Convenience wrapper for learn_gmrf_eda with regularization='lasso'.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values used for weighting (minimization assumed)
    params : dict, optional
        Additional parameters:
        - 'alpha': regularization for weight computation (default: 1e-10)
        - 'beta': scaling factor for exponential weights (default: 0.1)
        Population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Parameters (see learn_gmrf_eda)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'mean': weighted mean vector
        - 'cov': weighted covariance matrix
        - 'type': 'weighted_gaussian_full'
    """
    if params is None:
        params = {}

    alpha = params.get('alpha', 1e-10)
    beta = params.get('beta', 0.1)

    # Compute exponentially scaled fitness weights
    weights = (fitness - np.min(fitness) + alpha) / (np.max(fitness) - np.min(fitness) + alpha)
    weights = np.exp(beta * weights)
    weights = weights / np.sum(weights)

    # Compute weighted mean
    weighted_mean = np.sum(weights.reshape(-1, 1) * population, axis=0)

    # Compute weighted covariance
    centered_pop = population - weighted_mean
    weighted_cov = np.dot((weights.reshape(-1, 1) * centered_pop).T, centered_pop)

    # Ensure positive definiteness
    n_vars = population.shape[1]
    weighted_cov += np.eye(n_vars) * 1e-6

    return {
        'mean': weighted_mean,
        'cov': weighted_cov,
        'type': 'weighted_gaussian_full'
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
        GMRF-EDA model
    """
    if params is None:
        params = {}
    params['regularization'] = 'lasso'
    return learn_gmrf_eda(population, fitness, params)


def learn_gmrf_eda_elasticnet(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn GMRF-EDA using Elastic Net (L1 + L2) regularization.

    Convenience wrapper for learn_gmrf_eda with regularization='elasticnet'.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Parameters (see learn_gmrf_eda)

    Returns
    -------
    model : dict
        GMRF-EDA model
    """
    if params is None:
        params = {}
    params['regularization'] = 'elasticnet'
    return learn_gmrf_eda(population, fitness, params)


def learn_gmrf_eda_lars(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn GMRF-EDA using LARS (Least Angle Regression).

    Convenience wrapper for learn_gmrf_eda with regularization='lars'.

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
        Population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Parameters (see learn_gmrf_eda)

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
        GMRF-EDA model
    """
    if params is None:
        params = {}
    params['regularization'] = 'lars'
    return learn_gmrf_eda(population, fitness, params)
