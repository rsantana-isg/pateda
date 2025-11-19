"""
Gaussian Markov Random Field (GMRF) EDA Learning

This module implements GMRF-EDA, a hybrid algorithm that combines:
1. Regularized regression for learning variable dependencies
2. Affinity propagation for clustering variables into cliques
3. Multivariate Gaussian estimation for each clique

This creates a Marginal Product Model (MPM) where cliques don't overlap
and each clique models a subset of interacting variables.

References:
- Karshenas, H., Santana, R., Bielza, C., & Larrañaga, P. (2012).
  "Continuous Estimation of Distribution Algorithms Based on Factorized
  Gaussian Markov Networks." In Markov Networks in Evolutionary
  Computation (pp. 157-173). Springer.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import Lasso, ElasticNet, Lars, LassoLars


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
    Learn GMRF-EDA using LASSO (L1) regularization.

    Convenience wrapper for learn_gmrf_eda with regularization='lasso'.

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
    params['regularization'] = 'lars'
    return learn_gmrf_eda(population, fitness, params)
