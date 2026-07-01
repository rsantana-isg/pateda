"""
Extraction of interaction networks from the Gaussian models learned by
continuous EDAs.

For a multivariate Gaussian ``x ~ N(mu, Sigma)`` the **inverse covariance**
(precision) matrix ``Theta = Sigma^{-1}`` encodes the conditional-independence
structure of the variables: ``Theta_{ij} = 0`` iff ``x_i`` and ``x_j`` are
conditionally independent given the remaining variables.  The support of
``Theta`` therefore defines the *Gaussian graphical model* (GGM) — an undirected
interaction network that is the continuous analogue of the (directed) Bayesian
network learned by discrete EDAs.  This module extracts that network so that it
can be analysed with :mod:`pateda.knowledge_extraction.network_measures` and
**combined** with networks extracted from Bayesian networks.

References
----------
* A. S. Sundaramoorthy, S. K. Varanasi, B. Huang, et al.,
  "Sparse Inverse Covariance Estimation for Causal Inference in Process Data
  Analytics", IEEE Transactions on Control Systems Technology, 30(3):1268-1280,
  2022.  (Proposition 1: ``Theta_{ij}=0`` ⇔ conditional independence; Section
  II-C: likelihood-score orientation of the undirected GGM into a causal graph.)
* M. Drton, M. H. Maathuis, "Structure Learning in Graphical Modeling",
  Annual Review of Statistics and Its Application, 2017.
* J. Friedman, T. Hastie, R. Tibshirani, "Sparse inverse covariance estimation
  with the graphical lasso", Biostatistics, 9(3):432-441, 2008.
* R. Santana, R. Armañanzas, C. Bielza, P. Larrañaga, "Network measures for
  information extraction in evolutionary algorithms", IJCIS 6(6), 2013
  (the network-measure framework these networks feed into).

Author: Roberto Santana (roberto.santana@ehu.eus)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.covariance import graphical_lasso as _sk_graphical_lasso
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is a declared dependency
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Reading Gaussian parameters from a learned model
# ---------------------------------------------------------------------------

def extract_gaussian_parameters(model: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Robustly read ``(mean, covariance)`` from a learned continuous model.

    Supports the pateda ``GaussianModel`` (whose ``parameters`` dict uses keys
    ``mean``/``means`` and ``cov``/``covariance``/``covariances``), GMRF-EDA
    dicts (``clique_models`` are assembled into a block-diagonal covariance),
    plain dicts, and objects exposing ``mean`` / ``covariance`` attributes.
    Returns ``(None, None)`` when no Gaussian parameters can be found.
    """
    params = getattr(model, "parameters", model)

    def _get(d, *keys):
        if isinstance(d, dict):
            for k in keys:
                if k in d and d[k] is not None:
                    return np.asarray(d[k], dtype=float)
        return None

    mean = _get(params, "mean", "means", "mu")
    cov = _get(params, "cov", "covariance", "covariances", "sigma")

    # Object-attribute fallback.
    if cov is None:
        for attr in ("covariance", "cov", "sigma"):
            if hasattr(model, attr) and getattr(model, attr) is not None:
                cov = np.asarray(getattr(model, attr), dtype=float)
                break
    if mean is None:
        for attr in ("mean", "mu"):
            if hasattr(model, attr) and getattr(model, attr) is not None:
                mean = np.asarray(getattr(model, attr), dtype=float)
                break

    # GMRF-EDA: block-diagonal covariance from per-clique Gaussians.
    if cov is None and isinstance(params, dict) and "clique_models" in params and "cliques" in params:
        cliques = params["cliques"]
        clique_models = params["clique_models"]
        n = int(max(max(c) for c in cliques)) + 1
        cov = np.zeros((n, n))
        mean = np.zeros(n)
        for clique, cm in zip(cliques, clique_models):
            idx = np.asarray(clique, dtype=int)
            cov[np.ix_(idx, idx)] = np.asarray(cm["cov"], dtype=float)
            mean[idx] = np.asarray(cm["mean"], dtype=float)

    # A 1-D ``sigma`` is interpreted as standard deviations (diagonal model).
    if cov is not None and cov.ndim == 1:
        cov = np.diag(cov ** 2)

    return mean, cov


# ---------------------------------------------------------------------------
# Covariance -> precision -> partial correlations
# ---------------------------------------------------------------------------

def covariance_to_precision(cov: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """Invert a covariance matrix to the precision matrix ``Theta = Sigma^{-1}``.

    A small ridge is added to the diagonal for numerical stability (pseudo-inverse
    is used as a last resort).
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    reg = cov + np.eye(n) * regularization
    try:
        return np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(reg)


def partial_correlation_matrix(precision: np.ndarray) -> np.ndarray:
    """Partial-correlation matrix from a precision matrix.

    ``rho_{ij} = -Theta_{ij} / sqrt(Theta_{ii} Theta_{jj})`` — the strength of the
    conditional (direct) dependence between ``i`` and ``j`` given the rest.
    """
    precision = np.asarray(precision, dtype=float)
    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    pc = -precision / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def glasso_precision(
    cov: np.ndarray, alpha: float = 0.05, max_iter: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse precision matrix via the graphical lasso (Friedman et al. 2008).

    Operates directly on an (empirical) covariance matrix.  Returns
    ``(covariance, precision)``.  Falls back to a plain inverse if sklearn is
    unavailable or the solver fails.
    """
    cov = np.asarray(cov, dtype=float)
    if not _HAS_SKLEARN:
        return cov, covariance_to_precision(cov)
    # graphical_lasso needs a well-scaled, positive-definite input.
    scaled = cov + np.eye(cov.shape[0]) * 1e-4
    try:
        cov_, prec_ = _sk_graphical_lasso(scaled, alpha=alpha, max_iter=max_iter)
        return cov_, prec_
    except Exception:
        return cov, covariance_to_precision(cov)


# ---------------------------------------------------------------------------
# Gaussian interaction network
# ---------------------------------------------------------------------------

def gaussian_interaction_network(
    model_or_cov: Any,
    method: str = "partial_correlation",
    threshold: float = 0.1,
    alpha: float = 0.05,
    regularization: float = 1e-6,
) -> Dict[str, Any]:
    """
    Extract the undirected Gaussian interaction network from a learned model.

    Following the Gaussian-graphical-model interpretation of the inverse
    covariance matrix (Sundaramoorthy et al. 2022, Prop. 1), an edge ``i—j`` is
    placed whenever the conditional dependence between ``i`` and ``j`` is
    non-negligible.

    Parameters
    ----------
    model_or_cov : Any
        A learned model (e.g. ``GaussianModel``) or a covariance matrix.
    method : str
        * ``'partial_correlation'`` — threshold ``|rho_{ij}|`` from the precision
          matrix (conditional dependencies).
        * ``'precision'`` — connect non-zero entries of the precision matrix.
        * ``'glasso'`` — sparse precision via the graphical lasso, then connect
          its non-zero entries.
        * ``'correlation'`` — threshold the marginal correlation matrix.
    threshold : float
        Edge threshold for the correlation / partial-correlation methods.
    alpha : float
        Regularization strength for the graphical lasso.
    regularization : float
        Ridge added before inverting the covariance.

    Returns
    -------
    dict
        ``adjacency`` (symmetric 0/1), ``weights`` (signed edge strengths),
        ``precision``, ``partial_correlation``, ``correlation``, ``method``,
        ``n_edges`` and ``directed = False``.
    """
    if isinstance(model_or_cov, np.ndarray) and model_or_cov.ndim == 2 \
            and model_or_cov.shape[0] == model_or_cov.shape[1]:
        cov = np.asarray(model_or_cov, dtype=float)
    else:
        _, cov = extract_gaussian_parameters(model_or_cov)
        if cov is None:
            raise ValueError("Could not extract a covariance matrix from the model")

    n = cov.shape[0]
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    correlation = cov / np.outer(std, std)

    if method == "glasso":
        _, precision = glasso_precision(cov, alpha=alpha)
    else:
        precision = covariance_to_precision(cov, regularization)
    pcorr = partial_correlation_matrix(precision)

    adjacency = np.zeros((n, n), dtype=int)
    weights = np.zeros((n, n))

    if method == "correlation":
        strength = correlation
        for i in range(n):
            for j in range(i + 1, n):
                if abs(strength[i, j]) >= threshold:
                    adjacency[i, j] = adjacency[j, i] = 1
                    weights[i, j] = weights[j, i] = strength[i, j]
    elif method in ("partial_correlation", "precision", "glasso"):
        use_threshold = threshold if method == "partial_correlation" else 1e-8
        for i in range(n):
            for j in range(i + 1, n):
                val = pcorr[i, j]
                connect = (abs(val) >= use_threshold) if method == "partial_correlation" \
                    else (abs(precision[i, j]) > use_threshold)
                if connect:
                    adjacency[i, j] = adjacency[j, i] = 1
                    weights[i, j] = weights[j, i] = val
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return {
        "adjacency": adjacency,
        "weights": weights,
        "precision": precision,
        "partial_correlation": pcorr,
        "correlation": correlation,
        "covariance": cov,
        "method": method,
        "n_edges": int(adjacency.sum() // 2),
        "directed": False,
    }


# ---------------------------------------------------------------------------
# Causal orientation (undirected GGM -> directed graph)
# ---------------------------------------------------------------------------

def orient_edges_likelihood_score(
    adjacency: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """
    Orient the edges of a Gaussian network with the likelihood score.

    For a Gaussian pair, the score of the causal direction ``i -> j`` is
    ``L_{i->j} = log Var(residual of j regressed on i) + log Var(x_i)`` and the
    edge is oriented towards the smaller score (Sundaramoorthy et al. 2022,
    Section II-C, Eq. 15).  For a bivariate Gaussian the residual variance of
    ``j`` given ``i`` is ``Sigma_{jj}(1 - rho_{ij}^2)``.  Producing a *directed*
    graph makes the Gaussian network directly comparable with the directed
    Bayesian networks learned by discrete EDAs.

    Returns a directed 0/1 adjacency matrix.
    """
    adjacency = np.asarray(adjacency)
    cov = np.asarray(cov, dtype=float)
    n = adjacency.shape[0]
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    directed = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0:
                continue
            rho2 = float(np.clip(corr[i, j] ** 2, 0.0, 1.0 - 1e-12))
            var_i, var_j = cov[i, i], cov[j, j]
            score_i_to_j = np.log(var_j * (1 - rho2)) + np.log(var_i)
            score_j_to_i = np.log(var_i * (1 - rho2)) + np.log(var_j)
            if score_i_to_j <= score_j_to_i:
                directed[i, j] = 1
            else:
                directed[j, i] = 1
    return directed


# ---------------------------------------------------------------------------
# Combining Gaussian networks with Bayesian-network (or other) networks
# ---------------------------------------------------------------------------

def compare_networks(adjacency_a: np.ndarray, adjacency_b: np.ndarray) -> Dict[str, Any]:
    """
    Compare two interaction networks on their undirected skeletons.

    Useful for relating a Gaussian-covariance network to a Bayesian-network
    structure (or to a known problem-interaction graph).  Returns the number of
    common / unique edges, the Jaccard similarity and the union / intersection
    adjacency matrices.
    """
    a = (np.asarray(adjacency_a) != 0).astype(int)
    b = (np.asarray(adjacency_b) != 0).astype(int)
    # Work on undirected skeletons.
    a = ((a + a.T) != 0).astype(int)
    b = ((b + b.T) != 0).astype(int)
    n = a.shape[0]
    tri = np.triu_indices(n, k=1)
    ea, eb = a[tri], b[tri]
    common = int(np.sum((ea == 1) & (eb == 1)))
    only_a = int(np.sum((ea == 1) & (eb == 0)))
    only_b = int(np.sum((ea == 0) & (eb == 1)))
    union = common + only_a + only_b
    return {
        "common_edges": common,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "jaccard": (common / union) if union else 0.0,
        "union": ((a + b) != 0).astype(int),
        "intersection": ((a * b) != 0).astype(int),
    }


def combine_networks(
    adjacency_a: np.ndarray, adjacency_b: np.ndarray, mode: str = "union"
) -> np.ndarray:
    """Combine two skeletons via ``'union'``, ``'intersection'`` or ``'agreement'``.

    ``'agreement'`` is an alias of ``'intersection'`` (edges supported by both
    the Gaussian and the Bayesian-network analyses).
    """
    a = ((np.asarray(adjacency_a) + np.asarray(adjacency_a).T) != 0).astype(int)
    b = ((np.asarray(adjacency_b) + np.asarray(adjacency_b).T) != 0).astype(int)
    if mode == "union":
        return ((a + b) != 0).astype(int)
    if mode in ("intersection", "agreement"):
        return ((a * b) != 0).astype(int)
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Evolution across generations
# ---------------------------------------------------------------------------

def gaussian_network_evolution(
    models: Sequence[Any],
    method: str = "partial_correlation",
    threshold: float = 0.1,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Extract the Gaussian interaction network of every generation.

    Returns
    -------
    dict
        ``adjacencies`` (list of 0/1 matrices), ``partial_correlations``,
        ``precisions``, ``covariances`` and ``n_edges`` (per-generation array).
        The adjacency list can be fed directly to
        :func:`pateda.knowledge_extraction.network_measures.compute_measures_evolution`.
    """
    adjacencies, pcorrs, precisions, covs, n_edges = [], [], [], [], []
    for model in models:
        try:
            net = gaussian_interaction_network(
                model, method=method, threshold=threshold, alpha=alpha
            )
            adjacencies.append(net["adjacency"])
            pcorrs.append(net["partial_correlation"])
            precisions.append(net["precision"])
            covs.append(net["covariance"])
            n_edges.append(net["n_edges"])
        except Exception:
            adjacencies.append(np.zeros((0, 0), dtype=int))
            pcorrs.append(np.zeros((0, 0)))
            precisions.append(np.zeros((0, 0)))
            covs.append(np.zeros((0, 0)))
            n_edges.append(0)
    return {
        "adjacencies": adjacencies,
        "partial_correlations": pcorrs,
        "precisions": precisions,
        "covariances": covs,
        "n_edges": np.array(n_edges),
        "n_generations": len(models),
        "method": method,
    }
