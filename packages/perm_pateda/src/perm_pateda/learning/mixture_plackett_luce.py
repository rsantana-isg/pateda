"""
Mixture of Plackett-Luce Models — Learning via Spectral EM (EM-LSR)

Implements the algorithm from:
    Nguyen & Zhang (2023). "Efficient and Accurate Learning of Mixtures of
    Plackett-Luce Models." AAAI-23.

The algorithm has two phases:
    1. Spectral Initialization (Algorithms 1–3 in the paper)
       - Embed each ranking as a binary pairwise vector
       - SVD + adaptive dimension reduction + k-means clustering
       - Least-squares parameter estimation per cluster
    2. EM Refinement with Weighted LSR (Algorithms 4–5)
       - E-step: compute posterior component membership probabilities
       - M-step: weighted Luce Spectral Ranking (exact MLE, not surrogate)

Interface mirrors LearnPlackettLuce so it can be dropped in as a replacement.

References:
    [1] Nguyen & Zhang (2023). AAAI-23. arXiv:2302.05343
    [2] Maystre & Grossglauser (2015). "Fast and accurate inference of
        Plackett-Luce models." NeurIPS 28.  (original LSR)
    [3] Luce (1959); Plackett (1975). (base PL model)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Phase 1 — Spectral Initialization
# ---------------------------------------------------------------------------

def _pairwise_embedding(population: np.ndarray) -> np.ndarray:
    """
    Embed each ranking as a binary pairwise comparison vector.

    For a permutation of n items, produces a vector of length n*(n-1)/2.
    Entry (i,j) with i<j is 1 if item i is ranked before item j, else 0.
    This is Equation (7) in the paper.

    Args:
        population: (m, n) array of permutations (0-indexed item indices)

    Returns:
        X: (m, n*(n-1)//2) binary matrix
    """
    m, n = population.shape
    n_pairs = n * (n - 1) // 2
    X = np.zeros((m, n_pairs), dtype=np.float32)

    # Precompute position of each item in each ranking: pos[l, item] = rank position
    # population[l, p] = item at position p  =>  pos[l, item] = position of that item
    pos = np.empty_like(population)
    for l in range(m):
        pos[l, population[l]] = np.arange(n)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # X[l, idx] = 1 iff item i appears before item j in ranking l
            X[:, idx] = (pos[:, i] < pos[:, j]).astype(np.float32)
            idx += 1

    return X


def _spectral_clustering(
    X: np.ndarray,
    K: int,
    threshold: Optional[float] = None,
    random_state: int = 0,
) -> np.ndarray:
    """
    Spectral clustering with adaptive dimension reduction (Algorithm 1).

    Performs SVD on the pairwise embedding matrix, selects the number of
    dimensions r̂ adaptively via singular value gaps, then runs k-means.

    Args:
        X:            (m, n_pairs) pairwise embedding matrix
        K:            number of mixture components
        threshold:    gap threshold T for dimension selection.
                      If None, uses T = sqrt(n) * sqrt(m + n*log(n)) / m
                      as suggested in Algorithm 3.
        random_state: seed for k-means

    Returns:
        labels: (m,) integer cluster assignments in {0, ..., K-1}
    """
    m, _ = X.shape

    # SVD (economy)
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: random labels
        rng = np.random.default_rng(random_state)
        return rng.integers(0, K, size=m)

    # Adaptive rank selection: find largest r̂ ≤ K with gap > T
    if threshold is None:
        # T from Algorithm 3 header: sqrt(n) * sqrt(m + n*log(n))  (normalized by m)
        n_items = int(round((1 + np.sqrt(1 + 8 * X.shape[1])) / 2))
        threshold = np.sqrt(n_items) * np.sqrt(m + n_items * np.log(n_items + 1)) / m

    r_hat = 1  # at minimum use 1 dimension
    for a in range(1, K):
        if a < len(s) and (s[a - 1] - s[a]) >= threshold:
            r_hat = a

    # Project onto top r̂ right singular vectors: XV_{1:r̂}
    # Shape: (m, r_hat)
    proj = X @ Vt[:r_hat].T

    # K-means on the projected rows
    labels = _kmeans(proj, K, random_state=random_state)
    return labels


def _kmeans(
    Z: np.ndarray,
    K: int,
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    
    kmeans = KMeans(
        n_clusters=K, 
        init='k-means++', 
        n_init=n_init, 
        max_iter=max_iter, 
        random_state=random_state
    )
    return kmeans.fit_predict(Z)

def _estimate_pairwise_probs(
    population: np.ndarray,
    labels: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Estimate pairwise preference probability P̂ᵏᵢⱼ per cluster (Equation 8).
 
    P̂ᵏᵢⱼ = fraction of rankings in cluster k where item i is ranked before item j.
 
    Returns:
        P_hat: (K, n, n) array where P_hat[k, i, j] = P̂ᵏᵢⱼ
    """
    m, n = population.shape
    P_hat = np.zeros((K, n, n), dtype=np.float64)
    counts = np.zeros(K, dtype=int)
 
    # Position array
    pos = np.empty_like(population)
    for l in range(m):
        pos[l, population[l]] = np.arange(n)
 
    for l in range(m):
        k = labels[l]
        counts[k] += 1
        for i in range(n):
            for j in range(n):
                if i != j and pos[l, i] < pos[l, j]:
                    P_hat[k, i, j] += 1.0
 
    for k in range(K):
        if counts[k] > 0:
            P_hat[k] /= counts[k]
 
    return P_hat
 
 
def _estimate_pairwise_probs_fast(
    population: np.ndarray,
    labels: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Vectorized version of _estimate_pairwise_probs for large n.
    """
    m, n = population.shape
    P_hat = np.zeros((K, n, n), dtype=np.float64)
    counts = np.zeros(K, dtype=int)
 
    pos = np.empty_like(population)
    for l in range(m):
        pos[l, population[l]] = np.arange(n)
 
    for k in range(K):
        mask = labels == k
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        pos_k = pos[mask]  # (count_k, n)
        # P_hat[k, i, j] = mean over rankings l in cluster k of 1[pos_l[i] < pos_l[j]]
        # Vectorized: for each pair (i,j), compare columns
        for i in range(n):
            P_hat[k, i, :] = (pos_k[:, i:i+1] < pos_k).mean(axis=0)
        np.fill_diagonal(P_hat[k], 0.0)
 
    return P_hat
 
 
def _least_squares_params(P_hat: np.ndarray, n: int) -> np.ndarray:
    """
    Recover utility parameters θ̂ from pairwise preference matrix via
    least squares on logit-transformed probabilities (Algorithm 2).
 
    Solves: θ̂ = argmin_{θ: Σθᵢ=0} Σᵢ≠ⱼ (φ̂ᵢⱼ - (θᵢ - θⱼ))²
    where φ̂ᵢⱼ = logit(P̂ᵢⱼ) = log(P̂ᵢⱼ / (1 - P̂ᵢⱼ))
 
    The constraint Σθᵢ = 0 is enforced by projecting the solution.
 
    Returns:
        theta: (n,) normalized utility vector
    """
    # Clamp probabilities away from 0 and 1 to avoid log(0)
    P_safe = np.clip(P_hat, 1e-6, 1.0 - 1e-6)
 
    # Build system: for each ordered pair (i,j), equation: θᵢ - θⱼ = φ̂ᵢⱼ
    # This is a rank-1-deficient linear system; we add the constraint Σθᵢ = 0
    # Efficient form: use the normal equations of the least squares problem.
    #
    # The objective is: Σᵢ≠ⱼ (φ̂ᵢⱼ - θᵢ + θⱼ)²
    # Taking derivative w.r.t. θₖ and setting to zero:
    # 2(n-1) * Σⱼ≠ₖ [(θₖ - θⱼ) - φ̂ₖⱼ] = 0  for each k
    #
    # Matrix form: A @ θ = b
    # Aₖₖ = 2(n-1),  Aₖⱼ = -2 for j≠k
    # bₖ = 2 * Σⱼ≠ₖ φ̂ₖⱼ
    #
    # Add sum constraint as extra equation: [1,1,...,1] @ θ = 0
 
    phi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                phi[i, j] = np.log(P_safe[i, j] / (1.0 - P_safe[i, j]))
 
    # Normal equations (n x n system, rank n-1)
    A = np.full((n, n), -2.0)
    np.fill_diagonal(A, 2.0 * (n - 1))
 
    b = 2.0 * phi.sum(axis=1)  # Σⱼ≠ₖ φ̂ₖⱼ for each k
 
    # Replace last equation with sum constraint: Σθᵢ = 0
    A[-1, :] = 1.0
    b[-1] = 0.0
 
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.zeros(n)
 
    # Normalize: subtract mean so Σθᵢ = 0
    theta -= theta.mean()
    return theta
 
 
def _spectral_init(
    population: np.ndarray,
    K: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full spectral initialization (Algorithm 3).
 
    Returns:
        theta_init: (n, K) initial utility parameters (log-scale)
        labels:     (m,) cluster assignments used
    """
    m, n = population.shape
 
    # Step 1: pairwise embedding
    X = _pairwise_embedding(population)
 
    # Step 2: spectral clustering
    labels = _spectral_clustering(X, K, random_state=random_state)
 
    # Ensure every cluster has at least one sample (fix empty clusters)
    for k in range(K):
        if (labels == k).sum() == 0:
            # Assign the farthest point from the biggest cluster
            big = np.bincount(labels).argmax()
            idx = np.where(labels == big)[0]
            labels[idx[0]] = k
 
    # Step 3: pairwise probability estimation
    P_hat = _estimate_pairwise_probs_fast(population, labels, K)
 
    # Step 4: least-squares parameter estimation per cluster
    theta_init = np.zeros((n, K))
    for k in range(K):
        if (labels == k).sum() > 0:
            theta_init[:, k] = _least_squares_params(P_hat[k], n)
 
    return theta_init, labels
 
 
# ---------------------------------------------------------------------------
# Phase 2 — EM with Weighted LSR
# ---------------------------------------------------------------------------
 
def _pl_log_likelihood(perm: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute log P_PL(π | θ) for a single permutation.
 
    log P_PL(π) = Σᵢ₌₁ⁿ⁻¹ [θ_{πᵢ} - log(Σⱼ₌ᵢⁿ exp(θ_{πⱼ}))]
    """
    n = len(perm)
    log_prob = 0.0
    # Use log-sum-exp for numerical stability
    suffix_theta = theta[perm]  # utilities in ranking order
    for i in range(n - 1):
        log_prob += suffix_theta[i] - _logsumexp(suffix_theta[i:])
    return log_prob
 
 
def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = a.max()
    return a_max + np.log(np.sum(np.exp(a - a_max)))
 
 
def _pl_log_likelihood_batch(
    population: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """
    Compute log P_PL(πₗ | θ) for all m rankings at once.
 
    Returns:
        log_probs: (m,) array
    """
    m, n = population.shape
    log_probs = np.zeros(m)
 
    for l in range(m):
        log_probs[l] = _pl_log_likelihood(population[l], theta)
 
    return log_probs


def _e_step(
    population: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    E-step: compute posterior component membership probabilities.
 
    q^k_l = P(z_l = k | π_l, θ) ∝ β_k · P_PL(π_l | θ^k)
 
    Uses log-space arithmetic for numerical stability.
 
    Args:
        population: (m, n) permutations
        theta:      (n, K) utility parameters
        beta:       (K,) mixing proportions
 
    Returns:
        q: (m, K) posterior probabilities (each row sums to 1)
    """
    m, n = population.shape
    K = theta.shape[1]
 
    # log_q[l, k] = log β_k + log P_PL(π_l | θ^k)
    log_q = np.zeros((m, K))
    for k in range(K):
        log_q[:, k] = np.log(beta[k] + 1e-300) + _pl_log_likelihood_batch(
            population, theta[:, k]
        )
 
    # Normalize in log space (subtract row max for stability)
    log_q -= log_q.max(axis=1, keepdims=True)
    q = np.exp(log_q)
    q /= q.sum(axis=1, keepdims=True)
 
    return q


def _weighted_lsr(
    population: np.ndarray,
    weights: np.ndarray,
    theta_init: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Weighted Luce Spectral Ranking (Algorithm 4).
 
    Finds the maximum weighted log-likelihood estimate of PL parameters:
        θ_MLE = argmax_θ Σₗ wₗ · log P_PL(πₗ | θ)
 
    Uses the LSR Markov chain approach: constructs a weighted random walk
    on items where transition i→j is proportional to the sum of weights
    of choice breakings where j "beats" i. The stationary distribution
    gives the MLE.
 
    Args:
        population:  (m, n) permutations
        weights:     (m,) non-negative sample weights
        theta_init:  (n,) initial log-utility estimate
        max_iter:    maximum power-iteration steps
        tol:         convergence threshold on log-util L1 change
 
    Returns:
        theta: (n,) normalized log-utility vector (sums to 0)
    """
    m, n = population.shape
 
    # Build choice breaking: for ranking l at position p, item π_{l,p}
    # "beats" all items π_{l,p+1}, ..., π_{l,n-1}.
    # Transition weight M[j→i] += w_l / (Σ_{q≥p} exp(θ_{π_{l,q}}))
    # We iterate: update θ → recompute M → find stationary distribution.
 
    theta = theta_init.copy()
 
    # Normalization constant d for the Markov chain (must be large enough
    # that all diagonal entries are non-negative). We use d = m*n as a
    # conservative upper bound; the paper uses a similar strategy.
    d = float(m * n)
 
    for _ in range(max_iter):
        # Build transition matrix M (n x n)
        # M[i, j] = (1/d) * Σ_{(j,A,l) ∈ B: i ∈ A} w_l / Σ_{q∈A} exp(θ_q)
        # i.e., rate at which the chain goes from state i to state j is
        # proportional to how often j "beats" i in the weighted data.
 
        M = np.zeros((n, n))

        # Vectorizado: el triple bucle (ranking x posicion x loser) se
        # reemplaza por suffix log-sum-exp con np.logaddexp y fancy indexing.
        for l in range(m):
            if weights[l] < 1e-12:
                continue
            perm = population[l]
            w_l = weights[l]

            # Suffix log-sum-exp vectorizado - sin bucle Python interno
            suffix_theta = theta[perm]
            suffix_lse = np.zeros(n)
            suffix_lse[-1] = suffix_theta[-1]
            for p in range(n - 2, -1, -1):
                suffix_lse[p] = np.logaddexp(suffix_theta[p], suffix_lse[p + 1])

            # contributions[p] = w_l / exp(suffix_lse[p])  para p = 0..n-2
            denoms = np.exp(suffix_lse[:-1])
            contribs = np.where(denoms > 1e-300, w_l / denoms, 0.0)

            # Acumular en M: para cada posicion p, perm[p] gana a perm[p+1:]
            # M[loser, winner] += contribs[p]  - bucle solo sobre posiciones
            for p in range(n - 1):
                if contribs[p] == 0.0:
                    continue
                M[perm[p + 1:], perm[p]] += contribs[p]
 
        # Convert to proper row-stochastic form
        # M[i,i] = 1 - (1/d) * Σ_{j≠i} d*M[j,i] ... but we built M as rates
        # Rescale: off-diagonal entries are M[i,j] (rate i→j)
        # Diagonal: M[i,i] = 1 - Σ_{j≠i} M[i,j] (after scaling by 1/d)
        row_sums = M.sum(axis=1)
        max_rate = row_sums.max()
        if max_rate < 1e-12:
            break
        # Normalize so max row sum ≤ 1 (lazy chain)
        scale = 1.0 / (max_rate + 1e-12)
        M_stoch = M * scale
        np.fill_diagonal(M_stoch, 0.0)
        row_sums_scaled = M_stoch.sum(axis=1)
        np.fill_diagonal(M_stoch, np.maximum(1.0 - row_sums_scaled, 0.0))
 
        # Power iteration para la distribución estacionaria.
        # Vectorizado: M_stoch ya es (n,n) numpy, cada paso es un matmul.
        # Límite reducido de 200 → 50; la cadena converge en <20 pasos típicamente.
        p_stat = np.ones(n) / n
        for _ in range(50):
            p_new = p_stat @ M_stoch          # un solo matmul, sin bucle Python
            p_new = np.maximum(p_new, 1e-300)
            p_new /= p_new.sum()
            if np.linalg.norm(p_new - p_stat, 1) < 1e-8:
                break
            p_stat = p_new
        p_stat = p_new
 
        # Recover log utilities from stationary distribution
        theta_new = np.log(p_stat)
        theta_new -= theta_new.mean()  # normalize: Σθᵢ = 0
 
        if np.linalg.norm(theta_new - theta, 1) < tol:
            theta = theta_new
            break
        theta = theta_new
 
    return theta
 
 
def _m_step(
    population: np.ndarray,
    q: np.ndarray,
    theta_prev: np.ndarray,
    max_iter_lsr: int = 100,
    tol_lsr: float = 1e-6,
) -> np.ndarray:
    """
    M-step: run Weighted LSR independently for each component k.
 
    For component k, the sample weight of ranking l is q^k_l.
 
    Args:
        population:  (m, n) permutations
        q:           (m, K) posterior probabilities from E-step
        theta_prev:  (n, K) current parameter estimates (warm start)
        max_iter_lsr, tol_lsr: LSR convergence settings
 
    Returns:
        theta_new: (n, K) updated parameter estimates
    """
    n, K = theta_prev.shape
    theta_new = np.zeros_like(theta_prev)
 
    for k in range(K):
        theta_new[:, k] = _weighted_lsr(
            population,
            weights=q[:, k],
            theta_init=theta_prev[:, k],
            max_iter=max_iter_lsr,
            tol=tol_lsr,
        )
 
    return theta_new



class LearnPlackettLuceMixture:
    def __init__(
        self,
        n_components: int = 2,
        max_em_iter: int = 10,   # era 50 — 10 basta para convergencia en EDA
        tol_em: float = 1e-4,   # era 1e-5 — menos preciso pero mucho más rápido
        max_iter_lsr: int = 20,  # era 100 — LSR converge rápido con warm start
        tol_lsr: float = 1e-4,  # era 1e-6
        use_spectral_init: bool = True,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.max_em_iter = max_em_iter
        self.tol_em = tol_em
        self.max_iter_lsr = max_iter_lsr
        self.tol_lsr = tol_lsr
        self.use_spectral_init = use_spectral_init
        self.random_state = random_state

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Learn method to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,  # ignorados, los parámetros vienen del __init__
    ) -> Dict[str, Any]:
        data = np.asarray(population, dtype=int)
        m, n = data.shape
        K = self.n_components  # ← antes era parámetro, ahora self

        rng = np.random.default_rng(self.random_state)  # ← idem

        # ----------------------------------------------------------------
        # Phase 1: Initialization
        # ----------------------------------------------------------------
        if self.use_spectral_init and m >= K * 2:
            theta, _ = _spectral_init(data, K, random_state=self.random_state)
        else:
            theta = rng.standard_normal((n, K))
            theta -= theta.mean(axis=0, keepdims=True)

        beta = np.ones(K) / K

        # ----------------------------------------------------------------
        # Phase 2: EM iterations
        # ----------------------------------------------------------------
        for em_iter in range(self.max_em_iter):  # ← self
            beta_old = beta.copy()

            # E-step
            q = _e_step(data, theta, beta)

            # Update mixing weights
            beta = q.mean(axis=0)
            beta = np.maximum(beta, 1e-9)
            beta /= beta.sum()

            # M-step
            theta = _m_step(
                data, q, theta,
                self.max_iter_lsr,  # ← self
                self.tol_lsr,       # ← self
            )

            # Convergencia
            if np.linalg.norm(beta - beta_old, 1) < self.tol_em:  # ← self
                break

        # ----------------------------------------------------------------
        # Package results
        # ----------------------------------------------------------------
        weights_per_component = []
        for k in range(K):
            theta_k = theta[:, k]
            w_k = np.exp(theta_k - theta_k.max())
            w_k /= w_k.sum()
            weights_per_component.append(w_k)

        dominant_k = int(np.argmax(beta))

        return {
            "weights_per_component": weights_per_component,
            "mixing_weights": beta,
            "theta": theta,
            "n_components": K,
            "model_type": "plackett_luce_mixture",
            "weights": weights_per_component[dominant_k],
        }