"""
Mutual Information computation for Markov network learning

Implements efficient MI computation for discrete variables, used by MN-FDA,
MN-FDAG, and MOA algorithms.

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/FDA.cpp, IntTreeModel
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats


def compute_pairwise_mi(
    population: np.ndarray,
    var_i: int,
    var_j: int,
    card_i: int,
    card_j: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mutual information MI(Xi, Xj) for a single variable pair

    MI(Xi, Xj) = ∑∑ p(xi, xj) * log(p(xi, xj) / (p(xi) * p(xj)))

    Args:
        population: Population array (n_samples, n_vars)
        var_i: Index of first variable
        var_j: Index of second variable
        card_i: Cardinality of variable i
        card_j: Cardinality of variable j
        weights: Sample weights (default: uniform)

    Returns:
        Mutual information MI(Xi, Xj)
    """
    n_samples = population.shape[0]

    # Default to uniform weights
    if weights is None:
        weights = np.ones(n_samples) / n_samples
    else:
        # Normalize weights
        weights = weights / np.sum(weights)

    # Compute joint frequencies p(xi, xj)
    joint_freq = np.zeros((card_i, card_j))
    for sample_idx in range(n_samples):
        xi = int(population[sample_idx, var_i])
        xj = int(population[sample_idx, var_j])
        joint_freq[xi, xj] += weights[sample_idx]

    # Compute marginal frequencies
    marginal_i = np.sum(joint_freq, axis=1)  # p(xi)
    marginal_j = np.sum(joint_freq, axis=0)  # p(xj)

    # Compute MI
    mi = 0.0
    for xi in range(card_i):
        for xj in range(card_j):
            p_joint = joint_freq[xi, xj]
            p_i = marginal_i[xi]
            p_j = marginal_j[xj]

            # Avoid log(0)
            if p_joint > 0 and p_i > 0 and p_j > 0:
                mi += p_joint * np.log2(p_joint / (p_i * p_j))

    return mi


def compute_mutual_information_matrix(
    population: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute pairwise mutual information matrix for all variable pairs

    Uses efficient computation by iterating through all pairs once.
    Result is symmetric: MI[i,j] = MI[j,i]

    Args:
        population: Population array (n_samples, n_vars)
        cardinality: Variable cardinalities (n_vars,)
        weights: Sample weights (default: uniform)

    Returns:
        MI matrix (n_vars, n_vars) where MI[i,j] = MI(Xi, Xj)
        Diagonal is set to 0
    """
    n_vars = population.shape[1]
    mi_matrix = np.zeros((n_vars, n_vars))

    # Compute MI for all pairs (upper triangle, then mirror)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = compute_pairwise_mi(
                population, i, j, int(cardinality[i]), int(cardinality[j]), weights
            )
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # Symmetric

    return mi_matrix


def g_test_statistic(
    mi: float, n_samples: int, card_i: int, card_j: int
) -> Tuple[float, float]:
    """
    Compute G-test statistic and p-value for independence test

    G-test of independence:
        G(Xi, Xj) = 2 * N * MI(Xi, Xj) * ln(2)

    The G-statistic follows a chi-square distribution with
    degrees of freedom: df = (card_i - 1) * (card_j - 1)

    Reference: Algorithm 2, Santana (2013)
    C++ implementation: FDA.cpp:1610-1632 (LearnMatrixGTest)

    Args:
        mi: Mutual information MI(Xi, Xj) in bits (log2)
        n_samples: Number of samples
        card_i: Cardinality of variable i
        card_j: Cardinality of variable j

    Returns:
        Tuple of (G_statistic, p_value)
        - G_statistic: G-test statistic value
        - p_value: Probability of observing this G under null hypothesis
    """
    # Convert MI from bits to nats (natural log)
    mi_nats = mi * np.log(2)

    # G-statistic: G = 2*N*MI (in nats)
    g_statistic = 2 * n_samples * mi_nats

    # Degrees of freedom
    df = (card_i - 1) * (card_j - 1)

    # Chi-square p-value
    p_value = 1.0 - stats.chi2.cdf(g_statistic, df)

    return g_statistic, p_value


def compute_g_test_matrix(
    population: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute G-test statistics for all variable pairs

    Used by MN-FDAG (MN-FDA with G-test).

    Args:
        population: Population array (n_samples, n_vars)
        cardinality: Variable cardinalities (n_vars,)
        weights: Sample weights (default: uniform)
        alpha: Significance level for independence test

    Returns:
        Tuple of (g_matrix, adjacency_matrix)
        - g_matrix: Matrix of G-statistics (n_vars, n_vars)
        - adjacency_matrix: Binary matrix where [i,j]=1 if variables are dependent
                           (i.e., p_value < alpha)
    """
    n_vars = population.shape[1]
    n_samples = population.shape[0]

    # First compute MI matrix
    mi_matrix = compute_mutual_information_matrix(population, cardinality, weights)

    # Then compute G-statistics and p-values
    g_matrix = np.zeros((n_vars, n_vars))
    adjacency = np.zeros((n_vars, n_vars), dtype=int)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = mi_matrix[i, j]
            g_stat, p_value = g_test_statistic(
                mi, n_samples, int(cardinality[i]), int(cardinality[j])
            )

            g_matrix[i, j] = g_stat
            g_matrix[j, i] = g_stat

            # Reject null hypothesis (variables are dependent) if p < alpha
            if p_value < alpha:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    # Set diagonal to 1 (variable is dependent on itself)
    np.fill_diagonal(adjacency, 1)

    return g_matrix, adjacency


def chi_square_test(
    mi: float, n_samples: int, threshold: float = 0.05
) -> Tuple[float, bool]:
    """
    Chi-square test for independence (simplified, assumes df=1)

    Used by base MN-FDA (without G-test).

    Reference: C++ implementation FDA.cpp:1635-1672 (LearnMatrix)

    Args:
        mi: Mutual information MI(Xi, Xj)
        n_samples: Number of samples
        threshold: Significance threshold (probability)

    Returns:
        Tuple of (chi_square_value, is_dependent)
        - chi_square_value: Chi-square statistic
        - is_dependent: True if variables are dependent
    """
    # Convert MI to chi-square (approximation for df=1)
    # Chi^2 ≈ 2*N*MI*ln(2)
    chi_square = 2 * n_samples * mi * np.log(2)

    # Critical value for chi-square with df=1
    df = 1
    critical_value = stats.chi2.ppf(1 - threshold, df)

    is_dependent = chi_square > critical_value

    return chi_square, is_dependent


def compute_conditional_mi(
    population: np.ndarray,
    var_i: int,
    var_j: int,
    cond_vars: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute conditional mutual information MI(Xi; Xj | Z)

    MI(Xi; Xj | Z) = ∑ p(z) * MI(Xi; Xj | Z=z)

    Useful for graph refinement (Algorithm 2, step 2).

    Args:
        population: Population array (n_samples, n_vars)
        var_i: Index of first variable
        var_j: Index of second variable
        cond_vars: Indices of conditioning variables
        cardinality: Variable cardinalities
        weights: Sample weights

    Returns:
        Conditional mutual information
    """
    n_samples = population.shape[0]

    if weights is None:
        weights = np.ones(n_samples) / n_samples
    else:
        weights = weights / np.sum(weights)

    # Get unique configurations of conditioning variables
    cond_configs = population[:, cond_vars]
    unique_configs = np.unique(cond_configs, axis=0)

    cond_mi = 0.0

    for config in unique_configs:
        # Find samples matching this conditioning configuration
        mask = np.all(cond_configs == config, axis=1)
        sub_pop = population[mask]
        sub_weights = weights[mask]

        if len(sub_pop) > 1:  # Need at least 2 samples
            # Normalize sub-weights
            sub_weights = sub_weights / np.sum(sub_weights)

            # Compute MI for this subset
            mi = compute_pairwise_mi(
                sub_pop,
                var_i,
                var_j,
                int(cardinality[var_i]),
                int(cardinality[var_j]),
                sub_weights,
            )

            # Weight by probability of this configuration
            p_config = np.sum(weights[mask])
            cond_mi += p_config * mi

    return cond_mi
