"""
Probability table computation for Markov network cliques

Computes marginal and conditional probability tables for factorized models.

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- C++ implementation: cpp_EDAs/FDA.cpp (ComputeMarg, FDACallProb)
- MATEDA-2.0: CalcProbSubsetVars.m
"""

from typing import List, Optional
import numpy as np

from pateda.learning.utils.conversions import (
    find_acc_card,
    num_convert_card,
)


def compute_clique_table(
    population: np.ndarray,
    clique_vars: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True,
) -> np.ndarray:
    """
    Compute probability table for a single clique

    For a clique with variables [X1, X2, ..., Xk], computes:
        P(X1, X2, ..., Xk)

    Table is indexed using mixed-radix conversion.

    Args:
        population: Population array (n_samples, n_vars)
        clique_vars: Variable indices in this clique
        cardinality: Variable cardinalities (n_vars,)
        weights: Sample weights (default: uniform)
        prior: Whether to use Laplace prior (default: True)

    Returns:
        Probability table (flattened) of size prod(cardinalities)
    """
    n_samples = population.shape[0]
    n_clique_vars = len(clique_vars)

    # Get cardinalities for clique variables
    clique_cards = cardinality[clique_vars].astype(int)

    # Compute accumulated cardinalities for indexing
    acc_card = find_acc_card(n_clique_vars, clique_cards)

    # Number of configurations
    n_configs = int(np.prod(clique_cards))

    # Initialize frequency table
    if prior:
        # Laplace prior: add 1 to all counts
        freq = np.ones(n_configs)
        total_weight = n_samples + n_configs
    else:
        freq = np.zeros(n_configs)
        total_weight = n_samples

    # Default weights
    if weights is None:
        weights = np.ones(n_samples)
    else:
        # Adjust total weight for weighted case
        if prior:
            total_weight = np.sum(weights) + n_configs
        else:
            total_weight = np.sum(weights)

    # Count configurations
    for sample_idx in range(n_samples):
        # Get configuration for this sample
        config = population[sample_idx, clique_vars].astype(int)

        # Convert configuration to table index
        table_idx = num_convert_card(config, n_clique_vars, acc_card)

        # Add weight
        freq[table_idx] += weights[sample_idx]

    # Normalize to probabilities
    prob_table = freq / total_weight

    return prob_table


def compute_conditional_table(
    population: np.ndarray,
    overlap_vars: np.ndarray,
    new_vars: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True,
) -> np.ndarray:
    """
    Compute conditional probability table P(new_vars | overlap_vars)

    Table dimensions: (n_overlap_configs, n_new_configs)
    where:
        n_overlap_configs = prod(cardinality[overlap_vars])
        n_new_configs = prod(cardinality[new_vars])

    Args:
        population: Population array (n_samples, n_vars)
        overlap_vars: Variables in overlap (conditioning variables)
        new_vars: Variables to be sampled (conditioned variables)
        cardinality: Variable cardinalities (n_vars,)
        weights: Sample weights (default: uniform)
        prior: Whether to use Laplace prior

    Returns:
        Conditional probability table (n_overlap_configs, n_new_configs)
    """
    n_samples = population.shape[0]
    n_overlap = len(overlap_vars)
    n_new = len(new_vars)

    # Get cardinalities
    overlap_cards = cardinality[overlap_vars].astype(int)
    new_cards = cardinality[new_vars].astype(int)

    # Accumulated cardinalities for indexing
    overlap_acc = find_acc_card(n_overlap, overlap_cards)
    new_acc = find_acc_card(n_new, new_cards)

    # Number of configurations
    n_overlap_configs = int(np.prod(overlap_cards))
    n_new_configs = int(np.prod(new_cards))

    # Initialize frequency table
    if prior:
        # Laplace prior
        freq = np.ones((n_overlap_configs, n_new_configs))
    else:
        freq = np.zeros((n_overlap_configs, n_new_configs))

    # Default weights
    if weights is None:
        weights = np.ones(n_samples)

    # Count configurations
    for sample_idx in range(n_samples):
        # Get overlap configuration
        overlap_config = population[sample_idx, overlap_vars].astype(int)
        overlap_idx = num_convert_card(overlap_config, n_overlap, overlap_acc)

        # Get new variable configuration
        new_config = population[sample_idx, new_vars].astype(int)
        new_idx = num_convert_card(new_config, n_new, new_acc)

        # Add weight
        freq[overlap_idx, new_idx] += weights[sample_idx]

    # Normalize each row (conditional on overlap configuration)
    prob_table = np.zeros((n_overlap_configs, n_new_configs))

    for overlap_idx in range(n_overlap_configs):
        row_sum = np.sum(freq[overlap_idx, :])
        if row_sum > 0:
            prob_table[overlap_idx, :] = freq[overlap_idx, :] / row_sum
        else:
            # If no samples for this overlap config, use uniform
            prob_table[overlap_idx, :] = 1.0 / n_new_configs

    return prob_table


def compute_clique_tables(
    population: np.ndarray,
    cliques: List[np.ndarray],
    structure: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True,
) -> List[np.ndarray]:
    """
    Compute probability tables for all cliques in factorized structure

    Uses the structure format from FactorizedModel:
    structure[i] = [n_overlap, n_new, overlap_vars..., new_vars...]

    For root cliques (n_overlap = 0):
        Computes marginal P(new_vars)

    For overlapping cliques (n_overlap > 0):
        Computes conditional P(new_vars | overlap_vars)

    Reference:
    - Algorithm 2, step 5: "Find the marginal probabilities for the cliques"
    - C++ FDA.cpp:1874-1920 (CreateMarg, ComputeMarg)

    Args:
        population: Population array (n_samples, n_vars)
        cliques: List of cliques (for reference, not directly used)
        structure: Factorized structure array (n_cliques, max_row_size)
        cardinality: Variable cardinalities (n_vars,)
        weights: Sample weights
        prior: Whether to use Laplace prior

    Returns:
        List of probability tables, one per clique
        - Root cliques: 1D array of marginal probabilities
        - Overlapping cliques: 2D array (n_overlap_configs, n_new_configs)
    """
    n_cliques = structure.shape[0]
    tables = []

    for c in range(n_cliques):
        n_overlap = int(structure[c, 0])
        n_new = int(structure[c, 1])

        if n_overlap == 0:
            # Root clique: compute marginal distribution
            new_vars = structure[c, 2 : 2 + n_new].astype(int)

            table = compute_clique_table(
                population, new_vars, cardinality, weights, prior
            )

        else:
            # Overlapping clique: compute conditional distribution
            overlap_vars = structure[c, 2 : 2 + n_overlap].astype(int)
            new_vars = structure[c, 2 + n_overlap : 2 + n_overlap + n_new].astype(int)

            table = compute_conditional_table(
                population, overlap_vars, new_vars, cardinality, weights, prior
            )

        tables.append(table)

    return tables


def compute_moa_clique_table(
    population: np.ndarray,
    var_i: int,
    neighbors: np.ndarray,
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True,
) -> np.ndarray:
    """
    Compute conditional probability table for MOA clique

    For MOA, each clique is: [Xi, neighbors of Xi]
    We need: P(Xi | neighbors)

    Args:
        population: Population array (n_samples, n_vars)
        var_i: Target variable
        neighbors: Neighbor variable indices
        cardinality: Variable cardinalities
        weights: Sample weights
        prior: Whether to use Laplace prior

    Returns:
        Conditional probability table P(Xi | neighbors)
        Shape: (n_neighbor_configs, cardinality[var_i])
    """
    if len(neighbors) == 0:
        # No neighbors: marginal distribution P(Xi)
        return compute_clique_table(
            population, np.array([var_i]), cardinality, weights, prior
        )
    else:
        # Conditional distribution P(Xi | neighbors)
        return compute_conditional_table(
            population,
            neighbors,
            np.array([var_i]),
            cardinality,
            weights,
            prior,
        )


def compute_moa_tables(
    population: np.ndarray,
    neighbors_list: List[np.ndarray],
    cardinality: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior: bool = True,
) -> List[np.ndarray]:
    """
    Compute all conditional probability tables for MOA

    For each variable Xi, computes P(Xi | neighbors_i)

    Reference: mainmoa.cpp:669 (UpdateModelProteinMPM)

    Args:
        population: Population array
        neighbors_list: List of neighbor arrays from find_k_neighbors()
        cardinality: Variable cardinalities
        weights: Sample weights
        prior: Whether to use Laplace prior

    Returns:
        List of conditional probability tables, one per variable
    """
    n_vars = len(neighbors_list)
    tables = []

    for var_i in range(n_vars):
        neighbors = neighbors_list[var_i]

        table = compute_moa_clique_table(
            population, var_i, neighbors, cardinality, weights, prior
        )

        tables.append(table)

    return tables
