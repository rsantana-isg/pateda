"""
Marginal probability learning utilities

Equivalent to MATEDA's FindMargProb.m and LearnFDAParameters.m
"""

import numpy as np
from typing import List, Tuple

from pateda.learning.utils.conversions import (
    find_acc_card,
    num_convert_card,
    index_convert_card,
)


def find_marginal_prob(
    population: np.ndarray, n_vars: int, cardinality: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculate univariate and bivariate marginal probabilities from population

    Equivalent to MATEDA's FindMargProb.m

    Args:
        population: Population data (n_samples, n_vars)
        n_vars: Number of variables
        cardinality: Cardinality of each variable

    Returns:
        Tuple of (univariate_probs, bivariate_probs)
        - univariate_probs: List of n_vars arrays, each of size Card[i]
        - bivariate_probs: List of arrays for each pair (i,j) where i<j
    """
    n_samples = population.shape[0]

    # Initialize probability lists
    univ_prob = []
    biv_prob = []

    # Calculate univariate probabilities
    for i in range(n_vars):
        # Count occurrences of each value
        counts = np.zeros(cardinality[i])
        for val in range(cardinality[i]):
            counts[val] = np.sum(population[:, i] == val)

        # Convert to probabilities
        probs = counts / n_samples
        univ_prob.append(probs)

    # Calculate bivariate probabilities for all pairs
    for i in range(n_vars):
        biv_row = []
        for j in range(n_vars):
            if j <= i:
                biv_row.append(None)
            else:
                # Count co-occurrences
                counts = np.zeros((cardinality[i], cardinality[j]))
                for sample in population:
                    counts[sample[i], sample[j]] += 1

                # Convert to probabilities and flatten
                probs = (counts / n_samples).flatten()
                biv_row.append(probs)

        biv_prob.append(biv_row)

    return univ_prob, biv_prob


def learn_fda_parameters(
    cliques: np.ndarray, population: np.ndarray, n_vars: int, cardinality: np.ndarray
) -> List[np.ndarray]:
    """
    Learn probability tables for FDA model given clique structure

    Equivalent to MATEDA's LearnFDAParameters.m

    Args:
        cliques: Clique structure matrix
                Each row: [n_overlap, n_new, overlap_indices..., new_indices...]
        population: Population to learn from
        n_vars: Number of variables
        cardinality: Cardinality of each variable

    Returns:
        List of probability tables, one per clique
    """
    n_samples = population.shape[0]
    n_cliques = cliques.shape[0]
    tables = []

    for c in range(n_cliques):
        n_overlap = int(cliques[c, 0])
        n_new = int(cliques[c, 1])

        # Extract variable indices
        if n_overlap > 0:
            overlap_vars = cliques[c, 2 : 2 + n_overlap].astype(int)
        else:
            overlap_vars = np.array([], dtype=int)

        new_vars = cliques[c, 2 + n_overlap : 2 + n_overlap + n_new].astype(int)
        all_vars = np.concatenate([overlap_vars, new_vars])

        # Calculate table dimensions
        if n_overlap > 0:
            overlap_card = cardinality[overlap_vars]
            n_overlap_configs = int(np.prod(overlap_card))
        else:
            n_overlap_configs = 1

        new_card = cardinality[new_vars]
        n_new_configs = int(np.prod(new_card))

        # Count occurrences
        if n_overlap > 0:
            # Conditional table: P(new | overlap)
            table = np.zeros((n_overlap_configs, n_new_configs))

            overlap_acc_card = find_acc_card(n_overlap, overlap_card)
            new_acc_card = find_acc_card(n_new, new_card)

            for sample in population:
                overlap_vals = sample[overlap_vars]
                new_vals = sample[new_vars]

                overlap_idx = num_convert_card(overlap_vals, n_overlap, overlap_acc_card)
                new_idx = num_convert_card(new_vals, n_new, new_acc_card)

                table[overlap_idx, new_idx] += 1

            # Normalize to get probabilities (with Laplace smoothing)
            for i in range(n_overlap_configs):
                count_sum = np.sum(table[i, :])
                if count_sum > 0:
                    # Laplace smoothing
                    table[i, :] = (table[i, :] + 1) / (count_sum + n_new_configs)
                else:
                    # Uniform if no data
                    table[i, :] = 1.0 / n_new_configs

        else:
            # Marginal table: P(new)
            table = np.zeros(n_new_configs)
            new_acc_card = find_acc_card(n_new, new_card)

            for sample in population:
                new_vals = sample[new_vars]
                new_idx = num_convert_card(new_vals, n_new, new_acc_card)
                table[new_idx] += 1

            # Normalize (with Laplace smoothing)
            table = (table + 1) / (n_samples + n_new_configs)

        tables.append(table)

    return tables
