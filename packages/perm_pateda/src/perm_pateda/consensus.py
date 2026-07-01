"""
Consensus permutation methods

This module implements methods to find consensus permutations from
a set of permutations, which are used in Mallows models and other
permutation-based EDAs.
"""

import numpy as np
from typing import Callable, Optional
from perm_pateda.distances import kendall_distance


def find_consensus_borda(population: np.ndarray) -> np.ndarray:
    """
    Find consensus permutation using Borda count method.

    The Borda count assigns scores based on positions: an item at position i
    gets score (n-i). The consensus ranks items by their total scores.

    Args:
        population: Population of permutations, shape (pop_size, n_vars)

    Returns:
        Consensus permutation

    Example:
        >>> pop = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]])
        >>> find_consensus_borda(pop)
        array([1, 2, 3])
    """
    pop_size, n_vars = population.shape

    # Calculate Borda scores for each item
    scores = np.zeros(n_vars)

    for perm in population:
        for pos, item in enumerate(perm):
            # Convert to 0-indexed if needed
            item_idx = item if np.min(population) == 0 else item - 1
            # Score decreases with position: position 0 gets highest score
            scores[item_idx] += (n_vars - pos)

    # Create consensus by sorting items by their scores (descending)
    # argsort gives indices in ascending order, so we reverse
    consensus = np.argsort(-scores)

    # Convert back to 1-indexed if input was 1-indexed
    if np.min(population) == 1:
        consensus = consensus + 1

    return consensus


def find_consensus_median(
    distance_func: Callable,
    population: np.ndarray,
) -> np.ndarray:
    """
    Find consensus permutation as the median (Kemeny optimal ranking).

    The consensus is the permutation that minimizes the sum of distances
    to all permutations in the population.

    This is an approximation using the population itself as candidates.
    For small populations, this finds the permutation in the population
    that has minimum total distance to all others.

    Args:
        distance_func: Distance function to use (e.g., kendall_distance)
        population: Population of permutations, shape (pop_size, n_vars)

    Returns:
        Consensus permutation (the median)

    Example:
        >>> from perm_pateda.distances import kendall_distance
        >>> pop = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]])
        >>> find_consensus_median(kendall_distance, pop)
        array([1, 2, 3])
    """
    pop_size, n_vars = population.shape

    # For each permutation in the population, calculate total distance to all others
    min_total_dist = float('inf')
    consensus_idx = 0

    for i in range(pop_size):
        total_dist = 0
        for j in range(pop_size):
            if i != j:
                total_dist += distance_func(population[i], population[j])

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            consensus_idx = i

    return population[consensus_idx].copy()


def _permutation_inverse(perm: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a permutation.

    Args:
        perm: A permutation

    Returns:
        The inverse permutation

    Example:
        >>> perm = np.array([2, 0, 1])
        >>> _permutation_inverse(perm)
        array([1, 2, 0])
    """
    inv = np.zeros_like(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def compose_permutations(perm1: np.ndarray, perm2: np.ndarray) -> np.ndarray:
    """
    Compose two permutations: result[i] = perm2[perm1[i]]

    Args:
        perm1: First permutation
        perm2: Second permutation

    Returns:
        Composition of perm1 and perm2

    Example:
        >>> perm1 = np.array([1, 0, 2])
        >>> perm2 = np.array([2, 1, 0])
        >>> compose_permutations(perm1, perm2)
        array([1, 2, 0])
    """
    return perm2[perm1]

def find_consensus_best(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Find consensus permutation by simply selecting the best individual.

    Args:
        population: Population of permutations, shape (pop_size, n_vars)
        fitness: Fitness values for the population. Assumes higher is better.

    Returns:
        The single permutation with the highest fitness value.
    """
    best_idx = np.argmax(fitness)
    return population[best_idx].copy()


def get_consensus(
    method: str, 
    population: np.ndarray, 
    fitness: Optional[np.ndarray] = None,
    distance_func: Optional[Callable] = None
) -> np.ndarray:
    """
    Dispatcher function to get the consensus permutation using the specified method.

    Args:
        method: The consensus method ("borda", "setmedian", or "best").
        population: Population of permutations.
        fitness: Fitness values (required for "best" method).
        distance_func: Distance function (required for "setmedian" method).

    Returns:
        The consensus permutation.
    """
    method = method.lower()
    
    if method == "borda":
        return find_consensus_borda(population)
        
    elif method == "setmedian":
        if distance_func is None:
            raise ValueError("distance_func must be provided for 'setmedian' consensus")
        return find_consensus_median(distance_func, population)
        
    elif method == "best":
        if fitness is None or len(fitness) == 0:
            raise ValueError("fitness array must be provided for 'best' consensus")
        return find_consensus_best(population, fitness)
        
    else:
        raise ValueError(f"Unknown consensus method: {method}. Use 'borda', 'setmedian', or 'best'.")
