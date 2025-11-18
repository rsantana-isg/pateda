"""
Pareto-based utilities for multi-objective optimization

This module implements Pareto dominance checking, Pareto set identification,
and related operations for multi-objective optimization, equivalent to MATEDA's
FindParetoSet.m and related functions.
"""

from typing import Tuple
import numpy as np


def pareto_dominates(obj1: np.ndarray, obj2: np.ndarray, maximize: bool = True) -> bool:
    """
    Check if obj1 Pareto dominates obj2

    Args:
        obj1: First objective vector
        obj2: Second objective vector
        maximize: If True, assume maximization problem. If False, minimization.

    Returns:
        True if obj1 dominates obj2, False otherwise

    Note:
        For maximization, obj1 dominates obj2 if:
        - obj1[i] >= obj2[i] for all i
        - obj1[j] > obj2[j] for at least one j
    """
    if maximize:
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)
    else:
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


def find_pareto_set(
    fitness: np.ndarray, maximize: bool = True, return_mask: bool = False
) -> np.ndarray:
    """
    Identify the set of non-dominated solutions

    This function is equivalent to MATEDA's FindParetoSet.m

    Args:
        fitness: Fitness matrix (pop_size, n_objectives)
        maximize: If True, assume maximization problem. If False, minimization.
        return_mask: If True, return boolean mask. If False, return indices.

    Returns:
        If return_mask is True: Boolean mask of non-dominated solutions
        If return_mask is False: Indices of non-dominated solutions

    Example:
        >>> fitness = np.array([[1, 2], [2, 1], [0.5, 0.5], [1.5, 1.5]])
        >>> pareto_idx = find_pareto_set(fitness)
        >>> print(pareto_idx)  # Should show indices of non-dominated solutions
    """
    pop_size = fitness.shape[0]

    # For single objective, just find the best
    if fitness.shape[1] == 1:
        if maximize:
            best_idx = np.argmax(fitness[:, 0])
        else:
            best_idx = np.argmin(fitness[:, 0])
        if return_mask:
            mask = np.zeros(pop_size, dtype=bool)
            mask[best_idx] = True
            return mask
        else:
            return np.array([best_idx])

    # For multi-objective, check dominance
    dominated = np.zeros(pop_size, dtype=bool)

    for i in range(pop_size):
        if dominated[i]:
            continue

        for j in range(i + 1, pop_size):
            if dominated[j]:
                continue

            # Check if i dominates j
            if maximize:
                if np.all(fitness[i] >= fitness[j]) and np.any(fitness[i] > fitness[j]):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(fitness[j] >= fitness[i]) and np.any(fitness[j] > fitness[i]):
                    dominated[i] = True
                    break
            else:
                if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                    dominated[j] = True
                elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                    dominated[i] = True
                    break

    if return_mask:
        return ~dominated
    else:
        return np.where(~dominated)[0]


def pareto_ranking(
    fitness: np.ndarray, maximize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign Pareto ranks to solutions (iterative front extraction)

    This function is equivalent to MATEDA's Pareto_ordering.m

    Args:
        fitness: Fitness matrix (pop_size, n_objectives)
        maximize: If True, assume maximization problem. If False, minimization.

    Returns:
        Tuple of (ranks, ordered_indices):
        - ranks: Rank for each individual (0 = best front)
        - ordered_indices: Indices ordered by Pareto fronts

    Example:
        >>> fitness = np.array([[1, 2], [2, 1], [0.5, 0.5], [1.5, 1.5]])
        >>> ranks, order = pareto_ranking(fitness)
    """
    pop_size = fitness.shape[0]
    ranks = np.full(pop_size, -1)
    remaining = np.ones(pop_size, dtype=bool)
    current_rank = 0
    ordered_indices = []

    while np.any(remaining):
        # Find Pareto front in remaining individuals
        remaining_fitness = fitness[remaining]
        remaining_indices = np.where(remaining)[0]

        # Find non-dominated in remaining set
        pareto_mask = find_pareto_set(remaining_fitness, maximize, return_mask=True)
        pareto_indices = remaining_indices[pareto_mask]

        # Assign rank
        ranks[pareto_indices] = current_rank
        ordered_indices.extend(pareto_indices)

        # Remove from remaining
        remaining[pareto_indices] = False
        current_rank += 1

    return ranks, np.array(ordered_indices)


def fitness_ranking(fitness: np.ndarray, maximize: bool = True) -> np.ndarray:
    """
    Order individuals by average ranking across objectives

    This function is equivalent to MATEDA's fitness_ordering.m
    For single-objective: simple sorting
    For multi-objective: ranks each objective separately, then averages ranks

    Args:
        fitness: Fitness matrix (pop_size, n_objectives)
        maximize: If True, assume maximization problem. If False, minimization.

    Returns:
        Indices of individuals ordered by fitness (best first)

    Example:
        >>> fitness = np.array([[1, 2], [2, 1], [1.5, 1.5]])
        >>> order = fitness_ranking(fitness)
    """
    pop_size = fitness.shape[0]
    n_objectives = fitness.shape[1]

    if n_objectives == 1:
        # Single objective: simple sort
        if maximize:
            return np.argsort(fitness[:, 0])[::-1]
        else:
            return np.argsort(fitness[:, 0])

    # Multi-objective: average ranking
    cumulative_rank = np.zeros(pop_size)

    for obj_idx in range(n_objectives):
        # Sort by this objective
        if maximize:
            sorted_indices = np.argsort(fitness[:, obj_idx])[::-1]
        else:
            sorted_indices = np.argsort(fitness[:, obj_idx])

        # Assign ranks (1 to pop_size)
        ranks = np.empty(pop_size, dtype=float)
        ranks[sorted_indices] = np.arange(1, pop_size + 1)

        cumulative_rank += ranks

    # Order by cumulative rank (lower is better)
    return np.argsort(cumulative_rank)
