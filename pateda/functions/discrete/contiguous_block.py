"""
Contiguous Block Problem

This module implements a fitness function for finding a contiguous block of K ones
in a binary vector. The problem requires finding K consecutive ones in a binary
vector of length n (where n >= k).

This is a pedagogical problem useful for demonstrating:
- Seeding methods (e.g., initializing with exactly k ones)
- Local optimization (moving the block of ones to be contiguous)
- Repairing procedures (maintaining the k-ones constraint after sampling)

Author: Created for PATEDA examples
Date: 2025
"""

import numpy as np


def contiguous_block(x: np.ndarray, k: int = None) -> float:
    """
    Contiguous block fitness function.

    Finds the maximum length of a contiguous block of ones in a binary vector.
    The optimal solution has k consecutive ones (and n-k zeros).

    Args:
        x: Binary vector of length n
        k: Expected number of ones (optional, used for normalization)

    Returns:
        Fitness value: maximum length of contiguous ones found

    Examples:
        >>> contiguous_block(np.array([0, 0, 1, 1, 1, 0, 0]))
        3.0
        >>> contiguous_block(np.array([1, 0, 1, 0, 1, 0, 1]))
        1.0
        >>> contiguous_block(np.array([1, 1, 1, 1, 1, 0, 0]))
        5.0
    """
    if len(x) == 0:
        return 0.0

    # Find all contiguous blocks of ones
    max_block = 0
    current_block = 0

    for bit in x:
        if bit == 1:
            current_block += 1
            max_block = max(max_block, current_block)
        else:
            current_block = 0

    return float(max_block)


def contiguous_block_with_penalty(x: np.ndarray, k: int) -> float:
    """
    Contiguous block fitness with penalty for wrong number of ones.

    This variant penalizes solutions that don't have exactly k ones,
    which can be useful when used without a repairing method.

    Args:
        x: Binary vector of length n
        k: Target number of ones

    Returns:
        Fitness value: max contiguous block length - penalty for wrong k

    Examples:
        >>> contiguous_block_with_penalty(np.array([0, 1, 1, 1, 0]), k=3)
        3.0
        >>> contiguous_block_with_penalty(np.array([1, 1, 1, 1, 0]), k=3)
        3.0  # penalty for having 4 ones instead of 3
    """
    n_ones = np.sum(x)
    max_block = contiguous_block(x, k)

    # Penalty for wrong number of ones
    penalty = abs(n_ones - k)

    return float(max_block - penalty)


def create_contiguous_block_function(k: int, with_penalty: bool = False):
    """
    Factory function to create a contiguous block fitness function with fixed k.

    This is useful for creating a fitness function that can be passed to EDA
    without needing to specify k each time.

    Args:
        k: Target number of ones
        with_penalty: If True, use penalty version

    Returns:
        Fitness function that takes only x as argument

    Examples:
        >>> fitness_func = create_contiguous_block_function(k=5)
        >>> fitness_func(np.array([0, 1, 1, 1, 1, 1, 0]))
        5.0
    """
    if with_penalty:
        return lambda x: contiguous_block_with_penalty(x, k)
    else:
        return lambda x: contiguous_block(x, k)
