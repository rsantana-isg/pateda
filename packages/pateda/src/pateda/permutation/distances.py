"""
Permutation distance metrics

This module implements various distance metrics for permutations,
which are used in Mallows models and other permutation-based EDAs.

References:
    [1] P. Diaconis: Group representations in probability and statistics.
        Institute of Mathematical Statistics Volume 11, 1988
    [2] J. Ceberio, A. Mendiburu, J.A Lozano: Introducing the Mallows Model
        on Estimation of Distribution Algorithms. ICONIP 2011
"""

import numpy as np
from typing import Union


def kendall_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Calculate the Kendall-tau distance between two permutations.

    The Kendall distance measures the number of pairs of items which have
    opposing ordering. In other words, it's the minimum number of adjacent
    transpositions needed to transform perm1 into perm2.

    Args:
        perm1: First permutation (1-indexed or 0-indexed)
        perm2: Second permutation (1-indexed or 0-indexed)

    Returns:
        The Kendall distance between the two permutations

    Example:
        >>> perm1 = np.array([1, 2, 3, 4])
        >>> perm2 = np.array([2, 1, 4, 3])
        >>> kendall_distance(perm1, perm2)
        2
    """
    perm1 = np.array(perm1, dtype=int)
    perm2 = np.array(perm2, dtype=int)

    # Get inverse of perm2
    invperm2 = np.argsort(perm2)

    # Compose perm1 with inverse of perm2
    composition = perm1[invperm2]

    # Calculate the v-vector (Lehmer code) and sum it
    return int(np.sum(_v_vector(composition)))


def _v_vector(perm: np.ndarray) -> np.ndarray:
    """
    Calculate the v-vector (Lehmer code) for a permutation.

    The v-vector v[i] counts how many elements to the right of position i
    are smaller than perm[i].

    Args:
        perm: A permutation

    Returns:
        The v-vector of the permutation
    """
    n = len(perm)
    v = np.zeros(n, dtype=int)

    for i in range(n):
        v[i] = np.sum(perm[i] > perm[i+1:])

    return v


def cayley_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Calculate the Cayley distance between two permutations.

    The Cayley distance measures the minimum number of swaps (not necessarily
    adjacent) needed to transform perm1 into perm2.

    Args:
        perm1: First permutation (1-indexed or 0-indexed)
        perm2: Second permutation (1-indexed or 0-indexed)

    Returns:
        The Cayley distance between the two permutations

    Example:
        >>> perm1 = np.array([1, 2, 3, 4])
        >>> perm2 = np.array([2, 1, 4, 3])
        >>> cayley_distance(perm1, perm2)
        2

    References:
        [1] E. Irurozki, B. Calvo, J.A Lozano: Sampling and learning mallows
            and generalized mallows models under the cayley distance. Tech. Rep., 2013
        [2] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: Extending
            Distance-based Ranking Models in EDAs. CEC 2014
    """
    perm1 = np.array(perm1, dtype=int)
    perm2 = np.array(perm2, dtype=int)

    # Get inverse of perm2
    invperm2 = np.argsort(perm2)

    # Compose perm1 with inverse of perm2
    composition = perm1[invperm2]

    # Calculate the x-vector and sum it
    return int(np.sum(_x_vector(composition)))


def _x_vector(perm: np.ndarray) -> np.ndarray:
    """
    Calculate the x-vector for a permutation (used in Cayley distance).

    The x-vector x[i] is 0 if perm[i] == i, and 1 otherwise.
    This counts the number of positions that are not fixed points.

    Args:
        perm: A permutation

    Returns:
        The x-vector of the permutation
    """
    n = len(perm)
    x = np.zeros(n, dtype=int)

    for i in range(n):
        if perm[i] != i:
            x[i] = 1

    return x


def ulam_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Calculate the Ulam distance between two permutations.

    The Ulam distance is n minus the length of the longest increasing
    subsequence (LIS) when viewing perm1 as a sequence after composing
    with the inverse of perm2.

    Args:
        perm1: First permutation (1-indexed or 0-indexed)
        perm2: Second permutation (1-indexed or 0-indexed)

    Returns:
        The Ulam distance between the two permutations

    Example:
        >>> perm1 = np.array([1, 2, 3, 4])
        >>> perm2 = np.array([2, 1, 3, 4])
        >>> ulam_distance(perm1, perm2)
        1

    References:
        [1] S. M. Ulam: Monte Carlo calculations in problems of mathematical
            physics. Modern Mathematics for the Engineer, 1961
    """
    perm1 = np.array(perm1, dtype=int)
    perm2 = np.array(perm2, dtype=int)

    n = len(perm1)

    # Get inverse of perm2
    invperm2 = np.argsort(perm2)

    # Compose perm1 with inverse of perm2
    composition = perm1[invperm2]

    # Find longest increasing subsequence
    lis_length = _longest_increasing_subsequence(composition)

    return n - lis_length


def _longest_increasing_subsequence(sequence: np.ndarray) -> int:
    """
    Calculate the length of the longest increasing subsequence.

    Uses dynamic programming with binary search for O(n log n) complexity.

    Args:
        sequence: A sequence of integers

    Returns:
        The length of the longest increasing subsequence
    """
    n = len(sequence)
    if n == 0:
        return 0

    # tail[i] is the smallest tail element for all increasing subsequences of length i+1
    tail = []

    for num in sequence:
        # Binary search for the position to insert/replace
        left, right = 0, len(tail)
        while left < right:
            mid = (left + right) // 2
            if tail[mid] < num:
                left = mid + 1
            else:
                right = mid

        # If left == len(tail), append; otherwise replace
        if left == len(tail):
            tail.append(num)
        else:
            tail[left] = num

    return len(tail)


def hamming_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two permutations.

    The Hamming distance counts the number of positions at which
    the corresponding elements are different.

    Args:
        perm1: First permutation
        perm2: Second permutation

    Returns:
        The Hamming distance between the two permutations

    Example:
        >>> perm1 = np.array([1, 2, 3, 4])
        >>> perm2 = np.array([1, 3, 2, 4])
        >>> hamming_distance(perm1, perm2)
        2
    """
    return int(np.sum(perm1 != perm2))


def _x_vector_cycles(perm: np.ndarray) -> np.ndarray:
    """
    Calculate the cycle-based x-vector for a permutation (for Mallows model).

    The x-vector decomposes the permutation into cycles. For each cycle,
    the largest item gets x[i] = 0, all other items in the cycle get x[i] = 1.

    This is used in the Mallows model with Cayley distance for learning and sampling.

    Args:
        perm: A permutation (0-indexed)

    Returns:
        The x-vector of the permutation (length n-1)

    References:
        [1] E. Irurozki, B. Calvo, J.A Lozano: Sampling and learning mallows
            and generalized mallows models under the cayley distance. Tech. Rep., 2013
    """
    n = len(perm)
    x = np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    num_visited = 0

    while num_visited < n:
        # Find next unvisited item
        item = np.where(~visited)[0][0]

        # Trace the cycle starting from this item
        max_item_in_cycle = -1
        while not visited[item]:
            if item > max_item_in_cycle:
                max_item_in_cycle = item
            visited[item] = True
            num_visited += 1
            item = perm[item]

        # Mark largest item in cycle with 0
        x[max_item_in_cycle] = 0

    # Return x-vector without last position (always 0)
    return x[:-1]


def _generate_perm_from_x(x: np.ndarray, n: int) -> np.ndarray:
    """
    Generate a random permutation from an x-vector (inverse of x_vector_cycles).

    This is used for sampling in the Mallows model with Cayley distance.

    Args:
        x: The x-vector (length n-1)
        n: Size of the permutation

    Returns:
        A permutation consistent with the x-vector

    References:
        [1] E. Irurozki, B. Calvo, J.A Lozano: Sampling and learning mallows
            and generalized mallows models under the cayley distance. Tech. Rep., 2013
    """
    perm = np.arange(n, dtype=int)  # Start with identity permutation [0, 1, 2, ..., n-1]

    for pos in range(n - 1):
        if x[pos] == 1:
            # Randomly swap position 'pos' with a position in range [pos+1, n-1]
            random_pos = np.random.randint(pos + 1, n)
            perm[pos], perm[random_pos] = perm[random_pos], perm[pos]

    return perm
