"""
Additively Decomposable Benchmark Functions

This module contains various additively decomposable benchmark functions
ported from the C++ EDA implementation. These functions are commonly used
for testing Estimation of Distribution Algorithms (EDAs).

The functions are organized by category:
- K-Deceptive variants
- Deceptive-3 variants
- Hierarchical functions (HIFF, fhtrap1)
- Cuban functions
- Polytree functions
"""

import numpy as np
from typing import Union, Callable


# ============================================================================
# K-Deceptive Functions
# ============================================================================

def k_deceptive_partition(vector: np.ndarray, k: int) -> float:
    """
    Evaluate a single partition of the K-deceptive function

    Args:
        vector: Binary vector for this partition
        k: Size of the partition

    Returns:
        Fitness value for this partition

    Note:
        - If all bits are 1 (sum==k), returns k (global optimum)
        - Otherwise, returns k - 1 - sum(vector) (deceptive local optima)
    """
    s = np.sum(vector)
    if s == k:
        return float(k)
    else:
        return float(k - 1 - s)


def k_deceptive(vector: np.ndarray, k: int = 3) -> float:
    """
    K-Deceptive function

    Deceptive function with k variables per partition.
    Each partition either has all 1s (optimal) or is deceptive.

    Args:
        vector: Binary solution vector (1D array)
        k: Number of variables in each partition (default: 3)

    Returns:
        Sum of deceptive values over all partitions

    Example:
        >>> x = np.array([1, 1, 1, 0, 0, 0])
        >>> k_deceptive(x, k=3)
        5.0  # First partition: 3, second partition: 2
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    val = 0.0

    for i in range(0, n, k):
        partition = vector[i:i + k]
        val += k_deceptive_partition(partition, k)

    return val


def gen_k_decep(vector: np.ndarray, k: int = 3, cardinality: int = 2) -> float:
    """
    Generalized K-Deceptive function for integer values

    Extension of k-deceptive for multi-valued (non-binary) variables.

    Args:
        vector: Integer solution vector (1D array)
        k: Number of variables in each partition (default: 3)
        cardinality: Number of possible values for each variable (default: 2)

    Returns:
        Sum of generalized deceptive values over all partitions

    Note:
        - Optimal: sum of partition equals (cardinality-1)*k
        - Second best: sum equals 0
        - Otherwise: penalized based on distance from optimal
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0
    optimal = (cardinality - 1) * k

    for i in range(0, n, k):
        partition_sum = np.sum(vector[i:i + k])

        if partition_sum == optimal:
            total += optimal
        elif partition_sum == 0:
            total += (optimal - 1)
        else:
            total += (optimal - partition_sum - 1)

    return total


def gen_k_decep_overlap(vector: np.ndarray, k: int = 3,
                        cardinality: int = 2, overlap: int = 1) -> float:
    """
    Generalized K-Deceptive function with overlapping partitions

    Similar to gen_k_decep but with overlapping partitions.

    Args:
        vector: Integer solution vector (1D array)
        k: Number of variables in each partition (default: 3)
        cardinality: Number of possible values for each variable (default: 2)
        overlap: Number of overlapping variables between partitions (default: 1)

    Returns:
        Sum of generalized deceptive values over all overlapping partitions
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    optimal = (cardinality - 1) * k
    num_groups = 1 + (n - k) // (k - overlap)

    total = 0.0
    pos = 0

    for i in range(num_groups):
        partition_sum = np.sum(vector[pos:pos + k])

        if partition_sum == optimal:
            total += optimal
        elif partition_sum == 0:
            total += (optimal - 1)
        else:
            total += (optimal - partition_sum - 1)

        pos += (k - overlap)

    return total


# ============================================================================
# Deceptive-3 Functions
# ============================================================================

# Lookup table for 3-bit deceptive function
DECEP3_TABLE = np.array([0.9, 0.8, 0.8, 0.0, 0.8, 0.0, 0.0, 1.0])


def decep3(vector: np.ndarray, overlap: bool = True) -> float:
    """
    Deceptive-3 function using lookup table

    Args:
        vector: Binary solution vector (1D array)
        overlap: If True, use overlapping partitions (step=2),
                 if False, use non-overlapping (step=3)

    Returns:
        Sum of deceptive values from lookup table

    Note:
        Uses a lookup table indexed by the 3-bit value
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0
    step = 2 if overlap else 3

    for i in range(0, n - 2, step):
        # Compute index: buff[i] + 2*buff[i+1] + 4*buff[i+2]
        idx = vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2]
        total += DECEP3_TABLE[idx]

    return total


# Marta's deceptive-3 lookup table
DECEP_MARTA_TABLE = np.array([0.553827, 0.049179, 0.856078, 1.09925,
                               0.980221, -0.298355, 0.370961, -0.192739])


def decep_marta3(vector: np.ndarray) -> float:
    """
    Marta's Deceptive-3 function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of deceptive values using Marta's lookup table
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 3):
        # Index: 4*buff[i] + 2*buff[i+1] + buff[i+2]
        idx = 4 * vector[i] + 2 * vector[i + 1] + vector[i + 2]
        total += DECEP_MARTA_TABLE[idx]

    return total


# New Marta's deceptive-3 lookup table
DECEP_MARTA3_NEW_TABLE = np.array([1.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.5])


def decep_marta3_new(vector: np.ndarray) -> float:
    """
    New Marta's Deceptive-3 function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of deceptive values using new Marta's lookup table
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 3):
        # Index: 4*buff[i] + 2*buff[i+1] + buff[i+2]
        idx = 4 * vector[i] + 2 * vector[i + 1] + vector[i + 2]
        total += DECEP_MARTA3_NEW_TABLE[idx]

    return total


# MH's deceptive-3 lookup table
DECEP3_MH_TABLE = np.array([2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0])


def decep3_mh(vector: np.ndarray) -> float:
    """
    MH's Deceptive-3 function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of MH's deceptive values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 3):
        # Index: buff[i] + 2*buff[i+1] + 4*buff[i+2]
        idx = vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2]
        total += DECEP3_MH_TABLE[idx]

    return total


def two_peaks_decep3(vector: np.ndarray) -> float:
    """
    Two-Peaks Deceptive-3 function

    This function has two peaks depending on the first bit value.

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of deceptive values (depends on first bit)
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    if vector[0] == 0:
        # Use standard deceptive-3
        for i in range(1, n, 3):
            if i + 2 < n:
                idx = vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2]
                total += DECEP3_TABLE[idx]
    else:
        # Use inverted deceptive-3
        for i in range(1, n, 3):
            if i + 2 < n:
                idx = vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2]
                total += 1.0 - DECEP3_TABLE[idx]

    return total


# Venturini's deceptive lookup table
DECEP_VENTURINI_TABLE = np.array([0.2, 1.4, 1.6, 2.9, 3.0, 1.0, 0.8, 0.6])


def decep_venturini(vector: np.ndarray) -> float:
    """
    Venturini's Deceptive function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of Venturini's deceptive values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 3):
        # Index: buff[i+2] + 2*buff[i+1] + 4*buff[i]
        if i + 2 < n:
            idx = vector[i + 2] + 2 * vector[i + 1] + 4 * vector[i]
            total += DECEP_VENTURINI_TABLE[idx]

    return total


# ============================================================================
# Hard Deceptive-5 Function
# ============================================================================

# Lookup table for hard deceptive-5
HARD_DECEP5_TABLE = np.array([0.9, 0.85, 0.8, 0.75, 0.0, 1.0])


def hard_decep5(vector: np.ndarray) -> float:
    """
    Hard Deceptive-5 function

    A harder version of the 5-bit deceptive function.

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of hard deceptive-5 values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 5):
        if i + 4 < n:
            u = np.sum(vector[i:i + 5])
            total += HARD_DECEP5_TABLE[int(u)]

    return total


# ============================================================================
# Hierarchical Functions
# ============================================================================

def hiff(vector: np.ndarray) -> float:
    """
    Hierarchical If and only If (HIFF) function

    A hierarchical function that rewards building blocks at multiple scales.
    Works with problem sizes that are powers of 2.

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        HIFF fitness value

    Note:
        Problem size must be a power of 2 (e.g., 16, 32, 64, 128, 256, 512, 1024)
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    powers = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

    auxbuff = np.zeros(1024, dtype=int)
    auxn = n
    total = float(n)  # Base level contribution
    j = 1

    while auxn >= 2:
        level_idx = 0

        if auxn == n:
            # First level - process original vector
            for k in range(0, n, 2):
                nv = vector[k] + vector[k + 1]
                total += (nv == 2 or nv == 0) * powers[j]

                if nv == 2:
                    auxbuff[level_idx] = 1
                elif nv == 0:
                    auxbuff[level_idx] = 0
                else:
                    auxbuff[level_idx] = -10
                level_idx += 1

            auxn = n // 2
        elif auxn == 2:
            # Final level
            nv = auxbuff[0] + auxbuff[1]
            if nv == 2 or nv == 0:
                total += powers[j]
            auxn //= 2
        else:
            # Intermediate levels
            for k in range(0, auxn, 2):
                nv = auxbuff[k] + auxbuff[k + 1]
                total += (nv == 2 or nv == 0) * powers[j]

                if nv == 2:
                    auxbuff[level_idx] = 1
                elif nv == 0:
                    auxbuff[level_idx] = 0
                else:
                    auxbuff[level_idx] = -10
                level_idx += 1

            auxn //= 2

        j += 1

    return total


def fhtrap1(vector: np.ndarray) -> float:
    """
    Hierarchical Trap-3 function (fhtrap1)

    Similar to HIFF but using trap-3 building blocks.
    Works with problem sizes that are powers of 3.

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Hierarchical trap fitness value

    Note:
        Problem size must be a power of 3 (e.g., 9, 27, 81, 243, 729)
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    powers = np.array([1, 3, 9, 27, 81, 243, 729])

    auxbuff = np.zeros(729, dtype=int)
    auxn = n
    total = 0.0
    j = 1

    while auxn >= 3:
        level_idx = 0

        if auxn == n:
            # First level - process original vector
            for k in range(0, n, 3):
                nv = vector[k] + vector[k + 1] + vector[k + 2]
                total += (nv == 3 or nv == 0) * powers[j]

                if nv == 3:
                    auxbuff[level_idx] = 1
                elif nv == 0:
                    auxbuff[level_idx] = 0
                else:
                    auxbuff[level_idx] = -10
                level_idx += 1

            auxn = n // 3
        elif auxn == 3:
            # Final level
            nv = auxbuff[0] + auxbuff[1] + auxbuff[2]
            if nv == 3 or nv == 0:
                total += powers[j]
            auxn //= 3
        else:
            # Intermediate levels
            for k in range(0, auxn, 3):
                nv = auxbuff[k] + auxbuff[k + 1] + auxbuff[k + 2]
                total += (nv == 3 or nv == 0) * powers[j]

                if nv == 3:
                    auxbuff[level_idx] = 1
                elif nv == 0:
                    auxbuff[level_idx] = 0
                else:
                    auxbuff[level_idx] = -10
                level_idx += 1

            auxn //= 3

        j += 1

    return total


# ============================================================================
# Polytree Functions (Ochoa)
# ============================================================================

# Ochoa's Polytree-3 lookup table
FIRST_POLYTREE3 = np.array([1.042, -0.736, 0.357, -1.421,
                             -0.083, 0.092, -0.768, -0.592])


def first_polytree3_ochoa(vector: np.ndarray, overlap: bool = False) -> float:
    """
    Ochoa's First Polytree-3 function

    Args:
        vector: Binary solution vector (1D array)
        overlap: If True, use overlapping partitions (step=2),
                 if False, use non-overlapping (step=3)

    Returns:
        Sum of polytree values from lookup table
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0
    step = 2 if overlap else 3

    for i in range(0, n - 2, step):
        # Index: buff[i] + 2*buff[i+1] + 4*buff[i+2]
        idx = vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2]
        total += FIRST_POLYTREE3[idx]

    return total


# Ochoa's Polytree-5 lookup table
FIRST_POLYTREE5 = np.array([
    -1.141, 1.334, -5.353, -1.700, 0.063, -0.815, -0.952, -0.652,
    0.753, 1.723, -4.964, -1.311, 1.454, 0.576, 0.439, 0.739,
    -3.527, 1.051, -7.738, -4.085, 1.002, 0.124, -0.013, 0.286,
    -6.664, -4.189, -10.876, -7.223, -1.133, -2.011, -2.148, -1.849
])


def first_polytree5_ochoa(vector: np.ndarray) -> float:
    """
    Ochoa's First Polytree-5 function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of polytree-5 values from lookup table
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 5):
        if i + 4 < n:
            # Index: buff[i] + 2*buff[i+1] + 4*buff[i+2] + 8*buff[i+3] + 16*buff[i+4]
            idx = (vector[i] + 2 * vector[i + 1] + 4 * vector[i + 2] +
                   8 * vector[i + 3] + 16 * vector[i + 4])
            total += FIRST_POLYTREE5[idx]

    return total


# ============================================================================
# Cuban Functions (Fc2, Fc3, Fc4, Fc5)
# ============================================================================

# F5Muhl lookup table
F3_CUBAN1 = np.array([0.595, 0.2, 0.595, 0.1, 1.0, 0.05, 0.09, 0.15])


def f5_muhl(partition: np.ndarray) -> float:
    """
    F5Muhl building block for Fc2

    Args:
        partition: 5-bit binary partition

    Returns:
        Fitness contribution
    """
    # Define the target patterns
    if np.array_equal(partition, [0, 0, 0, 0, 1]):
        return 3.0
    elif np.array_equal(partition, [0, 0, 0, 1, 1]):
        return 2.0
    elif np.array_equal(partition, [0, 0, 1, 1, 1]):
        return 1.0
    elif np.array_equal(partition, [1, 1, 1, 1, 1]):
        return 3.5
    elif np.array_equal(partition, [0, 0, 0, 0, 0]):
        return 4.0
    else:
        return 0.0


def fc2(vector: np.ndarray) -> float:
    """
    Fc2 function - F5Muhl multimodal function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of F5Muhl values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 5):
        if i + 4 < n:
            partition = vector[i:i + 5]
            total += f5_muhl(partition)

    return total


def g_multimodal(partition: np.ndarray) -> float:
    """
    Helper function for F5Multimodal

    Returns 1 if sum is odd, 0 if even
    """
    return 1.0 if int(np.sum(partition)) % 2 == 1 else 0.0


def f5_multimodal(partition: np.ndarray) -> float:
    """
    F5Multimodal building block for Fc3

    Args:
        partition: 5-bit binary partition

    Returns:
        Fitness contribution
    """
    return np.sum(partition) + 2 * g_multimodal(partition)


def fc3(vector: np.ndarray) -> float:
    """
    Fc3 function - F5Multimodal function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of F5Multimodal values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, 5):
        if i + 4 < n:
            partition = vector[i:i + 5]
            total += f5_multimodal(partition)

    return total


def f5_cuban1(partition: np.ndarray) -> float:
    """
    F5Cuban1 building block for Fc4

    Args:
        partition: 5-bit binary partition

    Returns:
        Fitness contribution
    """
    if partition[1] == partition[3] and partition[2] == partition[4]:
        idx = 4 * partition[0] + 2 * partition[1] + partition[2]
        return 4 * F3_CUBAN1[idx]
    else:
        return 0.0


def fc4(vector: np.ndarray) -> float:
    """
    Fc4 function - F5Cuban1 function

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of F5Cuban1 values

    Note:
        Processes (n-1)/4 partitions of 4 bits each
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    m = (n - 1) // 4
    total = 0.0

    for i in range(m):
        partition = vector[4 * i:4 * i + 5]
        total += f5_cuban1(partition)

    return total


def f5_cuban2(partition: np.ndarray) -> float:
    """
    F5Cuban2 building block for Fc5

    Args:
        partition: 5-bit binary partition

    Returns:
        Fitness contribution
    """
    if partition[4] == 0:
        return np.sum(partition)
    else:
        if partition[0] == 0:
            return 0.0
        else:
            return np.sum(partition) - 2


def fc5(vector: np.ndarray) -> float:
    """
    Fc5 function - Combined Cuban functions

    Args:
        vector: Binary solution vector (1D array)

    Returns:
        Sum of combined Cuban function values
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    m = (n - 5) // 8

    total = f5_cuban1(vector[0:5])

    for i in range(m):
        total += f5_cuban2(vector[4 * (2 * i + 1):4 * (2 * i + 1) + 5])
        total += f5_cuban1(vector[4 * (2 * i + 2):4 * (2 * i + 2) + 5])

    return total


# ============================================================================
# Factory Functions for EDA Integration
# ============================================================================

def create_k_deceptive_function(k: int = 3):
    """Create a K-deceptive objective function for use with EDA"""
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([k_deceptive(population, k)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = k_deceptive(population[i], k)
        return fitness
    return objective


def create_hiff_function():
    """Create a HIFF objective function for use with EDA"""
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([hiff(population)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = hiff(population[i])
        return fitness
    return objective


def create_decep3_function(overlap: bool = True):
    """Create a Decep3 objective function for use with EDA"""
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([decep3(population, overlap)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = decep3(population[i], overlap)
        return fitness
    return objective


def create_polytree3_function(overlap: bool = False):
    """Create a Polytree-3 objective function for use with EDA"""
    def objective(population: np.ndarray) -> np.ndarray:
        if population.ndim == 1:
            return np.array([first_polytree3_ochoa(population, overlap)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = first_polytree3_ochoa(population[i], overlap)
        return fitness
    return objective
