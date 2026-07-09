"""
Summation Cancellation function for optimization benchmarking

Based on Section 3.4 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997).
The goal is to maximize the inverse of the sum of absolute cumulative parameters.
Parameters are encoded using 5 bits each, in either standard binary or Gray code.
"""

import numpy as np


def decode_binary_to_real(x: np.ndarray, bits_per_param: int = 5, encoding: str = 'binary') -> np.ndarray:
    """
    Decode a binary vector into a vector of real numbers.
    Each parameter is represented by bits_per_param bits and mapped
    uniformly to the range [-0.16, 0.15].

    Args:
        x: Binary vector.
        bits_per_param: Number of bits per parameter (default: 5).
        encoding: 'binary' for standard binary, 'gray' for Gray coding.

    Returns:
        1D array of decoded real parameters.
    """
    n_vars = len(x)
    if n_vars % bits_per_param != 0:
        raise ValueError(f"Vector length ({n_vars}) must be a multiple of bits_per_param ({bits_per_param})")

    n_params = n_vars // bits_per_param
    s = np.zeros(n_params)
    max_val = (1 << bits_per_param) - 1

    # Mapping to [-0.16, 0.15]
    min_range = -0.16
    max_range = 0.15
    spacing = (max_range - min_range) / max_val

    for i in range(n_params):
        block = x[i * bits_per_param : (i + 1) * bits_per_param]

        if encoding == 'gray':
            # Decode Gray code to binary
            bin_block = np.zeros_like(block)
            bin_block[0] = block[0]
            for j in range(1, len(block)):
                bin_block[j] = bin_block[j-1] ^ block[j]
            block = bin_block

        # Convert binary block (MSB first) to integer
        val = 0
        for j, bit in enumerate(block):
            val += int(bit) * (1 << (bits_per_param - 1 - j))

        s[i] = min_range + val * spacing

    return s


def summation_cancellation_real(s: np.ndarray) -> float:
    """
    Evaluate Summation Cancellation directly on real-valued parameters.

    Args:
        s: 1D array of real parameters.

    Returns:
        Fitness value.
    """
    n = len(s)
    if n == 0:
        return 0.0

    y = np.zeros(n)
    y[0] = s[0]
    for i in range(1, n):
        y[i] = s[i] + y[i - 1]

    c = 1e-5
    sum_abs_y = np.sum(np.abs(y))
    return float(1.0 / (c + sum_abs_y))


def summation_cancellation(x: np.ndarray, encoding: str = 'binary') -> float:
    """
    Evaluate Summation Cancellation on a binary vector by decoding it first.

    Args:
        x: Binary vector.
        encoding: 'binary' or 'gray'.

    Returns:
        Fitness value.
    """
    if x.ndim == 2:
        x = x.flatten()

    s = decode_binary_to_real(x, bits_per_param=5, encoding=encoding)
    return summation_cancellation_real(s)


def create_summation_cancellation_objective_function(encoding: str = 'binary'):
    """
    Create a Summation Cancellation objective function for use with EDAs

    Args:
        encoding: 'binary' or 'gray'.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Summation Cancellation for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([summation_cancellation(population, encoding)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = summation_cancellation(population[i], encoding)

        return fitness

    return objective
