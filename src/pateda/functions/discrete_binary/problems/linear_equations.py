"""
System of Linear Equations function for optimization benchmarking

Based on Section 3.4 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997).
The goal is to solve DX = B (minimize absolute difference |DX - B|) where D is a matrix,
B is a vector, and X is an integer vector encoded in binary/Gray code.
"""

import numpy as np


class LinearEquationsInstance:
    """
    Represents an instance of the System of Linear Equations problem.
    """
    def __init__(self, D: np.ndarray, B: np.ndarray, target_X: np.ndarray = None,
                 bits_per_param: int = 9):
        """
        Args:
            D: Coefficient matrix of shape (m_eqs, n_vars).
            B: Constant vector of shape (m_eqs,).
            target_X: The target integer solution vector of shape (n_vars,), if known.
            bits_per_param: Number of bits used to encode each integer variable.
                A binary solution vector therefore has ``n_vars * bits_per_param`` bits.
        """
        self.D = D
        self.B = B
        self.target_X = target_X
        self.bits_per_param = bits_per_param
        self.m_eqs, self.n_vars = D.shape


def generate_random_linear_equations(n_vars: int = 9, bits_per_param: int = 9) -> LinearEquationsInstance:
    """
    Generate a random System of Linear Equations instance.
    Generates a random coefficient matrix D and a random target solution target_X,
    then sets B = D * target_X to guarantee a solution exists.

    Args:
        n_vars: Number of variables (default: 9).
        bits_per_param: Number of bits per variable (default: 9, range [-256, 255]).

    Returns:
        A LinearEquationsInstance object.
    """
    # D is a random matrix with integers in [-10, 10]
    D = np.random.randint(-10, 11, size=(n_vars, n_vars)).astype(float)

    # Generate a target solution in the range [-2**(bits-1), 2**(bits-1)-1]
    offset = 1 << (bits_per_param - 1)
    target_X = np.random.randint(-offset, offset, size=n_vars)

    # Set B = D @ target_X
    B = D @ target_X

    return LinearEquationsInstance(D, B, target_X, bits_per_param=bits_per_param)


def decode_binary_to_integers(x: np.ndarray, bits_per_param: int = 9, encoding: str = 'binary') -> np.ndarray:
    """
    Decode a binary vector into a vector of integers.
    Each parameter is represented by bits_per_param bits and mapped to
    the integer range [-2**(bits-1), 2**(bits-1)-1].

    Args:
        x: Binary vector.
        bits_per_param: Number of bits per parameter (default: 9).
        encoding: 'binary' or 'gray'.

    Returns:
        1D array of decoded integers.
    """
    n_vars = len(x)
    if n_vars % bits_per_param != 0:
        raise ValueError(f"Vector length ({n_vars}) must be a multiple of bits_per_param ({bits_per_param})")

    n_params = n_vars // bits_per_param
    X = np.zeros(n_params, dtype=int)
    offset = 1 << (bits_per_param - 1)

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

        X[i] = val - offset

    return X


def eval_linear_equations_real(X: np.ndarray, instance: LinearEquationsInstance) -> float:
    """
    Evaluate the error (sum of absolute differences) for a given integer vector X.

    Args:
        X: The proposed integer vector solution.
        instance: The LinearEquationsInstance to evaluate against.

    Returns:
        Summed absolute difference between D*X and B (fitness to minimize).
    """
    pred_B = instance.D @ X
    error = np.sum(np.abs(pred_B - instance.B))
    return float(error)


def eval_linear_equations(x: np.ndarray, instance: LinearEquationsInstance, encoding: str = 'binary') -> float:
    """
    Evaluate the error for a binary vector by decoding it first.

    Args:
        x: Binary vector.
        instance: The LinearEquationsInstance to evaluate against.
        encoding: 'binary' or 'gray'.

    Returns:
        Summed absolute difference between D*X and B (fitness to minimize).
    """
    if x.ndim == 2:
        x = x.flatten()

    X = decode_binary_to_integers(x, bits_per_param=instance.bits_per_param, encoding=encoding)
    return eval_linear_equations_real(X, instance)


def create_linear_equations_objective_function(instance: LinearEquationsInstance, encoding: str = 'binary'):
    """
    Create a System of Linear Equations objective function for use with EDAs.
    Note: The goal is to minimize this error.

    Args:
        instance: The LinearEquationsInstance to evaluate against.
        encoding: 'binary' or 'gray'.

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate System of Linear Equations for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([eval_linear_equations(population, instance, encoding)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_linear_equations(population[i], instance, encoding)

        return fitness

    return objective
