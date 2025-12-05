"""
Continuous Benchmark Functions for Optimization

This module provides standard benchmark functions for continuous optimization.
All functions are defined as minimization problems.
"""

import numpy as np
from typing import Union


def sphere(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Sphere function: f(x) = sum(x_i^2)

    Global minimum: f(0, ..., 0) = 0
    Domain: typically [-5.12, 5.12]^n

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def rastrigin(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Global minimum: f(0, ..., 0) = 0
    Domain: typically [-5.12, 5.12]^n
    Highly multimodal with many local optima

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    else:
        n = x.shape[1]
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def rosenbrock(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Global minimum: f(1, ..., 1) = 0
    Domain: typically [-5, 10]^n
    Valley-shaped, difficult to optimize

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    else:
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def ackley(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Ackley function

    Global minimum: f(0, ..., 0) = 0
    Domain: typically [-32.768, 32.768]^n
    Highly multimodal with a nearly flat outer region

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        return term1 + term2 + 20 + np.e
    else:
        n = x.shape[1]
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / n)
        return term1 + term2 + 20 + np.e


def griewank(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Griewank function

    Global minimum: f(0, ..., 0) = 0
    Domain: typically [-600, 600]^n
    Multimodal with many local optima

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        i = np.arange(1, n + 1)
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        return sum_term - prod_term + 1
    else:
        n = x.shape[1]
        i = np.arange(1, n + 1)
        sum_term = np.sum(x**2, axis=1) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(i)), axis=1)
        return sum_term - prod_term + 1


def schwefel(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Schwefel function

    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    Domain: typically [-500, 500]^n
    Highly multimodal with a distant global optimum

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        n = x.shape[1]
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def levy(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Levy function

    Global minimum: f(1, ..., 1) = 0
    Domain: typically [-10, 10]^n
    Multimodal

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    def _levy_single(x):
        w = 1 + (x - 1) / 4
        n = len(w)

        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        return term1 + term2 + term3

    if x.ndim == 1:
        return _levy_single(x)
    else:
        return np.array([_levy_single(row) for row in x])


def michalewicz(x: np.ndarray, m: int = 10) -> Union[float, np.ndarray]:
    """
    Michalewicz function

    Global minimum: depends on dimension (e.g., -1.8013 for n=2, -9.66 for n=10)
    Domain: [0, π]^n
    Multimodal with steep valleys

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)
    m : int
        Steepness parameter (default: 10)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        i = np.arange(1, n + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))
    else:
        n = x.shape[1]
        i = np.arange(1, n + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m), axis=1)


def zakharov(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Zakharov function

    Global minimum: f(0, ..., 0) = 0
    Domain: typically [-5, 10]^n

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        n = len(x)
        i = np.arange(1, n + 1)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2**2 + sum2**4
    else:
        n = x.shape[1]
        i = np.arange(1, n + 1)
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(0.5 * i * x, axis=1)
        return sum1 + sum2**2 + sum2**4


def sum_function(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Simple sum function: f(x) = sum(x_i)

    Used for testing and simple optimization problems.
    Global minimum: at lower bounds

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix (pop_size, n_vars)

    Returns
    -------
    float or np.ndarray
        Function value(s)
    """
    if x.ndim == 1:
        return np.sum(x)
    else:
        return np.sum(x, axis=1)


# Dictionary of all available functions
CONTINUOUS_FUNCTIONS = {
    'sphere': sphere,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock,
    'ackley': ackley,
    'griewank': griewank,
    'schwefel': schwefel,
    'levy': levy,
    'michalewicz': michalewicz,
    'zakharov': zakharov,
    'sum': sum_function,
}


def get_function(name: str):
    """
    Get a continuous benchmark function by name.

    Parameters
    ----------
    name : str
        Name of the function

    Returns
    -------
    callable
        The benchmark function

    Raises
    ------
    ValueError
        If function name is not recognized
    """
    if name not in CONTINUOUS_FUNCTIONS:
        raise ValueError(
            f"Unknown function: {name}. "
            f"Available: {list(CONTINUOUS_FUNCTIONS.keys())}"
        )
    return CONTINUOUS_FUNCTIONS[name]
