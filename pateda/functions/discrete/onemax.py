"""
OneMax function

Simple test function that sums all bits. Optimal solution is all 1s.
"""

import numpy as np


def onemax(x: np.ndarray) -> float:
    """
    OneMax function: sum of all elements

    This is one of the simplest test functions for binary optimization.
    The optimal solution is all ones.

    Args:
        x: Binary vector (or integer vector)

    Returns:
        Sum of all elements
    """
    return float(np.sum(x))
