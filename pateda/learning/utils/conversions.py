"""
Utility functions for cardinality conversions

These functions handle conversions between different representations
of discrete variables, equivalent to MATEDA's utility functions.
"""

import numpy as np
from typing import List


def find_acc_card(size: int, card: np.ndarray) -> np.ndarray:
    """
    Calculate accumulated cardinalities for index conversions

    Args:
        size: Number of variables
        card: Cardinalities of variables

    Returns:
        Accumulated cardinalities array
    """
    acc_card = np.zeros(size, dtype=int)
    acc_card[0] = 1
    for i in range(1, size):
        acc_card[i] = acc_card[i - 1] * card[i - 1]
    return acc_card


def num_convert_card(values: np.ndarray, size: int, acc_card: np.ndarray) -> int:
    """
    Convert variable values to a single index

    Args:
        values: Variable values (size,)
        size: Number of variables
        acc_card: Accumulated cardinalities

    Returns:
        Single index representing the combination
    """
    return int(np.dot(values, acc_card))


def index_convert_card(index: int, size: int, acc_card: np.ndarray) -> np.ndarray:
    """
    Convert single index to variable values

    Args:
        index: Single index
        size: Number of variables
        acc_card: Accumulated cardinalities

    Returns:
        Variable values array
    """
    values = np.zeros(size, dtype=int)
    remaining = index

    for i in range(size - 1, -1, -1):
        values[i] = remaining // acc_card[i]
        remaining = remaining % acc_card[i]

    return values
