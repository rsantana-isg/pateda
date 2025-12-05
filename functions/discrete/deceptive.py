"""
Deceptive3 function

A non-linear test function with deceptive properties, introduced by Goldberg.
The function is designed to mislead greedy algorithms toward local optima.
"""

import numpy as np


def _deceptive3_block(block: np.ndarray) -> float:
    """
    Evaluate a single 3-bit deceptive block

    The deceptive function has the following values:
    - sum=3 (111): 1.0 (global optimum)
    - sum=2 (e.g., 110): 0.0 (deceptive - looks bad despite being close to optimum)
    - sum=1 (e.g., 100): 0.8 (looks good - deceptive local optimum)
    - sum=0 (000): 0.9 (almost as good as sum=1)

    Args:
        block: 3-element binary array

    Returns:
        Fitness value for this block
    """
    s = int(np.sum(block))

    if s == 3:
        return 1.0
    elif s == 1:
        return 0.8
    elif s == 2:
        return 0.0
    else:  # s == 0
        return 0.9


def deceptive3(x: np.ndarray) -> float:
    """
    Non-overlapping Deceptive3 function

    Decomposes the input into non-overlapping 3-bit blocks and sums
    the deceptive function value for each block.

    f(x) = f_d(x[0:3]) + f_d(x[3:6]) + ... + f_d(x[-3:])

    The function requires that len(x) is a multiple of 3.

    Args:
        x: Binary vector with length that is a multiple of 3

    Returns:
        Sum of deceptive function values over all blocks

    Raises:
        ValueError: If length of x is not a multiple of 3
    """
    n_vars = len(x)

    if n_vars % 3 != 0:
        raise ValueError(f"Deceptive3 requires vector length to be multiple of 3, got {n_vars}")

    val = 0.0

    # Evaluate each 3-bit block
    for i in range(0, n_vars, 3):
        block = x[i : i + 3]
        val += _deceptive3_block(block)

    return val
