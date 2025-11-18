"""
Sampling utility functions
"""

import numpy as np


def stochastic_universal_sampling(n_samples: int, cum_probs: np.ndarray) -> np.ndarray:
    """
    Stochastic Universal Sampling (SUS)

    A low-variance sampling method that selects samples according to a probability
    distribution. Similar to roulette wheel selection but with evenly spaced pointers.

    Equivalent to MATEDA's sus.m

    Args:
        n_samples: Number of samples to draw
        cum_probs: Cumulative probability distribution (must end with 1.0)

    Returns:
        Array of indices indicating which class each sample belongs to
    """
    # Start at random position
    pointer = np.random.rand()
    step = 1.0 / n_samples

    indices = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Find which bin this pointer falls into
        idx = np.searchsorted(cum_probs, pointer, side="right")
        indices[i] = idx

        # Move to next evenly-spaced pointer
        pointer += step
        if pointer > 1.0:
            pointer -= 1.0

    # Randomize order to avoid bias
    np.random.shuffle(indices)

    return indices
