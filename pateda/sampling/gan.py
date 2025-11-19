"""
GAN Model Sampling for Continuous EDAs

This module provides sampling algorithms for Generative Adversarial Network (GAN) based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch

# Import network classes from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.gan import GANGenerator


def sample_gan(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a trained GAN model.

    Parameters
    ----------
    model : dict
        Model containing generator state and parameters
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'temperature': scaling factor for sampling diversity (default: 1.0)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    latent_dim = model['latent_dim']
    input_dim = model['input_dim']
    hidden_dims_g = model['hidden_dims_g']
    ranges = model['ranges']

    # Get temperature parameter for controlling diversity
    temperature = params.get('temperature', 1.0) if params is not None else 1.0

    # Recreate generator
    generator = GANGenerator(latent_dim, input_dim, hidden_dims_g)
    generator.load_state_dict(model['generator_state'])
    generator.eval()

    # Sample from latent space
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim) * temperature
        norm_samples = generator(z).numpy()

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population
