"""
DAE Model Sampling for Discrete EDAs

This module provides sampling algorithms for Denoising Autoencoder (DAE) based
probabilistic models used in combinatorial optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch

# Import DAE classes from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.dae import DenoisingAutoencoder, MultiLayerDAE, corrupt_binary


def sample_dae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a DAE model using iterative refinement.

    This implements the sampling algorithm from the DAE-EDA paper:
    repeatedly corrupt and reconstruct to generate new solutions.

    Parameters
    ----------
    model : dict
        Model containing DAE state and parameters
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'n_refinement_steps': number of corruption-reconstruction iterations (default: 10)
        - 'corruption_level': noise level during sampling (default: 0.1)
        - 'threshold': binarization threshold (default: 0.5)
        - 'init_strategy': 'random' or 'ones' (default: 'random')

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    input_dim = model['input_dim']
    hidden_dim = model['hidden_dim']

    if params is None:
        params = {}

    n_refinement_steps = params.get('n_refinement_steps', 10)
    corruption_level = params.get('corruption_level', 0.1)
    threshold = params.get('threshold', 0.5)
    init_strategy = params.get('init_strategy', 'random')

    # Recreate DAE
    dae = DenoisingAutoencoder(input_dim, hidden_dim)
    dae.load_state_dict(model['dae_state'])
    dae.eval()

    with torch.no_grad():
        # Initialize samples
        if init_strategy == 'random':
            samples = torch.rand(n_samples, input_dim)
        elif init_strategy == 'ones':
            samples = torch.ones(n_samples, input_dim) * 0.5
        else:
            samples = torch.rand(n_samples, input_dim)

        # Binarize initial samples
        samples = (samples > threshold).float()

        # Iterative refinement: s steps of corruption and reconstruction
        for step in range(n_refinement_steps):
            # Corrupt
            corrupted = corrupt_binary(samples, corruption_level)

            # Reconstruct
            reconstructed = dae(corrupted)

            # Binarize
            samples = (reconstructed > threshold).float()

        population = samples.numpy()

    # Convert to integer binary
    population = population.astype(int)

    return population


def sample_dae_probabilistic(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a DAE model using probabilistic sampling.

    Instead of hard thresholding, sample from Bernoulli distribution
    using the DAE's output probabilities.

    Parameters
    ----------
    model : dict
        Model containing DAE state and parameters
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'n_refinement_steps': number of iterations (default: 10)
        - 'corruption_level': noise level (default: 0.1)
        - 'init_strategy': initialization strategy (default: 'random')

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    input_dim = model['input_dim']
    hidden_dim = model['hidden_dim']

    if params is None:
        params = {}

    n_refinement_steps = params.get('n_refinement_steps', 10)
    corruption_level = params.get('corruption_level', 0.1)
    init_strategy = params.get('init_strategy', 'random')

    # Recreate DAE
    dae = DenoisingAutoencoder(input_dim, hidden_dim)
    dae.load_state_dict(model['dae_state'])
    dae.eval()

    with torch.no_grad():
        # Initialize samples
        if init_strategy == 'random':
            samples = torch.rand(n_samples, input_dim)
            samples = (samples > 0.5).float()
        else:
            samples = torch.ones(n_samples, input_dim) * 0.5

        # Iterative refinement with probabilistic sampling
        for step in range(n_refinement_steps):
            # Corrupt
            corrupted = corrupt_binary(samples, corruption_level)

            # Reconstruct to get probabilities
            probs = dae(corrupted)

            # Sample from Bernoulli
            samples = torch.bernoulli(probs)

        population = samples.numpy()

    # Convert to integer binary
    population = population.astype(int)

    return population


def sample_multilayer_dae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a multi-layer DAE model.

    Parameters
    ----------
    model : dict
        Model containing multi-layer DAE state
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (same as sample_dae)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    input_dim = model['input_dim']
    hidden_dims = model['hidden_dims']

    if params is None:
        params = {}

    n_refinement_steps = params.get('n_refinement_steps', 10)
    corruption_level = params.get('corruption_level', 0.1)
    threshold = params.get('threshold', 0.5)
    init_strategy = params.get('init_strategy', 'random')

    # Recreate multi-layer DAE
    dae = MultiLayerDAE(input_dim, hidden_dims)
    dae.load_state_dict(model['dae_state'])
    dae.eval()

    with torch.no_grad():
        # Initialize samples
        if init_strategy == 'random':
            samples = torch.rand(n_samples, input_dim)
        else:
            samples = torch.ones(n_samples, input_dim) * 0.5

        # Binarize initial samples
        samples = (samples > threshold).float()

        # Iterative refinement
        for step in range(n_refinement_steps):
            # Corrupt
            corrupted = corrupt_binary(samples, corruption_level)

            # Reconstruct
            reconstructed = dae(corrupted)

            # Binarize
            samples = (reconstructed > threshold).float()

        population = samples.numpy()

    # Convert to integer binary
    population = population.astype(int)

    return population


def sample_dae_from_seeds(
    model: Dict[str, Any],
    seeds: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a DAE model starting from seed solutions.

    This is useful for local search around good solutions.

    Parameters
    ----------
    model : dict
        Model containing DAE state and parameters
    seeds : np.ndarray
        Seed solutions of shape (n_seeds, n_vars)
    params : dict, optional
        Sampling parameters:
        - 'n_refinement_steps': number of iterations (default: 5)
        - 'corruption_level': noise level (default: 0.2)
        - 'threshold': binarization threshold (default: 0.5)

    Returns
    -------
    population : np.ndarray
        Refined population of shape (n_seeds, n_vars)
    """
    input_dim = model['input_dim']
    hidden_dim = model['hidden_dim']

    if params is None:
        params = {}

    n_refinement_steps = params.get('n_refinement_steps', 5)
    corruption_level = params.get('corruption_level', 0.2)
    threshold = params.get('threshold', 0.5)

    # Recreate DAE
    dae = DenoisingAutoencoder(input_dim, hidden_dim)
    dae.load_state_dict(model['dae_state'])
    dae.eval()

    with torch.no_grad():
        # Convert seeds to tensor
        samples = torch.FloatTensor(seeds)

        # Iterative refinement starting from seeds
        for step in range(n_refinement_steps):
            # Corrupt
            corrupted = corrupt_binary(samples, corruption_level)

            # Reconstruct
            reconstructed = dae(corrupted)

            # Binarize
            samples = (reconstructed > threshold).float()

        population = samples.numpy()

    # Convert to integer binary
    population = population.astype(int)

    return population
