"""
VAE Model Sampling for Continuous EDAs

This module provides sampling algorithms for Variational Autoencoder (VAE) based
probabilistic models used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import network classes from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.vae import VAEDecoder, ConditionalDecoder


def sample_vae(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a basic VAE model.

    Parameters
    ----------
    model : dict
        Model containing decoder state and parameters
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters (not used)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    latent_dim = model['latent_dim']
    input_dim = model['input_dim']
    hidden_dims = model['hidden_dims']
    ranges = model['ranges']

    # Recreate decoder
    decoder = VAEDecoder(latent_dim, input_dim, list(reversed(hidden_dims)))
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    # Sample from latent space
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        norm_samples = decoder(z).numpy()

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_extended_vae(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from an Extended VAE (E-VAE) model.

    This function samples similarly to the basic VAE, but the model
    has been trained with a fitness predictor that can be used for
    surrogate-based filtering.

    Parameters
    ----------
    model : dict
        Model containing decoder and predictor states
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'use_predictor': whether to use fitness predictor for filtering (default: False)
        - 'predictor_percentile': percentile of predicted fitness to keep (default: 50)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    latent_dim = model['latent_dim']
    input_dim = model['input_dim']
    hidden_dims = model['hidden_dims']
    ranges = model['ranges']

    use_predictor = params.get('use_predictor', False) if params is not None else False

    # Recreate decoder
    decoder = VAEDecoder(latent_dim, input_dim, list(reversed(hidden_dims)))
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    if use_predictor:
        # Import predictor class
        from learning.vae import FitnessPredictor

        n_objectives = model['n_objectives']
        predictor = FitnessPredictor(latent_dim, n_objectives)
        predictor.load_state_dict(model['predictor_state'])
        predictor.eval()

        percentile = params.get('predictor_percentile', 50)

        # Generate more samples and filter
        oversample_factor = 2
        n_total_samples = n_samples * oversample_factor

        with torch.no_grad():
            z = torch.randn(n_total_samples, latent_dim)
            pred_fitness = predictor(z).numpy()

            # Select samples with best predicted fitness
            fitness_score = pred_fitness.sum(axis=1) if pred_fitness.shape[1] > 1 else pred_fitness.flatten()
            threshold = np.percentile(fitness_score, 100 - percentile)
            selected_idx = np.where(fitness_score <= threshold)[0][:n_samples]

            # If not enough samples, just take the first n_samples
            if len(selected_idx) < n_samples:
                selected_idx = np.arange(min(n_samples, len(z)))

            z_selected = z[selected_idx]
            norm_samples = decoder(z_selected).numpy()
    else:
        # Sample normally
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            norm_samples = decoder(z).numpy()

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_conditional_extended_vae(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a Conditional Extended VAE (CE-VAE) model.

    This function allows fitness-conditioned sampling by specifying
    desired fitness values.

    Parameters
    ----------
    model : dict
        Model containing conditional decoder state and parameters
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'target_fitness': target fitness values to condition on (default: best fitness from training)
        - 'fitness_noise': noise to add to target fitness (default: 0.1)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    latent_dim = model['latent_dim']
    input_dim = model['input_dim']
    n_objectives = model['n_objectives']
    hidden_dims = model['hidden_dims']
    ranges = model['ranges']
    fitness_min = model['fitness_min']
    fitness_max = model['fitness_max']
    fitness_range = model['fitness_range']

    # Recreate conditional decoder
    conditional_decoder = ConditionalDecoder(latent_dim, n_objectives, input_dim, list(reversed(hidden_dims)))
    conditional_decoder.load_state_dict(model['conditional_decoder_state'])
    conditional_decoder.eval()

    # Determine target fitness
    if params is not None and 'target_fitness' in params:
        target_fitness = params['target_fitness']
        if isinstance(target_fitness, (int, float)):
            target_fitness = np.array([[target_fitness]])
        else:
            target_fitness = np.array(target_fitness).reshape(1, -1)
    else:
        # Default: target the best fitness (minimum, assuming minimization)
        target_fitness = fitness_min.reshape(1, -1)

    # Normalize target fitness
    norm_target_fitness = (target_fitness - fitness_min) / fitness_range
    norm_target_fitness = np.clip(norm_target_fitness, 0, 1)

    # Add noise to fitness if requested
    fitness_noise = params.get('fitness_noise', 0.1) if params is not None else 0.1
    noise = np.random.randn(n_samples, n_objectives) * fitness_noise
    norm_fitness_samples = norm_target_fitness + noise
    norm_fitness_samples = np.clip(norm_fitness_samples, 0, 1)

    # Sample from latent space
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        fitness_tensor = torch.FloatTensor(norm_fitness_samples)
        norm_samples = conditional_decoder(z, fitness_tensor).numpy()

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population
