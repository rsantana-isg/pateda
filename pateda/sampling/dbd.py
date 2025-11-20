"""
Diffusion-by-Deblending (DbD) Sampling for Continuous EDAs

This module implements sampling algorithms for alpha-deblending diffusion models
used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import network class from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.dbd import AlphaDeblendingMLP


def iterative_deblending_sampling(
    model: nn.Module,
    x0: torch.Tensor,
    num_iterations: int
) -> torch.Tensor:
    """
    Implements the iterative alpha-deblending sampling algorithm (Algorithm 3).

    Parameters
    ----------
    model : nn.Module
        Trained alpha-deblending network
    x0 : torch.Tensor
        Initial samples from source distribution of shape (n_samples, input_dim)
    num_iterations : int
        Number of iterations T

    Returns
    -------
    x1 : torch.Tensor
        Final samples approximating target distribution
    """
    # Create alpha schedule: linearly spaced from 0 to 1
    alpha_values = np.linspace(0, 1, num=num_iterations + 1)
    x_alpha = x0.clone()

    n_samples = x0.shape[0]

    # Iteratively update samples
    for t in range(num_iterations):
        alpha_t = alpha_values[t]
        alpha_t_plus_1 = alpha_values[t + 1]

        # Create alpha tensor
        alpha_t_tensor = torch.full((n_samples, 1), alpha_t, dtype=torch.float32, device=x0.device)

        # Predict the difference (x1 - x0)
        with torch.no_grad():
            predicted_diff = model(x_alpha, alpha_t_tensor)

        # Update x_alpha using Algorithm 3 update rule:
        # x_{alpha_{t+1}} = x_{alpha_t} + (alpha_{t+1} - alpha_t) * D_theta(x_{alpha_t}, alpha_t)
        x_alpha = x_alpha + (alpha_t_plus_1 - alpha_t) * predicted_diff

    return x_alpha


def sample_dbd(
    model: Dict[str, Any],
    p0: np.ndarray,
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from a trained alpha-deblending diffusion model.

    Generates new population by running the iterative alpha-deblending process,
    starting from source distribution p0 and iteratively deblending to target
    distribution p1.

    Parameters
    ----------
    model : dict
        Model dictionary containing:
        - 'model_state': network state dict
        - 'input_dim': input dimension
        - 'hidden_dims': hidden layer dimensions
        - 'ranges': data normalization ranges
    p0 : np.ndarray
        Source distribution samples of shape (n_samples_p0, n_vars)
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'num_iterations': number of deblending iterations (default: 10)
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if rng is None:
        rng = np.random.default_rng()

    if params is None:
        params = {}

    # Extract model parameters
    input_dim = model['input_dim']
    hidden_dims = model['hidden_dims']
    ranges = model['ranges']
    num_iterations = params.get('num_iterations', 10)

    # Sample from p0
    if len(p0) >= n_samples:
        indices = rng.choice(len(p0), n_samples, replace=False)
    else:
        indices = rng.choice(len(p0), n_samples, replace=True)

    p0_samples = p0[indices]

    # Normalize if model was trained with normalization
    if ranges is not None:
        range_diff = ranges[1] - ranges[0]
        range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)
        p0_norm = (p0_samples - ranges[0]) / range_diff
    else:
        p0_norm = p0_samples

    # Recreate network
    deblending_model = AlphaDeblendingMLP(input_dim, hidden_dims)
    deblending_model.load_state_dict(model['model_state'])
    deblending_model.eval()

    # Convert to tensor
    x0 = torch.FloatTensor(p0_norm)

    # Sample using iterative deblending
    with torch.no_grad():
        x1 = iterative_deblending_sampling(deblending_model, x0, num_iterations)
        samples = x1.cpu().numpy()

    # Denormalize
    if ranges is not None:
        range_diff = ranges[1] - ranges[0]
        range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)
        population = samples * range_diff + ranges[0]
    else:
        population = samples

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_dbd_from_univariate(
    model: Dict[str, Any],
    population: np.ndarray,
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from DbD model starting from univariate Gaussian approximation.

    This is used for DbD-UC and DbD-US variants where p0 is a univariate
    approximation of the population.

    Parameters
    ----------
    model : dict
        Trained DbD model
    population : np.ndarray
        Population to approximate with univariate Gaussian
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Bounds for each variable
    params : dict, optional
        Additional parameters including 'num_iterations'
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    samples : np.ndarray
        Generated samples
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute univariate Gaussian parameters
    mean = np.mean(population, axis=0)
    std = np.std(population, axis=0)
    std = np.maximum(std, 1e-8)  # Avoid zero std

    # Sample from univariate Gaussian
    p0_samples = rng.normal(
        loc=mean[np.newaxis, :],
        scale=std[np.newaxis, :],
        size=(n_samples, len(mean))
    )

    # Use standard sampling with these p0 samples
    return sample_dbd(model, p0_samples, n_samples, bounds, params, rng)
