"""
Simplified Denoising Diffusion Sampling with ReLU Activation

This module provides sampling for the simplified dendiff variant that uses
ReLU activation and raw timestep input.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import network class from learning module
from pateda_nn.learning.dendiff_relu import SimpleDenoisingMLP


def p_sample_relu(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    n_timesteps: int,
    betas: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    posterior_variance: torch.Tensor
) -> torch.Tensor:
    """
    Single step of reverse diffusion process for SimpleDenoisingMLP.

    Same as p_sample but passes n_timesteps to the model for raw timestep normalization.

    Parameters
    ----------
    model : nn.Module
        Trained SimpleDenoisingMLP network
    x_t : torch.Tensor
        Current noisy sample at timestep t
    t : torch.Tensor
        Current timestep indices
    t_index : int
        Actual timestep value (0 to n_timesteps-1)
    n_timesteps : int
        Total number of timesteps (for normalization)
    betas : torch.Tensor
        Beta schedule
    sqrt_recip_alphas : torch.Tensor
        Precomputed √(1/α_t) values
    sqrt_one_minus_alphas_cumprod : torch.Tensor
        Precomputed √(1-ᾱ_t) values
    posterior_variance : torch.Tensor
        Precomputed posterior variance

    Returns
    -------
    x_t_minus_1 : torch.Tensor
        Denoised sample at timestep t-1
    """
    # Predict noise (pass n_timesteps for normalization)
    predicted_noise = model(x_t, t, n_timesteps)

    # Extract coefficients
    sqrt_recip_alpha_t = sqrt_recip_alphas[t_index]
    beta_t = betas[t_index]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]

    # Compute mean of p(x_{t-1} | x_t)
    model_mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)

    if t_index == 0:
        # No noise at t=0
        return model_mean
    else:
        # Add noise proportional to posterior variance
        posterior_variance_t = posterior_variance[t_index]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def p_sample_loop_relu(
    model: nn.Module,
    shape: tuple,
    n_timesteps: int,
    diffusion_params: Dict[str, np.ndarray]
) -> torch.Tensor:
    """
    Full reverse diffusion sampling loop for SimpleDenoisingMLP.

    Parameters
    ----------
    model : nn.Module
        Trained SimpleDenoisingMLP network
    shape : tuple
        Shape of samples to generate (n_samples, input_dim)
    n_timesteps : int
        Number of diffusion timesteps
    diffusion_params : dict
        Precomputed diffusion parameters

    Returns
    -------
    samples : torch.Tensor
        Generated samples of shape (n_samples, input_dim)
    """
    # Convert numpy arrays to tensors
    betas = torch.FloatTensor(diffusion_params['betas'])
    sqrt_recip_alphas = torch.FloatTensor(diffusion_params['sqrt_recip_alphas'])
    sqrt_one_minus_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_one_minus_alphas_cumprod'])
    posterior_variance = torch.FloatTensor(diffusion_params['posterior_variance'])

    # Start from pure noise
    x_t = torch.randn(shape)

    # Reverse diffusion process
    model.eval()
    with torch.no_grad():
        for t_index in reversed(range(n_timesteps)):
            t = torch.full((shape[0],), t_index, dtype=torch.long)
            x_t = p_sample_relu(
                model, x_t, t, t_index, n_timesteps,
                betas, sqrt_recip_alphas,
                sqrt_one_minus_alphas_cumprod, posterior_variance
            )

    return x_t


def sample_dendiff_relu(
    model_data: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a trained SimpleDenoisingMLP diffusion model.

    Parameters
    ----------
    model_data : dict
        Dictionary containing trained model data from learn_dendiff_relu()
        Must contain:
        - 'model_state': model state dict
        - 'input_dim': input dimension
        - 'n_timesteps': number of timesteps
        - 'diffusion_params': diffusion parameters
        - 'hidden_dims': hidden layer dimensions
        - 'list_init_functs': initialization functions
        - 'ranges': normalization ranges

    n_samples : int
        Number of samples to generate

    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable.
        If provided, samples will be clipped to these bounds.

    params : dict, optional
        Additional parameters (currently unused, for API compatibility)

    Returns
    -------
    samples : np.ndarray
        Generated samples of shape (n_samples, input_dim) in original space
    """
    # Extract model parameters
    input_dim = model_data['input_dim']
    n_timesteps = model_data['n_timesteps']
    diffusion_params = model_data['diffusion_params']
    hidden_dims = model_data['hidden_dims']
    list_init_functs = model_data['list_init_functs']
    ranges = model_data['ranges']

    # Recreate model architecture
    model = SimpleDenoisingMLP(
        input_dim,
        hidden_dims,
        list_init_functs=list_init_functs
    )

    # Load trained weights
    model.load_state_dict(model_data['model_state'])

    # Generate samples
    shape = (n_samples, input_dim)
    norm_samples = p_sample_loop_relu(model, shape, n_timesteps, diffusion_params)

    # Convert to numpy and denormalize
    norm_samples = norm_samples.numpy()
    norm_samples = np.clip(norm_samples, 0, 1)  # Ensure in [0, 1]

    # Denormalize to original space
    range_diff = ranges[1] - ranges[0]
    samples = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        samples = np.clip(samples, bounds[0], bounds[1])

    return samples
