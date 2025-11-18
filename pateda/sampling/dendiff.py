"""
Denoising Diffusion Model Sampling for Continuous EDAs

This module provides sampling algorithms for Denoising Diffusion Probabilistic Models (DDPM)
used in continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import network classes from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.dendiff import DenoisingMLP


def p_sample(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    betas: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    posterior_variance: torch.Tensor
) -> torch.Tensor:
    """
    Single step of reverse diffusion process: sample x_{t-1} from x_t.

    Implements one step of Algorithm 2 (Sampling) from Ho et al. (2020):
    x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

    Parameters
    ----------
    model : nn.Module
        Trained denoising network
    x_t : torch.Tensor
        Current noisy sample at timestep t
    t : torch.Tensor
        Current timestep indices
    t_index : int
        Actual timestep value (0 to n_timesteps-1)
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
    # Predict noise
    predicted_noise = model(x_t, t)

    # Extract coefficients
    sqrt_recip_alpha_t = sqrt_recip_alphas[t_index]
    beta_t = betas[t_index]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]

    # Compute mean of p(x_{t-1} | x_t)
    # μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
    model_mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)

    if t_index == 0:
        # No noise at t=0
        return model_mean
    else:
        # Add noise proportional to posterior variance
        posterior_variance_t = posterior_variance[t_index]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def p_sample_loop(
    model: nn.Module,
    shape: tuple,
    n_timesteps: int,
    diffusion_params: Dict[str, np.ndarray]
) -> torch.Tensor:
    """
    Full reverse diffusion sampling loop.

    Implements Algorithm 2 (Sampling) from Ho et al. (2020):
    1. x_T ~ N(0, I)
    2. for t = T, ..., 1:
           x_{t-1} = p_sample(x_t, t)
    3. return x_0

    Parameters
    ----------
    model : nn.Module
        Trained denoising network
    shape : tuple
        Shape of samples to generate (n_samples, input_dim)
    n_timesteps : int
        Number of diffusion timesteps
    diffusion_params : dict
        Precomputed diffusion parameters

    Returns
    -------
    x_0 : torch.Tensor
        Generated samples
    """
    device = next(model.parameters()).device

    # Convert diffusion params to tensors
    betas = torch.FloatTensor(diffusion_params['betas']).to(device)
    sqrt_recip_alphas = torch.FloatTensor(diffusion_params['sqrt_recip_alphas']).to(device)
    sqrt_one_minus_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_one_minus_alphas_cumprod']).to(device)
    posterior_variance = torch.FloatTensor(diffusion_params['posterior_variance']).to(device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    # Reverse diffusion
    for t_index in reversed(range(n_timesteps)):
        # Create timestep tensor
        t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)

        # One step of reverse diffusion
        x = p_sample(
            model, x, t, t_index,
            betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance
        )

    return x


def sample_dendiff(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a trained denoising diffusion model.

    Generates new population by running the reverse diffusion process,
    starting from pure Gaussian noise and iteratively denoising.

    Parameters
    ----------
    model : dict
        Model dictionary containing:
        - 'model_state': denoising network state dict
        - 'input_dim': input dimension
        - 'n_timesteps': number of diffusion timesteps
        - 'diffusion_params': precomputed diffusion parameters
        - 'hidden_dims': hidden layer dimensions
        - 'time_emb_dim': time embedding dimension
        - 'ranges': data normalization ranges
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds for each variable
    params : dict, optional
        Additional parameters:
        - 'clip_denoised': whether to clip to [0, 1] during sampling (default: True)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    # Extract model parameters
    input_dim = model['input_dim']
    n_timesteps = model['n_timesteps']
    diffusion_params = model['diffusion_params']
    hidden_dims = model['hidden_dims']
    time_emb_dim = model['time_emb_dim']
    ranges = model['ranges']

    # Recreate denoising network
    denoising_model = DenoisingMLP(input_dim, time_emb_dim, hidden_dims)
    denoising_model.load_state_dict(model['model_state'])
    denoising_model.eval()

    # Sample from diffusion model
    with torch.no_grad():
        shape = (n_samples, input_dim)
        norm_samples = p_sample_loop(denoising_model, shape, n_timesteps, diffusion_params)
        norm_samples = norm_samples.cpu().numpy()

    # Clip to [0, 1] if requested
    if params is not None and params.get('clip_denoised', True):
        norm_samples = np.clip(norm_samples, 0, 1)

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population


def sample_dendiff_fast(
    model: Dict[str, Any],
    n_samples: int,
    bounds: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Fast sampling using fewer diffusion steps (DDIM-like approach).

    Uses a strided sampling schedule to reduce the number of denoising steps
    while maintaining quality. Useful for faster generation during EDA iterations.

    Parameters
    ----------
    model : dict
        Model dictionary (same as sample_dendiff)
    n_samples : int
        Number of samples to generate
    bounds : np.ndarray, optional
        Array of shape (2, n_vars) with [min, max] bounds
    params : dict, optional
        Additional parameters:
        - 'ddim_steps': number of sampling steps (default: 50)
        - 'ddim_eta': noise level parameter (0 for deterministic, 1 for stochastic)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    # Extract parameters
    ddim_steps = params.get('ddim_steps', 50)
    ddim_eta = params.get('ddim_eta', 0.0)

    input_dim = model['input_dim']
    n_timesteps = model['n_timesteps']
    diffusion_params = model['diffusion_params']
    hidden_dims = model['hidden_dims']
    time_emb_dim = model['time_emb_dim']
    ranges = model['ranges']

    # Create strided timestep schedule
    # Sample uniformly spaced timesteps
    step_size = n_timesteps // ddim_steps
    timesteps = np.arange(0, n_timesteps, step_size)

    # Recreate denoising network
    denoising_model = DenoisingMLP(input_dim, time_emb_dim, hidden_dims)
    denoising_model.load_state_dict(model['model_state'])
    denoising_model.eval()

    device = next(denoising_model.parameters()).device

    # Convert diffusion params
    alphas_cumprod = torch.FloatTensor(diffusion_params['alphas_cumprod']).to(device)

    # Start from pure noise
    x = torch.randn(n_samples, input_dim, device=device)

    with torch.no_grad():
        # Reverse through strided timesteps
        for i in reversed(range(len(timesteps))):
            t_index = timesteps[i]
            t = torch.full((n_samples,), t_index, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = denoising_model(x, t)

            # DDIM update
            alpha_cumprod_t = alphas_cumprod[t_index]

            if i > 0:
                alpha_cumprod_t_prev = alphas_cumprod[timesteps[i - 1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - ddim_eta ** 2) * predicted_noise

            # Random noise
            noise = ddim_eta * torch.randn_like(x) if i > 0 else 0

            # Update x
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + noise

        norm_samples = x.cpu().numpy()

    # Clip to [0, 1]
    norm_samples = np.clip(norm_samples, 0, 1)

    # Denormalize
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    population = norm_samples * range_diff + ranges[0]

    # Apply bounds if provided
    if bounds is not None:
        population = np.clip(population, bounds[0], bounds[1])

    return population
