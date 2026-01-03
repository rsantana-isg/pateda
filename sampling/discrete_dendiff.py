"""
Sampling from Discrete Denoising Diffusion Models

This module provides sampling functions for discrete dendiff variants:
1. Gumbel-Softmax variant
2. Corruption/Denoising variant

The sampling process iteratively denoises random or corrupted binary data
to generate new solutions.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import network classes
from pateda.learning.discrete_dendiff_gumbel import (
    DiscreteDenoisingMLP, gumbel_softmax
)
from pateda.learning.discrete_dendiff_corruption import (
    CorruptionDenoisingMLP
)


def sample_discrete_dendiff_gumbel(
    model_data: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from discrete dendiff model with Gumbel-Softmax.
    
    Reverse diffusion process:
    1. Start with random binary data
    2. Iteratively denoise from t=T to t=0
    3. At each step, predict clean data and partially denoise
    
    Parameters
    ----------
    model_data : dict
        Trained model from learn_discrete_dendiff_gumbel
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'temperature': sampling temperature (default: 0.5)
        - 'n_steps': number of denoising steps (default: use all timesteps)
        - 'deterministic': use argmax instead of sampling (default: False)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    # Extract model parameters
    input_dim = model_data['input_dim']
    n_timesteps = model_data['n_timesteps']
    hidden_dims = model_data['hidden_dims']
    time_emb_dim = model_data['time_emb_dim']
    diffusion_params = model_data['diffusion_params']
    list_act_functs = model_data.get('list_act_functs', None)
    list_init_functs = model_data.get('list_init_functs', None)
    
    # Sampling parameters
    temperature = params.get('temperature', model_data.get('final_temperature', 0.5))
    n_steps = params.get('n_steps', n_timesteps)
    deterministic = params.get('deterministic', False)
    
    # Recreate model
    model = DiscreteDenoisingMLP(
        input_dim, time_emb_dim, hidden_dims,
        list_act_functs=list_act_functs,
        list_init_functs=list_init_functs
    )
    model.load_state_dict(model_data['model_state'])
    model.eval()
    
    # Diffusion parameters
    alphas_cumprod = torch.FloatTensor(diffusion_params['alphas_cumprod'])
    
    # Start from random binary data
    x_t = torch.randint(0, 2, (n_samples, input_dim), dtype=torch.float32)
    
    # Create timestep schedule (possibly strided for faster sampling)
    if n_steps < n_timesteps:
        timestep_schedule = np.linspace(n_timesteps-1, 0, n_steps, dtype=int)
    else:
        timestep_schedule = list(reversed(range(n_timesteps)))
    
    with torch.no_grad():
        for t_idx in timestep_schedule:
            t = torch.full((n_samples,), t_idx, dtype=torch.long)
            
            # Predict clean data probabilities
            logits = model(x_t, t)  # Shape: [batch, n_vars, 2]
            
            if deterministic:
                # Use argmax (most likely value)
                x_pred = logits.argmax(dim=-1).float()
            else:
                # Sample using Gumbel-Softmax
                probs = gumbel_softmax(logits, temperature, hard=True)
                x_pred = probs[:, :, 1]  # Take probability of bit being 1
            
            # Progressive denoising: mix predicted clean data with current noisy data
            # As we approach t=0, rely more on prediction
            if t_idx > 0:
                # Corruption level at this timestep
                alpha_bar_t = alphas_cumprod[t_idx]
                # Safe access: only access t_idx - 1 when t_idx > 0
                alpha_bar_prev = alphas_cumprod[t_idx - 1]
                
                # Mix: move toward predicted clean data
                # Use probabilistic mixing
                mixing_prob = (alpha_bar_prev / (alpha_bar_t + 1e-8)).clamp(0, 1)
                
                # Decide whether to keep predicted or add noise
                keep_pred = torch.rand_like(x_t) < mixing_prob
                x_t = torch.where(keep_pred, x_pred, x_t)
            else:
                # Final step: use prediction directly
                x_t = x_pred
    
    # Convert to numpy and ensure binary
    samples = x_t.numpy()
    samples = (samples > 0.5).astype(np.float32)
    
    return samples


def sample_discrete_dendiff_corruption(
    model_data: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from corruption-based discrete dendiff.
    
    Reverse process:
    1. Start with highly corrupted (random) binary data
    2. Iteratively denoise using the trained model
    3. At each step, predict original values and reduce corruption
    
    Parameters
    ----------
    model_data : dict
        Trained model from learn_discrete_dendiff_corruption
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'n_steps': number of denoising steps (default: use all)
        - 'temperature': sampling temperature (default: 0.5)
        - 'deterministic': use thresholding instead of sampling
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    # Extract model parameters
    input_dim = model_data['input_dim']
    n_timesteps = model_data['n_timesteps']
    hidden_dims = model_data['hidden_dims']
    time_emb_dim = model_data['time_emb_dim']
    corruption_rates = np.array(model_data['corruption_rates'])
    list_act_functs = model_data.get('list_act_functs', None)
    list_init_functs = model_data.get('list_init_functs', None)
    
    # Sampling parameters
    n_steps = params.get('n_steps', n_timesteps)
    temperature = params.get('temperature', 0.5)
    deterministic = params.get('deterministic', False)
    
    # Recreate model
    model = CorruptionDenoisingMLP(
        input_dim, time_emb_dim, hidden_dims,
        list_act_functs=list_act_functs,
        list_init_functs=list_init_functs
    )
    model.load_state_dict(model_data['model_state'])
    model.eval()
    
    # Start from random binary data (maximally corrupted)
    x_t = torch.randint(0, 2, (n_samples, input_dim), dtype=torch.float32)
    
    # Create timestep schedule
    if n_steps < n_timesteps:
        timestep_schedule = np.linspace(n_timesteps-1, 0, n_steps, dtype=int)
    else:
        timestep_schedule = list(reversed(range(n_timesteps)))
    
    with torch.no_grad():
        for t_idx in timestep_schedule:
            t = torch.full((n_samples,), t_idx, dtype=torch.long)
            
            # Predict clean data probabilities
            logits = model(x_t, t)  # Shape: [batch, n_vars]
            probs = torch.sigmoid(logits)
            
            if deterministic:
                # Threshold at 0.5
                x_pred = (probs > 0.5).float()
            else:
                # Sample from Bernoulli with temperature
                probs_tempered = torch.sigmoid(logits / temperature)
                x_pred = torch.bernoulli(probs_tempered)
            
            # Progressive denoising: gradually trust the prediction more
            if t_idx > 0:
                # Current corruption rate
                corruption_t = corruption_rates[t_idx]
                # Safe access: only access t_idx - 1 when t_idx > 0
                corruption_prev = corruption_rates[t_idx - 1]
                
                # Mixing factor: as corruption decreases, trust prediction more
                trust_factor = 1.0 - corruption_prev / (corruption_t + 1e-8)
                trust_factor = max(0.0, min(1.0, trust_factor))
                
                # Mix current noisy data with prediction
                keep_pred = torch.rand_like(x_t) < trust_factor
                x_t = torch.where(keep_pred, x_pred, x_t)
            else:
                # Final step: fully trust prediction
                x_t = x_pred
    
    # Convert to numpy
    samples = x_t.numpy()
    samples = (samples > 0.5).astype(np.float32)
    
    return samples


def sample_discrete_dendiff_fast(
    model_data: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Fast sampling with fewer denoising steps.
    
    Uses a strided timestep schedule to reduce computational cost
    while maintaining reasonable sample quality.
    
    Parameters
    ----------
    model_data : dict
        Trained model dictionary
    n_samples : int
        Number of samples
    params : dict, optional
        Parameters:
        - 'n_steps': number of steps (default: 10)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples
    """
    if params is None:
        params = {}
    
    # Use fast sampling with few steps
    fast_params = params.copy()
    fast_params['n_steps'] = params.get('n_steps', 10)
    
    # Dispatch to appropriate sampler based on model type
    model_type = model_data.get('type', 'discrete_dendiff_gumbel')
    
    if model_type == 'discrete_dendiff_gumbel':
        return sample_discrete_dendiff_gumbel(model_data, n_samples, fast_params)
    elif model_type == 'discrete_dendiff_corruption':
        return sample_discrete_dendiff_corruption(model_data, n_samples, fast_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
