"""
Shared utilities for discrete dendiff implementations.

This module contains common components used across multiple discrete dendiff variants
to reduce code duplication and ensure consistency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time step embedding for diffusion models.
    
    This is a shared implementation used by all dendiff variants.
    """
    
    def __init__(self, embed_dim: int):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embeddings.
        
        Parameters
        ----------
        timesteps : torch.Tensor
            Timestep indices (batch_size,)
        
        Returns
        -------
        embeddings : torch.Tensor
            Time embeddings (batch_size, embed_dim)
        """
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        
        return embeddings


def make_noise_schedule(
    schedule: str,
    n_timesteps: int,
    start_value: float = 0.0001,
    end_value: float = 0.5,
    schedule_name: str = 'beta'
) -> np.ndarray:
    """
    Create noise/beta schedule for diffusion process.
    
    This unified function handles different scheduling strategies used across
    all dendiff variants (beta, corruption, noise, etc.).
    
    Parameters
    ----------
    schedule : str
        'linear' or 'cosine'
    n_timesteps : int
        Number of timesteps
    start_value : float
        Starting noise/beta/corruption level
    end_value : float
        Ending noise/beta/corruption level
    schedule_name : str, optional
        Name of the schedule (for cosine schedule scaling), e.g., 'beta', 'corruption', 'noise'
    
    Returns
    -------
    schedule_values : np.ndarray
        Noise/beta values at each timestep
    """
    if schedule == 'linear':
        values = np.linspace(start_value, end_value, n_timesteps, dtype=np.float32)
    elif schedule == 'cosine':
        # Cosine schedule for smoother noise progression
        if schedule_name == 'beta':
            # For beta schedules in Gumbel/Deterministic variants
            timesteps = np.arange(n_timesteps + 1, dtype=np.float32) / n_timesteps
            alphas_cumprod = np.cos((timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            values = np.clip(betas * end_value, start_value, end_value)
        else:
            # For corruption/noise schedules
            t = np.arange(n_timesteps, dtype=np.float32) / (n_timesteps - 1)
            values = start_value + (end_value - start_value) * (1 - np.cos(t * np.pi)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Must be 'linear' or 'cosine'")
    
    return values


def add_noise_binary(
    x: torch.Tensor,
    noise_rate: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to binary data by flipping bits.
    
    This is used across multiple dendiff variants for the forward diffusion process.
    
    Parameters
    ----------
    x : torch.Tensor
        Original binary data (batch_size, n_vars)
    noise_rate : torch.Tensor
        Noise/flip probability (batch_size, 1) or scalar
    
    Returns
    -------
    noisy : torch.Tensor
        Noisy binary data
    """
    # Sample flip mask
    flip_mask = (torch.rand_like(x) < noise_rate).float()
    
    # XOR to flip bits (works for binary values)
    noisy = (x + flip_mask) % 2
    
    return noisy


def compute_diffusion_params(betas: np.ndarray) -> dict:
    """
    Precompute diffusion parameters from beta schedule.
    
    Used by variants that need alpha and alpha_cumprod values.
    
    Parameters
    ----------
    betas : np.ndarray
        Beta schedule (noise rates)
    
    Returns
    -------
    params : dict
        Dictionary containing:
        - 'betas': original beta values
        - 'alphas': 1 - betas
        - 'alphas_cumprod': cumulative product of alphas
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
    }


def binarize_samples(samples: torch.Tensor) -> np.ndarray:
    """
    Convert soft samples to hard binary values.
    
    Ensures consistent binarization across all sampling functions.
    
    Parameters
    ----------
    samples : torch.Tensor
        Soft sample values (typically in [0, 1])
    
    Returns
    -------
    binary_samples : np.ndarray
        Hard binary values (0 or 1) as integers
    """
    binary_np = (samples.numpy() > 0.5).astype(int)
    return binary_np
