"""
Denoising Diffusion Model Learning for Continuous EDAs

This module provides learning algorithms for Denoising Diffusion Probabilistic Models (DDPM)
used in continuous optimization. Implementation based on the papers:
- "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, ICML 2021)
- "DiffImpute: Tabular Data Imputation with Denoising Diffusion Probabilistic Model" (Nazzal et al., 2024)

The module implements a multi-layer perceptron (MLP) based denoising diffusion model
for learning distributions over continuous solution vectors in EDAs.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time step embedding for diffusion models.

    Encodes the timestep t into a continuous representation that allows
    the network to distinguish different noise levels in the diffusion process.
    """

    def __init__(self, embed_dim: int):
        """
        Initialize time embedding layer.

        Parameters
        ----------
        embed_dim : int
            Dimension of the time embedding (should be even)
        """
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.

        Parameters
        ----------
        timesteps : torch.Tensor
            Tensor of shape (batch_size,) containing timestep indices

        Returns
        -------
        embeddings : torch.Tensor
            Tensor of shape (batch_size, embed_dim) containing sinusoidal embeddings
        """
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class DenoisingMLP(nn.Module):
    """
    Multi-layer perceptron for denoising diffusion models.

    This network takes noisy data x_t and timestep t as input, and predicts
    the noise ε that was added to create x_t from the clean data x_0.
    """

    def __init__(
        self,
        input_dim: int,
        time_emb_dim: int = 32,
        hidden_dims: list = None
    ):
        """
        Initialize denoising MLP.

        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        time_emb_dim : int
            Dimension of time step embedding (default: 32)
        hidden_dims : list, optional
            List of hidden layer dimensions (default: [128, 64])
        """
        super(DenoisingMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims

        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # MLP layers
        layers = []
        prev_dim = input_dim + time_emb_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())  # Swish activation (x * sigmoid(x))
            prev_dim = hidden_dim

        # Output layer (predicts noise)
        layers.append(nn.Linear(prev_dim, input_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy input and timestep.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy input of shape (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices of shape (batch_size,)

        Returns
        -------
        predicted_noise : torch.Tensor
            Predicted noise of shape (batch_size, input_dim)
        """
        # Embed timestep
        t_emb = self.time_embed(t)

        # Concatenate noisy input with time embedding
        h = torch.cat([x_t, t_emb], dim=1)

        # Pass through MLP
        predicted_noise = self.mlp(h)

        return predicted_noise


def make_beta_schedule(
    schedule: str,
    n_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cosine_s: float = 8e-3
) -> np.ndarray:
    """
    Create beta schedule for diffusion process.

    Parameters
    ----------
    schedule : str
        Type of schedule: 'linear' or 'cosine'
    n_timesteps : int
        Number of diffusion timesteps
    beta_start : float
        Starting beta value (for linear schedule)
    beta_end : float
        Ending beta value (for linear schedule)
    cosine_s : float
        Offset parameter for cosine schedule

    Returns
    -------
    betas : np.ndarray
        Array of beta values of shape (n_timesteps,)
    """
    if schedule == 'linear':
        # Linear schedule from Ho et al. (2020)
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)

    elif schedule == 'cosine':
        # Cosine schedule from Nichol & Dhariwal (2021)
        # More stable for lower resolutions / simpler data
        timesteps = np.arange(n_timesteps + 1, dtype=np.float32) / n_timesteps
        alphas_cumprod = np.cos((timesteps + cosine_s) / (1 + cosine_s) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)

    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


def compute_diffusion_params(betas: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Precompute diffusion parameters for efficiency.

    Parameters
    ----------
    betas : np.ndarray
        Beta schedule of shape (n_timesteps,)

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'betas': beta values
        - 'alphas': 1 - beta values
        - 'alphas_cumprod': cumulative product of alphas
        - 'sqrt_alphas_cumprod': square root of alphas_cumprod
        - 'sqrt_one_minus_alphas_cumprod': square root of (1 - alphas_cumprod)
        - 'sqrt_recip_alphas': square root of (1 / alphas)
        - 'posterior_variance': variance of q(x_{t-1} | x_t, x_0)
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = np.sqrt(1.0 / alphas)

    # Posterior variance for reverse process
    # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance
    }


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Forward diffusion process: sample x_t given x_0.

    q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)

    Using reparameterization: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε

    Parameters
    ----------
    x_0 : torch.Tensor
        Clean data of shape (batch_size, input_dim)
    t : torch.Tensor
        Timestep indices of shape (batch_size,)
    sqrt_alphas_cumprod : torch.Tensor
        Precomputed √ᾱ_t values
    sqrt_one_minus_alphas_cumprod : torch.Tensor
        Precomputed √(1 - ᾱ_t) values
    noise : torch.Tensor, optional
        Noise to add (if None, sample from N(0, I))

    Returns
    -------
    x_t : torch.Tensor
        Noisy data at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Extract coefficients for batch
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t][:, None]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None]

    # Apply forward diffusion
    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    return x_t


def learn_dendiff(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a denoising diffusion model from selected population.

    Implements Algorithm 1 (Training) from Ho et al. (2020):
    1. Sample x_0 from population
    2. Sample t ~ Uniform(1, T)
    3. Sample ε ~ N(0, I)
    4. Take gradient step on ∇_θ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)||²

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in basic DDPM)
    params : dict, optional
        Training parameters containing:
        - 'n_timesteps': number of diffusion steps (default: 1000)
        - 'beta_schedule': 'linear' or 'cosine' (default: 'linear')
        - 'beta_start': starting beta (default: 1e-4)
        - 'beta_end': ending beta (default: 0.02)
        - 'hidden_dims': list of hidden layer dimensions (default: [128, 64])
        - 'time_emb_dim': time embedding dimension (default: 32)
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': batch size for training (default: 32)
        - 'learning_rate': learning rate (default: 1e-3)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'model_state': denoising network state dict
        - 'input_dim': input dimension
        - 'n_timesteps': number of diffusion timesteps
        - 'diffusion_params': precomputed diffusion parameters
        - 'hidden_dims': hidden layer dimensions
        - 'time_emb_dim': time embedding dimension
        - 'ranges': data normalization ranges
        - 'type': 'dendiff'
    """
    if params is None:
        params = {}

    # Extract parameters
    n_timesteps = params.get('n_timesteps', 1000)
    beta_schedule = params.get('beta_schedule', 'linear')
    beta_start = params.get('beta_start', 1e-4)
    beta_end = params.get('beta_end', 0.02)
    hidden_dims = params.get('hidden_dims', [128, 64])
    time_emb_dim = params.get('time_emb_dim', 32)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 1e-3)

    # Normalize data to [0, 1]
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)  # Prevent division by zero

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)

    # Convert to tensors
    data = torch.FloatTensor(norm_pop)
    input_dim = population.shape[1]

    # Create beta schedule and precompute parameters
    betas = make_beta_schedule(beta_schedule, n_timesteps, beta_start, beta_end)
    diffusion_params = compute_diffusion_params(betas)

    # Convert to tensors
    sqrt_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_alphas_cumprod'])
    sqrt_one_minus_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_one_minus_alphas_cumprod'])

    # Create denoising network
    model = DenoisingMLP(input_dim, time_emb_dim, hidden_dims)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            current_batch_size = len(batch)

            # Sample random timesteps
            t = torch.randint(0, n_timesteps, (current_batch_size,), dtype=torch.long)

            # Sample noise
            noise = torch.randn_like(batch)

            # Forward diffusion: create x_t
            x_t = q_sample(batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)

            # Predict noise
            predicted_noise = model(x_t, t)

            # Compute loss (simple objective from Ho et al.)
            loss = F.mse_loss(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Return model
    return {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'n_timesteps': n_timesteps,
        'diffusion_params': {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in diffusion_params.items()},
        'hidden_dims': hidden_dims,
        'time_emb_dim': time_emb_dim,
        'ranges': ranges,
        'type': 'dendiff'
    }
