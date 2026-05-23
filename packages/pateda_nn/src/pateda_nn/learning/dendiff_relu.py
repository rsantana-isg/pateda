"""
Simplified Denoising Diffusion Model with ReLU Activation

This module provides a simplified variant of dendiff that uses:
- ReLU activation instead of SiLU/Swish
- Raw timestep as additional input instead of sinusoidal positional encoding

This simpler variant is useful for:
- Faster training (ReLU is computationally cheaper than SiLU)
- Smaller model (no time embedding layer)
- Benchmarking and comparison studies
- Understanding the impact of architectural choices

The simplified architecture may be sufficient for lower-dimensional problems
or when computational efficiency is more important than maximum expressiveness.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pateda_nn.learning.nn_utils import (
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
)

# Import diffusion helper functions from main dendiff module
from pateda_nn.learning.dendiff import (
    make_beta_schedule,
    compute_diffusion_params,
    q_sample
)


class SimpleDenoisingMLP(nn.Module):
    """
    Simplified MLP for denoising diffusion with ReLU activation.

    Key differences from DenoisingMLP:
    1. Uses ReLU activation instead of SiLU
    2. Takes raw timestep as scalar input instead of sinusoidal embedding
    3. No TimeEmbedding layer needed (simpler architecture)

    This results in a smaller, faster model that may be sufficient for
    many continuous optimization problems.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    hidden_dims : list, optional
        List of hidden layer dimensions (default: [128, 64]).
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_init_functs: List[str] = None
    ):
        super(SimpleDenoisingMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        # Use ReLU for all hidden layers (no need for list_act_functs parameter)
        list_act_functs = ['relu'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        # MLP layers
        # Input: concatenation of noisy data and raw timestep (normalized to [0,1])
        layers = []
        prev_dim = input_dim + 1  # +1 for the timestep

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(nn.ReLU())  # Fixed ReLU activation
            prev_dim = hidden_dim

        # Output layer (predicts noise)
        layers.append(nn.Linear(prev_dim, input_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, n_timesteps: int) -> torch.Tensor:
        """
        Predict noise given noisy input and timestep.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy input of shape (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices of shape (batch_size,)
        n_timesteps : int
            Total number of timesteps (for normalization)

        Returns
        -------
        predicted_noise : torch.Tensor
            Predicted noise of shape (batch_size, input_dim)
        """
        # Normalize timestep to [0, 1] range
        t_normalized = t.float() / float(n_timesteps - 1)
        t_normalized = t_normalized.unsqueeze(1)  # Shape: (batch_size, 1)

        # Concatenate noisy input with normalized timestep
        h = torch.cat([x_t, t_normalized], dim=1)

        # Pass through MLP
        predicted_noise = self.mlp(h)

        return predicted_noise


def learn_dendiff_relu(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a simplified denoising diffusion model with ReLU activation.

    This variant uses:
    - ReLU activation instead of SiLU
    - Raw timestep input instead of sinusoidal positional encoding

    Implements the same training algorithm as learn_dendiff but with
    the simplified architecture.

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
        - 'hidden_dims': list of hidden layer dimensions
          (default: computed from n_vars and pop_size)
        - 'list_init_functs': list of initialization functions for hidden layers
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': batch size for training (default: max(8, n_vars/50))
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
        - 'list_init_functs': initialization functions used
        - 'ranges': data normalization ranges
        - 'type': 'dendiff_relu'
    """
    if params is None:
        params = {}

    # Extract dimensions
    pop_size = population.shape[0]
    input_dim = population.shape[1]

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(input_dim, pop_size)
    default_batch_size = compute_default_batch_size(input_dim, pop_size)

    # Extract parameters with defaults
    n_timesteps = params.get('n_timesteps', 1000)
    beta_schedule = params.get('beta_schedule', 'linear')
    beta_start = params.get('beta_start', 1e-4)
    beta_end = params.get('beta_end', 0.02)
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 1e-3)

    # Extract initialization function list
    list_init_functs = params.get('list_init_functs', None)

    # Normalize data to [0, 1]
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)  # Prevent division by zero

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)

    # Convert to tensors
    data = torch.FloatTensor(norm_pop)

    # Create beta schedule and precompute parameters
    betas = make_beta_schedule(beta_schedule, n_timesteps, beta_start, beta_end)
    diffusion_params = compute_diffusion_params(betas)

    # Convert to tensors
    sqrt_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_alphas_cumprod'])
    sqrt_one_minus_alphas_cumprod = torch.FloatTensor(diffusion_params['sqrt_one_minus_alphas_cumprod'])

    # Create simplified denoising network (ReLU + raw timestep)
    model = SimpleDenoisingMLP(
        input_dim, hidden_dims,
        list_init_functs=list_init_functs
    )

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

            # Forward diffusion: create noisy samples
            x_t = q_sample(batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)

            # Predict noise
            predicted_noise = model(x_t, t, n_timesteps)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / n_batches
            #print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

    # Return model state and parameters
    return {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'n_timesteps': n_timesteps,
        'diffusion_params': diffusion_params,
        'hidden_dims': hidden_dims,
        'list_init_functs': list_init_functs,
        'ranges': ranges,
        'type': 'dendiff_relu'
    }
