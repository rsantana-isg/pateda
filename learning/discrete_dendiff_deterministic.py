"""
Discrete Denoising Diffusion Model with Deterministic Softmax

This module implements a discrete dendiff variant using deterministic softmax
without Gumbel noise, providing cleaner gradients for optimization and network inversion.

Key Approach:
1. Uses softmax without stochastic Gumbel noise
2. Provides cleaner, more stable gradients
3. Better for optimization/inversion tasks (e.g., fitness surrogate inversion)
4. Faster convergence to local optima
5. No temperature tuning needed during optimization

Inspired by:
- Deterministic optimization in neural network inversion
- Fitness surrogate inversion techniques
- DISCRETE_DENDIFF_ANALYSIS.md recommendation for deterministic approaches
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
)
from pateda.learning.discrete_dendiff_utils import (
    TimeEmbedding,
    make_noise_schedule,
    compute_diffusion_params,
)


def deterministic_softmax(logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
    """
    Deterministic softmax without Gumbel noise.
    
    Parameters
    ----------
    logits : torch.Tensor
        Logits for binary choices (shape: [batch, n_vars, 2])
    hard : bool
        If True, use argmax to get hard samples
    
    Returns
    -------
    probs : torch.Tensor
        Probabilities or hard samples
    """
    probs = F.softmax(logits, dim=-1)
    
    if hard:
        # Hard selection using argmax
        probs_hard = torch.zeros_like(probs)
        probs_hard.scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
        # Straight-through: hard selection in forward, soft gradients in backward
        probs = (probs_hard - probs).detach() + probs
    
    return probs


class DeterministicDenoisingMLP(nn.Module):
    """
    Deterministic softmax-based denoising network for binary variables.
    
    Uses clean softmax probabilities without stochastic noise,
    providing stable gradients for optimization tasks.
    
    Parameters
    ----------
    input_dim : int
        Number of binary variables
    time_emb_dim : int
        Dimension of time embedding
    hidden_dims : list
        Hidden layer dimensions
    list_act_functs : list, optional
        Activation functions
    list_init_functs : list, optional
        Weight initialization functions
    """
    
    def __init__(
        self,
        input_dim: int,
        time_emb_dim: int = 32,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(DeterministicDenoisingMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        
        n_hidden = len(hidden_dims)
        
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden
        
        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # MLP layers
        layers = []
        prev_dim = input_dim + time_emb_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: predict logits for binary [0, 1] for each variable
        # Output shape: [batch, n_vars, 2]
        output_layer = nn.Linear(prev_dim, input_dim * 2)
        layers.append(output_layer)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict clean data probabilities given noisy input and timestep.
        
        Parameters
        ----------
        x_t : torch.Tensor
            Noisy binary input (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices (batch_size,)
        
        Returns
        -------
        logits : torch.Tensor
            Logits for binary variables (batch_size, input_dim, 2)
        """
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Concatenate
        h = torch.cat([x_t, t_emb], dim=1)
        
        # Predict logits
        out = self.mlp(h)
        
        # Reshape to [batch, n_vars, 2]
        logits = out.reshape(-1, self.input_dim, 2)
        
        return logits


# Helper function for deterministic variant's specific noise adding
def add_noise_binary_deterministic(
    x: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to binary data by flipping bits (deterministic variant).
    
    This is a wrapper that adapts the interface for the deterministic variant.
    
    Parameters
    ----------
    x : torch.Tensor
        Original binary data (batch_size, n_vars)
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas
    t : torch.Tensor
        Timestep indices
    
    Returns
    -------
    noisy : torch.Tensor
        Noisy binary data
    """
    alpha_bar_t = alphas_cumprod[t][:, None]
    noise = torch.rand_like(x)
    flip_mask = (noise > alpha_bar_t).float()
    noisy = (x + flip_mask) % 2
    return noisy


def learn_discrete_dendiff_deterministic(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn discrete denoising diffusion using deterministic softmax.
    
    This variant:
    1. Uses deterministic softmax without Gumbel noise
    2. Provides cleaner, more stable gradients
    3. Better for optimization and network inversion tasks
    4. Faster convergence to local optima
    5. No stochastic sampling during training
    
    Parameters
    ----------
    population : np.ndarray
        Binary population (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (not used)
    params : dict, optional
        Training parameters:
        - 'n_timesteps': number of diffusion steps (default: 100)
        - 'schedule': 'linear' or 'cosine' (default: 'linear')
        - 'beta_start': starting noise (default: 0.0001)
        - 'beta_end': ending noise (default: 0.3)
        - 'hidden_dims': hidden dimensions (default: computed)
        - 'time_emb_dim': time embedding dim (default: 32)
        - 'epochs': training epochs (default: 50)
        - 'batch_size': batch size (default: computed)
        - 'learning_rate': learning rate (default: 1e-3)
    
    Returns
    -------
    model : dict
        Trained model dictionary
    """
    if params is None:
        params = {}
    
    # Extract dimensions
    pop_size = population.shape[0]
    input_dim = population.shape[1]
    
    # Compute adaptive defaults
    default_hidden_dims = compute_default_hidden_dims(input_dim, pop_size)
    default_batch_size = compute_default_batch_size(input_dim, pop_size)
    
    # Extract parameters
    n_timesteps = params.get('n_timesteps', 100)
    schedule = params.get('schedule', 'linear')
    beta_start = params.get('beta_start', 0.0001)
    beta_end = params.get('beta_end', 0.3)
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    time_emb_dim = params.get('time_emb_dim', 32)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 1e-3)
    
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)
    
    # Ensure binary
    population = (population > 0.5).astype(np.float32)
    
    # Convert to tensors
    data = torch.FloatTensor(population)
    
    # Create beta schedule using shared utility
    betas = make_noise_schedule(schedule, n_timesteps, beta_start, beta_end, 'beta')
    diffusion_params = compute_diffusion_params(betas)
    
    # Convert to tensors
    alphas_cumprod = torch.FloatTensor(diffusion_params['alphas_cumprod'])
    
    # Create model
    model = DeterministicDenoisingMLP(
        input_dim, time_emb_dim, hidden_dims,
        list_act_functs=list_act_functs,
        list_init_functs=list_init_functs
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    
    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            current_batch_size = len(batch)
            
            # Sample random timesteps
            t = torch.randint(0, n_timesteps, (current_batch_size,), dtype=torch.long)
            
            # Add noise to the data
            x_noisy = add_noise_binary_deterministic(batch, alphas_cumprod, t)
            
            # Predict original data distribution
            logits = model(x_noisy, t)
            
            # Target: one-hot encoding of original binary values
            target_indices = batch.long()
            
            # Cross-entropy loss
            logits_flat = logits.reshape(-1, 2)
            target_flat = target_indices.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, target_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        'list_act_functs': list_act_functs if list_act_functs else ['relu'] * len(hidden_dims),
        'list_init_functs': list_init_functs if list_init_functs else ['default'] * len(hidden_dims),
        'time_emb_dim': time_emb_dim,
        'type': 'discrete_dendiff_deterministic'
    }
