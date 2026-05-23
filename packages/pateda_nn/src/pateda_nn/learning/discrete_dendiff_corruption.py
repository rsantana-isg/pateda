"""
Discrete Denoising Diffusion Model with Corruption/Denoising

This module implements a discrete dendiff variant based on corruption and denoising,
which is more aligned with discrete diffusion literature (e.g., masked language models).

Key Approach:
1. Corruption: Randomly flip bits based on corruption level
2. Network learns to denoise: predict original values from corrupted ones
3. Simpler than Gumbel-Softmax: uses direct probability prediction
4. More interpretable: corruption rate directly controls noise level

Inspired by:
- BERT-style masked language modeling
- Discrete denoising literature
- DISCRETE_DBD_ANALYSIS.md recommendation for corruption-based approaches
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pateda_nn.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
)


class TimeEmbedding(nn.Module):
    """Sinusoidal time step embedding."""
    
    def __init__(self, embed_dim: int):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        
        return embeddings


class CorruptionDenoisingMLP(nn.Module):
    """
    Corruption-based denoising network for binary variables.
    
    Simpler architecture that directly predicts bit probabilities
    without Gumbel-Softmax complexity.
    
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
        super(CorruptionDenoisingMLP, self).__init__()
        
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
        
        # Output: probability of each bit being 1
        output_layer = nn.Linear(prev_dim, input_dim)
        layers.append(output_layer)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict original bit probabilities from corrupted input.
        
        Parameters
        ----------
        x_t : torch.Tensor
            Corrupted binary input (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices (batch_size,)
        
        Returns
        -------
        logits : torch.Tensor
            Logits for bit probabilities (batch_size, input_dim)
        """
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Concatenate
        h = torch.cat([x_t, t_emb], dim=1)
        
        # Predict logits (will be passed through sigmoid)
        logits = self.mlp(h)
        
        return logits


def make_corruption_schedule(
    schedule: str,
    n_timesteps: int,
    corruption_start: float = 0.01,
    corruption_end: float = 0.5
) -> np.ndarray:
    """
    Create corruption schedule.
    
    Parameters
    ----------
    schedule : str
        'linear' or 'cosine'
    n_timesteps : int
        Number of timesteps
    corruption_start : float
        Starting corruption probability
    corruption_end : float
        Ending corruption probability
    
    Returns
    -------
    corruption_rates : np.ndarray
        Corruption probability at each timestep
    """
    if schedule == 'linear':
        rates = np.linspace(corruption_start, corruption_end, n_timesteps, dtype=np.float32)
    elif schedule == 'cosine':
        # Cosine schedule for smoother corruption
        t = np.arange(n_timesteps, dtype=np.float32) / (n_timesteps - 1)
        rates = corruption_start + (corruption_end - corruption_start) * (1 - np.cos(t * np.pi)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return rates


def corrupt_binary(
    x: torch.Tensor,
    corruption_rate: torch.Tensor
) -> torch.Tensor:
    """
    Corrupt binary data by flipping bits.
    
    Parameters
    ----------
    x : torch.Tensor
        Original binary data (batch_size, n_vars)
    corruption_rate : torch.Tensor
        Corruption probability (batch_size, 1) or scalar
    
    Returns
    -------
    corrupted : torch.Tensor
        Corrupted binary data
    """
    # Sample flip mask
    flip_mask = (torch.rand_like(x) < corruption_rate).float()
    
    # XOR to flip bits
    corrupted = (x + flip_mask) % 2
    
    return corrupted


def learn_discrete_dendiff_corruption(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn discrete denoising diffusion using corruption/denoising approach.
    
    This variant:
    1. Corrupts data by randomly flipping bits at rate proportional to timestep
    2. Trains network to predict original (uncorrupted) bit values
    3. Uses simple binary cross-entropy loss
    4. No Gumbel-Softmax needed - simpler training
    
    Parameters
    ----------
    population : np.ndarray
        Binary population (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (not used)
    params : dict, optional
        Training parameters:
        - 'n_timesteps': number of corruption levels (default: 50)
        - 'schedule': 'linear' or 'cosine' (default: 'linear')
        - 'corruption_start': min corruption (default: 0.01)
        - 'corruption_end': max corruption (default: 0.5)
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
    n_timesteps = params.get('n_timesteps', 50)
    schedule = params.get('schedule', 'linear')
    corruption_start = params.get('corruption_start', 0.01)
    corruption_end = params.get('corruption_end', 0.5)
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
    
    # Create corruption schedule
    corruption_rates = make_corruption_schedule(schedule, n_timesteps, corruption_start, corruption_end)
    corruption_rates_tensor = torch.FloatTensor(corruption_rates)
    
    # Create model
    model = CorruptionDenoisingMLP(
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
            
            # Get corruption rates for this batch
            corruption_rate = corruption_rates_tensor[t].unsqueeze(1)
            
            # Corrupt the data
            x_corrupted = corrupt_binary(batch, corruption_rate)
            
            # Predict original data
            logits = model(x_corrupted, t)
            
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Print progress (optional logging)
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     avg_loss = epoch_loss / n_batches
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    # Return model
    return {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'n_timesteps': n_timesteps,
        'corruption_rates': corruption_rates.tolist(),
        'hidden_dims': hidden_dims,
        'list_act_functs': list_act_functs if list_act_functs else ['relu'] * len(hidden_dims),
        'list_init_functs': list_init_functs if list_init_functs else ['default'] * len(hidden_dims),
        'time_emb_dim': time_emb_dim,
        'type': 'discrete_dendiff_corruption'
    }
