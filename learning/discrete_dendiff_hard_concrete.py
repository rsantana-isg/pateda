"""
Discrete Denoising Diffusion Model with Hard Concrete Distribution

This module implements a discrete dendiff variant using the Hard Concrete distribution,
which is a specialized version of the Concrete distribution that uses "stretching and folding"
to allow the model to produce exact zeros and ones.

Key Approach:
1. Concrete distribution: continuous relaxation of discrete variables
2. Hard Concrete: stretches distribution beyond [0,1] then clips to get exact 0s and 1s
3. Better than Gumbel-Softmax: can produce exact discrete values during training
4. Useful for: binary gating, regularization, and discrete optimization

Inspired by:
- Louizos et al. "Learning Sparse Neural Networks through L0 Regularization"
- Maddison et al. "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"
- Hard Concrete distribution for neural architecture search
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
    add_noise_binary,
    compute_diffusion_params,
)


def sample_hard_concrete(
    logits: torch.Tensor,
    temperature: float = 0.1,
    stretch_limits: tuple = (-0.1, 1.1),
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Sample from Hard Concrete distribution.
    
    The Hard Concrete distribution:
    1. Samples from stretched Concrete distribution with limits (gamma, zeta)
    2. Clips values to [0, 1] to get exact 0s and 1s in the support
    
    Parameters
    ----------
    logits : torch.Tensor
        Logits for binary variables (batch, n_vars)
    temperature : float
        Temperature parameter (lower = more discrete)
    stretch_limits : tuple
        (gamma, zeta) stretching limits, typically (-0.1, 1.1)
    eps : float
        Small constant for numerical stability
    
    Returns
    -------
    samples : torch.Tensor
        Samples from Hard Concrete distribution in [0, 1]
    """
    gamma, zeta = stretch_limits
    
    # Sample from uniform distribution
    u = torch.rand_like(logits)
    u = torch.clamp(u, eps, 1 - eps)
    
    # Compute stretched Concrete samples
    # Formula: s = sigmoid((log(u) - log(1-u) + logits) / temperature)
    log_u = torch.log(u)
    log_1_minus_u = torch.log(1 - u)
    
    s = torch.sigmoid((log_u - log_1_minus_u + logits) / temperature)
    
    # Stretch: map [0, 1] to [gamma, zeta]
    s_stretched = s * (zeta - gamma) + gamma
    
    # Hard concrete: clip to [0, 1] to get exact boundaries
    s_hard = torch.clamp(s_stretched, 0, 1)
    
    return s_hard


def hard_concrete_sample_with_ste(
    logits: torch.Tensor,
    temperature: float = 0.1,
    stretch_limits: tuple = (-0.1, 1.1),
    training: bool = True
) -> torch.Tensor:
    """
    Sample from Hard Concrete with straight-through in training.
    
    During training: uses soft Hard Concrete values for gradients
    During inference: uses hard binary values
    
    Parameters
    ----------
    logits : torch.Tensor
        Logits for binary variables
    temperature : float
        Temperature parameter
    stretch_limits : tuple
        Stretching limits
    training : bool
        Whether in training mode
    
    Returns
    -------
    samples : torch.Tensor
        Samples (soft during training, hard during inference)
    """
    s_hard = sample_hard_concrete(logits, temperature, stretch_limits)
    
    if not training:
        # During inference: binarize completely
        s_binary = (s_hard > 0.5).float()
        return s_binary
    else:
        # During training: use soft values directly (they already include 0s and 1s at boundaries)
        return s_hard


class HardConcreteDenoisingMLP(nn.Module):
    """
    Hard Concrete based denoising network for binary variables.
    
    Predicts logits that will be sampled using Hard Concrete distribution,
    which can produce exact 0s and 1s during training.
    
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
        super(HardConcreteDenoisingMLP, self).__init__()
        
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
        
        # Output: logits for Hard Concrete distribution
        output_layer = nn.Linear(prev_dim, input_dim)
        layers.append(output_layer)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict logits for Hard Concrete sampling.
        
        Parameters
        ----------
        x_t : torch.Tensor
            Noisy binary input (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices (batch_size,)
        
        Returns
        -------
        logits : torch.Tensor
            Logits for Hard Concrete distribution (batch_size, input_dim)
        """
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Concatenate
        h = torch.cat([x_t, t_emb], dim=1)
        
        # Predict logits
        logits = self.mlp(h)
        
        return logits


def learn_discrete_dendiff_hard_concrete(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn discrete denoising diffusion using Hard Concrete distribution.
    
    This variant:
    1. Uses Hard Concrete distribution for continuous relaxation
    2. Can produce exact 0s and 1s during training (not just at inference)
    3. Better for binary gating and regularization
    4. Stretching and folding mechanism allows exact discrete values
    
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
        - 'temperature': Hard Concrete temperature (default: 0.1)
        - 'stretch_limits': (gamma, zeta) for stretching (default: (-0.1, 1.1))
    
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
    temperature = params.get('temperature', 0.1)
    stretch_limits = params.get('stretch_limits', (-0.1, 1.1))
    
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)
    
    # Ensure binary
    population = (population > 0.5).astype(np.float32)
    
    # Convert to tensors
    data = torch.FloatTensor(population)
    
    # Create noise schedule using shared utility
    betas = make_noise_schedule(schedule, n_timesteps, beta_start, beta_end, 'beta')
    betas_tensor = torch.FloatTensor(betas)
    
    # Create model
    model = HardConcreteDenoisingMLP(
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
            
            # Get noise rates for this batch
            noise_rate = betas_tensor[t].unsqueeze(1)
            
            # Add noise to the data
            x_noisy = add_noise_binary(batch, noise_rate)
            
            # Predict logits for original data
            logits = model(x_noisy, t)
            
            # Sample from Hard Concrete (soft values with exact 0s and 1s at boundaries)
            x_pred = sample_hard_concrete(logits, temperature, stretch_limits)
            
            # Loss: MSE between predicted and original (using soft predictions)
            # This allows gradients to flow through the Hard Concrete sampling
            loss = F.mse_loss(x_pred, batch)
            
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
        'betas': betas.tolist(),
        'hidden_dims': hidden_dims,
        'list_act_functs': list_act_functs if list_act_functs else ['relu'] * len(hidden_dims),
        'list_init_functs': list_init_functs if list_init_functs else ['default'] * len(hidden_dims),
        'time_emb_dim': time_emb_dim,
        'temperature': temperature,
        'stretch_limits': stretch_limits,
        'type': 'discrete_dendiff_hard_concrete'
    }
