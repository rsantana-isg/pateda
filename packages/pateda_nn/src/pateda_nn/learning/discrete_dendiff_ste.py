"""
Discrete Denoising Diffusion Model with Straight-Through Estimator (STE)

This module implements a discrete dendiff variant using the Straight-Through Estimator,
which uses discrete (hard) values during the forward pass but treats the operation
as continuous during the backward pass to provide gradients.

Key Approach:
1. Forward pass: Use hard discrete values (0 or 1)
2. Backward pass: Gradient flows as if operation was continuous (identity function)
3. Simpler than Gumbel-Softmax: no temperature tuning needed
4. More stable: avoids biased gradients from Gumbel noise

Inspired by:
- Bengio et al. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"
- Straight-through estimators in binarized neural networks
- DISCRETE_DENDIFF_ANALYSIS.md recommendation for gradient estimators
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
from pateda_nn.learning.discrete_dendiff_utils import (
    TimeEmbedding,
    make_noise_schedule,
    add_noise_binary,
)


class StraightThroughBinarize(torch.autograd.Function):
    """
    Straight-Through Estimator for binary values.
    
    Forward: hard binarization (0 or 1)
    Backward: gradient passes through as if it was an identity function
    """
    
    @staticmethod
    def forward(ctx, input):
        """Binarize to 0 or 1 based on threshold 0.5"""
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradient through unchanged (identity)"""
        # Gradient flows as if the binarization was an identity function
        return grad_output


def straight_through_binarize(x):
    """
    Apply straight-through estimator to binarize values.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1]
    
    Returns
    -------
    binary : torch.Tensor
        Binarized tensor (0 or 1) with gradient support
    """
    return StraightThroughBinarize.apply(x)


class STEDenoisingMLP(nn.Module):
    """
    Straight-Through Estimator based denoising network for binary variables.
    
    Uses hard binary values in forward pass but allows gradient flow in backward pass.
    This provides unbiased gradients compared to Gumbel-Softmax.
    
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
        super(STEDenoisingMLP, self).__init__()
        
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
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """
        Predict original bit probabilities from corrupted input.
        
        Parameters
        ----------
        x_t : torch.Tensor
            Corrupted binary input (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices (batch_size,)
        use_ste : bool
            If True, applies STE to ensure x_t is binary in forward pass
        
        Returns
        -------
        logits : torch.Tensor
            Logits for bit probabilities (batch_size, input_dim)
        """
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Apply STE to input if requested (ensures hard binary values in forward)
        if use_ste and self.training:
            x_t = straight_through_binarize(x_t)
        
        # Concatenate
        h = torch.cat([x_t, t_emb], dim=1)
        
        # Predict logits (will be passed through sigmoid)
        logits = self.mlp(h)
        
        return logits


def learn_discrete_dendiff_ste(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn discrete denoising diffusion using Straight-Through Estimator.
    
    This variant:
    1. Uses hard binary values in forward pass (no soft relaxation)
    2. Allows gradients to flow in backward pass as if continuous
    3. Avoids biased gradients from Gumbel noise
    4. No temperature parameter needed
    
    Parameters
    ----------
    population : np.ndarray
        Binary population (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (not used)
    params : dict, optional
        Training parameters:
        - 'n_timesteps': number of noise levels (default: 50)
        - 'schedule': 'linear' or 'cosine' (default: 'linear')
        - 'noise_start': min noise (default: 0.01)
        - 'noise_end': max noise (default: 0.5)
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
    noise_start = params.get('noise_start', 0.01)
    noise_end = params.get('noise_end', 0.5)
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
    
    # Create noise schedule using shared utility
    noise_rates = make_noise_schedule(schedule, n_timesteps, noise_start, noise_end, 'noise')
    noise_rates_tensor = torch.FloatTensor(noise_rates)
    
    # Create model
    model = STEDenoisingMLP(
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
            noise_rate = noise_rates_tensor[t].unsqueeze(1)
            
            # Add noise to the data
            x_noisy = add_noise_binary(batch, noise_rate)
            
            # Predict original data
            logits = model(x_noisy, t, use_ste=True)
            
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, batch)
            
            # Backward pass (gradients flow through STE)
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
        'noise_rates': noise_rates.tolist(),
        'hidden_dims': hidden_dims,
        'list_act_functs': list_act_functs if list_act_functs else ['relu'] * len(hidden_dims),
        'list_init_functs': list_init_functs if list_init_functs else ['default'] * len(hidden_dims),
        'time_emb_dim': time_emb_dim,
        'type': 'discrete_dendiff_ste'
    }
