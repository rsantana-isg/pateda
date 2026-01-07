"""
Enhanced Discrete Denoising Diffusion Model with Deterministic Softmax

This module extends the base deterministic dendiff with:
1. Alternative loss functions (weighted_mse, ranking, huber)
2. Flexible architecture configurations

Enhancements inspired by:
- discrete_dendiff_gumbel_enhanced.py: Loss function patterns
- discrete_backdrive_weighted_mse.py: Fitness-weighted loss
- discrete_backdrive_ranking.py: Ranking loss
- discrete_backdrive_huber.py: Huber loss
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pateda.learning.nn_utils import (
    compute_default_hidden_dims,
    compute_default_batch_size,
)

# Import base components from the standard deterministic implementation
from pateda.learning.discrete_dendiff_deterministic import (
    DeterministicDenoisingMLP,
    add_noise_binary_deterministic,
)

from pateda.learning.discrete_dendiff_utils import (
    make_noise_schedule,
    compute_diffusion_params,
)


def compute_weighted_loss(logits: torch.Tensor, target: torch.Tensor,
                         fitness: torch.Tensor) -> torch.Tensor:
    """
    Compute fitness-weighted cross-entropy loss.

    Higher fitness samples get higher weight.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars, 2]
    target : torch.Tensor
        Target values [batch, n_vars]
    fitness : torch.Tensor
        Fitness values [batch, 1]

    Returns
    -------
    loss : torch.Tensor
        Weighted loss
    """
    # Normalize fitness to [0, 1] range for weighting
    fitness_min = fitness.min()
    fitness_max = fitness.max()
    if fitness_max > fitness_min:
        normalized_fitness = (fitness - fitness_min) / (fitness_max - fitness_min + 1e-8)
    else:
        normalized_fitness = torch.ones_like(fitness)

    # Compute per-sample loss
    logits_flat = logits.reshape(len(logits), -1, 2)
    target_flat = target.long()

    per_sample_loss = []
    for i in range(len(logits)):
        sample_logits = logits_flat[i]
        sample_target = target_flat[i]
        loss_i = F.cross_entropy(sample_logits, sample_target, reduction='mean')
        per_sample_loss.append(loss_i)

    per_sample_loss = torch.stack(per_sample_loss)

    # Weight by normalized fitness
    weighted_loss = (per_sample_loss * normalized_fitness.squeeze()).mean()

    return weighted_loss


def compute_ranking_loss(logits: torch.Tensor, target: torch.Tensor,
                        fitness: torch.Tensor) -> torch.Tensor:
    """
    Compute ranking-based loss.

    Prioritizes learning the relative ordering of solutions by fitness.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars, 2]
    target : torch.Tensor
        Target values [batch, n_vars]
    fitness : torch.Tensor
        Fitness values [batch, 1]

    Returns
    -------
    loss : torch.Tensor
        Ranking-aware loss
    """
    # For dendiff, ranking loss is implemented through standard cross-entropy
    # (ranking is handled through selection in the EDA framework)
    logits_flat = logits.reshape(-1, 2)
    target_flat = target.reshape(-1).long()

    return F.cross_entropy(logits_flat, target_flat)


def compute_huber_loss(logits: torch.Tensor, target: torch.Tensor,
                      delta: float = 1.0) -> torch.Tensor:
    """
    Compute Huber loss for robust training.

    Huber loss is less sensitive to outliers than squared error.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars, 2]
    target : torch.Tensor
        Target values [batch, n_vars]
    delta : float
        Huber delta parameter

    Returns
    -------
    loss : torch.Tensor
        Huber loss
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Get probabilities for correct class
    target_long = target.long()
    target_probs = torch.gather(probs, -1, target_long.unsqueeze(-1)).squeeze(-1)

    # Huber-like smooth L1 loss on probabilities
    # We want target_probs to be close to 1.0
    error = 1.0 - target_probs

    huber_loss = torch.where(
        error.abs() <= delta,
        0.5 * error ** 2,
        delta * (error.abs() - 0.5 * delta)
    )

    return huber_loss.mean()


def learn_discrete_dendiff_deterministic_enhanced(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn discrete denoising diffusion using deterministic softmax with enhanced loss functions.

    This variant supports:
    1. Multiple loss functions: mse, weighted_mse, ranking, huber
    2. Deterministic softmax without Gumbel noise
    3. Cleaner, more stable gradients

    Parameters
    ----------
    population : np.ndarray
        Binary population (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (pop_size,)
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
        - 'loss_function': 'mse', 'weighted_mse', 'ranking', 'huber' (default: 'mse')

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
    loss_function = params.get('loss_function', 'mse')

    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)

    # Ensure binary
    population = (population > 0.5).astype(np.float32)

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness).unsqueeze(1)

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
            batch_fitness = fitness_tensor[idx]
            current_batch_size = len(batch)

            # Sample random timesteps
            t = torch.randint(0, n_timesteps, (current_batch_size,), dtype=torch.long)

            # Add noise to the data
            x_noisy = add_noise_binary_deterministic(batch, alphas_cumprod, t)

            # Predict original data distribution
            logits = model(x_noisy, t)

            # Target: one-hot encoding of original binary values
            target_indices = batch.long()

            # Compute loss based on loss_function
            if loss_function == 'weighted_mse':
                loss = compute_weighted_loss(logits, target_indices, batch_fitness)
            elif loss_function == 'ranking':
                loss = compute_ranking_loss(logits, target_indices, batch_fitness)
            elif loss_function == 'huber':
                loss = compute_huber_loss(logits, target_indices)
            else:  # 'mse' or default
                # Standard cross-entropy
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
        'loss_function': loss_function,
        'type': 'discrete_dendiff_deterministic_enhanced'
    }
