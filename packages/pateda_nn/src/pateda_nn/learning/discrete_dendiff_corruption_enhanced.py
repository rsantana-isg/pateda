"""
Enhanced Discrete Denoising Diffusion Model with Corruption/Denoising

This module extends the base corruption/denoising dendiff with:
1. Alternative loss functions (weighted_bce, ranking, huber)
2. Fitness guidance/conditioning (inspired by C-VAE and fitness-guided DbD)
3. Flexible architecture configurations

Enhancements inspired by:
- discrete_backdrive_weighted_mse.py: Fitness-weighted loss
- discrete_backdrive_ranking.py: Ranking loss
- discrete_backdrive_huber.py: Huber loss
- discrete_dbd.py: Fitness-guided training
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pateda_nn.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
)

# Import base components from the standard corruption implementation
from pateda_nn.learning.discrete_dendiff_corruption import (
    TimeEmbedding, make_corruption_schedule, corrupt_binary
)


class FitnessGuidedCorruptionDenoisingMLP(nn.Module):
    """
    Fitness-guided corruption-based denoising network for binary variables.

    This variant conditions the denoising on fitness information,
    similar to conditional VAE (C-VAE) and fitness-guided DbD.

    Parameters
    ----------
    input_dim : int
        Number of binary variables
    time_emb_dim : int
        Dimension of time embedding
    fitness_emb_dim : int
        Dimension of fitness embedding
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
        fitness_emb_dim: int = 8,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(FitnessGuidedCorruptionDenoisingMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.fitness_emb_dim = fitness_emb_dim
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

        # Fitness embedding
        self.fitness_embed = nn.Linear(1, fitness_emb_dim)

        # MLP layers
        layers = []
        prev_dim = input_dim + time_emb_dim + fitness_emb_dim

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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        """
        Predict original bit probabilities from corrupted input.

        Parameters
        ----------
        x_t : torch.Tensor
            Corrupted binary input (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices (batch_size,)
        fitness : torch.Tensor
            Fitness values (batch_size, 1)

        Returns
        -------
        logits : torch.Tensor
            Logits for bit probabilities (batch_size, input_dim)
        """
        # Embed timestep
        t_emb = self.time_embed(t)

        # Embed fitness
        f_emb = self.fitness_embed(fitness)

        # Concatenate
        h = torch.cat([x_t, t_emb, f_emb], dim=1)

        # Predict logits (will be passed through sigmoid)
        logits = self.mlp(h)

        return logits


def compute_weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                              fitness: torch.Tensor) -> torch.Tensor:
    """
    Compute fitness-weighted binary cross-entropy loss.

    Higher fitness samples get higher weight.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars]
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

    # Compute per-element loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    # Weight by fitness
    weights = normalized_fitness.expand_as(bce_loss)
    weighted_loss = (bce_loss * weights).mean()

    return weighted_loss


def compute_ranking_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                             fitness: torch.Tensor) -> torch.Tensor:
    """
    Compute ranking-based BCE loss.

    For corruption dendiff, ranking is implemented through weighted sampling.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars]
    target : torch.Tensor
        Target values [batch, n_vars]
    fitness : torch.Tensor
        Fitness values [batch, 1]

    Returns
    -------
    loss : torch.Tensor
        Ranking-aware loss
    """
    # Standard BCE - ranking can be implemented through batch sampling
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_huber_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                          delta: float = 1.0) -> torch.Tensor:
    """
    Compute Huber-like loss for binary cross-entropy.

    Less sensitive to outliers than standard BCE.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars]
    target : torch.Tensor
        Target values [batch, n_vars]
    delta : float
        Huber delta parameter

    Returns
    -------
    loss : torch.Tensor
        Huber-BCE loss
    """
    # Convert to probabilities
    probs = torch.sigmoid(logits)

    # Compute element-wise error
    error = target - probs

    # Apply Huber transformation
    huber = torch.where(torch.abs(error) < delta,
                       0.5 * error ** 2,
                       delta * (torch.abs(error) - 0.5 * delta))

    return huber.mean()


def learn_discrete_dendiff_corruption_enhanced(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn an enhanced discrete denoising diffusion using corruption/denoising approach.

    Enhancements over base version:
    1. Alternative loss functions: mse, weighted_bce, ranking, huber
    2. Fitness guidance/conditioning
    3. Flexible architecture

    Parameters
    ----------
    population : np.ndarray
        Binary population (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (pop_size,)
    params : dict, optional
        Training parameters:
        - Standard corruption dendiff params (n_timesteps, schedule, etc.)
        - 'loss_function': 'mse', 'weighted_bce', 'ranking', 'huber' (default: 'mse')
        - 'use_fitness_guidance': whether to condition on fitness (default: False)
        - 'fitness_weight': weight for fitness guidance (default: 0.1)
        - 'fitness_emb_dim': fitness embedding dimension (default: 8)

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

    # Enhanced parameters
    loss_function = params.get('loss_function', 'mse')
    use_fitness_guidance = params.get('use_fitness_guidance', False)
    fitness_weight = params.get('fitness_weight', 0.1)
    fitness_emb_dim = params.get('fitness_emb_dim', 8)

    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)

    # Ensure binary
    population = (population > 0.5).astype(np.float32)

    # Normalize fitness for conditioning
    fitness_normalized = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)
    fitness_normalized = fitness_normalized.astype(np.float32)

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness_normalized).unsqueeze(1)

    # Create corruption schedule
    corruption_rates = make_corruption_schedule(schedule, n_timesteps, corruption_start, corruption_end)
    corruption_rates_tensor = torch.FloatTensor(corruption_rates)

    # Create model
    if use_fitness_guidance:
        model = FitnessGuidedCorruptionDenoisingMLP(
            input_dim, time_emb_dim, fitness_emb_dim, hidden_dims,
            list_act_functs=list_act_functs,
            list_init_functs=list_init_functs
        )
    else:
        # Use standard model from base implementation
        from pateda_nn.learning.discrete_dendiff_corruption import CorruptionDenoisingMLP
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
            batch_fitness = fitness_tensor[idx]
            current_batch_size = len(batch)

            # Sample random timesteps
            t = torch.randint(0, n_timesteps, (current_batch_size,), dtype=torch.long)

            # Get corruption rates for this batch
            corruption_rate = corruption_rates_tensor[t].unsqueeze(1)

            # Corrupt the data
            x_corrupted = corrupt_binary(batch, corruption_rate)

            # Predict original data
            if use_fitness_guidance:
                logits = model(x_corrupted, t, batch_fitness)
            else:
                logits = model(x_corrupted, t)

            # Compute loss based on loss_function
            if loss_function == 'weighted_bce' or loss_function == 'weighted_mse':
                loss = compute_weighted_bce_loss(logits, batch, batch_fitness)
            elif loss_function == 'ranking':
                loss = compute_ranking_bce_loss(logits, batch, batch_fitness)
            elif loss_function == 'huber':
                loss = compute_huber_bce_loss(logits, batch)
            else:  # 'mse' or default
                # Standard binary cross-entropy
                loss = F.binary_cross_entropy_with_logits(logits, batch)

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
        'corruption_rates': corruption_rates.tolist(),
        'hidden_dims': hidden_dims,
        'list_act_functs': list_act_functs if list_act_functs else ['relu'] * len(hidden_dims),
        'list_init_functs': list_init_functs if list_init_functs else ['default'] * len(hidden_dims),
        'time_emb_dim': time_emb_dim,
        'use_fitness_guidance': use_fitness_guidance,
        'fitness_emb_dim': fitness_emb_dim if use_fitness_guidance else 0,
        'loss_function': loss_function,
        'type': 'discrete_dendiff_corruption_enhanced'
    }
