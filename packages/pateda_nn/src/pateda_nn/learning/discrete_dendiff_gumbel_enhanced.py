"""
Enhanced Discrete Denoising Diffusion Model with Gumbel-Softmax

This module extends the base Gumbel-Softmax dendiff with:
1. Alternative loss functions (weighted_mse, ranking, huber)
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
import math

from pateda_nn.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
)

# Import base components from the standard gumbel implementation
from pateda_nn.learning.discrete_dendiff_gumbel import (
    TimeEmbedding, sample_gumbel, gumbel_softmax,
    make_beta_schedule_discrete, compute_diffusion_params_discrete,
    q_sample_discrete
)


class FitnessGuidedDiscreteDenoisingMLP(nn.Module):
    """
    Fitness-guided discrete denoising network for binary variables.

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
        super(FitnessGuidedDiscreteDenoisingMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.fitness_emb_dim = fitness_emb_dim
        self.hidden_dims = hidden_dims

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Fitness embedding: simple linear projection
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

        # Output layer: predict logits for binary [0, 1] for each variable
        # Output shape: [batch, n_vars, 2]
        output_layer = nn.Linear(prev_dim, input_dim * 2)
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        """
        Predict clean data probabilities given noisy input, timestep, and fitness.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy binary input of shape (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices of shape (batch_size,)
        fitness : torch.Tensor
            Fitness values of shape (batch_size, 1)

        Returns
        -------
        logits : torch.Tensor
            Logits for binary variables of shape (batch_size, input_dim, 2)
        """
        # Embed timestep
        t_emb = self.time_embed(t)

        # Embed fitness
        f_emb = self.fitness_embed(fitness)

        # Concatenate noisy input with time and fitness embeddings
        h = torch.cat([x_t, t_emb, f_emb], dim=1)

        # Pass through MLP
        out = self.mlp(h)

        # Reshape to [batch, n_vars, 2] for binary logits
        logits = out.reshape(-1, self.input_dim, 2)

        return logits


def compute_weighted_loss(logits: torch.Tensor, target: torch.Tensor,
                         fitness: torch.Tensor) -> torch.Tensor:
    """
    Compute fitness-weighted cross-entropy loss.

    Higher fitness samples get higher weight.
    Inspired by discrete_backdrive_weighted_mse.py

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
    logits_flat = logits.reshape(-1, 2)
    target_flat = target.reshape(-1).long()
    loss_per_sample = F.cross_entropy(logits_flat, target_flat, reduction='none')

    # Reshape back to [batch, n_vars]
    loss_per_sample = loss_per_sample.reshape(target.shape)

    # Weight by fitness (expand to match dimensions)
    weights = normalized_fitness.expand_as(loss_per_sample)
    weighted_loss = (loss_per_sample * weights).mean()

    return weighted_loss


def compute_ranking_loss(logits: torch.Tensor, target: torch.Tensor,
                        fitness: torch.Tensor, margin: float = 0.1, 
                        ranking_weight: float = 0.5) -> torch.Tensor:
    """
    Compute ranking-based loss.

    Prioritizes learning the relative ordering of solutions by fitness.
    Inspired by discrete_backdrive_ranking.py
    
    Uses pairwise ranking: for samples with higher fitness, encourage
    the model to produce predictions that would lead to higher fitness
    (lower cross-entropy loss).
    
    Note: This function has O(n²) space complexity for batch size n,
    as it creates matrices for all pairwise comparisons. For very large
    batch sizes (>256), this could cause memory issues. The batch sizes
    in dendiff are typically smaller due to the diffusion training process.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits [batch, n_vars, 2]
    target : torch.Tensor
        Target values [batch, n_vars]
    fitness : torch.Tensor
        Fitness values [batch, 1]
    margin : float
        Margin for ranking loss (default: 0.1)
    ranking_weight : float
        Weight for ranking vs standard cross-entropy (default: 0.5)

    Returns
    -------
    loss : torch.Tensor
        Ranking-aware loss
    """
    # Compute standard cross-entropy for each sample
    logits_flat = logits.reshape(-1, 2)
    target_flat = target.reshape(-1).long()
    
    # Get per-sample loss (reshaped to [batch, n_vars])
    ce_loss_per_element = F.cross_entropy(logits_flat, target_flat, reduction='none')
    ce_loss_per_sample = ce_loss_per_element.reshape(logits.shape[0], -1).mean(dim=1, keepdim=True)
    
    batch_size = fitness.shape[0]
    
    if batch_size < 2:
        # Not enough samples for pairwise comparison, use standard cross-entropy
        return ce_loss_per_sample.mean()
    
    # Compute pairwise ranking loss
    # For pairs where fitness[i] > fitness[j], encourage loss[i] < loss[j] + margin
    # This means the model should predict better for higher fitness samples
    
    # Compute all pairwise differences
    # Memory: O(batch_size²)
    fitness_diff = fitness - fitness.t()  # [batch, batch]: fitness[i] - fitness[j]
    loss_diff = ce_loss_per_sample - ce_loss_per_sample.t()  # [batch, batch]: loss[i] - loss[j]
    
    # For pairs where fitness[i] > fitness[j], we want loss[i] < loss[j] 
    # So we want loss_diff < 0, or equivalently: max(0, loss_diff + margin)
    valid_pairs = (fitness_diff > 0).float()
    
    # Hinge loss: encourage loss[i] to be at least margin lower than loss[j]
    hinge_loss = torch.clamp(loss_diff + margin, min=0)
    
    # Apply mask and average over valid pairs
    masked_loss = hinge_loss * valid_pairs
    n_valid_pairs = valid_pairs.sum()
    
    if n_valid_pairs > 0:
        pairwise_ranking_loss = masked_loss.sum() / n_valid_pairs
    else:
        # No valid pairs, use only standard cross-entropy
        pairwise_ranking_loss = 0.0
    
    # Combine standard cross-entropy with ranking loss
    standard_ce = ce_loss_per_sample.mean()
    total_loss = (1 - ranking_weight) * standard_ce + ranking_weight * pairwise_ranking_loss
    
    return total_loss


def compute_huber_loss(logits: torch.Tensor, target: torch.Tensor,
                      delta: float = 1.0) -> torch.Tensor:
    """
    Compute Huber loss for robust training.

    Huber loss is less sensitive to outliers than squared error.
    Inspired by discrete_backdrive_huber.py

    For cross-entropy, we use smooth approximation.

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

    # Negative log likelihood
    nll = -torch.log(target_probs + 1e-8)

    # Apply Huber transformation
    huber = torch.where(nll < delta,
                       0.5 * nll ** 2,
                       delta * (nll - 0.5 * delta))

    return huber.mean()


def learn_discrete_dendiff_gumbel_enhanced(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn an enhanced discrete denoising diffusion model using Gumbel-Softmax.

    Enhancements over base version:
    1. Alternative loss functions: mse, weighted_mse, ranking, huber
    2. Fitness guidance/conditioning
    3. Flexible architecture

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values of shape (pop_size,)
    params : dict, optional
        Training parameters:
        - Standard dendiff params (n_timesteps, beta_schedule, etc.)
        - 'loss_function': 'mse', 'weighted_mse', 'ranking', 'huber' (default: 'mse')
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
    n_timesteps = params.get('n_timesteps', 100)
    beta_schedule = params.get('beta_schedule', 'linear')
    beta_start = params.get('beta_start', 0.0001)
    beta_end = params.get('beta_end', 0.3)
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    time_emb_dim = params.get('time_emb_dim', 32)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 1e-3)
    temperature = params.get('temperature', 1.0)
    temperature_decay = params.get('temperature_decay', 0.99)
    min_temperature = params.get('min_temperature', 0.5)

    # Enhanced parameters
    loss_function = params.get('loss_function', 'mse')
    use_fitness_guidance = params.get('use_fitness_guidance', False)
    fitness_weight = params.get('fitness_weight', 0.1)
    fitness_emb_dim = params.get('fitness_emb_dim', 8)

    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)

    # Ensure binary values
    population = (population > 0.5).astype(np.float32)

    # Normalize fitness for conditioning
    fitness_normalized = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)
    fitness_normalized = fitness_normalized.astype(np.float32)

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness_normalized).unsqueeze(1)

    # Create beta schedule
    betas = make_beta_schedule_discrete(beta_schedule, n_timesteps, beta_start, beta_end)
    diffusion_params = compute_diffusion_params_discrete(betas)

    # Convert to tensors
    alphas_cumprod = torch.FloatTensor(diffusion_params['alphas_cumprod'])

    # Create denoising network
    if use_fitness_guidance:
        model = FitnessGuidedDiscreteDenoisingMLP(
            input_dim, time_emb_dim, fitness_emb_dim, hidden_dims,
            list_act_functs=list_act_functs,
            list_init_functs=list_init_functs
        )
    else:
        # Use standard model from base implementation
        from pateda_nn.learning.discrete_dendiff_gumbel import DiscreteDenoisingMLP
        model = DiscreteDenoisingMLP(
            input_dim, time_emb_dim, hidden_dims,
            list_act_functs=list_act_functs,
            list_init_functs=list_init_functs
        )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    current_temp = temperature

    for epoch in range(epochs):
        # Shuffle data
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

            # Forward diffusion: corrupt binary data
            x_t = q_sample_discrete(batch, t, alphas_cumprod)

            # Predict original data distribution
            if use_fitness_guidance:
                logits = model(x_t, t, batch_fitness)
            else:
                logits = model(x_t, t)

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

        # Temperature annealing
        current_temp = max(min_temperature, current_temp * temperature_decay)

    # Return model with all configuration
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
        'final_temperature': current_temp,
        'use_fitness_guidance': use_fitness_guidance,
        'fitness_emb_dim': fitness_emb_dim if use_fitness_guidance else 0,
        'loss_function': loss_function,
        'type': 'discrete_dendiff_gumbel_enhanced'
    }
