"""
Discrete Denoising Diffusion Model with Gumbel-Softmax

This module implements a discrete variant of the denoising diffusion model (dendiff)
for binary optimization problems using Gumbel-Softmax for differentiable discrete sampling.

Key Adaptations from Continuous Dendiff:
1. Binary noise is modeled as bit-flip probability rather than Gaussian noise
2. Uses Gumbel-Softmax for differentiable discrete sampling during training
3. Network predicts probabilities of each bit being 1 rather than continuous noise
4. Forward process corrupts by randomly flipping bits with probability based on timestep

Based on insights from:
- DISCRETE_VAE_ANALYSIS.md: Use of Gumbel-Softmax for discrete variables
- DISCRETE_DBD_ANALYSIS.md: Probabilistic blending approaches
- Continuous dendiff.py: Core diffusion framework
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


def sample_gumbel(shape, device=None, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from Gumbel-Softmax distribution
    
    Parameters
    ----------
    logits : torch.Tensor
        Logits for binary choices (shape: [batch, n_vars, 2])
    temperature : float
        Temperature parameter (lower = more discrete)
    hard : bool
        If True, return hard samples but use soft gradients (STE)
    
    Returns
    -------
    samples : torch.Tensor
        Samples from Gumbel-Softmax
    """
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    y = F.softmax(y / temperature, dim=-1)
    
    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    
    return y


class TimeEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion models."""
    
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


class DiscreteDenoisingMLP(nn.Module):
    """
    Discrete denoising network for binary variables.
    
    Predicts the probabilities for each bit given noisy input and timestep.
    Instead of predicting continuous noise, predicts binary probabilities.
    
    Parameters
    ----------
    input_dim : int
        Number of binary variables
    time_emb_dim : int
        Dimension of time embedding
    hidden_dims : list
        Hidden layer dimensions
    list_act_functs : list, optional
        Activation functions for hidden layers
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
        super(DiscreteDenoisingMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
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
            Noisy binary input of shape (batch_size, input_dim)
        t : torch.Tensor
            Timestep indices of shape (batch_size,)
        
        Returns
        -------
        logits : torch.Tensor
            Logits for binary variables of shape (batch_size, input_dim, 2)
        """
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Concatenate noisy input with time embedding
        h = torch.cat([x_t, t_emb], dim=1)
        
        # Pass through MLP
        out = self.mlp(h)
        
        # Reshape to [batch, n_vars, 2] for binary logits
        logits = out.reshape(-1, self.input_dim, 2)
        
        return logits


def make_beta_schedule_discrete(
    schedule: str,
    n_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.5
) -> np.ndarray:
    """
    Create beta schedule for discrete diffusion process.
    
    Beta represents the probability of flipping a bit at each timestep.
    
    Parameters
    ----------
    schedule : str
        Type of schedule: 'linear' or 'cosine'
    n_timesteps : int
        Number of diffusion timesteps
    beta_start : float
        Starting flip probability
    beta_end : float
        Ending flip probability (should be < 0.5 for meaningful signal)
    
    Returns
    -------
    betas : np.ndarray
        Array of bit-flip probabilities
    """
    if schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)
    elif schedule == 'cosine':
        # Cosine schedule adapted for discrete case
        timesteps = np.arange(n_timesteps + 1, dtype=np.float32) / n_timesteps
        alphas_cumprod = np.cos((timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Scale to reasonable range for bit flips
        betas = np.clip(betas * beta_end, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    
    return betas


def compute_diffusion_params_discrete(betas: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Precompute parameters for discrete diffusion.
    
    For discrete case:
    - beta_t: probability of flipping a bit at step t
    - alpha_t: probability of NOT flipping = 1 - beta_t
    - alpha_bar_t: probability that bit stays original after t steps
    
    Parameters
    ----------
    betas : np.ndarray
        Bit-flip probabilities
    
    Returns
    -------
    params : dict
        Diffusion parameters
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
    }


def q_sample_discrete(
    x_0: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Forward diffusion: corrupt binary data by flipping bits.
    
    At timestep t, each bit has probability (1 - alpha_bar_t) of being flipped
    from its original value.
    
    Parameters
    ----------
    x_0 : torch.Tensor
        Clean binary data (0 or 1) of shape (batch_size, input_dim)
    t : torch.Tensor
        Timestep indices of shape (batch_size,)
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas
    noise : torch.Tensor, optional
        Pre-sampled uniform noise in [0, 1]
    
    Returns
    -------
    x_t : torch.Tensor
        Corrupted binary data
    """
    if noise is None:
        noise = torch.rand_like(x_0)
    
    # Probability that bit stays original
    alpha_bar_t = alphas_cumprod[t][:, None]
    
    # Flip bit if noise > alpha_bar_t
    flip_mask = (noise > alpha_bar_t).float()
    
    # XOR with flip mask (binary flip)
    x_t = (x_0 + flip_mask) % 2
    
    return x_t


def learn_discrete_dendiff_gumbel(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a discrete denoising diffusion model using Gumbel-Softmax.
    
    Trains a network to predict the original (clean) binary values given
    corrupted binary inputs and the corruption timestep.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values (not used in basic model)
    params : dict, optional
        Training parameters:
        - 'n_timesteps': number of diffusion steps (default: 100)
        - 'beta_schedule': 'linear' or 'cosine' (default: 'linear')
        - 'beta_start': starting flip probability (default: 0.0001)
        - 'beta_end': ending flip probability (default: 0.3)
        - 'hidden_dims': hidden layer dimensions (default: computed)
        - 'time_emb_dim': time embedding dimension (default: 32)
        - 'epochs': training epochs (default: 50)
        - 'batch_size': batch size (default: computed)
        - 'learning_rate': learning rate (default: 1e-3)
        - 'temperature': Gumbel-Softmax temperature (default: 1.0)
        - 'temperature_decay': temperature decay per epoch (default: 0.99)
        - 'min_temperature': minimum temperature (default: 0.5)
    
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
    
    # Compute adaptive defaults to prevent overfitting
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
    
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)
    
    # Ensure binary values
    population = (population > 0.5).astype(np.float32)
    
    # Convert to tensors
    data = torch.FloatTensor(population)
    
    # Create beta schedule
    betas = make_beta_schedule_discrete(beta_schedule, n_timesteps, beta_start, beta_end)
    diffusion_params = compute_diffusion_params_discrete(betas)
    
    # Convert to tensors
    alphas_cumprod = torch.FloatTensor(diffusion_params['alphas_cumprod'])
    
    # Create denoising network
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
            current_batch_size = len(batch)
            
            # Sample random timesteps
            t = torch.randint(0, n_timesteps, (current_batch_size,), dtype=torch.long)
            
            # Forward diffusion: corrupt binary data
            x_t = q_sample_discrete(batch, t, alphas_cumprod)
            
            # Predict original data distribution
            logits = model(x_t, t)
            
            # Target: one-hot encoding of original binary values
            # Convert binary [0,1] to indices [0,1]
            target_indices = batch.long()
            
            # Cross-entropy loss
            # Reshape logits: [batch, n_vars, 2] -> [batch*n_vars, 2]
            # Reshape target: [batch, n_vars] -> [batch*n_vars]
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
        
        # Print progress (optional logging)
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     avg_loss = epoch_loss / n_batches
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Temp: {current_temp:.3f}')
    
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
        'type': 'discrete_dendiff_gumbel'
    }
