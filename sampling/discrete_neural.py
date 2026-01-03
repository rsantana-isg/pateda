"""
Sampling from Discrete Neural EDA Models

==============================================================================
OVERVIEW
==============================================================================

This module provides sampling functions for discrete neural network-based EDAs:
1. Discrete VAE (Binary and Categorical)
2. Discrete GAN (Binary and Categorical)
3. Discrete Backdrive (Binary and Categorical)

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import network definitions
from pateda.learning.discrete_vae import (
    BinaryVAEEncoder, BinaryVAEDecoder, CategoricalVAEDecoder,
    reparameterize, gumbel_softmax
)
from pateda.learning.discrete_gan import (
    BinaryGANGenerator, CategoricalGANGenerator, gumbel_softmax as gan_gumbel_softmax
)
from pateda.learning.discrete_backdrive import DiscreteBackdriveNet


def sample_binary_vae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary VAE model

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_vae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'sample_from_prior': sample from N(0,I) vs encoding data (default: True)
        - 'temperature': temperature for sampling (default: 0.5)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    sample_from_prior = params.get('sample_from_prior', True)
    temperature = params.get('temperature', 0.5)

    # Reconstruct decoder
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_dec = model['hidden_dims_dec']

    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    with torch.no_grad():
        # Sample from prior
        z = torch.randn(n_samples, latent_dim)

        # Decode
        logits = decoder(z)
        probs = torch.sigmoid(logits)

        # Sample binary values
        # Use Bernoulli sampling
        samples = torch.bernoulli(probs).numpy()

    return samples.astype(int)


def sample_categorical_vae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Categorical VAE model

    Parameters
    ----------
    model : dict
        Model dictionary from learn_categorical_vae
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'temperature': Gumbel-Softmax temperature (default: 0.5)

    Returns
    -------
    samples : np.ndarray
        Categorical samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    temperature = params.get('temperature', model.get('temperature', 0.5))

    # Reconstruct decoder
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    cardinality = model['cardinality']
    hidden_dims_dec = model['hidden_dims_dec']

    decoder = CategoricalVAEDecoder(latent_dim, n_vars, cardinality, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    with torch.no_grad():
        # Sample from prior
        z = torch.randn(n_samples, latent_dim)

        # Decode with hard sampling
        one_hot = decoder(z, temperature=temperature, hard=True)

        # Convert one-hot back to discrete values
        samples = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            start_idx = cum_card[i]
            end_idx = cum_card[i + 1]
            var_one_hot = one_hot[:, start_idx:end_idx]
            samples[:, i] = torch.argmax(var_one_hot, dim=1).numpy()

    return samples


def sample_binary_gan(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary GAN model

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_gan
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'hard_sample': use hard 0/1 samples (default: True)
        - 'threshold': probability threshold for hard sampling (default: 0.5)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    hard_sample = params.get('hard_sample', True)
    threshold = params.get('threshold', 0.5)

    # Reconstruct generator
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_g = model['hidden_dims_g']

    generator = BinaryGANGenerator(latent_dim, n_vars, hidden_dims_g)
    generator.load_state_dict(model['generator_state'])
    generator.eval()

    with torch.no_grad():
        # Sample latent noise
        z = torch.randn(n_samples, latent_dim)

        # Generate
        if hard_sample:
            probs = generator(z, hard_sample=False)
            samples = (probs > threshold).float().numpy()
        else:
            probs = generator(z, hard_sample=True)
            samples = probs.numpy()

    return samples.astype(int)


def sample_categorical_gan(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Categorical GAN model

    Parameters
    ----------
    model : dict
        Model dictionary from learn_categorical_gan
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'temperature': Gumbel-Softmax temperature (default: 0.5)

    Returns
    -------
    samples : np.ndarray
        Categorical samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    temperature = params.get('temperature', model.get('temperature', 0.5))

    # Reconstruct generator
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    cardinality = model['cardinality']
    hidden_dims_g = model['hidden_dims_g']

    generator = CategoricalGANGenerator(latent_dim, n_vars, cardinality, hidden_dims_g)
    generator.load_state_dict(model['generator_state'])
    generator.eval()

    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    with torch.no_grad():
        # Sample latent noise
        z = torch.randn(n_samples, latent_dim)

        # Generate with hard sampling
        one_hot = generator(z, temperature=temperature, hard=True)

        # Convert one-hot to discrete values
        samples = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            start_idx = cum_card[i]
            end_idx = cum_card[i + 1]
            var_one_hot = one_hot[:, start_idx:end_idx]
            samples[:, i] = torch.argmax(var_one_hot, dim=1).numpy()

    return samples


def sample_discrete_backdrive(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Discrete Backdrive model via network inversion

    Generates solutions by optimizing continuous relaxations to maximize
    predicted fitness, then projecting back to discrete values.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_discrete_backdrive
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'n_iterations': optimization iterations (default: 100)
        - 'learning_rate': optimization learning rate (default: 0.1)
        - 'init_method': initialization method: 'random', 'uniform', 'perturb_best', 'perturb_selected' (default: 'random')
        - 'temperature': Gumbel-Softmax temperature (default: 1.0)
        - 'temperature_decay': temperature decay per iteration (default: 0.99)
        - 'init_noise': noise level for perturbation methods (default: 0.1)
        - 'current_population': current population for perturb methods (required for 'perturb_best', 'perturb_selected')
        - 'current_fitness': current fitness for perturb methods (required for 'perturb_best', 'perturb_selected')

    Returns
    -------
    samples : np.ndarray
        Discrete samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    n_iterations = params.get('n_iterations', 100)
    learning_rate = params.get('learning_rate', 0.1)
    init_method = params.get('init_method', 'random')
    temperature = params.get('temperature', 1.0)
    temperature_decay = params.get('temperature_decay', 0.99)
    init_noise = params.get('init_noise', 0.1)

    # Reconstruct network
    n_vars = model['n_vars']
    cardinality = model['cardinality']
    hidden_layers = model['hidden_layers']
    use_embeddings = model['use_embeddings']
    embedding_dim = model.get('embedding_dim', 8)

    network = DiscreteBackdriveNet(
        n_vars, cardinality, hidden_layers, use_embeddings, embedding_dim
    )
    network.load_state_dict(model['network_state'])
    network.eval()

    # Fitness statistics for denormalization
    fitness_mean, fitness_std = model['fitness_stats']

    # Initialize solutions in continuous space
    # For each variable, maintain a distribution over its possible values
    if init_method == 'uniform':
        # Uniform distribution over categories
        logits = torch.zeros(n_samples, n_vars, int(np.max(cardinality)))
        for i in range(n_vars):
            card = int(cardinality[i])
            logits[:, i, :card] = 0.0  # Uniform logits
    elif init_method == 'perturb_best':
        # Initialize from best solution with perturbations
        current_population = params.get('current_population')
        current_fitness = params.get('current_fitness')
        
        if current_population is None or current_fitness is None:
            raise ValueError("'perturb_best' requires 'current_population' and 'current_fitness' in params")
        
        # Find best solution
        best_idx = np.argmax(current_fitness.flatten())
        best_solution = current_population[best_idx]
        
        # Convert best solution to one-hot logits
        logits = torch.zeros(n_samples, n_vars, int(np.max(cardinality)))
        for i in range(n_vars):
            card = int(cardinality[i])
            # Set logit for the best solution's value high
            value = int(best_solution[i])
            logits[:, i, value] = 5.0  # High logit for best value
            # Add noise to all logits
            logits[:, i, :card] += torch.randn(n_samples, card) * init_noise
    
    elif init_method == 'perturb_selected':
        # Initialize from top solutions with perturbations
        current_population = params.get('current_population')
        current_fitness = params.get('current_fitness')
        
        if current_population is None or current_fitness is None:
            raise ValueError("'perturb_selected' requires 'current_population' and 'current_fitness' in params")
        
        # Select top solutions
        n_select = min(n_samples, len(current_population) // 2)
        top_indices = np.argsort(current_fitness.flatten())[-n_select:]
        selected_solutions = current_population[top_indices]
        
        # Replicate and perturb
        n_copies = n_samples // n_select + 1
        repeated = np.tile(selected_solutions, (n_copies, 1))[:n_samples]
        
        # Convert to one-hot logits with noise
        logits = torch.zeros(n_samples, n_vars, int(np.max(cardinality)))
        for i in range(n_vars):
            card = int(cardinality[i])
            for j in range(n_samples):
                value = int(repeated[j, i])
                logits[j, i, value] = 5.0  # High logit for selected value
                # Add noise to all logits
                logits[j, i, :card] += torch.randn(card) * init_noise
    
    else:
        # Random initialization (default)
        logits = torch.randn(n_samples, n_vars, int(np.max(cardinality))) * 0.1

    logits.requires_grad = True

    # Optimizer
    optimizer = optim.Adam([logits], lr=learning_rate)

    current_temp = temperature

    # Optimization loop
    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Convert logits to soft samples using Gumbel-Softmax
        soft_samples = []
        for i in range(n_vars):
            card = int(cardinality[i])
            var_logits = logits[:, i, :card]

            # Gumbel-Softmax
            gumbel = -torch.log(-torch.log(torch.rand_like(var_logits) + 1e-20) + 1e-20)
            soft_sample = F.softmax((var_logits + gumbel) / current_temp, dim=-1)

            # Get hard index
            hard_sample = torch.zeros_like(soft_sample)
            hard_idx = soft_sample.argmax(dim=-1, keepdim=True)
            hard_sample.scatter_(-1, hard_idx, 1.0)

            # Straight-through estimator
            soft_samples.append((hard_sample - soft_sample).detach() + soft_sample)

        # Stack to [n_samples, n_vars, max_card]
        soft_samples_tensor = torch.stack(soft_samples, dim=1)

        # Convert to discrete indices for network input
        discrete_samples = torch.argmax(soft_samples_tensor, dim=-1)

        # Predict fitness
        predicted_fitness = network(discrete_samples)

        # Maximize predicted fitness (minimize negative fitness)
        loss = -predicted_fitness.mean()

        loss.backward()
        optimizer.step()

        # Decay temperature
        current_temp = max(0.1, current_temp * temperature_decay)

    # Final sampling with hard discretization
    with torch.no_grad():
        samples = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            card = int(cardinality[i])
            var_logits = logits[:, i, :card]
            samples[:, i] = torch.argmax(var_logits, dim=-1).numpy()

    return samples


def sample_binary_backdrive(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary Backdrive model (simplified interface)

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_backdrive
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters (same as sample_discrete_backdrive)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    return sample_discrete_backdrive(model, n_samples, params)


def sample_discrete_backdrive_adaptive(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Adaptive discrete backdrive sampling with multiple target fitness levels.

    This variant samples solutions targeting different fitness levels to maintain
    diversity while still focusing on high-fitness regions. It works by running
    multiple backdrive sampling operations with different target fitness percentiles.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_discrete_backdrive
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (extends sample_discrete_backdrive parameters):
        - 'target_levels': list of target fitness percentiles to sample from (default: [100, 90, 80])
        - 'level_fractions': fraction of samples per level (default: [0.5, 0.3, 0.2])
        - All other parameters from sample_discrete_backdrive

    Returns
    -------
    samples : np.ndarray
        Discrete samples of shape (n_samples, n_vars) with diverse fitness targets

    Notes
    -----
    This method helps maintain exploration by targeting multiple fitness levels
    rather than only the absolute best, which can help avoid premature convergence.
    
    Note: For discrete problems, target fitness levels don't have the same direct
    interpretation as in continuous optimization since we're optimizing discrete
    categorical logits. The approach still provides diversity by starting from
    different initial configurations.
    """
    if params is None:
        params = {}

    n_samples_total = n_samples
    target_levels = params.get('target_levels', [100, 90, 80])
    level_fractions = params.get('level_fractions', [0.5, 0.3, 0.2])

    # Normalize fractions
    level_fractions = np.array(level_fractions)
    level_fractions = level_fractions / level_fractions.sum()

    # Generate samples for each target level
    all_solutions = []

    for target_level, fraction in zip(target_levels, level_fractions):
        n_samples_level = int(n_samples_total * fraction)

        if n_samples_level > 0:
            # Create params for this level
            level_params = params.copy()
            level_params['n_samples'] = n_samples_level
            
            # For discrete backdrive, we interpret target levels as initialization diversity
            # Higher target level = more focused initialization (if using perturb methods)
            # Lower target level = more diverse/random initialization
            if target_level < 100:
                # Add more noise for lower target levels to increase diversity
                if 'init_noise' in level_params:
                    level_params['init_noise'] = level_params['init_noise'] * (1.0 + (100 - target_level) / 100.0)

            # Sample
            solutions = sample_discrete_backdrive(model, n_samples_level, level_params)
            all_solutions.append(solutions)

    # Concatenate all solutions
    if len(all_solutions) > 0:
        all_solutions = np.vstack(all_solutions)
    else:
        # Fallback to single sample if no levels generated samples
        return sample_discrete_backdrive(model, n_samples_total, params)

    # If we have more or fewer samples than needed due to rounding, adjust
    if len(all_solutions) > n_samples_total:
        all_solutions = all_solutions[:n_samples_total]
    elif len(all_solutions) < n_samples_total:
        # Fill remaining with top-level samples
        n_remaining = n_samples_total - len(all_solutions)
        extra_params = params.copy()
        extra_solutions = sample_discrete_backdrive(model, n_remaining, extra_params)
        all_solutions = np.vstack([all_solutions, extra_solutions])

    return all_solutions


def sample_binary_backdrive_adaptive(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Adaptive binary backdrive sampling (simplified interface)

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_backdrive
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters (same as sample_discrete_backdrive_adaptive)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    return sample_discrete_backdrive_adaptive(model, n_samples, params)
