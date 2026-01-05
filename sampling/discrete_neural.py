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
    ConditionalBinaryVAEEncoder, ConditionalBinaryVAEDecoder,
    DescriptorBinaryVAEEncoder, FitnessPredictor,
    reparameterize, gumbel_softmax
)
from pateda.learning.discrete_gan import (
    BinaryGANGenerator, CategoricalGANGenerator, gumbel_softmax as gan_gumbel_softmax
)
from pateda.learning.discrete_backdrive import DiscreteBackdriveNet
from pateda.learning.discrete_backdrive_descriptors import DescriptorBackdriveNet


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
        - 'deterministic': use deterministic sampling (argmax) instead of random (default: False)
        - 'use_fitness_guidance': use fitness predictor for E-VAE sampling (default: False)
        - 'target_fitness': target fitness for guided sampling (default: None)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    sample_from_prior = params.get('sample_from_prior', True)
    temperature = params.get('temperature', 0.5)
    deterministic = params.get('deterministic', False)
    use_fitness_guidance = params.get('use_fitness_guidance', False)
    target_fitness = params.get('target_fitness', None)

    # Reconstruct decoder
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_dec = model['hidden_dims_dec']

    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    # For E-VAE, optionally use fitness predictor
    fitness_predictor = None
    if use_fitness_guidance and model.get('type') == 'binary_evae':
        from pateda.learning.discrete_vae import FitnessPredictor
        fitness_predictor = FitnessPredictor(latent_dim, 1, 32)
        fitness_predictor.load_state_dict(model['fitness_predictor_state'])
        fitness_predictor.eval()

    with torch.no_grad():
        if use_fitness_guidance and fitness_predictor is not None and target_fitness is not None:
            # Sample multiple latent codes and select based on fitness
            n_candidates = n_samples * 10
            z_candidates = torch.randn(n_candidates, latent_dim)
            pred_fitness = fitness_predictor(z_candidates).squeeze()

            # Select samples closest to target fitness
            fitness_diff = torch.abs(pred_fitness - target_fitness)
            _, top_indices = torch.topk(fitness_diff, n_samples, largest=False)
            z = z_candidates[top_indices]
        else:
            # Standard sampling from prior
            z = torch.randn(n_samples, latent_dim)

        # Decode
        logits = decoder(z)
        probs = torch.sigmoid(logits)

        # Sample binary values
        if deterministic:
            # Deterministic sampling: use threshold at 0.5 for exploitation
            samples = (probs > 0.5).float().numpy()
        else:
            # Stochastic sampling: use Bernoulli for exploration
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


def sample_binary_cvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Conditional Binary VAE (C-VAE) model

    This variant allows conditioning on target fitness and statistics during sampling,
    enabling fitness-guided generation.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_cvae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'target_condition': target condition vector [fitness, mean, std] (default: None, uses max from training)
        - 'sample_from_prior': sample from N(0,I) vs encoding data (default: True)
        - 'deterministic': use deterministic sampling (argmax) instead of random (default: False)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    sample_from_prior = params.get('sample_from_prior', True)
    deterministic = params.get('deterministic', False)
    target_condition = params.get('target_condition', None)

    # Reconstruct networks
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    condition_dim = model['condition_dim']
    hidden_dims_dec = model['hidden_dims_dec']
    condition_means = model['condition_means']
    condition_stds = model['condition_stds']

    decoder = ConditionalBinaryVAEDecoder(latent_dim, n_vars, condition_dim, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    # Prepare target condition
    if target_condition is None:
        # Use maximum fitness from normalized conditions (approximately +1.0 std above mean)
        target_condition = np.array([1.5, 0.5, 0.3])  # [high fitness, medium mean, low std]

    # Ensure target_condition is normalized
    if len(target_condition) != condition_dim:
        raise ValueError(f"target_condition must have {condition_dim} dimensions")

    # Broadcast condition to all samples
    conditions = np.tile(target_condition, (n_samples, 1))
    conditions_tensor = torch.FloatTensor(conditions)

    with torch.no_grad():
        # Sample from prior
        z = torch.randn(n_samples, latent_dim)

        # Decode with condition
        logits = decoder(z, conditions_tensor)
        probs = torch.sigmoid(logits)

        # Sample binary values
        if deterministic:
            samples = (probs > 0.5).float().numpy()
        else:
            samples = torch.bernoulli(probs).numpy()

    return samples.astype(int)


def sample_binary_descvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Descriptor-augmented Binary VAE (Desc-VAE) model

    This variant uses descriptors during encoding but samples from standard decoder.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_descvae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'sample_from_prior': sample from N(0,I) (default: True)
        - 'deterministic': use deterministic sampling (argmax) instead of random (default: False)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    sample_from_prior = params.get('sample_from_prior', True)
    deterministic = params.get('deterministic', False)

    # Reconstruct decoder
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_dec = model['hidden_dims_dec']

    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    with torch.no_grad():
        # Sample from prior (descriptors only used during encoding/training)
        z = torch.randn(n_samples, latent_dim)

        # Decode
        logits = decoder(z)
        probs = torch.sigmoid(logits)

        # Sample binary values
        if deterministic:
            samples = (probs > 0.5).float().numpy()
        else:
            samples = torch.bernoulli(probs).numpy()

    return samples.astype(int)


def sample_binary_regvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Regression-focused Binary VAE (Reg-VAE) model

    This variant can use the fitness predictor to guide sampling by selecting
    latent codes with high predicted fitness.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_regvae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'use_fitness_guidance': use fitness predictor for guided sampling (default: False)
        - 'target_fitness': target fitness for guided sampling (default: None, uses max)
        - 'candidate_factor': generate this many candidates for filtering (default: 10)
        - 'deterministic': use deterministic sampling (argmax) instead of random (default: False)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    use_fitness_guidance = params.get('use_fitness_guidance', False)
    target_fitness = params.get('target_fitness', None)
    candidate_factor = params.get('candidate_factor', 10)
    deterministic = params.get('deterministic', False)

    # Reconstruct decoder
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_dec = model['hidden_dims_dec']

    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec)
    decoder.load_state_dict(model['decoder_state'])
    decoder.eval()

    # For fitness guidance, also load fitness predictor
    if use_fitness_guidance:
        fitness_predictor = FitnessPredictor(latent_dim, 1, 32)
        fitness_predictor.load_state_dict(model['fitness_predictor_state'])
        fitness_predictor.eval()

    with torch.no_grad():
        if use_fitness_guidance:
            # Sample multiple latent codes and select based on fitness
            n_candidates = n_samples * candidate_factor
            z_candidates = torch.randn(n_candidates, latent_dim)
            pred_fitness = fitness_predictor(z_candidates).squeeze()

            if target_fitness is not None:
                # Select samples closest to target fitness
                fitness_diff = torch.abs(pred_fitness - target_fitness)
                _, top_indices = torch.topk(fitness_diff, n_samples, largest=False)
            else:
                # Select samples with highest predicted fitness
                _, top_indices = torch.topk(pred_fitness, n_samples, largest=True)

            z = z_candidates[top_indices]
        else:
            # Standard sampling from prior
            z = torch.randn(n_samples, latent_dim)

        # Decode
        logits = decoder(z)
        probs = torch.sigmoid(logits)

        # Sample binary values
        if deterministic:
            samples = (probs > 0.5).float().numpy()
        else:
            samples = torch.bernoulli(probs).numpy()

    return samples.astype(int)


def sample_binary_momvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Moment-matching Binary VAE (Mom-VAE) model

    This variant samples from the standard VAE, which has been trained to match
    the statistical moments of the elite population.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_momvae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'sample_from_prior': sample from N(0,I) (default: True)
        - 'deterministic': use deterministic sampling (argmax) instead of random (default: False)

    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    sample_from_prior = params.get('sample_from_prior', True)
    deterministic = params.get('deterministic', False)

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
        if deterministic:
            samples = (probs > 0.5).float().numpy()
        else:
            samples = torch.bernoulli(probs).numpy()

    return samples.astype(int)


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

    Uses gradient-based optimization in continuous relaxation space (Gumbel-Softmax)
    to find solutions that maximize predicted fitness, then projects to discrete values.

    Parameters
    ----------
    model : dict
        Model dictionary from learn_discrete_backdrive
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'n_iterations': number of optimization iterations (default: 100)
        - 'learning_rate': learning rate for optimization (default: 0.01)
        - 'gradient_clip': gradient clipping max_norm (default: 1.0)
        - 'init_method': initialization method - 'random', 'uniform', 'perturb_best', 'perturb_selected' (default: 'random')
        - 'temperature': initial Gumbel-Softmax temperature (default: 2.0)
        - 'temperature_min': minimum temperature (default: 0.1)
        - 'temperature_schedule': 'cosine' or 'exponential' (default: 'cosine')
        - 'temperature_decay': temperature decay for exponential schedule (default: 0.99)
        - 'use_gumbel_noise': whether to use Gumbel noise during optimization (default: False)
        - 'init_noise': noise level for perturbation methods (default: 0.1)
        - 'bias_strength': strength of initialization bias for perturb methods (default: 5.0)
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
    # Reduced learning rate from 0.1 to 0.01
    learning_rate = params.get('learning_rate', 0.01)
    # Add gradient clipping
    gradient_clip = params.get('gradient_clip', 1.0)
    init_method = params.get('init_method', 'random')
    # Increased initial temperature from 1.0 to 2.0
    temperature = params.get('temperature', 2.0)
    # Add minimum temperature
    temperature_min = params.get('temperature_min', 0.1)
    # Add temperature schedule option
    temperature_schedule = params.get('temperature_schedule', 'cosine')
    temperature_decay = params.get('temperature_decay', 0.99)
    # Remove Gumbel noise during optimization by default
    # Rationale: Continuous Gumbel noise at every iteration introduces stochasticity that
    # hinders convergence. Deterministic softmax provides cleaner gradients for optimization.
    # Gumbel noise can still be enabled via parameter if needed for diversity.
    use_gumbel_noise = params.get('use_gumbel_noise', False)
    init_noise = params.get('init_noise', 0.1)
    bias_strength = params.get('bias_strength', 5.0)  # Strength of initialization bias

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
            # Validate value is within valid range
            if value < 0 or value >= card:
                raise ValueError(f"Invalid value {value} for variable {i} with cardinality {card}")
            logits[:, i, value] = bias_strength  # High logit for best value
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
                # Validate value is within valid range
                if value < 0 or value >= card:
                    raise ValueError(f"Invalid value {value} for variable {i} with cardinality {card}")
                logits[j, i, value] = bias_strength  # High logit for selected value
                # Add noise to all logits
                logits[j, i, :card] += torch.randn(card) * init_noise
    
    else:
        # Random initialization (default)
        logits = torch.randn(n_samples, n_vars, int(np.max(cardinality))) * 0.1

    logits.requires_grad = True

    # Optimizer
    optimizer = optim.Adam([logits], lr=learning_rate)

    # Initial temperature for cosine annealing
    current_temp = temperature

    # Optimization loop
    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Compute current temperature based on schedule
        if temperature_schedule == 'cosine':
            # Cosine annealing: starts at temperature, ends at temperature_min
            progress = iteration / n_iterations
            current_temp = temperature_min + 0.5 * (temperature - temperature_min) * (1 + np.cos(np.pi * progress))
        else:
            # Exponential decay
            current_temp = max(temperature_min, current_temp * temperature_decay)

        # Convert logits to soft samples
        soft_samples = []
        for i in range(n_vars):
            card = int(cardinality[i])
            var_logits = logits[:, i, :card]

            # Optionally add Gumbel noise (disabled by default per analysis)
            if use_gumbel_noise:
                gumbel = -torch.log(-torch.log(torch.rand_like(var_logits) + 1e-20) + 1e-20)
                soft_sample = F.softmax((var_logits + gumbel) / current_temp, dim=-1)
            else:
                # Use deterministic softmax during optimization
                soft_sample = F.softmax(var_logits / current_temp, dim=-1)

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
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_([logits], max_norm=gradient_clip)
        
        optimizer.step()

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


def sample_discrete_backdrive_descriptors(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Descriptor-based Backdrive model
    
    Generates solutions by sampling descriptors from the selected population
    distribution and feeding them through the learned network.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_discrete_backdrive_descriptors
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'selected_population': selected population for descriptor sampling (required)
        - 'selected_fitness': selected fitness for descriptor sampling (required)
        - 'descriptor_sampling': how to sample descriptors:
            * 'from_population': sample from actual selected population (default)
            * 'gaussian': fit Gaussian to descriptors and sample from it
            * 'uniform': uniform sampling in descriptor ranges
        - 'temperature': temperature for sigmoid sampling (default: 0.5)
        - 'deterministic': use deterministic thresholding (default: False)
        
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    # Get required population data
    selected_population = params.get('selected_population')
    selected_fitness = params.get('selected_fitness')
    
    if selected_population is None or selected_fitness is None:
        raise ValueError("'selected_population' and 'selected_fitness' required in params for descriptor sampling")
    
    descriptor_sampling = params.get('descriptor_sampling', 'from_population')
    temperature = params.get('temperature', 0.5)
    deterministic = params.get('deterministic', False)
    
    # Reconstruct network
    n_vars = model['n_vars']
    n_descriptors = model['n_descriptors']
    hidden_layers = model['hidden_layers']
    descriptor_means, descriptor_stds = model['descriptor_stats']
    
    network = DescriptorBackdriveNet(n_vars, n_descriptors, hidden_layers)
    network.load_state_dict(model['network_state'])
    network.eval()
    
    # Compute descriptors from selected population
    fitness_1d = selected_fitness.flatten()
    pop_descriptors = np.zeros((len(selected_population), 3))
    pop_descriptors[:, 0] = fitness_1d
    pop_descriptors[:, 1] = np.mean(selected_population, axis=1)
    pop_descriptors[:, 2] = np.std(selected_population, axis=1)
    
    # Sample target descriptors based on method
    if descriptor_sampling == 'from_population':
        # Randomly sample descriptors from the selected population
        indices = np.random.choice(len(pop_descriptors), size=n_samples, replace=True)
        target_descriptors = pop_descriptors[indices]
        
    elif descriptor_sampling == 'gaussian':
        # Fit Gaussian to descriptors and sample
        desc_mean = np.mean(pop_descriptors, axis=0)
        desc_cov = np.cov(pop_descriptors, rowvar=False)
        
        # Regularization constant for ensuring positive definite covariance
        COV_REGULARIZATION = 1e-6
        desc_cov += np.eye(3) * COV_REGULARIZATION
        
        target_descriptors = np.random.multivariate_normal(desc_mean, desc_cov, size=n_samples)
        
        # Clip to reasonable ranges based on actual population data
        # Fitness range from population
        target_descriptors[:, 0] = np.clip(target_descriptors[:, 0], 
                                           np.min(pop_descriptors[:, 0]), 
                                           np.max(pop_descriptors[:, 0]))
        # Mean range from population (for binary: 0 to 1)
        target_descriptors[:, 1] = np.clip(target_descriptors[:, 1], 
                                           max(0, np.min(pop_descriptors[:, 1])), 
                                           min(1, np.max(pop_descriptors[:, 1])))
        # Std range from population (non-negative, max is theoretical max for binary)
        target_descriptors[:, 2] = np.clip(target_descriptors[:, 2], 
                                           0, 
                                           min(0.5, np.max(pop_descriptors[:, 2])))
        
    elif descriptor_sampling == 'uniform':
        # Uniform sampling in descriptor ranges
        target_descriptors = np.zeros((n_samples, 3))
        for i in range(3):
            min_val = np.min(pop_descriptors[:, i])
            max_val = np.max(pop_descriptors[:, i])
            target_descriptors[:, i] = np.random.uniform(min_val, max_val, size=n_samples)
    else:
        raise ValueError(f"Unknown descriptor_sampling method: {descriptor_sampling}")
    
    # Normalize descriptors using stored statistics
    normalized_descriptors = (target_descriptors - descriptor_means) / descriptor_stds
    
    # Generate solutions from descriptors
    with torch.no_grad():
        desc_tensor = torch.FloatTensor(normalized_descriptors)
        logits = network(desc_tensor)
        probs = torch.sigmoid(logits)
        
        # Sample binary values
        if deterministic:
            # Deterministic: threshold at 0.5
            samples = (probs > 0.5).float().numpy()
        else:
            # Stochastic: Bernoulli sampling
            samples = torch.bernoulli(probs).numpy()
    
    return samples.astype(int)


def sample_binary_backdrive_descriptors(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary Descriptor-based Backdrive model (simplified interface)
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_backdrive_descriptors
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters (same as sample_discrete_backdrive_descriptors)
        
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    return sample_discrete_backdrive_descriptors(model, n_samples, params)


# ==============================================================================
# GAN Variant Sampling Functions
# ==============================================================================

def sample_binary_gan_cond_fit(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Conditional Fitness GAN
    
    Allows sampling at specific fitness percentiles to control
    exploration vs exploitation trade-off.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_gan_cond_fit
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'target_percentile': target fitness percentile in [0, 1] (default: 1.0 for best)
        - 'selected_fitness': selected population fitness for computing percentiles
        - 'hard_sample': use hard 0/1 samples (default: True)
        - 'threshold': probability threshold for hard sampling (default: 0.5)
        
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    from pateda.learning.discrete_gan import ConditionalBinaryGANGenerator
    
    target_percentile = params.get('target_percentile', 1.0)
    hard_sample = params.get('hard_sample', True)
    threshold = params.get('threshold', 0.5)
    selected_fitness = params.get('selected_fitness', None)
    
    # Reconstruct generator
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_g = model['hidden_dims_g']
    list_act_functs_g = model.get('list_act_functs_g', None)
    list_init_functs_g = model.get('list_init_functs_g', None)
    
    generator = ConditionalBinaryGANGenerator(
        latent_dim, n_vars, condition_dim=1,
        hidden_dims=hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    generator.load_state_dict(model['generator_state'])
    generator.eval()
    
    with torch.no_grad():
        # Sample latent noise
        z = torch.randn(n_samples, latent_dim)
        
        # Create condition tensor (all samples at target percentile)
        condition = torch.full((n_samples, 1), target_percentile)
        
        # Generate
        if hard_sample:
            probs = generator(z, condition, hard_sample=False)
            samples = (probs > threshold).float().numpy()
        else:
            probs = generator(z, condition, hard_sample=True)
            samples = probs.numpy()
    
    return samples.astype(int)


def sample_binary_gan_aux(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Auxiliary GAN with optional surrogate filtering
    
    Can use the fitness prediction head to pre-filter solutions
    and select the most promising ones.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_gan_aux
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'use_surrogate': use fitness head for filtering (default: False)
        - 'surrogate_factor': generate this many times more samples for filtering (default: 3)
        - 'hard_sample': use hard 0/1 samples (default: True)
        - 'threshold': probability threshold for hard sampling (default: 0.5)
        
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    from pateda.learning.discrete_gan import BinaryGANGenerator, BinaryAuxGANDiscriminator
    
    use_surrogate = params.get('use_surrogate', False)
    surrogate_factor = params.get('surrogate_factor', 3)
    hard_sample = params.get('hard_sample', True)
    threshold = params.get('threshold', 0.5)
    
    # Reconstruct generator
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_g = model['hidden_dims_g']
    list_act_functs_g = model.get('list_act_functs_g', None)
    list_init_functs_g = model.get('list_init_functs_g', None)
    
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    generator.load_state_dict(model['generator_state'])
    generator.eval()
    
    if use_surrogate:
        # Reconstruct discriminator with fitness head
        hidden_dims_d = model['hidden_dims_d']
        list_act_functs_d = model.get('list_act_functs_d', None)
        list_init_functs_d = model.get('list_init_functs_d', None)
        dropout = model.get('dropout', 0.5)
        
        discriminator = BinaryAuxGANDiscriminator(
            n_vars, hidden_dims_d,
            list_act_functs=list_act_functs_d,
            list_init_functs=list_init_functs_d,
            dropout=dropout
        )
        discriminator.load_state_dict(model['discriminator_state'])
        discriminator.eval()
        
        with torch.no_grad():
            # Generate more samples than needed
            n_candidates = n_samples * surrogate_factor
            z = torch.randn(n_candidates, latent_dim)
            probs = generator(z, hard_sample=False)
            
            # Predict fitness for all candidates
            _, fitness_pred = discriminator(probs, return_fitness=True)
            fitness_pred = fitness_pred.squeeze().numpy()
            
            # Select top predicted fitness samples
            top_indices = np.argsort(fitness_pred)[-n_samples:]
            
            if hard_sample:
                selected_probs = probs[top_indices]
                samples = (selected_probs > threshold).float().numpy()
            else:
                samples = probs[top_indices].numpy()
    else:
        # Standard sampling without surrogate filtering
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            
            if hard_sample:
                probs = generator(z, hard_sample=False)
                samples = (probs > threshold).float().numpy()
            else:
                probs = generator(z, hard_sample=True)
                samples = probs.numpy()
    
    return samples.astype(int)


def sample_binary_gan_hybrid_vae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Hybrid GAN-VAE (BiGAN) with Encoder
    
    Can use encoder for latent space operations like crossover and interpolation.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_gan_hybrid_vae
    n_samples : int
        Number of samples
    params : dict, optional
        Sampling parameters:
        - 'use_latent_crossover': use latent space crossover (default: False)
        - 'parent_population': parent solutions for crossover (required if use_latent_crossover=True)
        - 'crossover_prob': probability of crossover vs new random sample (default: 0.8)
        - 'hard_sample': use hard 0/1 samples (default: True)
        - 'threshold': probability threshold for hard sampling (default: 0.5)
        
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    from pateda.learning.discrete_gan import BinaryGANGenerator, BinaryGANEncoder
    
    use_latent_crossover = params.get('use_latent_crossover', False)
    parent_population = params.get('parent_population', None)
    crossover_prob = params.get('crossover_prob', 0.8)
    hard_sample = params.get('hard_sample', True)
    threshold = params.get('threshold', 0.5)
    
    # Reconstruct generator
    latent_dim = model['latent_dim']
    n_vars = model['n_vars']
    hidden_dims_g = model['hidden_dims_g']
    list_act_functs_g = model.get('list_act_functs_g', None)
    list_init_functs_g = model.get('list_init_functs_g', None)
    
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    generator.load_state_dict(model['generator_state'])
    generator.eval()
    
    if use_latent_crossover and parent_population is not None:
        # Reconstruct encoder
        hidden_dims_e = model['hidden_dims_e']
        list_act_functs_e = model.get('list_act_functs_e', None)
        list_init_functs_e = model.get('list_init_functs_e', None)
        
        encoder = BinaryGANEncoder(
            n_vars, latent_dim, hidden_dims_e,
            list_act_functs=list_act_functs_e,
            list_init_functs=list_init_functs_e
        )
        encoder.load_state_dict(model['encoder_state'])
        encoder.eval()
        
        with torch.no_grad():
            # Encode parent population to latent space
            parent_tensor = torch.FloatTensor(parent_population)
            parent_z = encoder(parent_tensor)
            
            # Generate latent codes via crossover or random
            latent_codes = []
            for _ in range(n_samples):
                if np.random.rand() < crossover_prob:
                    # Crossover: blend two random parents in latent space
                    idx1, idx2 = np.random.choice(len(parent_z), 2, replace=False)
                    alpha = np.random.rand()
                    z_child = alpha * parent_z[idx1] + (1 - alpha) * parent_z[idx2]
                    latent_codes.append(z_child)
                else:
                    # Random new sample
                    latent_codes.append(torch.randn(latent_dim))
            
            z = torch.stack(latent_codes)
            
            # Generate from latent codes
            if hard_sample:
                probs = generator(z, hard_sample=False)
                samples = (probs > threshold).float().numpy()
            else:
                probs = generator(z, hard_sample=True)
                samples = probs.numpy()
    else:
        # Standard sampling from prior
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            
            if hard_sample:
                probs = generator(z, hard_sample=False)
                samples = (probs > threshold).float().numpy()
            else:
                probs = generator(z, hard_sample=True)
                samples = probs.numpy()
    
    return samples.astype(int)


# ==============================================================================
# Enhanced VAE Sampling Functions (Based on DISCRETE_VAE_ANALYSIS.md)
# ==============================================================================

def sample_binary_bavae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Beta-Annealed VAE (BA-VAE)
    
    BA-VAE uses standard VAE sampling. The cyclical beta annealing is only
    applied during training to prevent posterior collapse.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_bavae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (same as sample_binary_vae)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    # BA-VAE uses standard VAE sampling
    return sample_binary_vae(model, n_samples, params)


def sample_binary_aavae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Adaptive-Architecture VAE (AA-VAE)
    
    AA-VAE uses standard VAE sampling. The adaptive architecture and dropout
    are only applied during training to prevent overfitting.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_aavae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (same as sample_binary_vae)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    # AA-VAE uses standard VAE sampling (dropout is disabled during inference)
    return sample_binary_vae(model, n_samples, params)


def sample_binary_fwvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Fitness-Weighted VAE (FW-VAE)
    
    FW-VAE uses standard VAE sampling. The fitness weighting is only applied
    during training to bias the latent space toward high-fitness regions.
    
    Parameters
    ----------
    model : dict
        Model dictionary from learn_binary_fwvae
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (same as sample_binary_vae)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    # FW-VAE uses standard VAE sampling
    return sample_binary_vae(model, n_samples, params)


def sample_binary_gsvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Greedy-Sampling VAE (GS-VAE) - Uses deterministic sampling
    
    This variant uses deterministic (greedy) sampling instead of stochastic
    Bernoulli sampling to reduce variance and improve exploitation.
    
    Parameters
    ----------
    model : dict
        Model dictionary from any binary VAE variant
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters (extends sample_binary_vae)
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    # Force deterministic sampling
    modified_params = params.copy()
    modified_params['deterministic'] = True
    
    return sample_binary_vae(model, n_samples, modified_params)


def sample_binary_hsvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Hybrid-Sampling VAE (HS-VAE) - Combines deterministic and stochastic
    
    This variant combines greedy (deterministic) sampling for exploitation with
    stochastic sampling for exploration.
    
    Parameters
    ----------
    model : dict
        Model dictionary from any binary VAE variant
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'exploration_ratio': fraction of samples to generate stochastically (default: 0.3)
        - Other parameters from sample_binary_vae
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    exploration_ratio = params.get('exploration_ratio', 0.3)
    
    # Calculate split
    n_exploit = int(n_samples * (1 - exploration_ratio))
    n_explore = n_samples - n_exploit
    
    # Exploitation: deterministic sampling
    exploit_params = params.copy()
    exploit_params['deterministic'] = True
    exploit_samples = sample_binary_vae(model, n_exploit, exploit_params)
    
    # Exploration: stochastic sampling
    explore_params = params.copy()
    explore_params['deterministic'] = False
    explore_params['temperature'] = params.get('temperature', 1.0)  # Higher temp for exploration
    explore_samples = sample_binary_vae(model, n_explore, explore_params)
    
    # Combine samples
    return np.vstack([exploit_samples, explore_samples])


def sample_binary_tcvae(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Temperature-Controlled VAE (TC-VAE)
    
    This variant uses adaptive temperature for exploration-exploitation balance.
    Temperature decreases over generations to gradually shift from exploration
    to exploitation.
    
    Parameters
    ----------
    model : dict
        Model dictionary from any binary VAE variant
    n_samples : int
        Number of samples to generate
    params : dict, optional
        Sampling parameters:
        - 'generation': current generation number (default: 0)
        - 'max_generations': total generations (default: 50)
        - 'temp_start': starting temperature (default: 1.0)
        - 'temp_end': ending temperature (default: 0.1)
        - Other parameters from sample_binary_vae
    
    Returns
    -------
    samples : np.ndarray
        Binary samples of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}
    
    # Get generation info
    generation = params.get('generation', 0)
    max_generations = params.get('max_generations', 50)
    temp_start = params.get('temp_start', 1.0)
    temp_end = params.get('temp_end', 0.1)
    
    # Compute adaptive temperature (linear annealing)
    progress = min(1.0, generation / max_generations)
    temperature = temp_start - (temp_start - temp_end) * progress
    
    # Sample with adaptive temperature
    modified_params = params.copy()
    modified_params['temperature'] = temperature
    
    return sample_binary_vae(model, n_samples, modified_params)
