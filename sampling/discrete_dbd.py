"""
Sampling from Discrete DbD Models

==============================================================================
OVERVIEW
==============================================================================

This module provides sampling functions for Discrete Diffusion-by-Deblending
(DbD) models. Sampling is done via iterative denoising, starting from a
random distribution and progressively denoising toward the learned distribution.

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from pateda.learning.discrete_dbd import BinaryDeblendingNet, CategoricalDeblendingNet


def sample_binary_dbd(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD model via iterative denoising

    Starting from random binary samples (alpha=0), progressively
    denoise toward the learned distribution (alpha=1).

    Args:
        model: Model dictionary from learn_binary_dbd
        n_samples: Number of samples to generate
        params: Sampling parameters:
            - 'n_steps': number of denoising steps (default: 20)
            - 'temperature': softmax temperature (default: 1.0)
            - 'init_method': 'random' or 'uniform' (default: 'random')

    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    # CRITICAL FIX: More denoising steps for smoother transitions
    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)
    init_method = params.get('init_method', 'random')

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize samples
        if init_method == 'uniform':
            x = torch.ones(n_samples, n_vars) * 0.5
        else:
            x = torch.rand(n_samples, n_vars)

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]  # From 0 to 1

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # CRITICAL FIX: Network now predicts DIFFERENCE
            # Use it as velocity to update x
            predicted_diff = network(x, alpha)

            # Update: x_new = x + (alpha_t+1 - alpha_t) * diff
            delta_alpha = alpha_val - alpha_current
            x = x + delta_alpha * predicted_diff

            # Clip to [0, 1] range and sample
            x = torch.clamp(x, 0.0, 1.0)
            probs = x / temperature if temperature != 1.0 else x
            probs = torch.clamp(probs, 0.0, 1.0)
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)


def sample_categorical_dbd(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Categorical DbD model via iterative denoising

    Args:
        model: Model dictionary from learn_categorical_dbd
        n_samples: Number of samples
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 10)
            - 'temperature': softmax temperature (default: 1.0)

    Returns:
        samples: Categorical samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 10)
    temperature = params.get('temperature', 1.0)

    # Reconstruct network
    n_vars = model['n_vars']
    cardinality = model['cardinality']
    hidden_dims = model['hidden_dims']
    total_categories = int(np.sum(cardinality))

    network = CategoricalDeblendingNet(n_vars, cardinality, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    with torch.no_grad():
        # Initialize with uniform distribution (one-hot)
        x_onehot = torch.zeros(n_samples, total_categories)
        for i in range(n_vars):
            card = int(cardinality[i])
            # Random initialization
            init_vals = torch.randint(0, card, (n_samples,))
            for j in range(n_samples):
                x_onehot[j, cum_card[i] + init_vals[j]] = 1.0

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for alpha_val in alphas:
            alpha = torch.full((n_samples,), alpha_val)

            # Predict target distribution
            logits = network(x_onehot, alpha)

            # Sample new x_onehot
            x_onehot_new = torch.zeros_like(x_onehot)
            for i in range(n_vars):
                start_idx = cum_card[i]
                end_idx = cum_card[i + 1]
                var_logits = logits[:, start_idx:end_idx]

                # Sample with temperature
                probs = F.softmax(var_logits / temperature, dim=-1)
                samples = torch.multinomial(probs, 1).squeeze(1)

                # One-hot encode
                for j in range(n_samples):
                    x_onehot_new[j, start_idx + samples[j]] = 1.0

            x_onehot = x_onehot_new

        # Convert one-hot back to discrete values
        samples = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            start_idx = cum_card[i]
            end_idx = cum_card[i + 1]
            var_onehot = x_onehot[:, start_idx:end_idx]
            samples[:, i] = torch.argmax(var_onehot, dim=1).numpy()

    return samples


def sample_binary_dbd_from_seeds(
    model: Dict[str, Any],
    seed_samples: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD starting from seed samples

    Instead of random initialization, start from given seed samples
    and refine them using the learned denoising model.

    Args:
        model: Model dictionary
        seed_samples: Initial samples [n_samples, n_vars]
        params: Sampling parameters:
            - 'n_steps': refinement steps (default: 5)
            - 'temperature': sampling temperature (default: 1.0)
            - 'start_alpha': starting alpha value (default: 0.5)

    Returns:
        samples: Refined samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 5)
    temperature = params.get('temperature', 1.0)
    start_alpha = params.get('start_alpha', 0.5)

    n_samples = seed_samples.shape[0]

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        x = torch.FloatTensor(seed_samples)

        # Denoise from start_alpha to 1.0
        alphas = np.linspace(start_alpha, 1, n_steps)

        for alpha_val in alphas:
            alpha = torch.full((n_samples,), alpha_val)

            # Predict
            logits = network(x, alpha)
            probs = torch.sigmoid(logits / temperature)

            # Sample
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)


# ==============================================================================
# Sampling Functions for DbD Variants
# ==============================================================================

def sample_binary_dbd_cs(
    model: Dict[str, Any],
    n_samples: int,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-CS model (Current to Selected)

    DbD-CS starts from selected population samples and refines them
    toward higher fitness solutions.

    Args:
        model: Model dictionary from learn_binary_dbd_cs
        n_samples: Number of samples to generate
        selected_pop: Selected population for initialization
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 20)
            - 'temperature': softmax temperature (default: 1.0)

    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from selected population
        sel_indices = np.random.randint(0, len(selected_pop), size=n_samples)
        x = torch.FloatTensor(selected_pop[sel_indices])

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            predicted_diff = network(x, alpha)

            # Update using difference
            delta_alpha = alpha_val - alpha_current
            x = x + delta_alpha * predicted_diff

            # Clip and sample
            x = torch.clamp(x, 0.0, 1.0)
            probs = x / temperature if temperature != 1.0 else x
            probs = torch.clamp(probs, 0.0, 1.0)
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)


def sample_binary_dbd_cd(
    model: Dict[str, Any],
    n_samples: int,
    current_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-CD model (Current to Closest in selected - Distance)

    DbD-CD starts from current population samples and refines them.

    Args:
        model: Model dictionary from learn_binary_dbd_cd
        n_samples: Number of samples to generate
        current_pop: Current population for initialization
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 20)
            - 'temperature': softmax temperature (default: 1.0)

    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from current population
        cur_indices = np.random.randint(0, len(current_pop), size=n_samples)
        x = torch.FloatTensor(current_pop[cur_indices])

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            predicted_diff = network(x, alpha)

            # Update using difference
            delta_alpha = alpha_val - alpha_current
            x = x + delta_alpha * predicted_diff

            # Clip and sample
            x = torch.clamp(x, 0.0, 1.0)
            probs = x / temperature if temperature != 1.0 else x
            probs = torch.clamp(probs, 0.0, 1.0)
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)


def sample_binary_dbd_uc(
    model: Dict[str, Any],
    n_samples: int,
    current_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-UC model (Univariate to Current)

    DbD-UC starts from univariate marginal samples and learns to
    recover variable dependencies by denoising toward current population.

    Args:
        model: Model dictionary from learn_binary_dbd_uc
        n_samples: Number of samples to generate
        current_pop: Current population (not used, included for API consistency)
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 20)
            - 'temperature': softmax temperature (default: 1.0)

    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']
    marginal_probs = model['marginal_probs']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from univariate marginal distribution
        from pateda.learning.discrete_dbd import sample_from_univariate_binary
        x_init = sample_from_univariate_binary(marginal_probs, n_samples)
        x = torch.FloatTensor(x_init)

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            predicted_diff = network(x, alpha)

            # Update using difference
            delta_alpha = alpha_val - alpha_current
            x = x + delta_alpha * predicted_diff

            # Clip and sample
            x = torch.clamp(x, 0.0, 1.0)
            probs = x / temperature if temperature != 1.0 else x
            probs = torch.clamp(probs, 0.0, 1.0)
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)


def sample_binary_dbd_us(
    model: Dict[str, Any],
    n_samples: int,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-US model (Univariate to Selected)

    DbD-US starts from univariate marginal samples and learns to
    both recover variable dependencies and improve fitness by
    denoising toward selected population.

    Args:
        model: Model dictionary from learn_binary_dbd_us
        n_samples: Number of samples to generate
        selected_pop: Selected population (not used, included for API consistency)
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 20)
            - 'temperature': softmax temperature (default: 1.0)

    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}

    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)

    # Reconstruct network
    n_vars = model['n_vars']
    hidden_dims = model['hidden_dims']
    marginal_probs = model['marginal_probs']

    network = BinaryDeblendingNet(n_vars, hidden_dims)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from univariate marginal distribution
        from pateda.learning.discrete_dbd import sample_from_univariate_binary
        x_init = sample_from_univariate_binary(marginal_probs, n_samples)
        x = torch.FloatTensor(x_init)

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            predicted_diff = network(x, alpha)

            # Update using difference
            delta_alpha = alpha_val - alpha_current
            x = x + delta_alpha * predicted_diff

            # Clip and sample
            x = torch.clamp(x, 0.0, 1.0)
            probs = x / temperature if temperature != 1.0 else x
            probs = torch.clamp(probs, 0.0, 1.0)
            x = torch.bernoulli(probs)

    return x.numpy().astype(int)
