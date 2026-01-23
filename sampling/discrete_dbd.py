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
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F

from pateda.learning.discrete_dbd import BinaryDeblendingNet, CategoricalDeblendingNet


def sample_binary_dbd(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None,
    target_fitness: np.ndarray = None
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
        target_fitness: Optional target fitness values for conditional generation [n_samples]

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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
    network.load_state_dict(model['network_state'])
    network.eval()
    
    # Prepare fitness tensor if needed
    fitness_tensor = None
    if use_fitness_guidance:
        if target_fitness is None:
            # If no target fitness provided, use high fitness value for all samples
            # to encourage generation of high-quality solutions
            target_fitness = np.ones(n_samples) * 1.0  # Maximum normalized fitness
        fitness_tensor = torch.FloatTensor(target_fitness.reshape(-1, 1))

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
            if use_fitness_guidance:
                predicted_diff = network(x, alpha, fitness_tensor)
            else:
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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from selected population
        sel_indices = np.random.randint(0, len(selected_pop), size=n_samples)
        x = torch.FloatTensor(selected_pop[sel_indices])

        # Prepare fitness tensor if needed (use high fitness values for all samples during sampling)
        fitness_tensor = None
        if use_fitness_guidance:
            fitness_tensor = torch.ones(n_samples, 1) * 1.0  # Maximum normalized fitness

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            if use_fitness_guidance:
                predicted_diff = network(x, alpha, fitness_tensor)
            else:
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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from current population
        cur_indices = np.random.randint(0, len(current_pop), size=n_samples)
        x = torch.FloatTensor(current_pop[cur_indices])

        # Prepare fitness tensor if needed
        fitness_tensor = None
        if use_fitness_guidance:
            fitness_tensor = torch.ones(n_samples, 1) * 1.0  # Maximum normalized fitness

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            if use_fitness_guidance:
                predicted_diff = network(x, alpha, fitness_tensor)
            else:
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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from univariate marginal distribution
        from pateda.learning.discrete_dbd import sample_from_univariate_binary
        x_init = sample_from_univariate_binary(marginal_probs, n_samples)
        x = torch.FloatTensor(x_init)

        # Prepare fitness tensor if needed
        fitness_tensor = None
        if use_fitness_guidance:
            fitness_tensor = torch.ones(n_samples, 1) * 1.0  # Maximum normalized fitness

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            if use_fitness_guidance:
                predicted_diff = network(x, alpha, fitness_tensor)
            else:
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
    use_fitness_guidance = model.get('use_fitness_guidance', False)
    list_act_functs = model.get('list_act_functs', None)
    list_init_functs = model.get('list_init_functs', None)

    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance,
                                 list_act_functs, list_init_functs)
    network.load_state_dict(model['network_state'])
    network.eval()

    with torch.no_grad():
        # Initialize from univariate marginal distribution
        from pateda.learning.discrete_dbd import sample_from_univariate_binary
        x_init = sample_from_univariate_binary(marginal_probs, n_samples)
        x = torch.FloatTensor(x_init)

        # Prepare fitness tensor if needed
        fitness_tensor = None
        if use_fitness_guidance:
            fitness_tensor = torch.ones(n_samples, 1) * 1.0  # Maximum normalized fitness

        # Iterative denoising
        alphas = np.linspace(0, 1, n_steps + 1)[1:]

        for i, alpha_val in enumerate(alphas):
            alpha_current = alphas[i - 1] if i > 0 else 0.0
            alpha = torch.full((n_samples,), alpha_val)

            # Network predicts difference (velocity)
            if use_fitness_guidance:
                predicted_diff = network(x, alpha, fitness_tensor)
            else:
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


# ==============================================================================
# Sampling Functions for DbD-T Variants
# ==============================================================================

# Blending weights for combining denoised probability with conditional probability
# These control how much influence the denoising process has vs the learned conditional structure
DENOISED_WEIGHT = 0.7  # Weight for denoised continuous probability
CONDITIONAL_WEIGHT = 0.3  # Weight for conditional probability from learned structure


def sample_from_continuous_probabilities(
    continuous_values: np.ndarray,
    conditional_probs: list,
    k: int,
    prob_min: float = 0.01,
    prob_max: float = 0.99,
    denoised_weight: float = DENOISED_WEIGHT,
    conditional_weight: float = CONDITIONAL_WEIGHT,
    parent_structure: Optional[Dict[int, List[int]]] = None
) -> np.ndarray:
    """
    Sample binary values from continuous probability values using conditional probabilities
    
    The continuous values represent probabilities and should be bounded to ensure diversity.
    
    Parameters:
    -----------
    continuous_values : np.ndarray
        Continuous values [n_samples, n_vars] representing probabilities
    conditional_probs : list
        List of conditional probability tables for each variable
    k : int
        Order of Markov chain used
    prob_min : float
        Minimum probability bound (default: 0.01)
    prob_max : float
        Maximum probability bound (default: 0.99)
    denoised_weight : float
        Weight for denoised probability (default: 0.7)
    conditional_weight : float
        Weight for conditional probability (default: 0.3)
    parent_structure : dict, optional
        Dictionary mapping variable index to list of parent variable indices.
        If None, uses previous k variables in given order.
        
    Returns:
    --------
    np.ndarray
        Binary samples [n_samples, n_vars]
    """
    n_samples, n_vars = continuous_values.shape
    binary_samples = np.zeros((n_samples, n_vars), dtype=int)
    
    # Bound probabilities to ensure diversity
    bounded_probs = np.clip(continuous_values, prob_min, prob_max)
    
    for var in range(n_vars):
        cpd = conditional_probs[var]
        
        # Determine parent variables for this variable
        if parent_structure is not None:
            # Use MI-based parent structure
            parent_vars = parent_structure.get(var, [])
        else:
            # Use original order: previous k variables
            if var < k:
                parent_vars = []
            else:
                n_parents = min(k, var)
                parent_vars = list(range(var - n_parents, var))
        
        # If no parents, use marginal probabilities
        if len(parent_vars) == 0:
            # cpd is stored as [P(X=0), P(X=1)]
            # Blend with the denoised continuous value
            for sample_idx in range(n_samples):
                # Get marginal P(X_i = 1) from learned distribution
                # Convert to float to prevent numpy array indexing errors
                marginal_prob_1 = float(cpd[1])
                
                # Get denoised probability
                denoised_prob_1 = bounded_probs[sample_idx, var]
                
                # Blend the two probabilities
                final_prob = denoised_weight * denoised_prob_1 + conditional_weight * marginal_prob_1
                final_prob = np.clip(final_prob, prob_min, prob_max)
                
                binary_samples[sample_idx, var] = 1 if np.random.rand() < final_prob else 0
        else:
            # Use conditional probabilities
            for sample_idx in range(n_samples):
                # Calculate parent configuration index
                config_idx = 0
                for i, parent_var in enumerate(parent_vars):
                    config_idx += int(binary_samples[sample_idx, parent_var]) * (2 ** i)
                
                # Get conditional probability P(X_i = 1 | parents)
                # Convert to float to prevent numpy array indexing errors
                prob_1_given_parents = float(cpd[config_idx, 1])
                
                # Use the continuous value as a modulation factor
                # This allows the denoising process to influence the sampling
                modulated_prob = bounded_probs[sample_idx, var]
                
                # Blend between conditional probability and modulated probability
                final_prob = denoised_weight * modulated_prob + conditional_weight * prob_1_given_parents
                final_prob = np.clip(final_prob, prob_min, prob_max)
                
                binary_samples[sample_idx, var] = 1 if np.random.rand() < final_prob else 0
    
    return binary_samples


def sample_binary_dbd_cs_t(
    model: Dict[str, Any],
    n_samples: int,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-CS-T model (Current to Selected with Transformation)
    
    Args:
        model: Model dictionary from learn_binary_dbd_cs_t
        n_samples: Number of samples to generate
        selected_pop: Selected population for initialization
        params: Sampling parameters:
            - 'n_steps': denoising steps (default: 20)
            - 'temperature': sampling temperature (default: 1.0)
            - 'prob_min': minimum probability bound (default: 0.01)
            - 'prob_max': maximum probability bound (default: 0.99)
            - 'num_iterations': number of deblending iterations (default: 10)
    
    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}
    
    n_steps = params.get('n_steps', 20)
    temperature = params.get('temperature', 1.0)
    prob_min = params.get('prob_min', 0.01)
    prob_max = params.get('prob_max', 0.99)
    num_iterations = params.get('num_iterations', 10)
    
    k = model['k']
    conditional_probs = model['conditional_probs']
    parent_structure = model.get('parent_structure', None)
    
    # Transform selected population to continuous using conditional probabilities
    # This provides the p0 source distribution for continuous DbD sampling
    from pateda.learning.discrete_dbd import transform_binary_to_continuous
    p0_continuous = transform_binary_to_continuous(selected_pop, conditional_probs, k, parent_structure)
    
    # Use continuous DbD sampling to get continuous values
    from pateda.sampling.dbd import sample_dbd
    
    # Sample continuous values - note: sample_dbd expects (model, p0, n_samples, bounds, params, rng)
    continuous_samples = sample_dbd(model, p0_continuous, n_samples, bounds=None, 
                                   params={'num_iterations': num_iterations}, rng=None)
    
    # Clip to [0, 1] range (they should already be, but ensure)
    continuous_samples = np.clip(continuous_samples, 0.0, 1.0)
    
    # Convert continuous values to binary using conditional probabilities
    binary_samples = sample_from_continuous_probabilities(
        continuous_samples, conditional_probs, k, prob_min, prob_max, 
        parent_structure=parent_structure
    )
    
    return binary_samples


def sample_binary_dbd_cd_t(
    model: Dict[str, Any],
    n_samples: int,
    current_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-CD-T model (Current to Closest with Transformation)
    
    Args:
        model: Model dictionary from learn_binary_dbd_cd_t
        n_samples: Number of samples to generate
        current_pop: Current population for initialization
        params: Sampling parameters:
            - 'num_iterations': number of deblending iterations (default: 10)
            - 'prob_min': minimum probability bound (default: 0.01)
            - 'prob_max': maximum probability bound (default: 0.99)
    
    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}
    
    num_iterations = params.get('num_iterations', 10)
    prob_min = params.get('prob_min', 0.01)
    prob_max = params.get('prob_max', 0.99)
    
    k = model['k']
    conditional_probs = model['conditional_probs']
    parent_structure = model.get('parent_structure', None)
    
    # Transform current population to continuous using conditional probabilities
    from pateda.learning.discrete_dbd import transform_binary_to_continuous
    p0_continuous = transform_binary_to_continuous(current_pop, conditional_probs, k, parent_structure)
    
    # Use continuous DbD sampling
    from pateda.sampling.dbd import sample_dbd
    continuous_samples = sample_dbd(model, p0_continuous, n_samples, bounds=None,
                                   params={'num_iterations': num_iterations}, rng=None)
    continuous_samples = np.clip(continuous_samples, 0.0, 1.0)
    
    # Convert to binary
    binary_samples = sample_from_continuous_probabilities(
        continuous_samples, conditional_probs, k, prob_min, prob_max,
        parent_structure=parent_structure
    )
    
    return binary_samples


def sample_binary_dbd_uc_t(
    model: Dict[str, Any],
    n_samples: int,
    current_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-UC-T model (Univariate to Current with Transformation)
    
    Args:
        model: Model dictionary from learn_binary_dbd_uc_t
        n_samples: Number of samples to generate
        current_pop: Current population (not used, included for API consistency)
        params: Sampling parameters:
            - 'num_iterations': number of deblending iterations (default: 10)
            - 'prob_min': minimum probability bound (default: 0.01)
            - 'prob_max': maximum probability bound (default: 0.99)
    
    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}
    
    num_iterations = params.get('num_iterations', 10)
    prob_min = params.get('prob_min', 0.01)
    prob_max = params.get('prob_max', 0.99)
    
    k = model['k']
    conditional_probs = model['conditional_probs']
    marginal_probs = model['marginal_probs']
    
    # Sample from univariate distribution and transform to continuous
    from pateda.learning.discrete_dbd import sample_from_univariate_binary, compute_conditional_probabilities, transform_binary_to_continuous
    
    # Generate more samples for p0 to have a diverse source distribution
    n_p0_samples = max(n_samples, 500)
    p0_binary = sample_from_univariate_binary(marginal_probs, n_p0_samples)
    
    # Compute conditional probabilities for the p0 samples
    alpha_smooth = model.get('alpha_smooth', 0.1)
    conditional_probs_p0 = compute_conditional_probabilities(p0_binary.astype(int), k, alpha_smooth)
    
    # Transform to continuous (no parent_structure for UC variant)
    p0_continuous = transform_binary_to_continuous(p0_binary.astype(int), conditional_probs_p0, k, parent_structure=None)
    
    # Use continuous DbD sampling
    from pateda.sampling.dbd import sample_dbd
    continuous_samples = sample_dbd(model, p0_continuous, n_samples, bounds=None,
                                   params={'num_iterations': num_iterations}, rng=None)
    continuous_samples = np.clip(continuous_samples, 0.0, 1.0)
    
    # Convert to binary (no parent_structure for UC variant)
    binary_samples = sample_from_continuous_probabilities(
        continuous_samples, conditional_probs, k, prob_min, prob_max,
        parent_structure=None
    )
    
    return binary_samples


def sample_binary_dbd_us_t(
    model: Dict[str, Any],
    n_samples: int,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from Binary DbD-US-T model (Univariate to Selected with Transformation)
    
    Args:
        model: Model dictionary from learn_binary_dbd_us_t
        n_samples: Number of samples to generate
        selected_pop: Selected population (not used, included for API consistency)
        params: Sampling parameters:
            - 'num_iterations': number of deblending iterations (default: 10)
            - 'prob_min': minimum probability bound (default: 0.01)
            - 'prob_max': maximum probability bound (default: 0.99)
    
    Returns:
        samples: Binary samples [n_samples, n_vars]
    """
    if params is None:
        params = {}
    
    num_iterations = params.get('num_iterations', 10)
    prob_min = params.get('prob_min', 0.01)
    prob_max = params.get('prob_max', 0.99)
    
    k = model['k']
    conditional_probs = model['conditional_probs']
    marginal_probs = model['marginal_probs']
    
    # Sample from univariate distribution and transform to continuous
    from pateda.learning.discrete_dbd import sample_from_univariate_binary, compute_conditional_probabilities, transform_binary_to_continuous
    
    # Generate more samples for p0 to have a diverse source distribution
    n_p0_samples = max(n_samples, 500)
    p0_binary = sample_from_univariate_binary(marginal_probs, n_p0_samples)
    
    # Compute conditional probabilities for the p0 samples
    alpha_smooth = model.get('alpha_smooth', 0.1)
    conditional_probs_p0 = compute_conditional_probabilities(p0_binary.astype(int), k, alpha_smooth)
    
    # Transform to continuous (no parent_structure for US variant)
    p0_continuous = transform_binary_to_continuous(p0_binary.astype(int), conditional_probs_p0, k, parent_structure=None)
    
    # Use continuous DbD sampling
    from pateda.sampling.dbd import sample_dbd
    continuous_samples = sample_dbd(model, p0_continuous, n_samples, bounds=None,
                                   params={'num_iterations': num_iterations}, rng=None)
    continuous_samples = np.clip(continuous_samples, 0.0, 1.0)
    
    # Convert to binary (no parent_structure for US variant)
    binary_samples = sample_from_continuous_probabilities(
        continuous_samples, conditional_probs, k, prob_min, prob_max,
        parent_structure=None
    )
    
    return binary_samples
