"""
Backdrive-EDA sampling functions

This module implements sampling for Backdrive-EDA using network inversion (backdrive).
Given a trained MLP that predicts fitness from solutions, sampling is performed by:
1. Freezing the network weights
2. Initializing input solutions (randomly or from best solutions)
3. Clamping the output to a high fitness value
4. Using backpropagation to modify the inputs to achieve the target output

This is fundamentally different from traditional generative models - instead of
sampling from a learned distribution, we use gradient descent on the inputs to
"invert" the regression model.

References:
- Garciarena et al. (2020). "Envisioning the Benefits of Back-Drive in Evolutionary Algorithms"
- Baluja (2017). "Deep Learning for Explicitly Modeling Optimization Landscapes"
- Linden & Kindermann (1989). "Inversion of multilayer nets"
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

from pateda.core.models import NeuralNetworkModel


def sample_backdrive(
    n_vars: int,
    model: NeuralNetworkModel,
    cardinality: np.ndarray,
    current_population: np.ndarray,
    current_fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Sample new solutions using network inversion (backdrive).

    The sampling process works by:
    1. Initializing candidate solutions (randomly or from best solutions)
    2. Freezing the network weights
    3. Setting target output to high fitness value
    4. Using gradient descent to modify inputs to achieve target output

    Parameters
    ----------
    n_vars : int
        Number of variables
    model : NeuralNetworkModel
        Trained Backdrive model from learn_backdrive
    cardinality : np.ndarray
        Variable bounds/cardinalities
    current_population : np.ndarray
        Current population, shape (pop_size, n_vars)
    current_fitness : np.ndarray
        Current fitness values, shape (pop_size, 1) or (pop_size,)
    params : dict, optional
        Sampling parameters:
        - 'n_samples': number of samples to generate, default = len(current_population)
        - 'target_fitness': target fitness percentile (0-100), default = 100 (best)
        - 'backdrive_iterations': number of gradient steps, default = 500
        - 'backdrive_lr': learning rate for backdrive, default = 0.01
        - 'init_method': initialization method, default = 'perturb_best'
                       Options: 'random', 'perturb_best', 'perturb_selected'
        - 'init_noise': noise level for perturbation, default = 0.1
        - 'use_surrogate': oversample and filter using model predictions, default = False
        - 'oversample_factor': factor for oversampling, default = 5
        - 'clip_gradients': clip gradient magnitudes, default = True
        - 'gradient_clip': gradient clipping value, default = 1.0

    Returns
    -------
    np.ndarray
        Generated population, shape (n_samples, n_vars)

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.sampling.backdrive import sample_backdrive
    >>> from pateda.learning.backdrive import learn_backdrive
    >>> # Train model
    >>> n_vars = 10
    >>> pop = np.random.randn(100, n_vars)
    >>> fit = np.sum(pop**2, axis=1, keepdims=True)
    >>> cardinality = np.array([[-5]*n_vars, [5]*n_vars])
    >>> model = learn_backdrive(0, n_vars, cardinality, pop, fit)
    >>> # Sample new solutions
    >>> new_pop = sample_backdrive(n_vars, model, cardinality, pop, fit)
    """
    # Import PyTorch
    try:
        import torch
        import torch.optim as optim
    except ImportError:
        raise ImportError(
            "PyTorch is required for Backdrive-EDA. "
            "Install it with: pip install torch"
        )

    # Set default parameters
    default_params = {
        'n_samples': len(current_population),
        'target_fitness': 100,  # Target best fitness
        'backdrive_iterations': 500,
        'backdrive_lr': 0.01,
        'init_method': 'perturb_best',
        'init_noise': 0.1,
        'use_surrogate': False,
        'oversample_factor': 5,
        'clip_gradients': True,
        'gradient_clip': 1.0,
    }

    if params is not None:
        default_params.update(params)
    params = default_params

    # Extract model components
    network = model.parameters
    metadata = model.metadata

    # Get normalization parameters
    lower_bounds = metadata['lower_bounds']
    upper_bounds = metadata['upper_bounds']
    ranges = metadata['ranges']

    # Determine target fitness value (normalized)
    target_percentile = params['target_fitness']
    if target_percentile == 100:
        # Target the best fitness seen
        target_fitness_norm = 1.0
    else:
        # Target a percentile of the fitness range
        target_fitness_norm = target_percentile / 100.0

    # Determine number of samples to generate
    n_samples_base = params['n_samples']
    if params['use_surrogate']:
        # Oversample and filter using model predictions
        n_samples = n_samples_base * params['oversample_factor']
    else:
        n_samples = n_samples_base

    # Initialize candidate solutions
    if params['init_method'] == 'random':
        # Random initialization in normalized space
        init_solutions_norm = torch.rand(n_samples, n_vars, requires_grad=True)

    elif params['init_method'] == 'perturb_best':
        # Initialize from best solution with perturbations
        best_idx = np.argmax(current_fitness.flatten())
        best_solution = current_population[best_idx]

        # Normalize best solution
        best_solution_norm = (best_solution - lower_bounds) / ranges

        # Create perturbations
        noise = torch.randn(n_samples, n_vars) * params['init_noise']
        init_solutions_norm = torch.FloatTensor(best_solution_norm).unsqueeze(0) + noise
        init_solutions_norm = torch.clamp(init_solutions_norm, 0, 1)
        init_solutions_norm.requires_grad = True

    elif params['init_method'] == 'perturb_selected':
        # Initialize from top solutions with perturbations
        n_select = min(n_samples, len(current_population) // 2)
        top_indices = np.argsort(current_fitness.flatten())[-n_select:]

        selected_solutions = current_population[top_indices]
        selected_norm = (selected_solutions - lower_bounds) / ranges

        # Replicate and perturb
        n_copies = n_samples // n_select + 1
        repeated = np.tile(selected_norm, (n_copies, 1))[:n_samples]

        noise = torch.randn(n_samples, n_vars) * params['init_noise']
        init_solutions_norm = torch.FloatTensor(repeated) + noise
        init_solutions_norm = torch.clamp(init_solutions_norm, 0, 1)
        init_solutions_norm.requires_grad = True

    else:
        raise ValueError(f"Unknown init_method: {params['init_method']}")

    # Setup optimizer for backdrive (modifies inputs, not weights)
    optimizer = optim.Adam([init_solutions_norm], lr=params['backdrive_lr'])

    # Target output (high fitness)
    target_output = torch.FloatTensor([[target_fitness_norm]] * n_samples)

    # Freeze network weights
    network.eval()
    for param in network.parameters():
        param.requires_grad = False

    # Backdrive loop: modify inputs to achieve target output
    criterion = torch.nn.MSELoss()

    for iteration in range(params['backdrive_iterations']):
        optimizer.zero_grad()

        # Forward pass
        predictions = network(init_solutions_norm)

        # Compute loss (difference between predicted and target fitness)
        loss = criterion(predictions, target_output)

        # Backward pass (gradients flow to inputs, not weights)
        loss.backward()

        # Optionally clip gradients
        if params['clip_gradients']:
            torch.nn.utils.clip_grad_value_([init_solutions_norm], params['gradient_clip'])

        # Update inputs
        optimizer.step()

        # Clamp inputs to [0, 1]
        with torch.no_grad():
            init_solutions_norm.clamp_(0, 1)

    # Denormalize solutions
    with torch.no_grad():
        solutions_norm = init_solutions_norm.cpu().numpy()

    solutions = solutions_norm * ranges + lower_bounds

    # Clip to bounds
    solutions = np.clip(solutions, lower_bounds, upper_bounds)

    # If using surrogate filtering, select best predicted samples
    if params['use_surrogate']:
        # Get predictions from model
        with torch.no_grad():
            predictions_norm = network(torch.FloatTensor(solutions_norm)).numpy().flatten()

        # Denormalize predictions
        predictions = (predictions_norm * metadata['fitness_range']) + metadata['fitness_min']

        # Select top samples
        top_indices = np.argsort(predictions)[-n_samples_base:]
        solutions = solutions[top_indices]

    return solutions


def sample_backdrive_adaptive(
    n_vars: int,
    model: NeuralNetworkModel,
    cardinality: np.ndarray,
    current_population: np.ndarray,
    current_fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Adaptive backdrive sampling with multiple target fitness levels.

    This variant samples solutions targeting different fitness levels to maintain
    diversity while still focusing on high-fitness regions.

    Parameters
    ----------
    n_vars : int
        Number of variables
    model : NeuralNetworkModel
        Trained Backdrive model
    cardinality : np.ndarray
        Variable bounds/cardinalities
    current_population : np.ndarray
        Current population
    current_fitness : np.ndarray
        Current fitness values
    params : dict, optional
        Sampling parameters (same as sample_backdrive, plus):
        - 'target_levels': list of target fitness percentiles, default = [100, 90, 80]
        - 'level_fractions': fraction of samples per level, default = [0.5, 0.3, 0.2]

    Returns
    -------
    np.ndarray
        Generated population with diverse fitness targets

    Notes
    -----
    This method helps maintain exploration by targeting multiple fitness levels
    rather than only the absolute best, which can help avoid premature convergence.
    """
    default_params = {
        'n_samples': len(current_population),
        'target_levels': [100, 90, 80],
        'level_fractions': [0.5, 0.3, 0.2],
    }

    if params is not None:
        default_params.update(params)
    params = default_params

    n_samples_total = params['n_samples']
    target_levels = params['target_levels']
    level_fractions = params['level_fractions']

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
            level_params['target_fitness'] = target_level

            # Sample
            solutions = sample_backdrive(
                n_vars, model, cardinality,
                current_population, current_fitness,
                level_params
            )

            all_solutions.append(solutions)

    # Concatenate all solutions
    all_solutions = np.vstack(all_solutions)

    # If we have more or fewer samples than needed due to rounding, adjust
    if len(all_solutions) > n_samples_total:
        all_solutions = all_solutions[:n_samples_total]
    elif len(all_solutions) < n_samples_total:
        # Fill remaining with top-level samples
        n_remaining = n_samples_total - len(all_solutions)
        extra_params = params.copy()
        extra_params['n_samples'] = n_remaining
        extra_params['target_fitness'] = target_levels[0]

        extra_solutions = sample_backdrive(
            n_vars, model, cardinality,
            current_population, current_fitness,
            extra_params
        )

        all_solutions = np.vstack([all_solutions, extra_solutions])

    return all_solutions
