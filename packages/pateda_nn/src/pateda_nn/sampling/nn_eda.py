"""
Sampling from Fitness-Weighted Autoencoder (NN-EDA)

==============================================================================
OVERVIEW
==============================================================================

This module provides sampling functionality for the fitness-weighted autoencoder
EDA. Unlike other EDAs, NN-EDA has already generated and evaluated many solutions
during the learning phase. This sampling module provides two main functions:

1. Access to solutions already evaluated during training
2. Generate new solutions by applying the learned transformation

==============================================================================
SAMPLING STRATEGIES
==============================================================================

Strategy 1: Use Training Solutions
- Return the best solutions found during autoencoder training
- No additional function evaluations needed
- Acts as local optimization of the population

Strategy 2: Generate New Solutions
- Apply the learned autoencoder to transform current population
- Generate new candidate solutions
- Requires evaluation of generated solutions

Strategy 3: Hybrid Approach (Recommended)
- Combine best solutions from training
- Generate additional solutions via autoencoder
- Provides both exploitation (training solutions) and exploration (new samples)

==============================================================================
USAGE NOTES
==============================================================================

The NN-EDA approach is fundamentally different from other EDAs because:

1. Learning Phase = Optimization Phase:
   - Training the autoencoder generates and evaluates many solutions
   - These solutions are often better than the original population
   - Can be viewed as a local optimizer

2. Sampling Phase = Optional:
   - May not need additional sampling if training found good solutions
   - Sampling can provide diversity and exploration
   - Useful for maintaining population size

3. Evaluation Budget:
   - Most evaluations happen during learning
   - Sampling may require fewer evaluations
   - Total budget = training evaluations + sampling evaluations

==============================================================================
INTEGRATION WITH EDA LOOP
==============================================================================

Typical usage in an EDA iteration:

```python
# 1. Learn autoencoder (evaluates many solutions during training)
model_data = learn_nn_eda(population, fitness, objective_function, params)

# 2. Get solutions found during training
training_solutions = model_data['evaluated_solutions']
training_fitness = model_data['evaluated_fitness']

# 3. Optionally generate additional solutions
new_solutions = sample_nn_eda(model_data, n_samples=pop_size//2)
new_fitness = objective_function(new_solutions)

# 4. Combine and select
all_solutions = np.vstack([training_solutions, new_solutions])
all_fitness = np.hstack([training_fitness, new_fitness])

# 5. Select best for next generation
next_population = select_best(all_solutions, all_fitness, pop_size)
```

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from pateda_nn.learning.nn_eda import FitnessWeightedAutoencoder


def sample_nn_eda(
    model_data: Dict[str, Any],
    n_samples: int,
    source_population: Optional[np.ndarray] = None,
    source_fitness: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample new solutions from the trained NN-EDA autoencoder.

    This function applies the learned autoencoder transformation to generate
    new solutions. It can either:
    1. Transform provided source_population
    2. Use the evaluated solutions from training as source
    3. Sample randomly from the solution space

    Parameters
    ----------
    model_data : dict
        Dictionary returned by learn_nn_eda containing:
        - 'model_state': Trained model state dict
        - 'n_vars': Number of variables
        - 'hidden_dims': Hidden layer dimensions
        - 'list_act_functs': Activation functions
        - 'list_init_functs': Initialization functions
        - 'output_activation': Output activation
        - 'ranges': Data normalization ranges
        - 'fitness_range': Fitness normalization range
        - 'evaluated_solutions': Solutions from training
        - 'evaluated_fitness': Fitness from training

    n_samples : int
        Number of new solutions to generate

    source_population : np.ndarray, optional
        Source population to transform of shape (n_source, n_vars).
        If None, uses random selection from training solutions or random sampling.

    source_fitness : np.ndarray, optional
        Fitness values for source_population of shape (n_source,).
        Required if source_population is provided.

    params : dict, optional
        Sampling parameters:
        - 'use_training_solutions': bool, use training solutions as source (default: True)
        - 'add_noise': bool, add Gaussian noise to outputs (default: False)
        - 'noise_std': float, standard deviation of noise (default: 0.01)
        - 'n_iterations': int, apply transformation multiple times (default: 1)

    Returns
    -------
    new_solutions : np.ndarray
        Generated solutions of shape (n_samples, n_vars)
    """
    if params is None:
        params = {}

    # Extract model parameters
    n_vars = model_data['n_vars']
    hidden_dims = model_data['hidden_dims']
    list_act_functs = model_data['list_act_functs']
    list_init_functs = model_data['list_init_functs']
    output_activation = model_data['output_activation']
    ranges = model_data['ranges']
    fitness_range = model_data['fitness_range']

    use_training_solutions = params.get('use_training_solutions', True)
    add_noise = params.get('add_noise', False)
    noise_std = params.get('noise_std', 0.01)
    n_iterations = params.get('n_iterations', 1)

    # Recreate model
    model = FitnessWeightedAutoencoder(
        n_vars=n_vars,
        hidden_dims=hidden_dims,
        list_act_functs=list_act_functs,
        list_init_functs=list_init_functs,
        output_activation=output_activation
    )
    model.load_state_dict(model_data['model_state'])
    model.eval()

    # Determine source for transformation
    if source_population is not None:
        if source_fitness is None:
            raise ValueError("source_fitness must be provided when source_population is given")
        pop = source_population
        fit = source_fitness
    elif use_training_solutions and len(model_data['evaluated_solutions']) > 0:
        # Use random selection from training solutions
        all_solutions = model_data['evaluated_solutions']
        all_fitness = model_data['evaluated_fitness']

        # Sample with replacement if needed
        n_available = len(all_solutions)
        indices = np.random.choice(n_available, size=n_samples, replace=(n_samples > n_available))
        pop = all_solutions[indices]
        fit = all_fitness[indices]
    else:
        # Generate random solutions within ranges
        x_min, x_max = ranges[0], ranges[1]
        pop = np.random.uniform(x_min, x_max, size=(n_samples, n_vars))
        # Assign median fitness as placeholder
        f_min, f_max = fitness_range[0], fitness_range[1]
        fit = np.full(n_samples, (f_min + f_max) / 2)

    # Normalize source population and fitness
    x_min, x_max = ranges[0], ranges[1]
    range_diff = x_max - x_min
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    f_min, f_max = fitness_range[0], fitness_range[1]
    f_range = f_max - f_min
    if f_range < 1e-10:
        f_range = 1.0

    # Apply transformation iteratively
    current_pop = pop.copy()
    current_fit = fit.copy()

    with torch.no_grad():
        for iteration in range(n_iterations):
            # Normalize
            norm_pop = (current_pop - x_min) / range_diff
            norm_fit = (current_fit - f_min) / f_range

            # Create input [x, f(x)]
            input_data = np.hstack([norm_pop, norm_fit.reshape(-1, 1)])
            input_tensor = torch.FloatTensor(input_data)

            # Forward pass
            output_tensor = model(input_tensor)

            # Extract new solutions
            output_data = output_tensor.numpy()
            norm_new_solutions = output_data[:, :n_vars]
            # Note: we ignore the predicted fitness output[:, n_vars]
            # since we'll evaluate actual fitness later

            # Denormalize
            new_solutions = norm_new_solutions * range_diff + x_min

            # Clip to valid range
            new_solutions = np.clip(new_solutions, x_min, x_max)

            # Add noise if requested
            if add_noise:
                noise = np.random.randn(*new_solutions.shape) * noise_std * range_diff
                new_solutions = new_solutions + noise
                new_solutions = np.clip(new_solutions, x_min, x_max)

            # Update for next iteration
            current_pop = new_solutions
            # For next iteration, use predicted fitness (not evaluated)
            current_fit = output_data[:, n_vars] * f_range + f_min

    return new_solutions


def get_best_training_solutions(
    model_data: Dict[str, Any],
    n_solutions: int,
    return_fitness: bool = False
) -> np.ndarray:
    """
    Get the best solutions found during autoencoder training.

    Parameters
    ----------
    model_data : dict
        Dictionary returned by learn_nn_eda

    n_solutions : int
        Number of best solutions to return

    return_fitness : bool, optional
        If True, return (solutions, fitness) tuple

    Returns
    -------
    best_solutions : np.ndarray
        Best solutions of shape (n_solutions, n_vars)
        Or if return_fitness=True: (solutions, fitness) tuple
    """
    evaluated_solutions = model_data['evaluated_solutions']
    evaluated_fitness = model_data['evaluated_fitness']

    if len(evaluated_solutions) == 0:
        raise ValueError("No solutions were evaluated during training")

    # Sort by fitness (ascending for minimization)
    sorted_indices = np.argsort(evaluated_fitness)

    # Take best n_solutions
    n_available = len(evaluated_solutions)
    n_to_take = min(n_solutions, n_available)

    best_indices = sorted_indices[:n_to_take]
    best_solutions = evaluated_solutions[best_indices]
    best_fitness = evaluated_fitness[best_indices]

    # If need more, repeat the best ones
    if n_to_take < n_solutions:
        repeats_needed = n_solutions - n_to_take
        repeat_indices = np.random.choice(best_indices, size=repeats_needed, replace=True)
        best_solutions = np.vstack([best_solutions, evaluated_solutions[repeat_indices]])
        best_fitness = np.hstack([best_fitness, evaluated_fitness[repeat_indices]])

    if return_fitness:
        return best_solutions, best_fitness
    else:
        return best_solutions


def sample_nn_eda_hybrid(
    model_data: Dict[str, Any],
    n_samples: int,
    fraction_from_training: float = 0.5,
    source_population: Optional[np.ndarray] = None,
    source_fitness: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Hybrid sampling: combine best training solutions with new generated solutions.

    This is the recommended sampling strategy for NN-EDA. It combines:
    1. Elite solution from original population (if elitism enabled)
    2. Best solutions found during training (exploitation)
    3. New solutions generated by transforming source population (exploration)

    Parameters
    ----------
    model_data : dict
        Dictionary returned by learn_nn_eda

    n_samples : int
        Total number of samples to return

    fraction_from_training : float, optional
        Fraction of samples to take from training solutions (default: 0.5)
        Must be between 0 and 1.

    source_population : np.ndarray, optional
        Source population to transform for generating new solutions.
        Shape: (n_source, n_vars)
        If None, uses training solutions or random sampling.

    source_fitness : np.ndarray, optional
        Fitness values for source_population.
        Shape: (n_source,)
        Required if source_population is provided.

    params : dict, optional
        Parameters passed to sample_nn_eda for the generated portion

    Returns
    -------
    samples : np.ndarray
        Combined samples of shape (n_samples, n_vars)

    Notes
    -----
    If elitism is enabled in model_data, the elite solution is always included
    as the first solution in the returned array, and the number of other solutions
    is adjusted accordingly.

    When source_population is provided, new solutions are generated by applying
    the learned autoencoder transformation to the source population, which is
    more meaningful than using random solutions or training solutions.
    """
    if not (0 <= fraction_from_training <= 1):
        raise ValueError("fraction_from_training must be between 0 and 1")

    # Check if elitism is enabled
    elitism = model_data.get('elitism', False)
    elite_solution = model_data.get('elite_solution', None)

    # Adjust counts if elitism is enabled
    if elitism and elite_solution is not None:
        # Reserve one spot for elite
        n_remaining = n_samples - 1
        n_from_training = int(n_remaining * fraction_from_training)
        n_to_generate = n_remaining - n_from_training
    else:
        n_from_training = int(n_samples * fraction_from_training)
        n_to_generate = n_samples - n_from_training

    samples = []

    # Add elite solution first if elitism is enabled
    if elitism and elite_solution is not None:
        samples.append(elite_solution.reshape(1, -1))

    # Get best from training
    if n_from_training > 0:
        training_samples = get_best_training_solutions(model_data, n_from_training)
        samples.append(training_samples)

    # Generate new solutions by transforming source population
    if n_to_generate > 0:
        new_samples = sample_nn_eda(
            model_data,
            n_to_generate,
            source_population=source_population,
            source_fitness=source_fitness,
            params=params
        )
        samples.append(new_samples)

    # Combine
    if len(samples) > 1:
        combined = np.vstack(samples)
    else:
        combined = samples[0]

    return combined


def get_training_statistics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about solutions evaluated during training.

    Parameters
    ----------
    model_data : dict
        Dictionary returned by learn_nn_eda

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'n_evaluations': Total evaluations during training
        - 'best_fitness': Best fitness found
        - 'mean_fitness': Mean fitness of evaluated solutions
        - 'std_fitness': Std of fitness values
        - 'improvement_over_initial': Improvement over best initial solution
    """
    evaluated_fitness = model_data['evaluated_fitness']

    if len(evaluated_fitness) == 0:
        return {
            'n_evaluations': 0,
            'best_fitness': None,
            'mean_fitness': None,
            'std_fitness': None,
        }

    stats = {
        'n_evaluations': model_data['n_evaluations'],
        'best_fitness': float(np.min(evaluated_fitness)),
        'mean_fitness': float(np.mean(evaluated_fitness)),
        'std_fitness': float(np.std(evaluated_fitness)),
        'median_fitness': float(np.median(evaluated_fitness)),
        'worst_fitness': float(np.max(evaluated_fitness)),
    }

    return stats
