"""
Neural Network-based EDA with Fitness-Weighted Autoencoder

==============================================================================
OVERVIEW
==============================================================================

This module implements a neural network-based EDA that uses an autoencoder
architecture with a unique fitness-weighted loss function. Unlike traditional
autoencoders, this approach:

1. Takes both solution and fitness as input: [x, f(x)]
2. Outputs both solution and predicted fitness: [x', f'(x')]
3. Uses a fitness-weighted reconstruction loss
4. Evaluates predicted solutions during training
5. Stores all evaluated solutions as a form of local optimization

==============================================================================
ARCHITECTURE
==============================================================================

The autoencoder learns a transformation from [x, f(x)] to [x', f'(x')]
where ideally f(x') < f(x) (for minimization problems).

Input:
- Solution x: vector of n variables
- Fitness f(x): scalar value

Output:
- New solution x': vector of n variables
- Predicted fitness f'(x'): scalar value

Network Structure:
- Input layer: (n_vars + 1) dimensions [x, f(x)]
- Hidden layers: Configurable architecture
- Output layer: (n_vars + 1) dimensions [x', f'(x')]

==============================================================================
LOSS FUNCTION
==============================================================================

The loss function has a unique structure that biases learning toward
better solutions:

    Loss = MSE([x, f(x)], [x', f'(x')]) * (f(x') - f(x))

Where:
- MSE([x, f(x)], [x', f'(x')]): Reconstruction loss
- (f(x') - f(x)): Fitness improvement factor

Key Properties:
1. When f(x') < f(x) (improvement): Loss is negative (gradient descent minimizes)
2. When f(x') > f(x) (degradation): Loss is positive (penalized)
3. f(x') must be evaluated using the actual objective function during training

==============================================================================
UNIQUE FEATURES FOR EDAs
==============================================================================

1. **In-Training Evaluation**:
   - Unlike other EDAs, this method evaluates solutions during neural network
     training, not just during sampling
   - Each forward pass generates a new solution that gets evaluated
   - Acts as a local optimizer on the current population

2. **Solution Collection**:
   - All solutions evaluated during training are stored
   - These solutions can be returned as part of the learning output
   - Provides immediate improvement without additional sampling

3. **Fitness-Aware Learning**:
   - The network learns to transform solutions toward better fitness
   - Loss function explicitly encourages fitness improvement
   - Not just distribution learning, but optimization-aware learning

==============================================================================
USAGE IN EDA LOOP
==============================================================================

Two possible integration approaches:

Approach 1: Learning as Local Optimization
1. Learn autoencoder on current population
2. Collect best solutions found during training
3. Optionally sample additional solutions from learned model
4. Combine training-time and sampling-time solutions

Approach 2: Two-Phase Optimization
1. Learn autoencoder (collecting improved solutions)
2. Sample new population from learned model
3. Use training solutions as elite solutions
4. Merge and select for next generation

==============================================================================
IMPLEMENTATION DETAILS
==============================================================================

Training Process:
1. For each batch:
   a. Forward pass: [x, f(x)] → [x', f'(x')]
   b. Evaluate f(x') using objective function
   c. Compute MSE reconstruction loss
   d. Compute weighted loss: MSE * (f(x') - f(x))
   e. Backpropagate and update weights
   f. Store evaluated solutions

Solution Storage:
- All (x', f(x')) pairs evaluated during training
- Can be filtered by fitness threshold
- Provides immediate optimization benefit

Parameters:
- hidden_dims: Network architecture
- epochs: Training epochs
- batch_size: Mini-batch size
- learning_rate: Optimizer learning rate
- list_act_functs: Activation functions per layer
- list_init_functs: Weight initialization per layer

==============================================================================
REFERENCES
==============================================================================

This implementation is inspired by optimization-aware neural network approaches
where the network learning process itself performs optimization, similar to:
- Meta-learning for optimization (learning to optimize)
- Neural architecture search with evaluation during training
- Learned optimizers that improve during training

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


class FitnessWeightedAutoencoder(nn.Module):
    """
    Autoencoder that takes [x, f(x)] as input and outputs [x', f'(x')].

    Parameters
    ----------
    n_vars : int
        Number of variables in the solution vector
    hidden_dims : List[int]
        Dimensions of hidden layers
    list_act_functs : List[str], optional
        Activation function for each hidden layer
    list_init_functs : List[str], optional
        Initialization function for each layer
    output_activation : str, optional
        Activation for output layer (default: 'linear' for solutions, 'linear' for fitness)
    """

    def __init__(
        self,
        n_vars: int,
        hidden_dims: List[int],
        list_act_functs: Optional[List[str]] = None,
        list_init_functs: Optional[List[str]] = None,
        output_activation: Optional[str] = None
    ):
        super(FitnessWeightedAutoencoder, self).__init__()

        self.n_vars = n_vars
        self.hidden_dims = hidden_dims

        # Input dimension: n_vars + 1 (solution + fitness)
        # Output dimension: n_vars + 1 (new solution + predicted fitness)
        input_dim = n_vars + 1
        output_dim = n_vars + 1

        # Validate and set activation/initialization functions
        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        self.list_act_functs = list_act_functs
        self.list_init_functs = list_init_functs

        # Build network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)

            activation = get_activation(list_act_functs[i], in_features=hidden_dim)
            layers.append(activation)

            prev_dim = hidden_dim

        # Output layer
        output_linear = nn.Linear(prev_dim, output_dim)
        apply_weight_init(output_linear, list_init_functs[-1] if list_init_functs else 'default')
        layers.append(output_linear)

        # Optional output activation (usually linear for regression-like output)
        if output_activation:
            layers.append(get_activation(output_activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x_with_fitness: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x_with_fitness : torch.Tensor
            Input tensor of shape (batch_size, n_vars + 1)
            where [:, :n_vars] is the solution and [:, n_vars] is the fitness

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_vars + 1)
            where [:, :n_vars] is the new solution and [:, n_vars] is predicted fitness
        """
        return self.network(x_with_fitness)


def learn_nn_eda(
    population: np.ndarray,
    fitness: np.ndarray,
    objective_function: Callable[[np.ndarray], np.ndarray],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a fitness-weighted autoencoder model on the population.

    This function trains an autoencoder that transforms [x, f(x)] to [x', f'(x')]
    where the loss is weighted by fitness improvement. Importantly, all solutions
    generated during training are evaluated and stored, making this both a
    learning and optimization process.

    Parameters
    ----------
    population : np.ndarray
        Current population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values of shape (pop_size,)
    objective_function : Callable
        Function to evaluate fitness of generated solutions.
        Should accept array of shape (n_samples, n_vars) and return shape (n_samples,)
    params : dict, optional
        Training parameters:
        - 'hidden_dims': List[int], hidden layer dimensions (default: computed)
        - 'epochs': int, number of training epochs (default: 100)
        - 'batch_size': int, mini-batch size (default: computed)
        - 'learning_rate': float, Adam learning rate (default: 1e-3)
        - 'list_act_functs': List[str], activation functions (default: ['relu', 'relu'])
        - 'list_init_functs': List[str], weight initializations (default: ['default', 'default'])
        - 'output_activation': str, output activation (default: None)
        - 'fitness_weight_clip': float, clip fitness weights to avoid extreme values (default: 10.0)
        - 'store_all_solutions': bool, store all evaluated solutions (default: True)
        - 'elitism': bool, preserve best solution from original population (default: True)
        - 'verbose': bool, print training progress (default: True)

    Returns
    -------
    model_data : dict
        Dictionary containing:
        - 'model_state': Trained model state dict
        - 'n_vars': Number of variables
        - 'hidden_dims': Hidden layer dimensions
        - 'list_act_functs': Activation functions used
        - 'list_init_functs': Initialization functions used
        - 'output_activation': Output activation used
        - 'ranges': Data normalization ranges [min, max]
        - 'evaluated_solutions': All solutions evaluated during training (n_evaluated, n_vars)
        - 'evaluated_fitness': Fitness values of evaluated solutions (n_evaluated,)
        - 'n_evaluations': Total number of evaluations performed
        - 'training_history': List of epoch losses
        - 'elite_solution': Best solution from original population (if elitism=True)
        - 'elite_fitness': Fitness of elite solution (if elitism=True)
    """
    if params is None:
        params = {}

    # Extract parameters
    n_vars = population.shape[1]
    pop_size = population.shape[0]

    hidden_dims = params.get('hidden_dims', compute_default_hidden_dims(n_vars, pop_size))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', compute_default_batch_size(n_vars, pop_size))
    learning_rate = params.get('learning_rate', 1e-3)
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)
    output_activation = params.get('output_activation', None)
    fitness_weight_clip = params.get('fitness_weight_clip', 10.0)
    store_all_solutions = params.get('store_all_solutions', True)
    elitism = params.get('elitism', True)
    verbose = params.get('verbose', True)

    # Store elite solution if elitism is enabled
    if elitism:
        best_idx = np.argmin(fitness)
        elite_solution = population[best_idx].copy()
        elite_fitness = fitness[best_idx]
    else:
        elite_solution = None
        elite_fitness = None

    # Normalize data to [0, 1] range for each dimension
    x_min = population.min(axis=0)
    x_max = population.max(axis=0)
    ranges = np.array([x_min, x_max])

    # Avoid division by zero
    range_diff = x_max - x_min
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    norm_population = (population - x_min) / range_diff

    # Normalize fitness to [0, 1] as well
    f_min = fitness.min()
    f_max = fitness.max()
    f_range = f_max - f_min
    if f_range < 1e-10:
        f_range = 1.0
    norm_fitness = (fitness - f_min) / f_range

    # Create input: [x, f(x)]
    input_data = np.hstack([norm_population, norm_fitness.reshape(-1, 1)])

    # Convert to tensors
    input_tensor = torch.FloatTensor(input_data)

    # Create model
    model = FitnessWeightedAutoencoder(
        n_vars=n_vars,
        hidden_dims=hidden_dims,
        list_act_functs=list_act_functs,
        list_init_functs=list_init_functs,
        output_activation=output_activation
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Storage for evaluated solutions
    evaluated_solutions = []
    evaluated_fitness = []
    n_evaluations = 0
    training_history = []

    if verbose:
        print(f"Training NN-EDA model:")
        print(f"  Population size: {pop_size}")
        print(f"  Variables: {n_vars}")
        print(f"  Architecture: {[n_vars + 1] + hidden_dims + [n_vars + 1]}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        epoch_evaluations = 0

        # Shuffle data
        indices = np.random.permutation(pop_size)

        # Mini-batch training
        for i in range(0, pop_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_input = input_tensor[batch_indices]

            # Forward pass
            optimizer.zero_grad()
            batch_output = model(batch_input)

            # Extract predicted solutions and fitness
            pred_norm_solutions = batch_output[:, :n_vars]
            pred_norm_fitness = batch_output[:, n_vars]

            # Denormalize predicted solutions
            pred_solutions = pred_norm_solutions.detach().numpy() * range_diff + x_min

            # Evaluate actual fitness of predicted solutions
            actual_fitness = objective_function(pred_solutions)
            actual_fitness_tensor = torch.FloatTensor(actual_fitness)

            # Store evaluated solutions
            if store_all_solutions:
                evaluated_solutions.append(pred_solutions)
                evaluated_fitness.append(actual_fitness)

            n_evaluations += len(actual_fitness)
            epoch_evaluations += len(actual_fitness)

            # Normalize actual fitness for loss computation
            norm_actual_fitness = (actual_fitness - f_min) / f_range
            norm_actual_fitness_tensor = torch.FloatTensor(norm_actual_fitness)

            # Compute MSE reconstruction loss
            # Compare [x, f(x)] with [x', f'(x')]
            # But use actual fitness f(x') instead of predicted f'(x')
            output_with_actual_fitness = torch.cat([
                pred_norm_solutions,
                norm_actual_fitness_tensor.unsqueeze(1)
            ], dim=1)

            mse_loss = nn.functional.mse_loss(output_with_actual_fitness, batch_input, reduction='none')
            mse_loss = mse_loss.mean(dim=1)  # Mean over features

            # Compute fitness difference: f(x') - f(x)
            # Original fitness for this batch
            original_norm_fitness = batch_input[:, n_vars]
            original_fitness = original_norm_fitness * f_range + f_min

            # Fitness improvement factor (negative is good)
            fitness_diff = actual_fitness_tensor - original_fitness

            # Clip fitness difference to avoid extreme weights
            fitness_diff = torch.clamp(fitness_diff, -fitness_weight_clip, fitness_weight_clip)

            # Final loss: MSE * (f(x') - f(x))
            # When f(x') < f(x), loss is negative (good, but we minimize, so it becomes a reward)
            # When f(x') > f(x), loss is positive (bad, penalized)
            weighted_loss = mse_loss * fitness_diff

            # Take mean over batch
            loss = weighted_loss.mean()

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_epoch_loss = np.mean(epoch_losses)
        training_history.append(mean_epoch_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {mean_epoch_loss:.6f}, Evaluations = {epoch_evaluations}")

    # Concatenate all evaluated solutions
    if store_all_solutions and evaluated_solutions:
        all_evaluated_solutions = np.vstack(evaluated_solutions)
        all_evaluated_fitness = np.hstack(evaluated_fitness)
    else:
        all_evaluated_solutions = np.array([])
        all_evaluated_fitness = np.array([])

    if verbose:
        print(f"\nTraining completed:")
        print(f"  Total evaluations: {n_evaluations}")
        if len(all_evaluated_fitness) > 0:
            print(f"  Best fitness found: {all_evaluated_fitness.min():.6f}")
            print(f"  Best in original population: {fitness.min():.6f}")
            improvement = fitness.min() - all_evaluated_fitness.min()
            print(f"  Improvement: {improvement:.6f}")

    # Prepare model data
    model_data = {
        'model_state': model.state_dict(),
        'n_vars': n_vars,
        'hidden_dims': hidden_dims,
        'list_act_functs': model.list_act_functs,
        'list_init_functs': model.list_init_functs,
        'output_activation': output_activation,
        'ranges': ranges,
        'fitness_range': np.array([f_min, f_max]),
        'evaluated_solutions': all_evaluated_solutions,
        'evaluated_fitness': all_evaluated_fitness,
        'n_evaluations': n_evaluations,
        'training_history': training_history,
        'elite_solution': elite_solution,
        'elite_fitness': elite_fitness,
        'elitism': elitism,
    }

    return model_data
