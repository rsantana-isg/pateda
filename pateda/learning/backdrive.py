"""
Backdrive-EDA learning functions

This module implements learning for Backdrive-EDA, which trains a multi-layer
perceptron (MLP) to predict fitness from solutions. The key insight is that the
trained MLP can then be "back-driven" (network inversion) to generate new solutions
by propagating high fitness values backwards through the network to modify inputs.

References:
- Garciarena et al. (2020). "Envisioning the Benefits of Back-Drive in Evolutionary Algorithms"
- Baluja (2017). "Deep Learning for Explicitly Modeling Optimization Landscapes"
"""

import numpy as np
from typing import Dict, Any, Optional, List
import warnings

from pateda.core.models import NeuralNetworkModel
from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


def learn_backdrive(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_population: np.ndarray,
    selected_fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> NeuralNetworkModel:
    """
    Learn a Backdrive-EDA model by training an MLP to predict fitness from solutions.

    The network learns a regression from solution parameters (x) to fitness values (f).
    This mapping can then be inverted during sampling to generate high-fitness solutions.

    Parameters
    ----------
    generation : int
        Current generation number
    n_vars : int
        Number of variables in the problem
    cardinality : np.ndarray
        For continuous problems: shape (2, n_vars) with [min_values, max_values]
        For discrete problems: shape (n_vars,) with cardinalities
    selected_population : np.ndarray
        Selected individuals, shape (n_selected, n_vars)
    selected_fitness : np.ndarray
        Fitness values for selected individuals, shape (n_selected, 1) or (n_selected,)
    params : dict, optional
        Learning parameters:
        - 'hidden_layers': list of hidden layer sizes
          (default: computed from n_vars and pop_size)
        - 'list_act_functs': list of activation functions, one per hidden layer
        - 'list_init_functs': list of initialization functions, one per hidden layer
        - 'activation': single activation function for all layers (for backward compatibility)
        - 'epochs': number of training epochs, default 50
        - 'batch_size': batch size for training (default: max(8, n_vars/50))
        - 'learning_rate': learning rate for optimizer, default 0.001
        - 'optimizer': optimizer name, default 'adam'
        - 'validation_split': fraction for validation, default 0.2
        - 'early_stopping': enable early stopping, default True
        - 'patience': early stopping patience, default 10
        - 'transfer_weights': transfer weights from previous generation, default True
        - 'previous_model': model from previous generation for transfer learning

    Returns
    -------
    NeuralNetworkModel
        Trained model containing the MLP and training statistics

    Notes
    -----
    The network is trained to predict fitness from solutions. During sampling,
    the network weights are frozen and backpropagation is used to modify inputs
    to achieve high predicted fitness values (network inversion/backdrive).

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.backdrive import learn_backdrive
    >>> # Generate sample data
    >>> n_vars = 10
    >>> selected_pop = np.random.randn(100, n_vars)
    >>> selected_fit = np.sum(selected_pop**2, axis=1, keepdims=True)
    >>> cardinality = np.array([[-5]*n_vars, [5]*n_vars])
    >>> # Learn model
    >>> model = learn_backdrive(0, n_vars, cardinality, selected_pop, selected_fit)
    """
    # Import PyTorch (only when needed)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        raise ImportError(
            "PyTorch is required for Backdrive-EDA. "
            "Install it with: pip install torch"
        )

    # Compute defaults based on input dimensions
    pop_size = selected_population.shape[0]
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Set default parameters
    default_params = {
        'hidden_layers': default_hidden_dims,
        'list_act_functs': None,  # Will be set based on activation if not provided
        'list_init_functs': None,
        'activation': 'tanh',  # For backward compatibility
        'epochs': 50,
        'batch_size': default_batch_size,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 10,
        'transfer_weights': True,
        'previous_model': None,
    }

    if params is not None:
        default_params.update(params)
    params = default_params

    # Handle backward compatibility: if list_act_functs not provided, use activation
    if params['list_act_functs'] is None:
        params['list_act_functs'] = [params['activation']] * len(params['hidden_layers'])
    if params['list_init_functs'] is None:
        params['list_init_functs'] = ['default'] * len(params['hidden_layers'])

    # Normalize data
    # Store ranges for denormalization during sampling
    if cardinality.ndim == 2:
        # Continuous variables
        lower_bounds = cardinality[0, :]
        upper_bounds = cardinality[1, :]
    else:
        # Discrete variables - treat as continuous [0, card-1]
        lower_bounds = np.zeros(n_vars)
        upper_bounds = cardinality - 1

    # Normalize population to [0, 1]
    ranges = upper_bounds - lower_bounds
    ranges = np.where(ranges == 0, 1.0, ranges)  # Avoid division by zero
    normalized_pop = (selected_population - lower_bounds) / ranges

    # Normalize fitness to [0, 1]
    fitness_1d = selected_fitness.flatten()
    fitness_min = np.min(fitness_1d)
    fitness_max = np.max(fitness_1d)
    fitness_range = fitness_max - fitness_min
    if fitness_range == 0:
        fitness_range = 1.0
        warnings.warn("All fitness values are identical. Using range=1.0")

    normalized_fitness = (fitness_1d - fitness_min) / fitness_range

    # Convert to PyTorch tensors
    X = torch.FloatTensor(normalized_pop)
    y = torch.FloatTensor(normalized_fitness).unsqueeze(1)

    # Split into training and validation
    n_samples = len(X)
    n_val = int(n_samples * params['validation_split'])
    n_train = n_samples - n_val

    if n_val > 0:
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Create MLP architecture with configurable activations and initializations
    class BackdriveMLP(nn.Module):
        def __init__(self, input_dim, hidden_layers, list_act_functs, list_init_functs):
            super(BackdriveMLP, self).__init__()

            # Validate parameters
            list_act_functs, list_init_functs = validate_list_params(
                hidden_layers, list_act_functs, list_init_functs
            )

            layers = []
            prev_size = input_dim

            for i, hidden_size in enumerate(hidden_layers):
                linear = nn.Linear(prev_size, hidden_size)
                apply_weight_init(linear, list_init_functs[i])
                layers.append(linear)
                layers.append(get_activation(list_act_functs[i], in_features=hidden_size))
                prev_size = hidden_size

            # Output layer (fitness prediction)
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())  # Output in [0, 1]

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Create model with configurable activations and initializations
    model = BackdriveMLP(
        n_vars,
        params['hidden_layers'],
        params['list_act_functs'],
        params['list_init_functs']
    )

    # Transfer weights from previous generation if requested
    if params['transfer_weights'] and params['previous_model'] is not None:
        try:
            prev_model = params['previous_model'].parameters
            if isinstance(prev_model, BackdriveMLP):
                model.load_state_dict(prev_model.state_dict())
                print(f"  Transferred weights from generation {generation-1}")
        except Exception as e:
            warnings.warn(f"Could not transfer weights: {e}")

    # Setup optimizer
    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {params['optimizer']}")

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)

            # Early stopping
            if params['early_stopping']:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= params['patience']:
                        print(f"  Early stopping at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)
                        break

    # Calculate correlation between predictions and actual fitness
    model.eval()
    with torch.no_grad():
        all_predictions = model(X).numpy().flatten()

    correlation = np.corrcoef(all_predictions, normalized_fitness)[0, 1]

    # Create model structure description
    structure = {
        'input_dim': n_vars,
        'hidden_layers': params['hidden_layers'],
        'list_act_functs': params['list_act_functs'],
        'list_init_functs': params['list_init_functs'],
        'activation': params['activation'],  # For backward compatibility
        'output_dim': 1,
    }

    # Create metadata
    metadata = {
        'generation': generation,
        'n_samples': n_samples,
        'epochs_trained': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses else None,
        'correlation': correlation,
        'fitness_min': fitness_min,
        'fitness_max': fitness_max,
        'fitness_range': fitness_range,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'ranges': ranges,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    # Return trained model
    return NeuralNetworkModel(
        structure=structure,
        parameters=model,  # Store the trained PyTorch model
        metadata=metadata
    )
