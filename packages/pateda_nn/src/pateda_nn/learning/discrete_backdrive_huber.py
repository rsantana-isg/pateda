"""
Discrete Backdrive-EDA Learning with Huber Loss

==============================================================================
OVERVIEW
==============================================================================

This module implements a variant of Discrete Backdrive-EDA that uses
Huber loss instead of standard MSE loss. Huber loss is more robust to
outliers than MSE, using L2 loss for small errors and L1 loss for large errors.

This is beneficial when fitness values have outliers or when the fitness
distribution is not Gaussian, making the model more robust and less likely
to overfit to unusual fitness values.

==============================================================================
KEY DIFFERENCES FROM STANDARD BACKDRIVE
==============================================================================

Loss Function:
- Standard MSE: (pred - target)^2 for all errors
- Huber: 0.5 * (pred - target)^2 for |error| <= delta
         delta * (|pred - target| - 0.5 * delta) for |error| > delta

The Huber loss smoothly transitions between L2 (quadratic) for small errors
and L1 (linear) for large errors, providing robustness to outliers.

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from pateda_nn.learning.discrete_backdrive import (
    DiscreteBackdriveNet,
    compute_backdrive_hidden_dims,
)


def huber_loss(predictions: torch.Tensor, targets: torch.Tensor, 
               delta: float = 1.0) -> torch.Tensor:
    """
    Compute Huber loss
    
    Huber loss is L2 for small errors and L1 for large errors,
    providing robustness to outliers.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted fitness values [batch_size, 1]
    targets : torch.Tensor
        Target fitness values [batch_size, 1]
    delta : float
        Threshold for switching from L2 to L1 (default: 1.0)
    
    Returns
    -------
    loss : torch.Tensor
        Huber loss scalar
    """
    error = predictions - targets
    abs_error = torch.abs(error)
    
    # For |error| <= delta: 0.5 * error^2
    # For |error| > delta: delta * (|error| - 0.5 * delta)
    is_small = abs_error <= delta
    
    small_loss = 0.5 * error ** 2
    large_loss = delta * (abs_error - 0.5 * delta)
    
    loss = torch.where(is_small, small_loss, large_loss).mean()
    
    return loss


def learn_discrete_backdrive_huber(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Discrete Backdrive model with Huber loss

    Trains a neural network to predict fitness from discrete solutions,
    using Huber loss for robustness to outliers.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, 1)
    cardinality : np.ndarray
        Cardinality of each variable [n_vars]
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive plus):
        - 'hidden_layers': list of hidden layer sizes
          (default: computed dynamically to avoid overfitting)
        - 'use_embeddings': use embedding layers (default: True if max(card)>2)
        - 'embedding_dim': embedding dimension (default: 8)
        - 'epochs': number of training epochs (default: 100)
        - 'batch_size': batch size (default: max(8, n_vars/50))
        - 'learning_rate': learning rate (default: 0.001)
        - 'weight_decay': L2 regularization (default: 1e-3)
        - 'dropout': dropout rate (default: 0.45)
        - 'validation_split': validation fraction (default: 0.2)
        - 'early_stopping': enable early stopping (default: True)
        - 'patience': early stopping patience (default: 10)
        - 'huber_delta': delta parameter for Huber loss (default: 1.0)
        - 'pretrained_model': model dict from previous generation for transfer learning

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_backdrive_hidden_dims(n_vars, pop_size)
    hidden_layers = params.get('hidden_layers', default_hidden_dims)
    list_act_functs = params.get('list_act_functs', None)
    use_embeddings = params.get('use_embeddings', np.max(cardinality) > 2)
    embedding_dim = params.get('embedding_dim', 8)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', max(8, pop_size // 4))
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 1e-3)
    dropout = params.get('dropout', 0.45)
    validation_split = params.get('validation_split', 0.2)
    early_stopping = params.get('early_stopping', True)
    patience = params.get('patience', 10)
    huber_delta = params.get('huber_delta', 1.0)
    # Print per-epoch training loss only when explicitly requested.
    verbose = params.get('verbose', False)

    # Extract pretrained model for weight transfer
    pretrained_model = params.get('pretrained_model', None)

    # Normalize fitness
    fitness_1d = fitness.flatten()
    fitness_mean = np.mean(fitness_1d)
    fitness_std = np.std(fitness_1d)
    if fitness_std < 1e-10:
        fitness_std = 1.0
        warnings.warn("Fitness has zero std. Using std=1.0")

    normalized_fitness = (fitness_1d - fitness_mean) / fitness_std

    # Convert to tensors
    X = torch.LongTensor(population.astype(int))
    y = torch.FloatTensor(normalized_fitness).unsqueeze(1)

    # Split into training and validation
    n_samples = len(X)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    if n_val > 0:
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
    else:
        X_train = X
        y_train = y
        X_val = None
        y_val = None

    # Create network with configurable activation functions
    network = DiscreteBackdriveNet(
        n_vars, cardinality, hidden_layers, use_embeddings, embedding_dim, dropout,
        list_act_functs=list_act_functs
    )

    # Transfer weights from previous generation if provided
    if pretrained_model is not None:
        try:
            # Load state dict from pretrained model
            network.load_state_dict(pretrained_model['network_state'])
            #print("  Transferred weights from previous generation")
        except Exception as e:
            warnings.warn(f"Could not transfer weights: {e}")

    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate,
                          weight_decay=weight_decay)

    # Training loop
    network.train()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(len(X_train))

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            # Forward pass
            pred = network(batch_x)
            
            # Huber loss
            loss = huber_loss(pred, batch_y, delta=huber_delta)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        if X_val is not None:
            network.eval()
            with torch.no_grad():
                val_pred = network(X_val)
                val_loss = huber_loss(val_pred, y_val, delta=huber_delta).item()
            network.train()

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        #print(f"Early stopping at epoch {epoch+1}")
                        break

            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")
        else:
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}")

    # Return model
    model = {
        'network_state': network.state_dict(),
        'n_vars': n_vars,
        'cardinality': cardinality.copy(),
        'hidden_layers': hidden_layers,
        'use_embeddings': use_embeddings,
        'embedding_dim': embedding_dim if use_embeddings else None,
        'fitness_stats': (fitness_mean, fitness_std),
        'list_act_functs': list_act_functs if list_act_functs is not None else ['relu'] * len(hidden_layers),
        'type': 'discrete_backdrive_huber'
    }

    return model


def learn_binary_backdrive_huber(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary Backdrive model with Huber loss
    
    Simplified interface for binary problems.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive_huber)

    Returns
    -------
    model : dict
        Model dictionary
    """
    n_vars = population.shape[1]
    cardinality = np.full(n_vars, 2)  # All binary

    # Force no embeddings for pure binary
    if params is None:
        params = {}
    params['use_embeddings'] = False

    return learn_discrete_backdrive_huber(population, fitness, cardinality, params)
