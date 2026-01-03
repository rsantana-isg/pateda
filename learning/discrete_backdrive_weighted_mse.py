"""
Discrete Backdrive-EDA Learning with Fitness-Weighted MSE Loss

==============================================================================
OVERVIEW
==============================================================================

This module implements a variant of Discrete Backdrive-EDA that uses
fitness-weighted MSE loss instead of standard MSE loss. The idea is to
focus the model on learning the fitness landscape for high-fitness solutions,
which are more important for optimization.

The fitness-weighted MSE gives higher weight to errors on high-fitness solutions
and lower weight to errors on low-fitness solutions, allowing the model to
better predict fitness in the regions that matter most.

==============================================================================
KEY DIFFERENCES FROM STANDARD BACKDRIVE
==============================================================================

Loss Function:
- Standard: MSE(pred, target) = mean((pred - target)^2)
- Weighted: sum(weights * (pred - target)^2) where weights ∝ exp(fitness)

The weighted version focuses modeling capacity on high-fitness regions,
which is where the backdrive network inversion will search for solutions.

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from pateda.learning.discrete_backdrive import (
    DiscreteBackdriveNet,
    compute_backdrive_hidden_dims,
)


def fitness_weighted_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                              fitness_values: torch.Tensor) -> torch.Tensor:
    """
    Compute fitness-weighted MSE loss
    
    Higher fitness solutions get higher weight in the loss, focusing
    the model on accurately predicting fitness for good solutions.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted fitness values [batch_size, 1]
    targets : torch.Tensor
        Target fitness values [batch_size, 1]
    fitness_values : torch.Tensor
        Original fitness values for weighting [batch_size, 1]
    
    Returns
    -------
    loss : torch.Tensor
        Weighted MSE loss scalar
    """
    # Normalize fitness to [0, 1]
    fitness_min = fitness_values.min()
    fitness_max = fitness_values.max()
    
    if fitness_max - fitness_min < 1e-10:
        # All fitness values are the same, use uniform weights
        weights = torch.ones_like(fitness_values)
    else:
        fitness_norm = (fitness_values - fitness_min) / (fitness_max - fitness_min)
        
        # Compute weights: higher fitness -> higher weight
        # Use exponential weighting to emphasize top solutions
        weights = torch.exp(2.0 * fitness_norm)
    
    # Normalize weights to sum to batch size (for consistent scale with standard MSE)
    weights = weights / weights.mean()
    
    # Weighted MSE
    squared_errors = (predictions - targets) ** 2
    weighted_loss = (weights * squared_errors).mean()
    
    return weighted_loss


def learn_discrete_backdrive_weighted_mse(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Discrete Backdrive model with fitness-weighted MSE loss

    Trains a neural network to predict fitness from discrete solutions,
    using fitness-weighted MSE to focus on high-fitness regions.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, 1)
    cardinality : np.ndarray
        Cardinality of each variable [n_vars]
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive):
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
    # Keep original fitness for weighting
    y_orig = torch.FloatTensor(fitness_1d).unsqueeze(1)

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
        y_train_orig = y_orig[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        y_val_orig = y_orig[val_indices]
    else:
        X_train = X
        y_train = y
        y_train_orig = y_orig
        X_val = None
        y_val = None
        y_val_orig = None

    # Create network
    network = DiscreteBackdriveNet(
        n_vars, cardinality, hidden_layers, use_embeddings, embedding_dim, dropout
    )

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
            batch_y_orig = y_train_orig[idx]

            # Forward pass
            pred = network(batch_x)
            
            # Fitness-weighted MSE loss
            loss = fitness_weighted_mse_loss(pred, batch_y, batch_y_orig)

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
                val_loss = fitness_weighted_mse_loss(val_pred, y_val, y_val_orig).item()
            network.train()

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")
        else:
            if (epoch + 1) % 20 == 0:
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
        'type': 'discrete_backdrive_weighted_mse'
    }

    return model


def learn_binary_backdrive_weighted_mse(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary Backdrive model with fitness-weighted MSE
    
    Simplified interface for binary problems.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive_weighted_mse)

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

    return learn_discrete_backdrive_weighted_mse(population, fitness, cardinality, params)
