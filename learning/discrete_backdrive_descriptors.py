"""
Discrete Backdrive-EDA with Multi-Descriptor Learning

==============================================================================
OVERVIEW
==============================================================================

This module implements a variant of Backdrive-EDA where the neural network
predicts solutions from multiple descriptors rather than directly predicting
fitness from solutions.

The network learns to map: (fitness, mean(x), std(x)) → solution x

During training:
1. Compute descriptors from each solution in the selected population
2. Train network: descriptors → solution

During sampling:
1. Sample target descriptors from the distribution of selected solutions
2. Use network to generate solutions from these target descriptors
3. This gives the model more control over solution characteristics

==============================================================================
APPROACH
==============================================================================

Training Phase:
1. For each solution x in selected population:
   - Compute fitness f(x)
   - Compute mean(x) - average of variable values
   - Compute std(x) - standard deviation of variable values
2. Normalize descriptors and solutions
3. Train MLP: [fitness, mean, std] → solution x

Generation Phase:
1. Sample target descriptors from selected population statistics
2. Feed descriptors through network to generate solutions
3. Post-process to ensure discrete values

==============================================================================
REFERENCES
==============================================================================

Inspired by paper/backdrive/backdrive_models.py learn_backdrive_model()
which uses multiple descriptors (min, max, median, etc.)

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


class DescriptorBackdriveNet(nn.Module):
    """
    Neural network that generates solutions from descriptors
    
    Architecture: [fitness, mean, std] → hidden layers → solution
    
    This is the inverse of traditional backdrive which predicts fitness from solutions.
    """

    def __init__(self, n_vars: int, n_descriptors: int = 3,
                 hidden_layers: list = None, dropout: float = 0.2):
        super(DescriptorBackdriveNet, self).__init__()

        self.n_vars = n_vars
        self.n_descriptors = n_descriptors
        self.dropout = dropout

        if hidden_layers is None:
            # Default: gradually expand from descriptors to solution space
            hidden_layers = [max(16, n_vars // 2), max(32, n_vars)]

        # Build network: descriptors → hidden layers → solution
        layers = []
        prev_dim = n_descriptors

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Output layer: maps to n_vars binary values
        # Use sigmoid to output probabilities for binary variables
        layers.append(nn.Linear(prev_dim, n_vars))

        self.network = nn.Sequential(*layers)

    def forward(self, descriptors):
        """
        Forward pass: descriptors → solution logits
        
        Args:
            descriptors: [batch_size, n_descriptors] tensor of descriptor values
            
        Returns:
            Logits for binary variables [batch_size, n_vars]
        """
        return self.network(descriptors)


def learn_discrete_backdrive_descriptors(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Descriptor-based Backdrive model
    
    Trains a neural network to generate solutions from descriptors.
    
    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, 1)
    params : dict, optional
        Training parameters:
        - 'hidden_layers': list of hidden layer sizes (default: computed)
        - 'list_act_functs': list of activation functions for hidden layers
        - 'epochs': number of training epochs (default: 100)
        - 'batch_size': batch size (default: computed)
        - 'learning_rate': learning rate (default: 0.001)
        - 'weight_decay': L2 regularization (default: 1e-3)
        - 'dropout': dropout rate (default: 0.3)
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
    default_hidden_dims = [max(16, n_vars // 2), max(32, n_vars)]
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    hidden_layers = params.get('hidden_layers', default_hidden_dims)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 1e-3)
    dropout = params.get('dropout', 0.3)
    validation_split = params.get('validation_split', 0.2)
    early_stopping = params.get('early_stopping', True)
    patience = params.get('patience', 10)

    # Compute descriptors for each solution
    # Descriptor 0: fitness
    # Descriptor 1: mean of solution values
    # Descriptor 2: std of solution values
    
    fitness_1d = fitness.flatten()
    
    descriptors = np.zeros((pop_size, 3))
    descriptors[:, 0] = fitness_1d
    descriptors[:, 1] = np.mean(population, axis=1)
    descriptors[:, 2] = np.std(population, axis=1)
    
    # Store statistics for later normalization/denormalization
    descriptor_means = np.mean(descriptors, axis=0)
    descriptor_stds = np.std(descriptors, axis=0)
    descriptor_stds[descriptor_stds < 1e-10] = 1.0  # Avoid division by zero
    
    # Normalize descriptors
    normalized_descriptors = (descriptors - descriptor_means) / descriptor_stds
    
    # Normalize population (solutions) to [0, 1]
    # For binary variables, this is already in [0, 1]
    normalized_population = population.astype(float)
    
    # Convert to tensors
    X = torch.FloatTensor(normalized_descriptors)
    y = torch.FloatTensor(normalized_population)

    # Split into training and validation
    n_samples = len(X)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    if n_val > 0:
        # Random split
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

    # Create network
    network = DescriptorBackdriveNet(
        n_vars, n_descriptors=3, hidden_layers=hidden_layers, dropout=dropout
    )

    # Loss and optimizer
    # Use MSE loss for regression to binary values
    criterion = nn.MSELoss()
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
            loss = criterion(pred, batch_y)

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
                val_loss = criterion(val_pred, y_val).item()
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
            # Print progress (no validation)
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}")

    # Return model
    model = {
        'network_state': network.state_dict(),
        'n_vars': n_vars,
        'n_descriptors': 3,
        'hidden_layers': hidden_layers,
        'descriptor_stats': (descriptor_means, descriptor_stds),
        'type': 'discrete_backdrive_descriptors'
    }

    return model


def learn_binary_backdrive_descriptors(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary Backdrive Descriptors model (simplified interface)
    
    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive_descriptors)
        
    Returns
    -------
    model : dict
        Model dictionary
    """
    return learn_discrete_backdrive_descriptors(population, fitness, params)
