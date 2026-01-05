"""
Discrete Backdrive-EDA Learning with Ranking Loss

==============================================================================
OVERVIEW
==============================================================================

This module implements a variant of Discrete Backdrive-EDA that uses
ranking loss instead of standard MSE loss. The key idea is that for
optimization, the relative ranking of solutions is more important than
absolute fitness values.

The ranking loss ensures that if solution x1 has higher fitness than x2,
the network predicts higher fitness for x1 than x2, with a margin.

==============================================================================
KEY DIFFERENCES FROM STANDARD BACKDRIVE
==============================================================================

Loss Function:
- Standard: MSE(pred, target) = mean((pred - target)^2)
- Ranking: MarginRankingLoss enforcing pred(x1) > pred(x2) + margin when f(x1) > f(x2)

The ranking version focuses on learning the relative ordering of solutions,
which is what matters for finding the best solutions during network inversion.

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


def pairwise_ranking_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                          margin: float = 0.1) -> torch.Tensor:
    """
    Compute pairwise ranking loss
    
    For all pairs (i, j) where target[i] > target[j],
    encourage predictions[i] > predictions[j] + margin
    
    Note: This function has O(n²) space complexity for batch size n,
    as it creates matrices for all pairwise comparisons. For very large
    batch sizes (>256), consider using batched pairwise comparison or
    sampling pairs instead of all pairs.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted fitness values [batch_size, 1]
    targets : torch.Tensor
        Target fitness values [batch_size, 1]
    margin : float
        Margin for ranking (default: 0.1)
    
    Returns
    -------
    loss : torch.Tensor
        Ranking loss scalar
    """
    batch_size = predictions.shape[0]
    
    if batch_size < 2:
        # Not enough samples for pairwise comparison, fall back to MSE
        return F.mse_loss(predictions, targets)
    
    # Compute all pairwise comparisons
    # Shape: [batch_size, batch_size]
    # Memory: O(batch_size²)
    pred_diff = predictions - predictions.t()  # pred[i] - pred[j]
    target_diff = targets - targets.t()  # target[i] - target[j]
    
    # For pairs where target[i] > target[j], we want pred[i] > pred[j] + margin
    # loss = max(0, margin - (pred[i] - pred[j])) when target[i] > target[j]
    
    # Create mask for valid pairs (target[i] > target[j])
    valid_pairs = (target_diff > 0).float()
    
    # Compute hinge loss for each pair
    hinge_loss = torch.clamp(margin - pred_diff, min=0)
    
    # Apply mask and average
    masked_loss = hinge_loss * valid_pairs
    n_valid_pairs = valid_pairs.sum()
    
    if n_valid_pairs > 0:
        ranking_loss = masked_loss.sum() / n_valid_pairs
    else:
        # No valid pairs, use MSE
        ranking_loss = F.mse_loss(predictions, targets)
    
    return ranking_loss


def combined_ranking_mse_loss(predictions: torch.Tensor, targets: torch.Tensor,
                               ranking_weight: float = 0.5, margin: float = 0.1) -> torch.Tensor:
    """
    Combine ranking loss with MSE loss
    
    This helps ensure both correct ranking AND reasonable absolute predictions.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted fitness values
    targets : torch.Tensor
        Target fitness values
    ranking_weight : float
        Weight for ranking loss vs MSE (default: 0.5)
    margin : float
        Margin for ranking loss (default: 0.1)
    
    Returns
    -------
    loss : torch.Tensor
        Combined loss scalar
    """
    mse_loss = F.mse_loss(predictions, targets)
    rank_loss = pairwise_ranking_loss(predictions, targets, margin)
    
    total_loss = (1 - ranking_weight) * mse_loss + ranking_weight * rank_loss
    
    return total_loss


def learn_discrete_backdrive_ranking(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Discrete Backdrive model with ranking loss

    Trains a neural network to predict fitness from discrete solutions,
    using ranking loss to focus on learning the relative ordering of solutions.

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
        - 'ranking_weight': weight for ranking loss (default: 0.5)
        - 'ranking_margin': margin for ranking loss (default: 0.1)

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
    ranking_weight = params.get('ranking_weight', 0.5)
    ranking_margin = params.get('ranking_margin', 0.1)

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
            
            # Combined ranking and MSE loss
            loss = combined_ranking_mse_loss(pred, batch_y, 
                                            ranking_weight=ranking_weight,
                                            margin=ranking_margin)

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
                val_loss = combined_ranking_mse_loss(val_pred, y_val,
                                                     ranking_weight=ranking_weight,
                                                     margin=ranking_margin).item()
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
        'list_act_functs': list_act_functs if list_act_functs is not None else ['relu'] * len(hidden_layers),
        'type': 'discrete_backdrive_ranking'
    }

    return model


def learn_binary_backdrive_ranking(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary Backdrive model with ranking loss
    
    Simplified interface for binary problems.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive_ranking)

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

    return learn_discrete_backdrive_ranking(population, fitness, cardinality, params)
