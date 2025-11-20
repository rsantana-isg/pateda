"""
Discrete Backdrive-EDA Learning

==============================================================================
OVERVIEW
==============================================================================

This module implements Backdrive-EDA for discrete/binary optimization problems.
The backdrive approach trains a neural network to predict fitness from solutions,
then "back-drives" the network to generate new high-fitness solutions through
network inversion.

For discrete problems, we use:
1. **Embedding layers** for discrete inputs
2. **Gradient-based optimization** in embedding space
3. **Projection back to discrete values** (argmax or Gumbel-Softmax)

==============================================================================
DISCRETE BACKDRIVE APPROACH
==============================================================================

Training Phase:
1. Train MLP: discrete solution x → fitness f(x)
2. For discrete variables, use embeddings or one-hot encoding
3. Learn the fitness landscape approximation

Generation Phase (Network Inversion):
1. Initialize solution in continuous relaxation space
2. Freeze network weights
3. Optimize solution to maximize predicted fitness via backprop
4. Project continuous solution back to discrete values

==============================================================================
KEY TECHNIQUES
==============================================================================

1. **Embedding Layer**:
   - Maps discrete values to continuous vectors
   - Allows gradient flow for optimization
   - Each category gets a learnable embedding

2. **Gumbel-Softmax for Inversion**:
   - Continuous relaxation during optimization
   - Temperature annealing for discretization
   - Maintains differentiability

3. **Projection Methods**:
   - Argmax: Simple, no gradients
   - Gumbel-Softmax: Differentiable, temperature-controlled
   - Straight-through estimator: Hybrid approach

==============================================================================
USAGE CONSIDERATIONS
==============================================================================

When to use Discrete Backdrive-EDA:
- Problems where fitness landscape is smooth in discrete space
- Medium to large population sizes
- When other neural approaches (VAE, GAN) fail

Advantages:
- Directly optimizes for high fitness
- Can leverage problem structure through the fitness surrogate
- Natural for problems with complex fitness landscapes

Disadvantages:
- Quality depends on fitness model accuracy
- May get stuck in local optima
- Network inversion can be computationally expensive
- Less explored than VAE/GAN for discrete problems

==============================================================================
REFERENCES
==============================================================================

- Baluja, S. (2017). "Deep Learning for Explicitly Modeling Optimization
  Landscapes." arXiv:1703.07131.
- Garciarena, U., Santana, R., & Mendiburu, A. (2020). "Envisioning the
  benefits of back-drive in evolutionary algorithms."

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings


class DiscreteBackdriveNet(nn.Module):
    """
    Neural network for discrete backdrive with embeddings

    For binary variables: directly use 0/1 as input
    For categorical variables: use embedding layer
    """

    def __init__(self, n_vars: int, cardinality: np.ndarray,
                 hidden_layers: list = None, use_embeddings: bool = True,
                 embedding_dim: int = 8):
        super(DiscreteBackdriveNet, self).__init__()

        self.n_vars = n_vars
        self.cardinality = cardinality
        self.use_embeddings = use_embeddings

        if hidden_layers is None:
            hidden_layers = [128, 64]

        # Determine input dimension
        if use_embeddings and np.any(cardinality > 2):
            # Use embeddings for non-binary variables
            self.embeddings = nn.ModuleList()
            input_dim = 0
            for i, card in enumerate(cardinality):
                if card > 2:
                    emb = nn.Embedding(int(card), embedding_dim)
                    self.embeddings.append(emb)
                    input_dim += embedding_dim
                else:
                    self.embeddings.append(None)  # No embedding for binary
                    input_dim += 1
            self.embedding_dim = embedding_dim
        else:
            # Use one-hot or binary encoding
            self.embeddings = None
            input_dim = n_vars

        # Hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer (single fitness value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def encode_input(self, x):
        """
        Encode discrete input

        Args:
            x: discrete input [batch_size, n_vars]

        Returns:
            Encoded representation
        """
        if self.embeddings is None:
            return x.float()

        # Use embeddings for categorical variables
        encoded_parts = []
        for i in range(self.n_vars):
            if self.embeddings[i] is not None:
                # Categorical variable with embedding
                emb = self.embeddings[i](x[:, i].long())
                encoded_parts.append(emb)
            else:
                # Binary variable
                encoded_parts.append(x[:, i:i+1].float())

        return torch.cat(encoded_parts, dim=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: discrete input [batch_size, n_vars]

        Returns:
            Predicted fitness [batch_size, 1]
        """
        encoded = self.encode_input(x)
        return self.network(encoded)


def learn_discrete_backdrive(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Discrete Backdrive model

    Trains a neural network to predict fitness from discrete solutions.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, 1)
    cardinality : np.ndarray
        Cardinality of each variable [n_vars]
    params : dict, optional
        Training parameters:
        - 'hidden_layers': list of hidden layer sizes (default: [128, 64])
        - 'use_embeddings': use embedding layers (default: True if max(card)>2)
        - 'embedding_dim': embedding dimension (default: 8)
        - 'epochs': number of training epochs (default: 100)
        - 'batch_size': batch size (default: 32)
        - 'learning_rate': learning rate (default: 0.001)
        - 'weight_decay': L2 regularization (default: 1e-5)
        - 'validation_split': validation fraction (default: 0.2)
        - 'early_stopping': enable early stopping (default: True)
        - 'patience': early stopping patience (default: 10)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'network_state': trained network state dict
        - 'n_vars': number of variables
        - 'cardinality': variable cardinalities
        - 'hidden_layers': hidden layer configuration
        - 'use_embeddings': whether embeddings are used
        - 'embedding_dim': embedding dimension (if used)
        - 'fitness_stats': (mean, std) for normalization
        - 'type': 'discrete_backdrive'
    """
    if params is None:
        params = {}

    n_vars = population.shape[1]

    # Extract parameters
    hidden_layers = params.get('hidden_layers', [128, 64])
    use_embeddings = params.get('use_embeddings', np.max(cardinality) > 2)
    embedding_dim = params.get('embedding_dim', 8)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 1e-5)
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
    network = DiscreteBackdriveNet(
        n_vars, cardinality, hidden_layers, use_embeddings, embedding_dim
    )

    # Loss and optimizer
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
        'cardinality': cardinality.copy(),
        'hidden_layers': hidden_layers,
        'use_embeddings': use_embeddings,
        'embedding_dim': embedding_dim if use_embeddings else None,
        'fitness_stats': (fitness_mean, fitness_std),
        'type': 'discrete_backdrive'
    }

    return model


def learn_binary_backdrive(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary Backdrive model (simplified interface for binary problems)

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_discrete_backdrive)

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

    return learn_discrete_backdrive(population, fitness, cardinality, params)
