"""
DAE Model Learning for Discrete EDAs

This module provides learning algorithms for Denoising Autoencoder (DAE) based
probabilistic models used in combinatorial optimization. Implementation based on the paper:
"Denoising Autoencoders for Fast Combinatorial Black Box Optimization"
(Willemsen & Dockhorn, 2022).

The module implements:
1. DAE: Denoising autoencoder with corruption process
2. Iterative refinement sampling for combinatorial optimization
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for binary/discrete variables.

    The DAE learns to reconstruct clean inputs from corrupted versions,
    enabling it to learn robust representations for combinatorial optimization.
    """

    def __init__(self, input_dim: int, hidden_dim: int = None):
        """
        Initialize Denoising Autoencoder.

        Parameters
        ----------
        input_dim : int
            Dimension of input (total number of binary variables)
        hidden_dim : int, optional
            Dimension of hidden layer (default: input_dim // 2)
        """
        super(DenoisingAutoencoder, self).__init__()

        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 10)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder: x -> h
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Decoder: h -> z
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.

        h = sigmoid(x * W + b_h)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        h : torch.Tensor
            Hidden representation
        """
        return torch.sigmoid(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to reconstruction.

        z = sigmoid(h * W' + b_z)

        Parameters
        ----------
        h : torch.Tensor
            Hidden representation

        Returns
        -------
        z : torch.Tensor
            Reconstructed output
        """
        return torch.sigmoid(self.decoder(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed output
        """
        h = self.encode(x)
        z = self.decode(h)
        return z


def corrupt_binary(x: torch.Tensor, corruption_level: float = 0.1) -> torch.Tensor:
    """
    Apply salt & pepper noise corruption to binary inputs.

    Randomly flips bits with probability corruption_level.

    Parameters
    ----------
    x : torch.Tensor
        Clean binary input
    corruption_level : float
        Probability of flipping each bit (default: 0.1)

    Returns
    -------
    x_corrupted : torch.Tensor
        Corrupted input
    """
    # Create corruption mask
    mask = torch.rand_like(x) < corruption_level

    # Flip values where mask is True
    x_corrupted = x.clone()
    x_corrupted[mask] = 1 - x_corrupted[mask]

    return x_corrupted


def learn_dae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Denoising Autoencoder model from selected population.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with binary values {0, 1}
    fitness : np.ndarray
        Fitness values (not directly used in basic DAE training)
    params : dict, optional
        Training parameters containing:
        - 'hidden_dim': dimension of hidden layer (default: n_vars // 2)
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': mini-batch size (default: 32)
        - 'learning_rate': learning rate (default: 0.001)
        - 'corruption_level': noise corruption probability (default: 0.1)
        - 'loss_type': 'bce' or 'mse' (default: 'bce')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'dae_state': DAE network state dict
        - 'input_dim': input dimension
        - 'hidden_dim': hidden dimension
        - 'type': 'dae'
    """
    if params is None:
        params = {}

    n_vars = population.shape[1]

    # Extract parameters
    hidden_dim = params.get('hidden_dim', max(n_vars // 2, 10))
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)
    corruption_level = params.get('corruption_level', 0.1)
    loss_type = params.get('loss_type', 'bce')

    # Convert to tensors (assuming binary input)
    data = torch.FloatTensor(population)

    # Create DAE
    dae = DenoisingAutoencoder(n_vars, hidden_dim)

    # Optimizer
    optimizer = torch.optim.Adam(dae.parameters(), lr=learning_rate)

    # Loss function
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Training loop
    dae.train()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Corrupt input
            corrupted_batch = corrupt_binary(batch, corruption_level)

            # Forward pass
            reconstruction = dae(corrupted_batch)

            # Compute loss (reconstruct original, not corrupted)
            loss = criterion(reconstruction, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Return model
    return {
        'dae_state': dae.state_dict(),
        'input_dim': n_vars,
        'hidden_dim': hidden_dim,
        'type': 'dae'
    }


class MultiLayerDAE(nn.Module):
    """
    Multi-layer Denoising Autoencoder with deeper architecture.

    Useful for higher-dimensional problems.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None):
        """
        Initialize multi-layer DAE.

        Parameters
        ----------
        input_dim : int
            Dimension of input
        hidden_dims : list, optional
            List of hidden layer dimensions (default: [input_dim//2])
        """
        super(MultiLayerDAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, 10)]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.Sigmoid())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                next_dim = input_dim
            else:
                next_dim = hidden_dims[i - 1]

            decoder_layers.append(nn.Linear(hidden_dims[i], next_dim))
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-layer DAE."""
        h = self.encoder(x)
        z = self.decoder(h)
        return z


def learn_multilayer_dae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a multi-layer Denoising Autoencoder model.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with binary values
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_dae, plus):
        - 'hidden_dims': list of hidden layer dimensions

    Returns
    -------
    model : dict
        Dictionary containing model state
    """
    if params is None:
        params = {}

    n_vars = population.shape[1]

    # Extract parameters
    hidden_dims = params.get('hidden_dims', [max(n_vars // 2, 10)])
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)
    corruption_level = params.get('corruption_level', 0.1)
    loss_type = params.get('loss_type', 'bce')

    # Convert to tensors
    data = torch.FloatTensor(population)

    # Create multi-layer DAE
    dae = MultiLayerDAE(n_vars, hidden_dims)

    # Optimizer
    optimizer = torch.optim.Adam(dae.parameters(), lr=learning_rate)

    # Loss function
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Training loop
    dae.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(data))

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Corrupt input
            corrupted_batch = corrupt_binary(batch, corruption_level)

            # Forward pass
            reconstruction = dae(corrupted_batch)

            # Compute loss
            loss = criterion(reconstruction, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Return model
    return {
        'dae_state': dae.state_dict(),
        'input_dim': n_vars,
        'hidden_dims': hidden_dims,
        'type': 'multilayer_dae'
    }
