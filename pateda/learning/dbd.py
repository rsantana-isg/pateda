"""
Diffusion-by-Deblending (DbD) Learning for Continuous EDAs

This module implements learning algorithms for alpha-deblending diffusion models
used in continuous optimization. Based on the paper:
"Learning search distributions in estimation of distribution algorithms
with minimalist diffusion models"

The module implements MLP-based alpha-deblending models for learning distributions
over continuous solution vectors in EDAs using PyTorch.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaDeblendingMLP(nn.Module):
    """
    Multi-layer perceptron for alpha-deblending diffusion models.

    This network takes a blended sample x_alpha and blending parameter alpha as input,
    and predicts the difference vector (x1 - x0) that was used to create the blend.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None
    ):
        """
        Initialize alpha-deblending MLP.

        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        hidden_dims : list, optional
            List of hidden layer dimensions (default: [64, 64])
        """
        super(AlphaDeblendingMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # MLP layers
        # Input is x_alpha (input_dim) concatenated with alpha (1)
        layers = []
        prev_dim = input_dim + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (predicts difference x1 - x0)
        layers.append(nn.Linear(prev_dim, input_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_alpha: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Predict difference vector given blended input and alpha parameter.

        Parameters
        ----------
        x_alpha : torch.Tensor
            Blended input of shape (batch_size, input_dim)
        alpha : torch.Tensor
            Blending parameter of shape (batch_size, 1) or (batch_size,)

        Returns
        -------
        predicted_diff : torch.Tensor
            Predicted difference (x1 - x0) of shape (batch_size, input_dim)
        """
        # Ensure alpha has shape (batch_size, 1)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # Concatenate blended sample with alpha parameter
        h = torch.cat([x_alpha, alpha], dim=1)

        # Pass through MLP
        predicted_diff = self.mlp(h)

        return predicted_diff


def create_training_dataset(
    p0: np.ndarray,
    p1: np.ndarray,
    num_alpha_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training dataset for alpha-deblending.

    Parameters
    ----------
    p0 : np.ndarray
        Samples from source distribution of shape (n_samples, n_dims)
    p1 : np.ndarray
        Samples from target distribution of shape (n_samples, n_dims)
    num_alpha_samples : int
        Number of alpha values to sample per (x0, x1) pair

    Returns
    -------
    alpha : torch.Tensor
        Alpha values of shape (n_samples * num_alpha_samples, 1)
    x_alpha : torch.Tensor
        Blended samples of shape (n_samples * num_alpha_samples, n_dims)
    true_diff : torch.Tensor
        True difference vectors of shape (n_samples * num_alpha_samples, n_dims)
    """
    n, m = p0.shape

    # Repeat samples for multiple alpha values
    x0 = np.repeat(p0, num_alpha_samples, axis=0)
    x1 = np.repeat(p1, num_alpha_samples, axis=0)

    # Sample random alpha values
    alpha = np.random.uniform(low=0, high=1, size=n * num_alpha_samples).reshape(-1, 1)

    # Create blended samples
    x_alpha = (1 - alpha) * x0 + alpha * x1

    # Compute true difference
    true_diff = x1 - x0

    # Convert to tensors
    alpha_tensor = torch.FloatTensor(alpha)
    x_alpha_tensor = torch.FloatTensor(x_alpha)
    true_diff_tensor = torch.FloatTensor(true_diff)

    return alpha_tensor, x_alpha_tensor, true_diff_tensor


def learn_dbd(
    p0: np.ndarray,
    p1: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn an alpha-deblending diffusion model.

    Implements Algorithm 2 (Training) from the alpha-deblending paper:
    1. Sample x0 from p0, x1 from p1
    2. Sample alpha ~ Uniform(0, 1)
    3. Compute blended sample x_alpha = (1-alpha)*x0 + alpha*x1
    4. Train network to predict (x1 - x0) from (x_alpha, alpha)

    Parameters
    ----------
    p0 : np.ndarray
        Source distribution samples of shape (n_samples, n_dims)
    p1 : np.ndarray
        Target distribution samples of shape (n_samples, n_dims)
    params : dict, optional
        Training parameters containing:
        - 'num_alpha_samples': number of alpha samples per pair (default: 10)
        - 'hidden_dims': list of hidden layer dimensions (default: [64, 64])
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': batch size for training (default: 32)
        - 'learning_rate': learning rate (default: 1e-3)
        - 'normalize': whether to normalize data (default: True)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'model_state': network state dict
        - 'input_dim': input dimension
        - 'hidden_dims': hidden layer dimensions
        - 'ranges': data normalization ranges (if normalized)
        - 'type': 'dbd'
    """
    if params is None:
        params = {}

    # Extract parameters
    num_alpha_samples = params.get('num_alpha_samples', 10)
    hidden_dims = params.get('hidden_dims', [64, 64])
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 1e-3)
    normalize = params.get('normalize', True)

    # Normalize data if requested
    if normalize:
        ranges = np.vstack([np.min(np.vstack([p0, p1]), axis=0),
                           np.max(np.vstack([p0, p1]), axis=0)])
        range_diff = ranges[1] - ranges[0]
        range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

        p0_norm = (p0 - ranges[0]) / range_diff
        p1_norm = (p1 - ranges[0]) / range_diff
    else:
        p0_norm = p0.copy()
        p1_norm = p1.copy()
        ranges = None

    input_dim = p0.shape[1]

    # Create training dataset
    alpha, x_alpha, true_diff = create_training_dataset(p0_norm, p1_norm, num_alpha_samples)

    # Create model
    model = AlphaDeblendingMLP(input_dim, hidden_dims)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    dataset = torch.utils.data.TensorDataset(x_alpha, alpha, true_diff)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x_alpha, batch_alpha, batch_true_diff in dataloader:
            # Predict difference
            predicted_diff = model(batch_x_alpha, batch_alpha)

            # Compute loss
            loss = F.mse_loss(predicted_diff, batch_true_diff)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Return model
    return {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'ranges': ranges,
        'type': 'dbd'
    }


def find_closest_neighbors(
    source_matrix: np.ndarray,
    reference_matrix: np.ndarray
) -> np.ndarray:
    """
    Find the closest row in reference_matrix for each row in source_matrix
    based on mean squared error distance.

    Parameters
    ----------
    source_matrix : np.ndarray
        Source matrix of shape (N, m)
    reference_matrix : np.ndarray
        Reference matrix of shape (M, m)

    Returns
    -------
    closest_neighbors : np.ndarray
        Closest neighbors matrix of shape (N, m)
    """
    # Compute pairwise squared distances
    distances = np.sum(
        (source_matrix[:, np.newaxis, :] - reference_matrix[np.newaxis, :, :]) ** 2,
        axis=2
    )

    # Find the index of the closest reference row for each source row
    closest_indices = np.argmin(distances, axis=1)

    # Retrieve the closest rows from reference_matrix
    closest_neighbors = reference_matrix[closest_indices]

    return closest_neighbors


def sample_univariate_gaussian(population: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample from univariate Gaussian approximation of population.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars)
    n_samples : int
        Number of samples to generate

    Returns
    -------
    samples : np.ndarray
        Samples from univariate Gaussian of shape (n_samples, n_vars)
    """
    mean = np.mean(population, axis=0)
    std = np.std(population, axis=0)

    # Add small epsilon to avoid zero std
    std = np.maximum(std, 1e-8)

    # Sample from independent Gaussians
    samples = np.random.normal(
        loc=mean[np.newaxis, :],
        scale=std[np.newaxis, :],
        size=(n_samples, len(mean))
    )

    return samples
