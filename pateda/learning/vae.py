"""
VAE Model Learning for Continuous EDAs

This module provides learning algorithms for Variational Autoencoder (VAE) based
probabilistic models used in continuous optimization. Implementation based on the paper:
"Expanding variational autoencoders for learning and exploiting latent representations
in search distributions" (Garciarena et al., GECCO 2018).

The module implements three variants:
1. VAE: Basic variational autoencoder
2. E-VAE: Extended VAE with fitness predictor
3. CE-VAE: Conditioned Extended VAE with fitness-conditioned sampling
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class VAEEncoder(nn.Module):
    """
    Encoder network for VAE.

    Maps input x to latent distribution parameters (mean and log variance).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        super(VAEEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class VAEDecoder(nn.Module):
    """
    Decoder network for VAE.

    Maps latent variable z back to data space.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = None):
        super(VAEDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Assuming normalized input

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class FitnessPredictor(nn.Module):
    """
    Fitness predictor network for E-VAE and CE-VAE.

    Predicts fitness values from latent representation.
    """

    def __init__(self, latent_dim: int, n_objectives: int = 1, hidden_dim: int = 16):
        super(FitnessPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives)
        )

    def forward(self, z):
        return self.predictor(z)


class ConditionalDecoder(nn.Module):
    """
    Conditional decoder for CE-VAE.

    Takes concatenated [z, fitness] as input and reconstructs x.
    """

    def __init__(self, latent_dim: int, n_objectives: int, output_dim: int, hidden_dims: list = None):
        super(ConditionalDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32]

        layers = []
        prev_dim = latent_dim + n_objectives

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z, fitness):
        z_f = torch.cat([z, fitness], dim=1)
        return self.decoder(z_f)


def reparameterize(mean, logvar):
    """
    Reparameterization trick for VAE.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution

    Returns
    -------
    z : torch.Tensor
        Sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def vae_loss(recon_x, x, mean, logvar):
    """
    VAE loss function combining reconstruction loss and KL divergence.

    Parameters
    ----------
    recon_x : torch.Tensor
        Reconstructed input
    x : torch.Tensor
        Original input
    mean : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution

    Returns
    -------
    loss : torch.Tensor
        Total VAE loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def learn_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a basic VAE model from selected population.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in basic VAE)
    params : dict, optional
        Training parameters containing:
        - 'latent_dim': dimension of latent space (default: 5)
        - 'hidden_dims': list of hidden layer dimensions (default: [32, 16])
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': batch size for training (default: 32)
        - 'learning_rate': learning rate (default: 0.001)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'encoder_state': encoder network state dict
        - 'decoder_state': decoder network state dict
        - 'latent_dim': latent dimension
        - 'input_dim': input dimension
        - 'ranges': data normalization ranges
        - 'type': 'vae'
    """
    if params is None:
        params = {}

    # Extract parameters
    latent_dim = params.get('latent_dim', 5)
    hidden_dims = params.get('hidden_dims', [32, 16])
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)

    # Normalize data
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)  # Prevent division by zero

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)  # Ensure values are in [0, 1]

    # Convert to tensors
    data = torch.FloatTensor(norm_pop)
    input_dim = population.shape[1]

    # Create networks
    encoder = VAEEncoder(input_dim, latent_dim, hidden_dims)
    decoder = VAEDecoder(latent_dim, input_dim, list(reversed(hidden_dims)))

    # Optimizer
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    # Training loop
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon = decoder(z)

            # Compute loss
            loss = vae_loss(recon, batch, mean, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Return model
    return {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'ranges': ranges,
        'type': 'vae'
    }


def learn_extended_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn an Extended VAE (E-VAE) model with fitness predictor.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_vae)

    Returns
    -------
    model : dict
        Dictionary containing VAE components plus:
        - 'predictor_state': fitness predictor network state dict
        - 'type': 'extended_vae'
    """
    if params is None:
        params = {}

    # Extract parameters
    latent_dim = params.get('latent_dim', 5)
    hidden_dims = params.get('hidden_dims', [32, 16])
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)

    # Normalize data
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)

    # Normalize fitness
    if fitness.ndim == 1:
        fitness = fitness.reshape(-1, 1)

    fitness_min = np.min(fitness, axis=0)
    fitness_max = np.max(fitness, axis=0)
    fitness_range = fitness_max - fitness_min
    fitness_range = np.where(fitness_range < 1e-10, 1.0, fitness_range)

    norm_fitness = (fitness - fitness_min) / fitness_range

    # Convert to tensors
    data = torch.FloatTensor(norm_pop)
    fitness_tensor = torch.FloatTensor(norm_fitness)
    input_dim = population.shape[1]
    n_objectives = fitness.shape[1]

    # Create networks
    encoder = VAEEncoder(input_dim, latent_dim, hidden_dims)
    decoder = VAEDecoder(latent_dim, input_dim, list(reversed(hidden_dims)))
    predictor = FitnessPredictor(latent_dim, n_objectives)

    # Optimizer
    parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    # Training loop
    encoder.train()
    decoder.train()
    predictor.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(data))

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_fitness = fitness_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon = decoder(z)
            pred_fitness = predictor(z)

            # Compute losses
            loss_vae = vae_loss(recon, batch, mean, logvar)
            loss_fitness = nn.functional.mse_loss(pred_fitness, batch_fitness)
            loss = loss_vae + loss_fitness

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'predictor_state': predictor.state_dict(),
        'latent_dim': latent_dim,
        'input_dim': input_dim,
        'n_objectives': n_objectives,
        'hidden_dims': hidden_dims,
        'ranges': ranges,
        'fitness_min': fitness_min,
        'fitness_max': fitness_max,
        'fitness_range': fitness_range,
        'type': 'extended_vae'
    }


def learn_conditional_extended_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Conditional Extended VAE (CE-VAE) model.

    This model explicitly conditions the decoder on fitness values,
    allowing for fitness-conditioned sampling.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_vae)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'encoder_state': encoder network state dict
        - 'conditional_decoder_state': conditional decoder network state dict
        - 'predictor_state': fitness predictor network state dict
        - 'type': 'conditional_extended_vae'
    """
    if params is None:
        params = {}

    # Extract parameters
    latent_dim = params.get('latent_dim', 5)
    hidden_dims = params.get('hidden_dims', [32, 16])
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)

    # Normalize data
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)

    # Normalize fitness
    if fitness.ndim == 1:
        fitness = fitness.reshape(-1, 1)

    fitness_min = np.min(fitness, axis=0)
    fitness_max = np.max(fitness, axis=0)
    fitness_range = fitness_max - fitness_min
    fitness_range = np.where(fitness_range < 1e-10, 1.0, fitness_range)

    norm_fitness = (fitness - fitness_min) / fitness_range

    # Convert to tensors
    data = torch.FloatTensor(norm_pop)
    fitness_tensor = torch.FloatTensor(norm_fitness)
    input_dim = population.shape[1]
    n_objectives = fitness.shape[1]

    # Create networks
    encoder = VAEEncoder(input_dim, latent_dim, hidden_dims)
    conditional_decoder = ConditionalDecoder(latent_dim, n_objectives, input_dim, list(reversed(hidden_dims)))
    predictor = FitnessPredictor(latent_dim, n_objectives)

    # Optimizer
    parameters = list(encoder.parameters()) + list(conditional_decoder.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    # Training loop
    encoder.train()
    conditional_decoder.train()
    predictor.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(data))

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_fitness = fitness_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon = conditional_decoder(z, batch_fitness)
            pred_fitness = predictor(z)

            # Compute losses
            recon_loss = nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss_fitness = nn.functional.mse_loss(pred_fitness, batch_fitness)
            loss = recon_loss + kl_loss + loss_fitness

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        'encoder_state': encoder.state_dict(),
        'conditional_decoder_state': conditional_decoder.state_dict(),
        'predictor_state': predictor.state_dict(),
        'latent_dim': latent_dim,
        'input_dim': input_dim,
        'n_objectives': n_objectives,
        'hidden_dims': hidden_dims,
        'ranges': ranges,
        'fitness_min': fitness_min,
        'fitness_max': fitness_max,
        'fitness_range': fitness_range,
        'type': 'conditional_extended_vae'
    }
