"""
GAN Model Learning for Continuous EDAs

This module provides learning algorithms for Generative Adversarial Network (GAN) based
probabilistic models used in continuous optimization. Implementation based on the paper:
"Generative Adversarial Networks in Estimation of Distribution Algorithms for
Combinatorial Optimization" (Probst, 2016).

The GAN consists of two neural networks:
1. Generator G: Maps random noise z to data samples
2. Discriminator D: Classifies samples as real or generated

The networks are trained adversarially:
- D tries to distinguish real data from generated samples
- G tries to fool D by generating realistic samples
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class GANGenerator(nn.Module):
    """
    Generator network for GAN.

    Maps random noise z from latent space to data samples.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = None):
        super(GANGenerator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer with sigmoid activation for normalized data
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.generator = nn.Sequential(*layers)

    def forward(self, z):
        return self.generator(z)


class GANDiscriminator(nn.Module):
    """
    Discriminator network for GAN.

    Classifies input samples as real (from data) or fake (from generator).
    Outputs probability that input is real.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super(GANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: probability that input is real
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


def learn_gan(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a GAN model from selected population.

    The GAN is trained using alternating optimization:
    1. Train discriminator to distinguish real from generated samples
    2. Train generator to fool the discriminator

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in basic GAN, kept for API consistency)
    params : dict, optional
        Training parameters containing:
        - 'latent_dim': dimension of latent space (default: max(2, n_vars // 2))
        - 'hidden_dims_g': list of hidden layer dimensions for generator (default: [32, 64])
        - 'hidden_dims_d': list of hidden layer dimensions for discriminator (default: [64, 32])
        - 'epochs': number of training epochs (default: 100)
        - 'batch_size': batch size for training (default: 32)
        - 'learning_rate': learning rate for both networks (default: 0.0002)
        - 'beta1': Adam optimizer beta1 parameter (default: 0.5)
        - 'k_discriminator': number of discriminator updates per generator update (default: 1)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'generator_state': generator network state dict
        - 'discriminator_state': discriminator network state dict (for potential retraining)
        - 'latent_dim': latent dimension
        - 'input_dim': input dimension
        - 'ranges': data normalization ranges
        - 'type': 'gan'
    """
    if params is None:
        params = {}

    # Extract parameters
    input_dim = population.shape[1]
    latent_dim = params.get('latent_dim', max(2, input_dim // 2))
    hidden_dims_g = params.get('hidden_dims_g', [32, 64])
    hidden_dims_d = params.get('hidden_dims_d', [64, 32])
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)

    # Normalize data to [0, 1]
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)  # Prevent division by zero

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)  # Ensure values are in [0, 1]

    # Convert to tensors
    real_data = torch.FloatTensor(norm_pop)

    # Create networks
    generator = GANGenerator(latent_dim, input_dim, hidden_dims_g)
    discriminator = GANDiscriminator(input_dim, hidden_dims_d)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers (using Adam as recommended in paper)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    # Labels for real and fake data
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            current_batch_size = len(real_batch)

            # ============================
            # Train Discriminator
            # ============================
            for _ in range(k_discriminator):
                optimizer_d.zero_grad()

                # Train on real data
                labels_real = torch.full((current_batch_size, 1), real_label, dtype=torch.float32)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Train on fake data
                z = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(z)
                labels_fake = torch.full((current_batch_size, 1), fake_label, dtype=torch.float32)
                output_fake = discriminator(fake_batch.detach())  # detach to avoid backprop through G
                loss_d_fake = criterion(output_fake, labels_fake)

                # Total discriminator loss
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

                epoch_loss_d += loss_d.item()

            # ============================
            # Train Generator
            # ============================
            optimizer_g.zero_grad()

            # Generate fake data
            z = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(z)

            # Try to fool discriminator (label fake as real)
            labels_g = torch.full((current_batch_size, 1), real_label, dtype=torch.float32)
            output_g = discriminator(fake_batch)
            loss_g = criterion(output_g, labels_g)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_g += loss_g.item()
            n_batches += 1

    # Return model
    return {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'input_dim': input_dim,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'ranges': ranges,
        'type': 'gan'
    }
