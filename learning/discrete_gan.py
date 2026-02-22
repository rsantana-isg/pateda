"""
Discrete GAN Model Learning for Binary/Discrete EDAs

==============================================================================
OVERVIEW
==============================================================================

This module provides Generative Adversarial Network (GAN) implementations
specifically designed for discrete/binary optimization problems. It uses
Gumbel-Softmax and other techniques to enable gradient-based training with
discrete variables.

The module implements:
1. **Binary GAN**: For binary (0/1) variables
2. **Categorical GAN**: For multi-valued discrete variables

==============================================================================
KEY CHALLENGES WITH DISCRETE GANs
==============================================================================

GANs were originally designed for continuous data (e.g., images). Applying
them to discrete optimization presents challenges:

1. **Non-differentiable Sampling**: Discrete sampling breaks gradients
   - Solution: Gumbel-Softmax relaxation
   - Solution: Straight-Through Estimator

2. **Mode Collapse**: Generator produces limited diversity
   - Mitigation: Careful learning rate tuning
   - Mitigation: Feature matching loss

3. **Training Instability**: Discriminator/generator balance
   - Mitigation: Separate learning rates
   - Mitigation: Gradient penalty (Wasserstein GAN)

==============================================================================
ARCHITECTURE
==============================================================================

Binary GAN:
- Generator: noise z → hidden → binary probs (sigmoid)
- Discriminator: binary input → hidden → real/fake score
- Loss: Binary Cross-Entropy (adversarial)

Categorical GAN:
- Generator: noise z → hidden → category logits (Gumbel-Softmax)
- Discriminator: categorical input → hidden → real/fake score
- Loss: BCE or Wasserstein loss

==============================================================================
USAGE NOTES
==============================================================================

Performance Considerations:
- According to Santana (2017), GANs "did NOT produce competitive results"
  in EDAs compared to traditional methods
- More successful neural alternatives: VAEs, RBMs, Autoencoders
- Use GANs for research/exploration rather than production

When to try Discrete GAN-EDA:
- Exploratory research on neural models
- When traditional EDAs plateau
- Large population sizes (>100)
- GPU resources available

When NOT to use:
- Limited computational budget
- Small populations
- When interpretability matters
- Production optimization tasks

==============================================================================
REFERENCES
==============================================================================

- Probst, M. (2015). "Generative adversarial networks in estimation of
  distribution algorithms." arXiv:1509.09235.
- Yu, L., Zhang, W., Wang, J., & Yu, Y. (2017). "SeqGAN: Sequence generative
  adversarial nets with policy gradient." AAAI 2017. [Discrete sequences]
- Kusner, M.J., & Hernández-Lobato, J.M. (2016). "GANs for sequences of
  discrete elements with the Gumbel-softmax distribution." arXiv:1611.04051.

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Gumbel-Softmax sampling

    Args:
        logits: unnormalized log probabilities
        temperature: temperature parameter
        hard: if True, use straight-through estimator

    Returns:
        Soft or hard categorical samples
    """
    y = logits + sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)

    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y

    return y


class BinaryGANGenerator(nn.Module):
    """
    Generator for Binary GAN

    Maps random noise to binary probabilities

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    output_dim : int
        Dimension of the output (number of binary variables).
    hidden_dims : list, optional
        List of hidden layer dimensions.
    list_act_functs : list, optional
        List of activation functions, one per hidden layer.
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(BinaryGANGenerator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = latent_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output probabilities

        self.generator = nn.Sequential(*layers)

    def forward(self, z, hard_sample=False):
        """
        Generate binary samples

        Args:
            z: latent noise [batch_size, latent_dim]
            hard_sample: if True, return hard 0/1 samples

        Returns:
            Binary probabilities or hard samples
        """
        probs = self.generator(z)

        if hard_sample:
            # Use Gumbel-Softmax trick for binary
            # Reshape to [batch, n_vars, 2] for binary choice
            logits = torch.stack([1 - probs, probs], dim=-1)
            samples = gumbel_softmax(logits, temperature=0.5, hard=True)
            return samples[..., 1]  # Return the "1" probability

        return probs


class BinaryGANDiscriminator(nn.Module):
    """
    Discriminator for Binary GAN

    Classifies inputs as real or fake

    Parameters
    ----------
    input_dim : int
        Dimension of the input (number of binary variables).
    hidden_dims : list, optional
        List of hidden layer dimensions.
    list_act_functs : list, optional
        List of activation functions, one per hidden layer.
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(BinaryGANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['leaky_relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


class CategoricalGANGenerator(nn.Module):
    """
    Generator for Categorical GAN using Gumbel-Softmax

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    n_vars : int
        Number of categorical variables.
    cardinality : np.ndarray
        Cardinality of each variable.
    hidden_dims : list, optional
        List of hidden layer dimensions.
    list_act_functs : list, optional
        List of activation functions, one per hidden layer.
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer.
    """

    def __init__(
        self,
        latent_dim: int,
        n_vars: int,
        cardinality: np.ndarray,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(CategoricalGANGenerator, self).__init__()

        self.n_vars = n_vars
        self.cardinality = cardinality
        self.total_categories = int(np.sum(cardinality))

        if hidden_dims is None:
            hidden_dims = [128, 256]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = latent_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.total_categories))

        self.generator = nn.Sequential(*layers)

        # Cumulative indices for splitting output
        self.cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    def forward(self, z, temperature=1.0, hard=False):
        """
        Generate categorical samples

        Args:
            z: latent noise [batch_size, latent_dim]
            temperature: Gumbel-Softmax temperature
            hard: if True, return hard one-hot samples

        Returns:
            Soft or hard categorical samples [batch_size, total_categories]
        """
        logits = self.generator(z)

        # Apply Gumbel-Softmax to each variable separately
        outputs = []
        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            end_idx = self.cum_card[i + 1]
            var_logits = logits[:, start_idx:end_idx]
            var_sample = gumbel_softmax(var_logits, temperature, hard=hard)
            outputs.append(var_sample)

        return torch.cat(outputs, dim=1)


class CategoricalGANDiscriminator(nn.Module):
    """
    Discriminator for Categorical GAN

    Parameters
    ----------
    total_categories : int
        Total number of categories across all variables.
    hidden_dims : list, optional
        List of hidden layer dimensions.
    list_act_functs : list, optional
        List of activation functions, one per hidden layer.
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer.
    """

    def __init__(
        self,
        total_categories: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(CategoricalGANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['leaky_relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = total_categories

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


def learn_binary_gan(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary GAN model from selected population

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values (not directly used but kept for API consistency)
    params : dict, optional
        Training parameters:
        - 'latent_dim': latent space dimension (default: max(10, n_vars // 2))
        - 'hidden_dims_g': generator hidden dims
          (default: computed from n_vars and pop_size)
        - 'hidden_dims_d': discriminator hidden dims
          (default: computed from n_vars and pop_size, reversed)
        - 'list_act_functs_g': list of activation functions for generator
        - 'list_act_functs_d': list of activation functions for discriminator
        - 'list_init_functs_g': list of initialization functions for generator
        - 'list_init_functs_d': list of initialization functions for discriminator
        - 'epochs': training epochs (default: 200)
        - 'batch_size': batch size (default: max(8, n_vars/50))
        - 'learning_rate_g': generator learning rate (default: 0.0002)
        - 'learning_rate_d': discriminator learning rate (default: 0.0002)
        - 'beta1': Adam beta1 (default: 0.5)
        - 'k_discriminator': discriminator updates per generator (default: 1)

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters with new defaults
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)

    # Extract activation and initialization function lists
    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert to tensors (float for gradients)
    real_data = torch.FloatTensor(population)

    # Create networks with configurable activations and initializations
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    discriminator = BinaryGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

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

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Fake samples
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)  # Use soft samples
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake)

                # Combined discriminator loss
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)  # Want D to think it's real
            output_gen = discriminator(fake_batch)
            loss_g = criterion(output_gen, labels_gen)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        # Print progress
        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}")

    # Return model with all configuration
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'type': 'binary_gan'
    }

    return model


def learn_categorical_gan(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Categorical GAN model with Gumbel-Softmax

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values
    cardinality : np.ndarray
        Cardinality of each variable
    params : dict, optional
        Training parameters:
        - 'latent_dim': latent space dimension
        - 'hidden_dims_g': generator hidden dims
        - 'hidden_dims_d': discriminator hidden dims
        - 'list_act_functs_g': list of activation functions for generator
        - 'list_act_functs_d': list of activation functions for discriminator
        - 'list_init_functs_g': list of initialization functions for generator
        - 'list_init_functs_d': list of initialization functions for discriminator
        - 'temperature': Gumbel-Softmax temperature (default: 1.0)
        - 'temperature_decay': decay rate (default: 0.99)
        - 'min_temperature': minimum temperature (default: 0.5)

    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]
    total_categories = int(np.sum(cardinality))

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters with new defaults
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    temperature = params.get('temperature', 1.0)
    temperature_decay = params.get('temperature_decay', 0.99)
    min_temperature = params.get('min_temperature', 0.5)

    # Extract activation and initialization function lists
    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert population to one-hot encoding
    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)
    one_hot = np.zeros((len(population), total_categories))
    for i in range(n_vars):
        for j in range(len(population)):
            value = int(population[j, i])
            one_hot[j, cum_card[i] + value] = 1.0

    real_data = torch.FloatTensor(one_hot)

    # Create networks with configurable activations and initializations
    generator = CategoricalGANGenerator(
        latent_dim, n_vars, cardinality, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    discriminator = CategoricalGANDiscriminator(
        total_categories, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0
    current_temp = temperature

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Fake samples (use soft Gumbel-Softmax for discriminator)
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, temperature=current_temp, hard=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake)

                # Combined discriminator loss
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, temperature=current_temp, hard=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch)
            loss_g = criterion(output_gen, labels_gen)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        # Decay temperature
        current_temp = max(min_temperature, current_temp * temperature_decay)

        # Print progress
        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}, Temp={current_temp:.4f}")

    # Return model with all configuration
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'cardinality': cardinality,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'temperature': current_temp,
        'type': 'categorical_gan'
    }

    return model


# ==============================================================================
# GAN Variant 1: WGAN-GP (Wasserstein GAN with Gradient Penalty)
# ==============================================================================

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Compute gradient penalty for WGAN-GP

    Parameters
    ----------
    discriminator : nn.Module
        Discriminator network
    real_samples : torch.Tensor
        Real samples
    fake_samples : torch.Tensor
        Fake samples

    Returns
    -------
    penalty : torch.Tensor
        Gradient penalty
    """
    batch_size = real_samples.size(0)

    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_samples)

    # Interpolated samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad = True

    # Discriminator output for interpolates
    d_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class BinaryWGANDiscriminator(nn.Module):
    """
    Wasserstein GAN Discriminator (Critic) for Binary data

    Returns real-valued score (not probability)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        dropout: float = 0.5
    ):
        super(BinaryWGANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['leaky_relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid for WGAN - outputs real-valued score

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


def learn_binary_gan_wgan_gp(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN with Wasserstein loss and Gradient Penalty (WGAN-GP)

    WGAN-GP provides more stable training and addresses mode collapse by:
    - Using Wasserstein distance instead of JS divergence
    - Enforcing Lipschitz constraint via gradient penalty

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan, plus):
        - 'gradient_penalty_weight': weight for gradient penalty (default: 10.0)
        - 'n_critic': number of critic iterations per generator (default: 5)
        - 'dropout': dropout rate for discriminator (default: 0.5)

    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0001)
    learning_rate_d = params.get('learning_rate_d', 0.0001)
    gradient_penalty_weight = params.get('gradient_penalty_weight', 10.0)
    n_critic = params.get('n_critic', 5)
    dropout = params.get('dropout', 0.5)

    # Extract activation and initialization functions
    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert to tensors
    real_data = torch.FloatTensor(population)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    discriminator = BinaryWGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d,
        dropout=dropout
    )

    # Optimizers (RMSprop commonly used for WGAN, but Adam works too)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(0.5, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

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

            # Train Discriminator (Critic)
            for _ in range(n_critic):
                discriminator.zero_grad()

                # Real samples
                output_real = discriminator(real_batch)

                # Fake samples
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)
                output_fake = discriminator(fake_batch.detach())

                # Wasserstein loss
                loss_d = -torch.mean(output_real) + torch.mean(output_fake)

                # Gradient penalty
                gp = compute_gradient_penalty(discriminator, real_batch, fake_batch.detach())
                loss_d_total = loss_d + gradient_penalty_weight * gp

                loss_d_total.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            output_gen = discriminator(fake_batch)

            # Generator wants to maximize discriminator output for fake samples
            loss_g = -torch.mean(output_gen)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d_total.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        # Print progress
        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}")

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'type': 'binary_gan_wgan_gp'
    }

    return model


# ==============================================================================
# GAN Variant 2: Cond-Fit-GAN (Conditional Fitness GAN)
# ==============================================================================

class ConditionalBinaryGANGenerator(nn.Module):
    """
    Conditional Generator that takes fitness percentile as additional input
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        condition_dim: int = 1,  # Fitness percentile
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(ConditionalBinaryGANGenerator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = latent_dim + condition_dim  # Concatenate condition

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.generator = nn.Sequential(*layers)

    def forward(self, z, condition, hard_sample=False):
        """
        Generate conditioned on fitness percentile

        Args:
            z: latent noise [batch_size, latent_dim]
            condition: fitness percentile [batch_size, 1] in range [0, 1]
            hard_sample: if True, return hard 0/1 samples
        """
        # Concatenate condition
        x = torch.cat([z, condition], dim=1)
        probs = self.generator(x)

        if hard_sample:
            logits = torch.stack([1 - probs, probs], dim=-1)
            samples = gumbel_softmax(logits, temperature=0.5, hard=True)
            return samples[..., 1]

        return probs


def learn_binary_gan_cond_fit(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN conditioned on fitness percentile

    This variant allows the EDA to "ask" for solutions at specific fitness levels
    (e.g., "generate near-optimal" vs "exploratory" solutions)

    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan)

    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Normalize fitness to percentiles [0, 1]
    fitness_1d = fitness.flatten()
    fitness_ranks = np.argsort(np.argsort(fitness_1d))  # Rank
    fitness_percentiles = fitness_ranks / (len(fitness_1d) - 1)  # Normalize to [0, 1]

    # Convert to tensors
    real_data = torch.FloatTensor(population)
    fitness_cond = torch.FloatTensor(fitness_percentiles).unsqueeze(1)

    # Create networks
    generator = ConditionalBinaryGANGenerator(
        latent_dim, n_vars, condition_dim=1,
        hidden_dims=hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )

    # Discriminator remains unchanged (doesn't need condition)
    discriminator = BinaryGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )

    # Modify discriminator dropout
    for layer in discriminator.discriminator:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            cond_batch = fitness_cond[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Fake samples with condition
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, cond_batch, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, cond_batch, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch)
            loss_g = criterion(output_gen, labels_gen)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}")

    # Store fitness statistics for sampling
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'fitness_stats': (np.mean(fitness_1d), np.std(fitness_1d)),
        'type': 'binary_gan_cond_fit'
    }

    return model


# ==============================================================================
# GAN Variant 3: Aux-GAN (Auxiliary Classifier GAN with Fitness Prediction)
# ==============================================================================

class BinaryAuxGANDiscriminator(nn.Module):
    """
    Auxiliary Discriminator with dual heads:
    1. Real/Fake classification
    2. Fitness prediction (regression)
    
    This merges GAN distribution learning with fitness landscape approximation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        dropout: float = 0.5
    ):
        super(BinaryAuxGANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['leaky_relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            shared_layers.append(linear)
            shared_layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Real/Fake classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )

        # Fitness regression head
        self.fitness_head = nn.Linear(prev_dim, 1)

    def forward(self, x, return_fitness=False):
        """
        Forward pass
        
        Args:
            x: input samples
            return_fitness: if True, return both classification and fitness prediction
            
        Returns:
            If return_fitness=False: real/fake score
            If return_fitness=True: (real/fake score, fitness prediction)
        """
        features = self.shared(x)
        real_fake = self.classifier_head(features)
        
        if return_fitness:
            fitness_pred = self.fitness_head(features)
            return real_fake, fitness_pred
        
        return real_fake


def learn_binary_gan_aux(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN with Auxiliary Fitness Prediction Head
    
    The discriminator learns both:
    1. Real vs Fake classification (standard GAN objective)
    2. Fitness prediction (surrogate modeling objective)
    
    This combines density estimation and fitness landscape learning.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan, plus):
        - 'fitness_weight': weight for fitness prediction loss (default: 0.1)
        - 'dropout': dropout rate (default: 0.5)
        
    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    fitness_weight = params.get('fitness_weight', 0.1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Normalize fitness
    fitness_1d = fitness.flatten()
    fitness_mean = np.mean(fitness_1d)
    fitness_std = np.std(fitness_1d) + 1e-8
    fitness_normalized = (fitness_1d - fitness_mean) / fitness_std

    # Convert to tensors
    real_data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness_normalized).unsqueeze(1)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    
    discriminator = BinaryAuxGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d,
        dropout=dropout
    )

    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_fitness = nn.MSELoss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        epoch_loss_fitness = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            fitness_batch = fitness_tensor[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator with auxiliary fitness head
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples - get both outputs
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real, fitness_pred_real = discriminator(real_batch, return_fitness=True)
                loss_d_real = criterion_gan(output_real, labels_real)
                loss_fitness = criterion_fitness(fitness_pred_real, fitness_batch)

                # Fake samples - only classification
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach(), return_fitness=False)
                loss_d_fake = criterion_gan(output_fake, labels_fake)

                # Combined discriminator loss
                loss_d = loss_d_real + loss_d_fake + fitness_weight * loss_fitness
                loss_d.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch, return_fitness=False)
            loss_g = criterion_gan(output_gen, labels_gen)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_loss_fitness += loss_fitness.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            avg_loss_fitness = epoch_loss_fitness / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}, Fitness_loss={avg_loss_fitness:.4f}")

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'fitness_stats': (fitness_mean, fitness_std),
        'dropout': dropout,
        'type': 'binary_gan_aux'
    }

    return model


# ==============================================================================
# GAN Variant 4: Repulsion-GAN (Diversity-Promoting GAN)
# ==============================================================================

def compute_hamming_diversity_loss(samples):
    """
    Compute diversity loss as negative average Hamming distance
    
    Parameters
    ----------
    samples : torch.Tensor
        Binary samples [batch_size, n_vars]
        
    Returns
    -------
    loss : torch.Tensor
        Diversity loss (to be minimized = encourages diversity)
    """
    # Compute pairwise Hamming distances
    # samples: [batch_size, n_vars]
    batch_size = samples.size(0)
    
    # Expand dimensions for broadcasting
    s1 = samples.unsqueeze(1)  # [batch_size, 1, n_vars]
    s2 = samples.unsqueeze(0)  # [1, batch_size, n_vars]
    
    # Hamming distance: count of differing bits
    # Use absolute difference (works for soft probabilities too)
    diff = torch.abs(s1 - s2)  # [batch_size, batch_size, n_vars]
    hamming_dist = diff.sum(dim=2)  # [batch_size, batch_size]
    
    # Average Hamming distance (excluding diagonal)
    mask = 1 - torch.eye(batch_size)
    avg_hamming = (hamming_dist * mask).sum() / (batch_size * (batch_size - 1))
    
    # Return negative (we want to maximize diversity = minimize negative diversity)
    return -avg_hamming


def learn_binary_gan_repulsion(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN with Repulsion Loss for Diversity
    
    Adds a diversity-promoting term to the generator loss that encourages
    generated samples within a batch to be different from each other.
    This helps combat mode collapse.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan, plus):
        - 'repulsion_weight': weight for diversity loss (default: 0.1)
        - 'dropout': dropout rate (default: 0.5)
        
    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    repulsion_weight = params.get('repulsion_weight', 0.1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert to tensors
    real_data = torch.FloatTensor(population)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    
    discriminator = BinaryGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )
    
    # Modify discriminator dropout
    for layer in discriminator.discriminator:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        epoch_loss_diversity = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            current_batch_size = len(real_batch)

            # Skip if batch size is 1 (can't compute diversity)
            if current_batch_size < 2:
                continue

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Fake samples
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator with diversity loss
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch)
            loss_g_gan = criterion(output_gen, labels_gen)
            
            # Diversity loss (repulsion)
            loss_diversity = compute_hamming_diversity_loss(fake_batch)
            
            # Combined generator loss
            loss_g = loss_g_gan + repulsion_weight * loss_diversity

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g_gan.item()
            epoch_loss_diversity += loss_diversity.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            avg_loss_diversity = epoch_loss_diversity / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}, Diversity={avg_loss_diversity:.4f}")

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'type': 'binary_gan_repulsion'
    }

    return model


# ==============================================================================
# GAN Variant 5: Weighted-D-GAN (Fitness-Weighted Discriminator)
# ==============================================================================

def learn_binary_gan_weighted_d(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN with Fitness-Weighted Discriminator Loss
    
    Weights the discriminator's loss for real samples based on fitness,
    forcing the GAN to prioritize learning features of high-fitness solutions.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan)
        - 'dropout': dropout rate (default: 0.5)
        
    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Normalize fitness to weights [0, 1]
    fitness_1d = fitness.flatten()
    fitness_min = np.min(fitness_1d)
    fitness_max = np.max(fitness_1d)
    fitness_range = fitness_max - fitness_min
    if fitness_range > 0:
        fitness_weights = (fitness_1d - fitness_min) / fitness_range
    else:
        fitness_weights = np.ones_like(fitness_1d)

    # Convert to tensors
    real_data = torch.FloatTensor(population)
    weights_tensor = torch.FloatTensor(fitness_weights).unsqueeze(1)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    
    discriminator = BinaryGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )
    
    # Modify discriminator dropout
    for layer in discriminator.discriminator:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    # Loss (no reduction, we'll weight manually)
    criterion = nn.BCELoss(reduction='none')
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            weights_batch = weights_tensor[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator with weighted real loss
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples - weighted by fitness
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real_unweighted = criterion(output_real, labels_real)
                loss_d_real = (loss_d_real_unweighted * weights_batch).mean()

                # Fake samples - standard loss
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake).mean()

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch)
            loss_g = criterion(output_gen, labels_gen).mean()

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}")

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'type': 'binary_gan_weighted_d'
    }

    return model


# ==============================================================================
# GAN Variant 6: Statistic-Match GAN (Moment-Matching)
# ==============================================================================

def compute_statistic_matching_loss(fake_samples, real_samples):
    """
    Compute loss for matching first and second moments (mean and std)
    
    Parameters
    ----------
    fake_samples : torch.Tensor
        Generated samples
    real_samples : torch.Tensor
        Real samples from selected population
        
    Returns
    -------
    loss : torch.Tensor
        Statistic matching loss
    """
    # Match mean
    mean_fake = torch.mean(fake_samples, dim=0)
    mean_real = torch.mean(real_samples, dim=0)
    loss_mean = F.mse_loss(mean_fake, mean_real)
    
    # Match std
    std_fake = torch.std(fake_samples, dim=0)
    std_real = torch.std(real_samples, dim=0)
    loss_std = F.mse_loss(std_fake, std_real)
    
    return loss_mean + loss_std


def learn_binary_gan_statistic_match(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Binary GAN with Statistic Matching Loss
    
    Adds explicit statistical constraints to match mean and standard deviation
    of generated batches to real selected population. This acts as a regularizer
    to keep the neural model anchored to known search space properties.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan, plus):
        - 'stat_weight': weight for statistic matching loss (default: 0.1)
        - 'dropout': dropout rate (default: 0.5)
        
    Returns
    -------
    model : dict
        Model dictionary
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    stat_weight = params.get('stat_weight', 0.1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert to tensors
    real_data = torch.FloatTensor(population)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    
    discriminator = BinaryGANDiscriminator(
        n_vars, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )
    
    # Modify discriminator dropout
    for layer in discriminator.discriminator:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_g = 0
        epoch_loss_stat = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real samples
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch)
                loss_d_real = criterion(output_real, labels_real)

                # Fake samples
                noise = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(noise, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach())
                loss_d_fake = criterion(output_fake, labels_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator with statistic matching
            generator.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(noise, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch)
            loss_g_gan = criterion(output_gen, labels_gen)
            
            # Statistic matching loss
            loss_stat = compute_statistic_matching_loss(fake_batch, real_batch)
            
            # Combined generator loss
            loss_g = loss_g_gan + stat_weight * loss_stat

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g_gan.item()
            epoch_loss_stat += loss_stat.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            avg_loss_stat = epoch_loss_stat / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}, Stat_loss={avg_loss_stat:.4f}")

    # Store population statistics
    pop_mean = np.mean(population, axis=0)
    pop_std = np.std(population, axis=0)

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'population_stats': (pop_mean, pop_std),
        'type': 'binary_gan_statistic_match'
    }

    return model


# ==============================================================================
# GAN Variant 7: Hybrid-GAN-VAE (BiGAN with Encoder)
# ==============================================================================

class BinaryGANEncoder(nn.Module):
    """
    Encoder for Hybrid GAN-VAE (BiGAN)
    
    Maps solutions to latent space, enabling:
    - Crossover in latent space
    - Latent space interpolation
    - Solution encoding/reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(BinaryGANEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class BiGANDiscriminator(nn.Module):
    """
    BiGAN Discriminator that takes (x, z) pairs
    
    Discriminates between:
    - Real pairs: (real_x, E(real_x))
    - Fake pairs: (G(z), z)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        dropout: float = 0.5
    ):
        super(BiGANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_hidden = len(hidden_dims)

        # Validate and set defaults
        if list_act_functs is None:
            list_act_functs = ['leaky_relu'] * n_hidden
        if list_init_functs is None:
            list_init_functs = ['default'] * n_hidden

        list_act_functs, list_init_functs = validate_list_params(
            hidden_dims, list_act_functs, list_init_functs
        )

        layers = []
        prev_dim = input_dim + latent_dim  # Concatenate x and z

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x, z):
        """
        Discriminate (x, z) pairs
        
        Args:
            x: solutions [batch_size, input_dim]
            z: latent codes [batch_size, latent_dim]
        """
        xz = torch.cat([x, z], dim=1)
        return self.discriminator(xz)


def learn_binary_gan_hybrid_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Hybrid GAN-VAE (BiGAN) with Encoder
    
    Trains three networks:
    - Generator: z -> x
    - Encoder: x -> z
    - Discriminator: discriminates (x, E(x)) vs (G(z), z)
    
    This enables latent space operations like crossover and interpolation.
    
    Parameters
    ----------
    population : np.ndarray
        Binary population
    fitness : np.ndarray
        Fitness values
    params : dict, optional
        Training parameters (same as learn_binary_gan, plus):
        - 'hidden_dims_e': encoder hidden dims (default: same as generator)
        - 'list_act_functs_e': encoder activations
        - 'list_init_functs_e': encoder initializations
        - 'dropout': dropout rate (default: 0.5)
        
    Returns
    -------
    model : dict
        Model dictionary with encoder, generator, and discriminator
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters
    latent_dim = params.get('latent_dim', max(10, n_vars // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_e = params.get('hidden_dims_e', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 200)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate_g = params.get('learning_rate_g', 0.0002)
    learning_rate_e = params.get('learning_rate_e', 0.0002)
    learning_rate_d = params.get('learning_rate_d', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)
    dropout = params.get('dropout', 0.5)

    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_e = params.get('list_act_functs_e', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_e = params.get('list_init_functs_e', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Convert to tensors
    real_data = torch.FloatTensor(population)

    # Create networks
    generator = BinaryGANGenerator(
        latent_dim, n_vars, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    
    encoder = BinaryGANEncoder(
        n_vars, latent_dim, hidden_dims_e,
        list_act_functs=list_act_functs_e,
        list_init_functs=list_init_functs_e
    )
    
    discriminator = BiGANDiscriminator(
        n_vars, latent_dim, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d,
        dropout=dropout
    )

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g,
                             betas=(beta1, 0.999))
    optimizer_e = optim.Adam(encoder.parameters(), lr=learning_rate_e,
                             betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d,
                             betas=(beta1, 0.999))

    # Training loop
    generator.train()
    encoder.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(len(real_data))

        epoch_loss_d = 0
        epoch_loss_ge = 0
        n_batches = 0

        for i in range(0, len(real_data), batch_size):
            idx = perm[i:i+batch_size]
            real_batch = real_data[idx]
            current_batch_size = len(real_batch)

            # Train Discriminator
            for _ in range(k_discriminator):
                discriminator.zero_grad()

                # Real pairs: (real_x, E(real_x))
                z_encoded = encoder(real_batch)
                labels_real = torch.full((current_batch_size, 1), real_label)
                output_real = discriminator(real_batch, z_encoded)
                loss_d_real = criterion(output_real, labels_real)

                # Fake pairs: (G(z), z)
                z_random = torch.randn(current_batch_size, latent_dim)
                fake_batch = generator(z_random, hard_sample=False)
                labels_fake = torch.full((current_batch_size, 1), fake_label)
                output_fake = discriminator(fake_batch.detach(), z_random)
                loss_d_fake = criterion(output_fake, labels_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # Train Generator and Encoder jointly
            optimizer_g.zero_grad()
            optimizer_e.zero_grad()

            # Generator wants D to think (G(z), z) is real
            z_random = torch.randn(current_batch_size, latent_dim)
            fake_batch = generator(z_random, hard_sample=False)
            labels_gen = torch.full((current_batch_size, 1), real_label)
            output_gen = discriminator(fake_batch, z_random)
            loss_g = criterion(output_gen, labels_gen)

            # Encoder wants D to think (real_x, E(real_x)) is fake
            # (This encourages encoder to map to similar latent distribution as generator)
            z_encoded = encoder(real_batch)
            labels_enc = torch.full((current_batch_size, 1), fake_label)
            output_enc = discriminator(real_batch, z_encoded)
            loss_e = criterion(output_enc, labels_enc)

            loss_ge = loss_g + loss_e
            loss_ge.backward()
            optimizer_g.step()
            optimizer_e.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_ge += loss_ge.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_ge = epoch_loss_ge / n_batches
            #print(f"Epoch {epoch+1}/{epochs}: "
            #      f"D_loss={avg_loss_d:.4f}, GE_loss={avg_loss_ge:.4f}")

    # Return model
    model = {
        'generator_state': generator.state_dict(),
        'encoder_state': encoder.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_e': hidden_dims_e,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_e': list_act_functs_e if list_act_functs_e else ['relu'] * len(hidden_dims_e),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['leaky_relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_e': list_init_functs_e if list_init_functs_e else ['default'] * len(hidden_dims_e),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'type': 'binary_gan_hybrid_vae'
    }

    return model
