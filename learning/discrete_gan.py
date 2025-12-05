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
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}")

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
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"D_loss={avg_loss_d:.4f}, G_loss={avg_loss_g:.4f}, Temp={current_temp:.4f}")

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
