"""
Discrete VAE Model Learning for Binary/Discrete EDAs

==============================================================================
OVERVIEW
==============================================================================

This module provides Variational Autoencoder (VAE) implementations specifically
designed for discrete/binary optimization problems. Unlike continuous VAEs, these
use discrete representations and appropriate techniques for handling discrete
variables through neural networks.

The module implements:
1. **Binary VAE**: Uses Bernoulli distributions for binary variables
2. **Categorical VAE**: Uses Gumbel-Softmax for categorical variables
3. **Extended variants**: E-VAE and CE-VAE for fitness-guided generation

==============================================================================
KEY TECHNIQUES FOR DISCRETE VARIABLES
==============================================================================

Handling discrete variables in neural networks requires special techniques:

1. **Gumbel-Softmax (Categorical Variables)**:
   - Continuous relaxation of categorical sampling
   - Allows backpropagation through sampling operation
   - Temperature parameter controls discreteness
   - References: Jang et al. (2016), Maddison et al. (2016)

2. **Straight-Through Estimator**:
   - Forward pass uses hard (discrete) samples
   - Backward pass uses soft (continuous) gradients
   - Simpler alternative to Gumbel-Softmax

3. **Binary Bernoulli Sampling**:
   - For binary variables, use Bernoulli distribution
   - Sigmoid outputs represent probabilities
   - Can use Gumbel-Softmax or straight-through

==============================================================================
ARCHITECTURE
==============================================================================

For Binary Variables:
- Encoder: binary input → hidden → (μ, log σ²)  [latent parameters]
- Decoder: latent z → hidden → binary probs (sigmoid)
- Loss: Binary Cross-Entropy + KL Divergence

For Categorical Variables:
- Encoder: one-hot input → hidden → (μ, log σ²)
- Decoder: latent z → hidden → category logits (Gumbel-Softmax)
- Loss: Categorical Cross-Entropy + KL Divergence

==============================================================================
USAGE CONSIDERATIONS
==============================================================================

When to use Discrete VAE-EDA:
- Binary or categorical optimization problems
- Problems where latent structure might exist
- Medium to large population sizes (>50)
- When GPU acceleration is available

Advantages:
- Can learn complex dependencies
- Fast sampling after training
- GPU parallelization

Disadvantages:
- Requires hyperparameter tuning
- Training overhead per generation
- Less interpretable than traditional EDAs

==============================================================================
REFERENCES
==============================================================================

- Jang, E., Gu, S., & Poole, B. (2016). "Categorical reparameterization with
  Gumbel-Softmax." ICLR 2017.
- Maddison, C.J., Mnih, A., & Teh, Y.W. (2016). "The concrete distribution:
  A continuous relaxation of discrete random variables." ICLR 2017.
- Kusner, M.J., Paige, B., & Hernández-Lobato, J.M. (2017). "Grammar variational
  autoencoder." ICML 2017. [Application to discrete structures]

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
    compute_default_latent_dim,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """
    Sample from Gumbel-Softmax distribution

    Args:
        logits: unnormalized log probabilities [batch_size, n_categories]
        temperature: temperature parameter (lower = more discrete)

    Returns:
        Soft samples from Gumbel-Softmax
    """
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from Gumbel-Softmax distribution

    If hard=True, returns one-hot, but uses soft gradients
    (Straight-Through Estimator)
    """
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        # Straight through estimator
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y

    return y


class BinaryVAEEncoder(nn.Module):
    """
    Encoder for binary VAE

    Takes binary input and outputs latent distribution parameters

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    latent_dim : int
        Dimension of the latent space.
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
        latent_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(BinaryVAEEncoder, self).__init__()

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
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class BinaryVAEDecoder(nn.Module):
    """
    Decoder for binary VAE

    Takes latent code and outputs binary probabilities
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = None):
        super(BinaryVAEDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # No activation here - will use BCE with logits

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class CategoricalVAEDecoder(nn.Module):
    """
    Decoder for categorical VAE using Gumbel-Softmax

    Takes latent code and outputs categorical distributions
    for each variable
    """

    def __init__(self, latent_dim: int, n_vars: int, cardinality: np.ndarray,
                 hidden_dims: list = None):
        super(CategoricalVAEDecoder, self).__init__()

        self.n_vars = n_vars
        self.cardinality = cardinality
        self.total_categories = int(np.sum(cardinality))

        if hidden_dims is None:
            hidden_dims = [64, 128]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.total_categories))

        self.decoder = nn.Sequential(*layers)

        # Cumulative indices for splitting output
        self.cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    def forward(self, z, temperature=1.0, hard=False):
        """
        Forward pass with Gumbel-Softmax

        Args:
            z: latent code [batch_size, latent_dim]
            temperature: Gumbel-Softmax temperature
            hard: if True, use straight-through estimator

        Returns:
            Soft/hard categorical samples [batch_size, total_categories]
        """
        logits = self.decoder(z)

        # Apply Gumbel-Softmax to each variable separately
        outputs = []
        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            end_idx = self.cum_card[i + 1]
            var_logits = logits[:, start_idx:end_idx]
            var_sample = gumbel_softmax(var_logits, temperature, hard=hard)
            outputs.append(var_sample)

        return torch.cat(outputs, dim=1)


class FitnessPredictor(nn.Module):
    """
    Fitness predictor for E-VAE variants
    """

    def __init__(self, latent_dim: int, n_objectives: int = 1, hidden_dim: int = 32):
        super(FitnessPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_objectives)
        )

    def forward(self, z):
        return self.predictor(z)


def reparameterize(mean, logvar):
    """
    Reparameterization trick: z = μ + σ * ε
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence(mean, logvar):
    """
    KL divergence between N(μ, σ²) and N(0, 1)
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)


def learn_binary_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Binary VAE model from selected population

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters:
        - 'latent_dim': latent space dimension (default: max(2, n_vars/50))
        - 'hidden_dims_enc': encoder hidden dims
          (default: computed from n_vars and pop_size)
        - 'hidden_dims_dec': decoder hidden dims
          (default: computed from n_vars and pop_size, reversed)
        - 'list_act_functs_enc': list of activation functions for encoder
        - 'list_act_functs_dec': list of activation functions for decoder
        - 'list_init_functs_enc': list of initialization functions for encoder
        - 'list_init_functs_dec': list of initialization functions for decoder
        - 'epochs': training epochs (default: 100)
        - 'batch_size': batch size (default: max(8, n_vars/50))
        - 'learning_rate': learning rate (default: 0.001)
        - 'beta': KL divergence weight (default: 1.0)
        - 'use_extended': use fitness predictor (E-VAE) (default: False)
        - 'fitness_weight': weight for fitness prediction loss (default: 0.1)

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
    default_latent_dim = compute_default_latent_dim(n_vars)

    # Extract parameters with new defaults
    latent_dim = params.get('latent_dim', default_latent_dim)
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    beta = params.get('beta', 1.0)
    use_extended = params.get('use_extended', False)
    fitness_weight = params.get('fitness_weight', 0.1)

    # Extract activation and initialization function lists
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness.reshape(-1, 1))

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc)
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec)

    if use_extended:
        fitness_predictor = FitnessPredictor(latent_dim, 1, 32)
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(decoder.parameters()) +
            list(fitness_predictor.parameters()),
            lr=learning_rate
        )
    else:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate
        )

    # Training loop
    encoder.train()
    decoder.train()
    if use_extended:
        fitness_predictor.train()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_fit_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_fitness = fitness_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)

            # Reconstruction loss (BCE with logits)
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)

            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()

            # Total loss
            loss = recon_loss + beta * kl_loss

            # Extended VAE: add fitness prediction
            if use_extended:
                pred_fitness = fitness_predictor(z)
                fit_loss = F.mse_loss(pred_fitness, batch_fitness)
                loss = loss + fitness_weight * fit_loss
                epoch_fit_loss += fit_loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            if use_extended:
                avg_fit = epoch_fit_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Fit={avg_fit:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'type': 'binary_evae' if use_extended else 'binary_vae'
    }

    if use_extended:
        model['fitness_predictor_state'] = fitness_predictor.state_dict()

    return model


def learn_categorical_vae(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Categorical VAE model with Gumbel-Softmax

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values
    fitness : np.ndarray
        Fitness values
    cardinality : np.ndarray
        Cardinality of each variable
    params : dict, optional
        Training parameters (same as binary_vae, plus):
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

    n_vars = population.shape[1]
    total_categories = int(np.sum(cardinality))

    # Extract parameters
    latent_dim = params.get('latent_dim', max(2, n_vars // 4))
    hidden_dims_enc = params.get('hidden_dims_enc', [128, 64])
    hidden_dims_dec = params.get('hidden_dims_dec', [64, 128])
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.001)
    beta = params.get('beta', 1.0)
    temperature = params.get('temperature', 1.0)
    temperature_decay = params.get('temperature_decay', 0.99)
    min_temperature = params.get('min_temperature', 0.5)

    # Convert population to one-hot encoding
    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)
    one_hot = np.zeros((len(population), total_categories))
    for i in range(n_vars):
        for j in range(len(population)):
            value = int(population[j, i])
            one_hot[j, cum_card[i] + value] = 1.0

    data = torch.FloatTensor(one_hot)

    # Create networks
    encoder = BinaryVAEEncoder(total_categories, latent_dim, hidden_dims_enc)
    decoder = CategoricalVAEDecoder(latent_dim, n_vars, cardinality, hidden_dims_dec)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()

    current_temp = temperature

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon = decoder(z, temperature=current_temp, hard=False)

            # Reconstruction loss (categorical cross-entropy)
            recon_loss = F.binary_cross_entropy(recon, batch, reduction='sum') / len(batch)

            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()

            # Total loss
            loss = recon_loss + beta * kl_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1

        # Decay temperature
        current_temp = max(min_temperature, current_temp * temperature_decay)

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Temp={current_temp:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'cardinality': cardinality,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'temperature': current_temp,
        'type': 'categorical_vae'
    }

    return model
