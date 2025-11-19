"""
VAE Model Learning for Continuous EDAs

==============================================================================
OVERVIEW
==============================================================================

This module provides learning algorithms for Variational Autoencoder (VAE) based
probabilistic models used in continuous optimization EDAs. As discussed in Santana (2017),
VAEs represent a promising neural model approach for EDAs, offering advantages over GANs
in terms of efficiency and effectiveness.

The module implements three variants:
1. **VAE**: Basic variational autoencoder
2. **E-VAE**: Extended VAE with fitness predictor
3. **CE-VAE**: Conditioned Extended VAE with fitness-conditioned sampling

==============================================================================
VAE ARCHITECTURE
==============================================================================

VAEs learn a probabilistic encoder-decoder model:

1. **Encoder q(z|x)**: Maps input x to latent distribution parameters
   - Input: Data sample x (dimension n_vars)
   - Output: Mean μ(x) and log-variance log σ²(x) of latent distribution
   - Probabilistic: Represents uncertainty in latent representation

2. **Latent Space z**: Low-dimensional probabilistic representation
   - Sampled from q(z|x) = N(μ(x), σ²(x))
   - Dimension typically much smaller than input space
   - Regularized to be close to standard normal N(0, I)

3. **Decoder p(x|z)**: Maps latent variable z back to data space
   - Input: Latent sample z
   - Output: Reconstructed sample x̂
   - Learns to generate samples from latent representation

Training Objective (ELBO):
Maximize Evidence Lower Bound = Reconstruction Loss + KL Divergence
- Reconstruction: Encourages accurate reconstruction of inputs
- KL Divergence: Regularizes latent space to be close to N(0, I)

==============================================================================
RELATIONSHIP TO NEURAL MODELS IN EDAs (Santana, 2017)
==============================================================================

According to Section 6.2.1 of Santana (2017), VAEs have shown more promise in EDAs
compared to GANs:

**VAE Advantages Over GANs**:
1. **Generative model**: VAEs are true generative models (not just adversarial)
2. **Efficient sampling**: Direct sampling from latent space (no iterative process)
3. **Better EDA performance**: Autoencoders in Churchill et al. (2016) were "extremely fast"
   compared to Bayesian network learning, though less efficient in function evaluations

**VAE Variants for EDAs**:

1. **GA-dA** (Churchill et al., 2016): Denoising autoencoder as mutation distribution
   - Outperforms BOA on knapsack problem
   - Outperformed by BOA on hierarchical HIFF function
   - Conclusion: "Performance depends on how the neural model is used"

2. **Deep-Opt-GA** (Baluja, 2017): Deep neural networks (5-10 layers)
   - Acknowledges learning can be time-consuming
   - Evaluated on diverse problems but not compared directly to EDAs

**Key Insight** (from Santana, 2017):
"For the performance of the EDA, the class of model used might be as relevant as the
particular way it is used."

**Latent Representations in Optimization**:
A fundamental question (Santana, 2017, Section 6.2.1): "To what extent can a latent
representation of the optimization problem be efficiently exploited?"

In classification, latent features are crucial. In optimization, the role is less clear.
VAEs provide explicit latent representations that can potentially:
- Capture problem structure in compressed form
- Enable transfer learning across problem instances
- Support fitness prediction and guided sampling

==============================================================================
EXTENDED VAE VARIANTS
==============================================================================

1. **E-VAE** (Extended VAE):
   - Adds fitness predictor network f(z) that predicts fitness from latent code
   - Training optimizes both reconstruction and fitness prediction
   - Enables fitness-aware latent representations
   - Can guide sampling toward high-fitness regions

2. **CE-VAE** (Conditional Extended VAE):
   - Decoder conditioned on both z and desired fitness
   - Allows explicit fitness-conditioned generation: p(x|z, f)
   - Training learns to generate samples with specified fitness levels
   - Most sophisticated variant for optimization

==============================================================================
IMPLEMENTATION DETAILS
==============================================================================

Architecture:
- Encoder: n_vars → hidden[0] → hidden[1] → (μ, log σ²)
- Decoder: latent_dim → reversed_hidden → n_vars
- Fitness Predictor: latent_dim → hidden → n_objectives
- Conditional Decoder: (latent_dim + n_objectives) → hidden → n_vars

Default Configuration:
- Hidden dimensions: [32, 16] for encoder, [16, 32] for decoder
- Latent dimension: 5 (configurable)
- Activation functions: Tanh (encoder), ReLU (decoder), Sigmoid (output)

Training:
- Loss: Reconstruction (MSE) + KL Divergence + Fitness Prediction (for E-VAE/CE-VAE)
- Optimizer: Adam
- Data normalization: [0, 1] using min-max scaling

==============================================================================
USAGE CONSIDERATIONS
==============================================================================

When to use VAEs:
- Continuous optimization problems
- When structure learning from data is needed
- Medium to large population sizes (>50 samples)
- Problems where latent structure might exist

Variant Selection:
- **VAE**: Baseline, no fitness information
- **E-VAE**: When fitness prediction can guide search
- **CE-VAE**: When explicit fitness-conditioned generation is desired

Advantages vs. Traditional EDAs:
- Faster learning than Bayesian networks
- GPU parallelization possible
- Latent space can reveal problem structure
- Natural for transfer learning

Disadvantages vs. Traditional EDAs:
- Requires hyperparameter tuning
- Latent representations less interpretable than PGMs
- May not capture problem structure as explicitly as graphical models
- Computational overhead for neural network training

==============================================================================
REFERENCES
==============================================================================

- Garciarena, U., Santana, R., & Mendiburu, A. (2018). "Expanding variational
  autoencoders for learning and exploiting latent representations in search
  distributions." GECCO 2018.
  [Extended VAE variants for EDAs]

- Churchill, A.W., Sigtia, S., & Fernando, C. (2016). "Learning to generate genotypes
  with neural networks." arXiv:1604.04153.
  [GA-dA: Denoising autoencoders in EDAs]

- Santana, R. (2017). "Gray-box optimization and factorized distribution algorithms:
  where two worlds collide." arXiv:1707.03093, Section 6.2.1.
  [Comprehensive analysis of neural models in EDAs]

- Kingma, D.P., & Welling, M. (2013). "Auto-encoding variational bayes." ICLR 2014.
  [Original VAE paper]

==============================================================================
SEE ALSO
==============================================================================

Related neural model implementations in pateda:
- pateda.learning.dae: Denoising Autoencoders (simpler, efficient)
- pateda.learning.gan: Generative Adversarial Networks (less effective for EDAs)
- pateda.learning.rbm: Restricted Boltzmann Machines (discrete variables)
- pateda.sampling.vae: Sampling from trained VAE models

Traditional alternatives:
- pateda.learning.gaussian: Gaussian-based EDAs (explicit structure)
- pateda.learning.boa: Bayesian Optimization Algorithm (interpretable)
- pateda.learning.gaussian.learn_gmrf_eda: GMRF-EDA (structured Gaussian)
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
