"""
GAN Model Learning for Continuous EDAs

==============================================================================
OVERVIEW
==============================================================================

This module provides learning algorithms for Generative Adversarial Network (GAN) based
probabilistic models used in continuous optimization EDAs. As discussed in "Gray-box
optimization and factorized distribution algorithms" (Santana, 2017, Section 6.2.1),
neural models represent an emerging approach for capturing problem structure in EDAs,
offering alternatives to traditional probabilistic graphical models.

==============================================================================
GAN ARCHITECTURE
==============================================================================

The GAN consists of two neural networks trained adversarially:

1. **Generator G**: Maps random noise z from latent space to data samples
   - Input: Random noise vector z ~ N(0, I) of dimension latent_dim
   - Output: Generated samples in problem space (dimension n_vars)
   - Purpose: Learn to generate realistic samples that fool the discriminator

2. **Discriminator D**: Binary classifier distinguishing real from fake samples
   - Input: Sample x from either real data or generator
   - Output: Probability that x is real (vs. generated)
   - Purpose: Learn to distinguish real data from generated samples

Training Process (Adversarial Game):
- D tries to maximize its ability to distinguish real from generated samples
- G tries to minimize D's ability (maximize D's error on generated samples)
- At equilibrium, G generates samples indistinguishable from real data

==============================================================================
RELATIONSHIP TO NEURAL MODELS IN EDAs (Santana, 2017)
==============================================================================

According to the analysis in Section 6.2.1 of Santana (2017), neural models in EDAs
exhibit distinct characteristics compared to traditional PGMs:

**Fundamental Difference**:
- **PGMs** (Bayesian/Markov networks): Explicitly represent structure, interpretable
- **Neural Models** (GANs, VAEs, RBMs): Latent/distributed representations, less interpretable

**Key Question for EDAs**:
"To what extent can a latent representation of the optimization problem be efficiently
exploited?" (Santana, 2017, Section 6.2.1)

**Advantages of Neural Models** (from Section 6.2.1):
1. **Flexible models**: Don't require specifying dependency type a priori
2. **Efficient learning**: Usually faster than learning Bayesian networks
3. **GPU parallelization**: Dramatic efficiency gains possible
4. **Transfer learning**: Natural application for related problem instances

**Disadvantages of Neural Models** (from Section 6.2.1):
1. **Sampling complexity**: Can be cumbersome and costly (especially for GANs)
2. **Parameter sensitivity**: Very sensitive to initial parameters and hyperparameters
3. **Representation mismatch**: May not match problem representation (e.g., continuous vs. discrete)
4. **Large parameter count**: Can be very large (up to 2.5×n² for some models)
5. **Overfitting risk**: Possible when learning the models

**GAN-Specific Observations** (from Section 6.2.1):
- GANs tested in EDAs (Probst, 2015) did NOT produce competitive results
- Neither in number of fitness evaluations nor computational time
- Suggests GANs may not be ideal for optimization (despite success in other domains)

==============================================================================
IMPLEMENTATION DETAILS
==============================================================================

Architecture:
- Generator: latent_dim → hidden_dims[0] → hidden_dims[1] → n_vars
- Discriminator: n_vars → hidden_dims[0] → hidden_dims[1] → 1 (probability)
- Default hidden dimensions: [32, 64] for generator, [64, 32] for discriminator
- Activation functions: ReLU for hidden layers, Sigmoid for outputs

Training:
- Loss function: Binary Cross-Entropy (BCE)
- Optimizer: Adam (recommended in GAN literature)
- Alternating updates: k discriminator steps per generator step (default k=1)
- Data normalization: [0, 1] range using sigmoid output activation

==============================================================================
USAGE CONSIDERATIONS
==============================================================================

When to consider GANs in EDAs:
- Continuous optimization problems
- When population size is large enough to train neural networks
- When computational resources (GPU) are available
- Exploratory research on neural model capabilities

When NOT to use GANs:
- Limited computational budget (use traditional EDAs instead)
- Small population sizes (insufficient training data)
- Discrete optimization (use specialized models like softmax RBMs)
- When interpretability of structure is important

Based on Santana (2017) analysis, traditional PGM-based EDAs or other neural models
(VAEs, RBMs) may be more effective than GANs for most optimization scenarios.

==============================================================================
REFERENCES
==============================================================================

- Probst, M. (2015). "Generative adversarial networks in estimation of distribution
  algorithms for combinatorial optimization." arXiv:1509.09235.
  [Original application of GANs to EDAs]

- Santana, R. (2017). "Gray-box optimization and factorized distribution algorithms:
  where two worlds collide." arXiv:1707.03093, Section 6.2.1.
  [Comprehensive analysis of neural models in EDAs, including GANs]

- Goodfellow, I., et al. (2014). "Generative adversarial nets." NIPS 2014.
  [Original GAN paper]

==============================================================================
SEE ALSO
==============================================================================

Related neural model implementations in pateda:
- pateda.learning.vae: Variational Autoencoders (more successful in EDAs than GANs)
- pateda.learning.rbm: Restricted Boltzmann Machines (discrete variables)
- pateda.learning.dae: Denoising Autoencoders (efficient learning)
- pateda.sampling.gan: Sampling from trained GAN models

Traditional alternatives:
- pateda.learning.gaussian: Gaussian-based EDAs (simpler, often more effective)
- pateda.learning.boa: Bayesian Optimization Algorithm (interpretable structure)
- pateda.learning.gaussian.learn_gmrf_eda: GMRF-EDA (structured Gaussian models)
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
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


class GANGenerator(nn.Module):
    """
    Generator network for GAN.

    Maps random noise z from latent space to data samples.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space input.
    output_dim : int
        Dimension of the output (number of variables).
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
        super(GANGenerator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        n_hidden = len(hidden_dims)

        # Validate and set defaults for activation/initialization functions
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
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        # Output layer with sigmoid activation for normalized data
        output_layer = nn.Linear(prev_dim, output_dim)
        layers.append(output_layer)
        layers.append(nn.Sigmoid())

        self.generator = nn.Sequential(*layers)

    def forward(self, z):
        return self.generator(z)


class GANDiscriminator(nn.Module):
    """
    Discriminator network for GAN.

    Classifies input samples as real (from data) or fake (from generator).
    Outputs probability that input is real.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (number of variables).
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
        super(GANDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        n_hidden = len(hidden_dims)

        # Validate and set defaults for activation/initialization functions
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

        # Output layer: probability that input is real
        output_layer = nn.Linear(prev_dim, 1)
        layers.append(output_layer)
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
        - 'hidden_dims_g': list of hidden layer dimensions for generator
          (default: computed from n_vars and pop_size)
        - 'hidden_dims_d': list of hidden layer dimensions for discriminator
          (default: computed from n_vars and pop_size, reversed)
        - 'list_act_functs_g': list of activation functions for generator hidden layers
        - 'list_act_functs_d': list of activation functions for discriminator hidden layers
        - 'list_init_functs_g': list of initialization functions for generator hidden layers
        - 'list_init_functs_d': list of initialization functions for discriminator hidden layers
        - 'epochs': number of training epochs (default: 100)
        - 'batch_size': batch size for training (default: max(8, n_vars/50))
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
        - 'hidden_dims_g': generator hidden dimensions
        - 'hidden_dims_d': discriminator hidden dimensions
        - 'list_act_functs_g': generator activation functions
        - 'list_act_functs_d': discriminator activation functions
        - 'list_init_functs_g': generator initialization functions
        - 'list_init_functs_d': discriminator initialization functions
        - 'ranges': data normalization ranges
        - 'type': 'gan'
    """
    if params is None:
        params = {}

    # Extract dimensions
    pop_size = population.shape[0]
    input_dim = population.shape[1]

    # Compute default hidden dimensions based on input size and population size
    # First layer: max(5, n/10), Second layer: max(10, psize/10)
    default_hidden_dims = compute_default_hidden_dims(input_dim, pop_size)
    default_batch_size = compute_default_batch_size(input_dim, pop_size)

    # Extract parameters with new defaults
    latent_dim = params.get('latent_dim', max(2, input_dim // 2))
    hidden_dims_g = params.get('hidden_dims_g', default_hidden_dims)
    hidden_dims_d = params.get('hidden_dims_d', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.0002)
    beta1 = params.get('beta1', 0.5)
    k_discriminator = params.get('k_discriminator', 1)

    # Extract activation and initialization function lists
    list_act_functs_g = params.get('list_act_functs_g', None)
    list_act_functs_d = params.get('list_act_functs_d', None)
    list_init_functs_g = params.get('list_init_functs_g', None)
    list_init_functs_d = params.get('list_init_functs_d', None)

    # Normalize data to [0, 1]
    ranges = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])
    range_diff = ranges[1] - ranges[0]
    range_diff = np.where(range_diff < 1e-10, 1.0, range_diff)  # Prevent division by zero

    norm_pop = (population - ranges[0]) / range_diff
    norm_pop = np.clip(norm_pop, 0, 1)  # Ensure values are in [0, 1]

    # Convert to tensors
    real_data = torch.FloatTensor(norm_pop)

    # Create networks with configurable activations and initializations
    generator = GANGenerator(
        latent_dim, input_dim, hidden_dims_g,
        list_act_functs=list_act_functs_g,
        list_init_functs=list_init_functs_g
    )
    discriminator = GANDiscriminator(
        input_dim, hidden_dims_d,
        list_act_functs=list_act_functs_d,
        list_init_functs=list_init_functs_d
    )

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

    # Return model with all configuration
    return {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'input_dim': input_dim,
        'hidden_dims_g': hidden_dims_g,
        'hidden_dims_d': hidden_dims_d,
        'list_act_functs_g': list_act_functs_g if list_act_functs_g else ['relu'] * len(hidden_dims_g),
        'list_act_functs_d': list_act_functs_d if list_act_functs_d else ['relu'] * len(hidden_dims_d),
        'list_init_functs_g': list_init_functs_g if list_init_functs_g else ['default'] * len(hidden_dims_g),
        'list_init_functs_d': list_init_functs_d if list_init_functs_d else ['default'] * len(hidden_dims_d),
        'ranges': ranges,
        'type': 'gan'
    }
