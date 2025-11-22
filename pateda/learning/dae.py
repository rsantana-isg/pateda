"""
DAE Model Learning for Discrete EDAs

==============================================================================
OVERVIEW
==============================================================================

This module provides learning algorithms for Denoising Autoencoder (DAE) based
probabilistic models used in combinatorial optimization. Implementation based on the paper:
"Denoising Autoencoders for Fast Combinatorial Black Box Optimization"
(Willemsen & Dockhorn, 2022).

The module implements:
1. DAE: Denoising autoencoder with corruption process
2. Iterative refinement sampling for combinatorial optimization

==============================================================================
DAE ARCHITECTURE
==============================================================================

DAEs learn robust representations by reconstructing clean inputs from corrupted versions:

1. **Corruption Process**: Input x is corrupted with salt & pepper noise
   - Each bit is flipped with probability corruption_level
   - Creates corrupted input x̃

2. **Encoder q(h|x̃)**: Maps corrupted input to hidden representation
   - Input: Corrupted data x̃ (dimension n_vars)
   - Output: Hidden representation h
   - Learns robust features invariant to noise

3. **Decoder p(x|h)**: Maps hidden representation back to clean data
   - Input: Hidden representation h
   - Output: Reconstructed output z (probabilities)
   - Learns to denoise and reconstruct original input

Training Objective:
Minimize reconstruction loss between clean input x and reconstruction z
- BCE Loss: Binary cross-entropy (default for binary data)
- MSE Loss: Mean squared error (alternative)

==============================================================================
RELATIONSHIP TO NEURAL MODELS IN EDAs (Santana, 2017)
==============================================================================

According to Section 6.2.1 of Santana (2017), denoising autoencoders represent
a promising approach for EDAs:

**GA-dA** (Churchill et al., 2016): Denoising autoencoder as mutation distribution
- Outperforms BOA on knapsack problem
- "Extremely fast" compared to Bayesian network learning
- Performance depends on how the neural model is used

**Key Insight**: DAEs can be viewed as learning a transition kernel that
maps noisy solutions to cleaner ones, effectively learning the structure
of the fitness landscape.

==============================================================================
CONFIGURABLE ARCHITECTURE
==============================================================================

This implementation supports:
- 15 activation functions (relu, tanh, sigmoid, gelu, silu, etc.)
- 15 weight initialization methods (xavier, kaiming, orthogonal, etc.)
- Configurable hidden layer dimensions
- Dynamic defaults based on problem size

Default Configuration:
- Hidden dimensions: computed from input size and population
- Activation functions: sigmoid for encoder, sigmoid for decoder
- Initialization: PyTorch default (Kaiming uniform)

==============================================================================
REFERENCES
==============================================================================

- Willemsen, P. & Dockhorn, A. (2022). "Denoising Autoencoders for Fast
  Combinatorial Black Box Optimization."
  [DAE-EDA for combinatorial optimization]

- Churchill, A.W., Sigtia, S., & Fernando, C. (2016). "Learning to generate
  genotypes with neural networks." arXiv:1604.04153.
  [GA-dA: Denoising autoencoders in EDAs]

- Santana, R. (2017). "Gray-box optimization and factorized distribution
  algorithms: where two worlds collide." arXiv:1707.03093, Section 6.2.1.
  [Analysis of neural models in EDAs]

- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.A. (2008).
  "Extracting and composing robust features with denoising autoencoders."
  ICML 2008.
  [Original denoising autoencoder paper]

==============================================================================
SEE ALSO
==============================================================================

Related neural model implementations in pateda:
- pateda.learning.vae: Variational Autoencoders (probabilistic latent space)
- pateda.learning.gan: Generative Adversarial Networks
- pateda.learning.rbm: Restricted Boltzmann Machines
- pateda.sampling.dae: Sampling from trained DAE models
- pateda.learning.nn_utils: Shared neural network utilities
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for binary/discrete variables.

    The DAE learns to reconstruct clean inputs from corrupted versions,
    enabling it to learn robust representations for combinatorial optimization.

    Parameters
    ----------
    input_dim : int
        Dimension of input (total number of binary variables).
    hidden_dims : list, optional
        List of hidden layer dimensions. If None, uses [input_dim // 2].
    list_act_functs_enc : list, optional
        List of activation functions for encoder layers.
        One per hidden layer. Default: ['sigmoid'] * n_hidden.
    list_act_functs_dec : list, optional
        List of activation functions for decoder layers.
        One per hidden layer. Default: ['sigmoid'] * n_hidden.
    list_init_functs_enc : list, optional
        List of initialization functions for encoder layers.
        One per hidden layer. Default: ['default'] * n_hidden.
    list_init_functs_dec : list, optional
        List of initialization functions for decoder layers.
        One per hidden layer. Default: ['default'] * n_hidden.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs_enc: List[str] = None,
        list_act_functs_dec: List[str] = None,
        list_init_functs_enc: List[str] = None,
        list_init_functs_dec: List[str] = None
    ):
        """
        Initialize Denoising Autoencoder.

        Parameters
        ----------
        input_dim : int
            Dimension of input (total number of binary variables)
        hidden_dims : list, optional
            List of hidden layer dimensions (default: [input_dim // 2])
        list_act_functs_enc : list, optional
            Activation functions for encoder layers
        list_act_functs_dec : list, optional
            Activation functions for decoder layers
        list_init_functs_enc : list, optional
            Initialization functions for encoder layers
        list_init_functs_dec : list, optional
            Initialization functions for decoder layers
        """
        super(DenoisingAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, 10)]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        n_hidden = len(hidden_dims)

        # Set default activation functions (sigmoid for binary data)
        if list_act_functs_enc is None:
            list_act_functs_enc = ['sigmoid'] * n_hidden
        if list_act_functs_dec is None:
            list_act_functs_dec = ['sigmoid'] * n_hidden
        if list_init_functs_enc is None:
            list_init_functs_enc = ['default'] * n_hidden
        if list_init_functs_dec is None:
            list_init_functs_dec = ['default'] * n_hidden

        # Validate parameters
        list_act_functs_enc, list_init_functs_enc = validate_list_params(
            hidden_dims, list_act_functs_enc, list_init_functs_enc
        )
        list_act_functs_dec, list_init_functs_dec = validate_list_params(
            hidden_dims, list_act_functs_dec, list_init_functs_dec
        )

        # Store configuration
        self.list_act_functs_enc = list_act_functs_enc
        self.list_act_functs_dec = list_act_functs_dec
        self.list_init_functs_enc = list_init_functs_enc
        self.list_init_functs_dec = list_init_functs_dec

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs_enc[i])
            encoder_layers.append(linear)
            encoder_layers.append(get_activation(list_act_functs_enc[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                next_dim = input_dim
            else:
                next_dim = hidden_dims[i - 1]

            linear = nn.Linear(hidden_dims[i], next_dim)
            # Use decoder init functions in reverse order
            apply_weight_init(linear, list_init_functs_dec[len(hidden_dims) - 1 - i])
            decoder_layers.append(linear)
            # Use decoder activation functions in reverse order
            decoder_layers.append(get_activation(
                list_act_functs_dec[len(hidden_dims) - 1 - i],
                in_features=next_dim
            ))

        # Final output layer with sigmoid for binary output
        if len(decoder_layers) >= 2:
            # Replace the last activation with sigmoid for binary output
            decoder_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(*decoder_layers)

        # For backward compatibility, store hidden_dim as first hidden dimension
        self.hidden_dim = hidden_dims[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        h : torch.Tensor
            Hidden representation
        """
        return self.encoder(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to reconstruction.

        Parameters
        ----------
        h : torch.Tensor
            Hidden representation

        Returns
        -------
        z : torch.Tensor
            Reconstructed output
        """
        return self.decoder(h)

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
        - 'hidden_dims': list of hidden layer dimensions
          (default: computed from n_vars and pop_size)
        - 'hidden_dim': dimension of single hidden layer (backward compatible)
          (default: n_vars // 2)
        - 'list_act_functs_enc': list of activation functions for encoder
        - 'list_act_functs_dec': list of activation functions for decoder
        - 'list_init_functs_enc': list of initialization functions for encoder
        - 'list_init_functs_dec': list of initialization functions for decoder
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': mini-batch size (default: max(8, n_vars/50))
        - 'learning_rate': learning rate (default: 0.001)
        - 'corruption_level': noise corruption probability (default: 0.1)
        - 'loss_type': 'bce' or 'mse' (default: 'bce')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'dae_state': DAE network state dict
        - 'input_dim': input dimension
        - 'hidden_dims': list of hidden dimensions
        - 'hidden_dim': first hidden dimension (backward compatible)
        - 'list_act_functs_enc': encoder activation functions
        - 'list_act_functs_dec': decoder activation functions
        - 'list_init_functs_enc': encoder initialization functions
        - 'list_init_functs_dec': decoder initialization functions
        - 'type': 'dae'
    """
    if params is None:
        params = {}

    # Extract dimensions
    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters with new defaults
    # Support both 'hidden_dims' (list) and 'hidden_dim' (single value) for backward compatibility
    if 'hidden_dims' in params:
        hidden_dims = params['hidden_dims']
    elif 'hidden_dim' in params:
        hidden_dims = [params['hidden_dim']]
    else:
        hidden_dims = default_hidden_dims

    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    corruption_level = params.get('corruption_level', 0.1)
    loss_type = params.get('loss_type', 'bce')

    # Extract activation and initialization function lists
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # Convert to tensors (assuming binary input)
    data = torch.FloatTensor(population)

    # Create DAE with configurable architecture
    dae = DenoisingAutoencoder(
        n_vars,
        hidden_dims=hidden_dims,
        list_act_functs_enc=list_act_functs_enc,
        list_act_functs_dec=list_act_functs_dec,
        list_init_functs_enc=list_init_functs_enc,
        list_init_functs_dec=list_init_functs_dec
    )

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

    # Return model with all configuration
    return {
        'dae_state': dae.state_dict(),
        'input_dim': n_vars,
        'hidden_dims': hidden_dims,
        'hidden_dim': hidden_dims[0],  # Backward compatibility
        'list_act_functs_enc': dae.list_act_functs_enc,
        'list_act_functs_dec': dae.list_act_functs_dec,
        'list_init_functs_enc': dae.list_init_functs_enc,
        'list_init_functs_dec': dae.list_init_functs_dec,
        'type': 'dae'
    }


class MultiLayerDAE(nn.Module):
    """
    Multi-layer Denoising Autoencoder with deeper architecture.

    Useful for higher-dimensional problems. Supports configurable activation
    and initialization functions for each layer.

    Parameters
    ----------
    input_dim : int
        Dimension of input (total number of binary variables).
    hidden_dims : list, optional
        List of hidden layer dimensions. If None, uses [input_dim // 2].
    list_act_functs_enc : list, optional
        List of activation functions for encoder layers.
        One per hidden layer. Default: ['sigmoid'] * n_hidden.
    list_act_functs_dec : list, optional
        List of activation functions for decoder layers.
        One per hidden layer. Default: ['sigmoid'] * n_hidden.
    list_init_functs_enc : list, optional
        List of initialization functions for encoder layers.
        One per hidden layer. Default: ['default'] * n_hidden.
    list_init_functs_dec : list, optional
        List of initialization functions for decoder layers.
        One per hidden layer. Default: ['default'] * n_hidden.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs_enc: List[str] = None,
        list_act_functs_dec: List[str] = None,
        list_init_functs_enc: List[str] = None,
        list_init_functs_dec: List[str] = None
    ):
        """
        Initialize multi-layer DAE.

        Parameters
        ----------
        input_dim : int
            Dimension of input
        hidden_dims : list, optional
            List of hidden layer dimensions (default: [input_dim//2])
        list_act_functs_enc : list, optional
            Activation functions for encoder layers
        list_act_functs_dec : list, optional
            Activation functions for decoder layers
        list_init_functs_enc : list, optional
            Initialization functions for encoder layers
        list_init_functs_dec : list, optional
            Initialization functions for decoder layers
        """
        super(MultiLayerDAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, 10)]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        n_hidden = len(hidden_dims)

        # Set default activation functions (sigmoid for binary data)
        if list_act_functs_enc is None:
            list_act_functs_enc = ['sigmoid'] * n_hidden
        if list_act_functs_dec is None:
            list_act_functs_dec = ['sigmoid'] * n_hidden
        if list_init_functs_enc is None:
            list_init_functs_enc = ['default'] * n_hidden
        if list_init_functs_dec is None:
            list_init_functs_dec = ['default'] * n_hidden

        # Validate parameters
        list_act_functs_enc, list_init_functs_enc = validate_list_params(
            hidden_dims, list_act_functs_enc, list_init_functs_enc
        )
        list_act_functs_dec, list_init_functs_dec = validate_list_params(
            hidden_dims, list_act_functs_dec, list_init_functs_dec
        )

        # Store configuration
        self.list_act_functs_enc = list_act_functs_enc
        self.list_act_functs_dec = list_act_functs_dec
        self.list_init_functs_enc = list_init_functs_enc
        self.list_init_functs_dec = list_init_functs_dec

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs_enc[i])
            encoder_layers.append(linear)
            encoder_layers.append(get_activation(list_act_functs_enc[i], in_features=hidden_dim))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                next_dim = input_dim
            else:
                next_dim = hidden_dims[i - 1]

            linear = nn.Linear(hidden_dims[i], next_dim)
            # Use decoder init functions in reverse order
            apply_weight_init(linear, list_init_functs_dec[len(hidden_dims) - 1 - i])
            decoder_layers.append(linear)
            # Use decoder activation functions in reverse order
            decoder_layers.append(get_activation(
                list_act_functs_dec[len(hidden_dims) - 1 - i],
                in_features=next_dim
            ))

        # Final output layer with sigmoid for binary output
        if len(decoder_layers) >= 2:
            # Replace the last activation with sigmoid for binary output
            decoder_layers[-1] = nn.Sigmoid()

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
        Training parameters containing:
        - 'hidden_dims': list of hidden layer dimensions
          (default: computed from n_vars and pop_size)
        - 'list_act_functs_enc': list of activation functions for encoder
        - 'list_act_functs_dec': list of activation functions for decoder
        - 'list_init_functs_enc': list of initialization functions for encoder
        - 'list_init_functs_dec': list of initialization functions for decoder
        - 'epochs': number of training epochs (default: 50)
        - 'batch_size': mini-batch size (default: max(8, n_vars/50))
        - 'learning_rate': learning rate (default: 0.001)
        - 'corruption_level': noise corruption probability (default: 0.1)
        - 'loss_type': 'bce' or 'mse' (default: 'bce')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'dae_state': DAE network state dict
        - 'input_dim': input dimension
        - 'hidden_dims': list of hidden dimensions
        - 'list_act_functs_enc': encoder activation functions
        - 'list_act_functs_dec': decoder activation functions
        - 'list_init_functs_enc': encoder initialization functions
        - 'list_init_functs_dec': decoder initialization functions
        - 'type': 'multilayer_dae'
    """
    if params is None:
        params = {}

    # Extract dimensions
    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    default_batch_size = compute_default_batch_size(n_vars, pop_size)

    # Extract parameters with new defaults
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    corruption_level = params.get('corruption_level', 0.1)
    loss_type = params.get('loss_type', 'bce')

    # Extract activation and initialization function lists
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # Convert to tensors
    data = torch.FloatTensor(population)

    # Create multi-layer DAE with configurable architecture
    dae = MultiLayerDAE(
        n_vars,
        hidden_dims=hidden_dims,
        list_act_functs_enc=list_act_functs_enc,
        list_act_functs_dec=list_act_functs_dec,
        list_init_functs_enc=list_init_functs_enc,
        list_init_functs_dec=list_init_functs_dec
    )

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

    # Return model with all configuration
    return {
        'dae_state': dae.state_dict(),
        'input_dim': n_vars,
        'hidden_dims': hidden_dims,
        'list_act_functs_enc': dae.list_act_functs_enc,
        'list_act_functs_dec': dae.list_act_functs_dec,
        'list_init_functs_enc': dae.list_init_functs_enc,
        'list_init_functs_dec': dae.list_init_functs_dec,
        'type': 'multilayer_dae'
    }
