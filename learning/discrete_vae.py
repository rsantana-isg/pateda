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
from scipy import stats


# ==============================================================================
# Mutual Information and G-Test Utilities
# ==============================================================================

def compute_mutual_information_matrix(population: np.ndarray) -> np.ndarray:
    """
    Compute the mutual information matrix for binary variables.

    For binary variables X and Y, mutual information is:
    MI(X,Y) = sum_{x,y} P(x,y) * log(P(x,y) / (P(x)*P(y)))

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}

    Returns
    -------
    mi_matrix : np.ndarray
        Mutual information matrix of shape (n_vars, n_vars)
    """
    n_samples, n_vars = population.shape
    mi_matrix = np.zeros((n_vars, n_vars))

    # Add small epsilon to avoid log(0)
    eps = 1e-10

    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                # Self-MI is entropy
                p1 = np.mean(population[:, i])
                p0 = 1 - p1
                if p0 > eps and p1 > eps:
                    mi_matrix[i, j] = -p0 * np.log2(p0 + eps) - p1 * np.log2(p1 + eps)
            else:
                # Compute joint and marginal probabilities
                # P(X=x, Y=y) for x,y in {0,1}
                p00 = np.mean((population[:, i] == 0) & (population[:, j] == 0))
                p01 = np.mean((population[:, i] == 0) & (population[:, j] == 1))
                p10 = np.mean((population[:, i] == 1) & (population[:, j] == 0))
                p11 = np.mean((population[:, i] == 1) & (population[:, j] == 1))

                # Marginal probabilities
                p_i0 = p00 + p01
                p_i1 = p10 + p11
                p_j0 = p00 + p10
                p_j1 = p01 + p11

                # Compute mutual information
                mi = 0.0
                for px, py, pxy in [(p_i0, p_j0, p00), (p_i0, p_j1, p01),
                                     (p_i1, p_j0, p10), (p_i1, p_j1, p11)]:
                    if pxy > eps and px > eps and py > eps:
                        mi += pxy * np.log2(pxy / (px * py + eps) + eps)

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

    return mi_matrix


def compute_gtest_independence_matrix(population: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Compute G-Test (Likelihood Ratio Test) to test independence of binary variables.

    The G-Test statistic is:
    G = 2 * sum_{x,y} O_{x,y} * log(O_{x,y} / E_{x,y})

    where O_{x,y} are observed frequencies and E_{x,y} are expected frequencies
    under independence assumption.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    alpha : float
        Significance level for independence test (default: 0.05)

    Returns
    -------
    g_matrix : np.ndarray
        Binary matrix of shape (n_vars, n_vars) where G[i,j]=1 indicates
        variables i and j are independent (fail to reject H0), and
        G[i,j]=0 indicates they are dependent (reject H0)
    """
    n_samples, n_vars = population.shape
    g_matrix = np.zeros((n_vars, n_vars))

    eps = 1e-10

    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                # Variable is always dependent with itself
                g_matrix[i, j] = 0
            else:
                # Compute contingency table (2x2 for binary variables)
                n00 = np.sum((population[:, i] == 0) & (population[:, j] == 0))
                n01 = np.sum((population[:, i] == 0) & (population[:, j] == 1))
                n10 = np.sum((population[:, i] == 1) & (population[:, j] == 0))
                n11 = np.sum((population[:, i] == 1) & (population[:, j] == 1))

                # Marginal totals
                n_i0 = n00 + n01
                n_i1 = n10 + n11
                n_j0 = n00 + n10
                n_j1 = n01 + n11

                # Expected frequencies under independence
                e00 = (n_i0 * n_j0) / n_samples
                e01 = (n_i0 * n_j1) / n_samples
                e10 = (n_i1 * n_j0) / n_samples
                e11 = (n_i1 * n_j1) / n_samples

                # Compute G-statistic
                g_stat = 0.0
                for observed, expected in [(n00, e00), (n01, e01), (n10, e10), (n11, e11)]:
                    if observed > 0 and expected > eps:
                        g_stat += 2 * observed * np.log(observed / (expected + eps))

                # G-statistic follows chi-square distribution with df=1 for 2x2 table
                # Critical value for chi-square(1) at alpha=0.05 is 3.841
                critical_value = stats.chi2.ppf(1 - alpha, df=1)

                # If G < critical_value, fail to reject H0 (independence)
                # So G[i,j]=1 means variables are independent
                if g_stat < critical_value:
                    g_matrix[i, j] = 1
                    g_matrix[j, i] = 1
                else:
                    g_matrix[i, j] = 0
                    g_matrix[j, i] = 0

    return g_matrix


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


class SparseLinear(nn.Module):
    """
    Sparse linear layer with masked connections.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    mask : torch.Tensor
        Binary mask of shape (in_features, out_features) where 1 indicates connection
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register mask as buffer (not a parameter, but part of state)
        self.register_buffer('mask', mask.float())

        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        # Apply mask to weights before linear transformation
        masked_weight = self.weight * self.mask.t()
        return F.linear(x, masked_weight, self.bias)


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
    mi_layer : bool, optional
        If True, use MI-based sparse connectivity for first layer (default: False)
    g_matrix : np.ndarray, optional
        G-Test independence matrix for MI layer (required if mi_layer=True)
    activation_func : str, optional
        Activation function name for MI layer (required if mi_layer=True)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        mi_layer: bool = False,
        g_matrix: np.ndarray = None,
        activation_func: str = 'relu'
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

        # If MI layer is enabled, add sparse first layer
        if mi_layer:
            if g_matrix is None:
                raise ValueError("g_matrix must be provided when mi_layer=True")

            # First hidden layer has n neurons (same as input_dim)
            mi_hidden_dim = input_dim

            # Create connectivity mask:
            # - G[i,j]=1 means variables i and j are independent -> connection exists
            # - Additionally, each input i connects to neuron i (diagonal)
            connectivity_mask = torch.zeros(input_dim, mi_hidden_dim)
            for i in range(input_dim):
                for j in range(mi_hidden_dim):
                    if i == j or g_matrix[i, j] == 1:
                        connectivity_mask[i, j] = 1

            # Add sparse MI layer
            mi_linear = SparseLinear(input_dim, mi_hidden_dim, connectivity_mask)
            layers.append(mi_linear)
            layers.append(get_activation(activation_func, in_features=mi_hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = mi_hidden_dim

        # Add remaining hidden layers
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

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space
    output_dim : int
        Dimension of the output
    hidden_dims : list, optional
        List of hidden layer dimensions
    list_act_functs : list, optional
        List of activation functions, one per hidden layer
    list_init_functs : list, optional
        List of initialization functions, one per hidden layer
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(BinaryVAEDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

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
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
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


class ConditionalBinaryVAEEncoder(nn.Module):
    """
    Conditional encoder for C-VAE (conditioned on fitness and statistics)

    Parameters
    ----------
    input_dim : int
        Dimension of the binary input
    latent_dim : int
        Dimension of the latent space
    condition_dim : int
        Dimension of the conditioning vector (e.g., fitness + mean + std = 3)
    hidden_dims : list, optional
        List of hidden layer dimensions
    list_act_functs : list, optional
        List of activation functions
    list_init_functs : list, optional
        List of initialization functions
    mi_layer : bool, optional
        If True, use MI-based sparse connectivity for first layer (default: False)
    g_matrix : np.ndarray, optional
        G-Test independence matrix for MI layer (required if mi_layer=True)
    activation_func : str, optional
        Activation function name for MI layer (required if mi_layer=True)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        condition_dim: int = 3,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        mi_layer: bool = False,
        g_matrix: np.ndarray = None,
        activation_func: str = 'relu'
    ):
        super(ConditionalBinaryVAEEncoder, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.mi_layer = mi_layer

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

        # If MI layer is enabled, add sparse first layer on input only
        if mi_layer:
            if g_matrix is None:
                raise ValueError("g_matrix must be provided when mi_layer=True")

            # First hidden layer has n neurons (same as input_dim)
            mi_hidden_dim = input_dim

            # Create connectivity mask
            connectivity_mask = torch.zeros(input_dim, mi_hidden_dim)
            for i in range(input_dim):
                for j in range(mi_hidden_dim):
                    if i == j or g_matrix[i, j] == 1:
                        connectivity_mask[i, j] = 1

            # Store MI layer components separately
            self.mi_linear = SparseLinear(input_dim, mi_hidden_dim, connectivity_mask)
            self.mi_activation = get_activation(activation_func, in_features=mi_hidden_dim)
            self.mi_dropout = nn.Dropout(0.2)

            # After MI layer, combine with condition
            prev_dim = mi_hidden_dim + condition_dim
        else:
            # Concatenate input with condition from the start
            prev_dim = input_dim + condition_dim

        # Add remaining hidden layers
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

    def forward(self, x, condition):
        """
        Forward pass

        Args:
            x: binary input [batch_size, input_dim]
            condition: conditioning vector [batch_size, condition_dim]

        Returns:
            mean, logvar: latent distribution parameters
        """
        if self.mi_layer:
            # Process input through MI layer first
            h = self.mi_linear(x)
            h = self.mi_activation(h)
            h = self.mi_dropout(h)
            # Then concatenate with condition
            h = torch.cat([h, condition], dim=1)
        else:
            # Concatenate input and condition
            h = torch.cat([x, condition], dim=1)

        h = self.encoder(h)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class ConditionalBinaryVAEDecoder(nn.Module):
    """
    Conditional decoder for C-VAE (conditioned on fitness target)

    Takes latent code + condition and outputs binary probabilities
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        condition_dim: int = 3,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None
    ):
        super(ConditionalBinaryVAEDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

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
        # Concatenate latent with condition
        prev_dim = latent_dim + condition_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            apply_weight_init(linear, list_init_functs[i])
            layers.append(linear)
            layers.append(get_activation(list_act_functs[i], in_features=hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # No activation here - will use BCE with logits

        self.decoder = nn.Sequential(*layers)

    def forward(self, z, condition):
        """
        Forward pass

        Args:
            z: latent code [batch_size, latent_dim]
            condition: conditioning vector [batch_size, condition_dim]

        Returns:
            logits: output logits [batch_size, output_dim]
        """
        # Concatenate latent and condition
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)


class DescriptorBinaryVAEEncoder(nn.Module):
    """
    Descriptor-augmented encoder for Desc-VAE

    Takes binary input augmented with descriptors (fitness, mean, std)

    Parameters
    ----------
    input_dim : int
        Dimension of the binary input
    latent_dim : int
        Dimension of the latent space
    n_descriptors : int
        Number of descriptors (default: 3)
    hidden_dims : list, optional
        List of hidden layer dimensions
    list_act_functs : list, optional
        List of activation functions
    list_init_functs : list, optional
        List of initialization functions
    mi_layer : bool, optional
        If True, use MI-based sparse connectivity for first layer (default: False)
    g_matrix : np.ndarray, optional
        G-Test independence matrix for MI layer (required if mi_layer=True)
    activation_func : str, optional
        Activation function name for MI layer (required if mi_layer=True)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_descriptors: int = 3,
        hidden_dims: List[int] = None,
        list_act_functs: List[str] = None,
        list_init_functs: List[str] = None,
        mi_layer: bool = False,
        g_matrix: np.ndarray = None,
        activation_func: str = 'relu'
    ):
        super(DescriptorBinaryVAEEncoder, self).__init__()

        self.input_dim = input_dim
        self.n_descriptors = n_descriptors
        self.mi_layer = mi_layer

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

        # If MI layer is enabled, add sparse first layer on input only
        if mi_layer:
            if g_matrix is None:
                raise ValueError("g_matrix must be provided when mi_layer=True")

            # First hidden layer has n neurons (same as input_dim)
            mi_hidden_dim = input_dim

            # Create connectivity mask
            connectivity_mask = torch.zeros(input_dim, mi_hidden_dim)
            for i in range(input_dim):
                for j in range(mi_hidden_dim):
                    if i == j or g_matrix[i, j] == 1:
                        connectivity_mask[i, j] = 1

            # Store MI layer components separately
            self.mi_linear = SparseLinear(input_dim, mi_hidden_dim, connectivity_mask)
            self.mi_activation = get_activation(activation_func, in_features=mi_hidden_dim)
            self.mi_dropout = nn.Dropout(0.2)

            # After MI layer, combine with descriptors
            prev_dim = mi_hidden_dim + n_descriptors
        else:
            # Concatenate input with descriptors from the start
            prev_dim = input_dim + n_descriptors

        # Add remaining hidden layers
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

    def forward(self, x, descriptors):
        """
        Forward pass

        Args:
            x: binary input [batch_size, input_dim]
            descriptors: descriptor vector [batch_size, n_descriptors]

        Returns:
            mean, logvar: latent distribution parameters
        """
        if self.mi_layer:
            # Process input through MI layer first
            h = self.mi_linear(x)
            h = self.mi_activation(h)
            h = self.mi_dropout(h)
            # Then concatenate with descriptors
            h = torch.cat([h, descriptors], dim=1)
        else:
            # Concatenate input and descriptors
            h = torch.cat([x, descriptors], dim=1)

        h = self.encoder(h)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


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
          (default: computed dynamically to prevent overfitting)
        - 'hidden_dims_dec': decoder hidden dims
          (default: computed dynamically, reversed)
        - 'list_act_functs_enc': list of activation functions for encoder
        - 'list_act_functs_dec': list of activation functions for decoder
        - 'list_init_functs_enc': list of initialization functions for encoder
        - 'list_init_functs_dec': list of initialization functions for decoder
        - 'epochs': training epochs (default: 100)
        - 'batch_size': batch size (default: max(8, n_vars/50))
        - 'learning_rate': learning rate (default: 0.001)
        - 'beta_start': initial KL weight for annealing (default: 0.0)
        - 'beta_end': final KL weight for annealing (default: 1.0)
        - 'beta_annealing_epochs': epochs for beta annealing (default: epochs // 2)
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
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)

    # CRITICAL FIX: Dynamic hidden layer computation to prevent overfitting
    # Target: params ≈ 4-5x training samples
    # Two hidden layers: h1 = min(n_vars, selection_size), h2 computed dynamically
    h1 = min(n_vars, pop_size)
    # Estimate total params: (n_vars * h1) + (h1 * h2) + (h2 * latent_dim) + biases
    # Simplified: approximately n_vars * h1 + h1 * h2 + h2 * latent_dim
    # For target_params ≈ 4 * pop_size, solve for h2
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    # h2 ≈ (target_params - n_vars * h1) / (h1 + latent_dim)
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]

    # Extract parameters with new defaults
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)

    # Beta annealing parameters to prevent posterior collapse
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)

    use_extended = params.get('use_extended', False)
    fitness_weight = params.get('fitness_weight', 0.1)

    # Extract activation and initialization function lists
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness.reshape(-1, 1))

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)

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
        # Beta annealing to prevent posterior collapse
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end

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
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Fit={avg_fit:.4f}, Beta={beta:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Beta={beta:.4f}")

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


def learn_binary_cvae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Conditional Binary VAE (C-VAE) model

    This variant conditions both the encoder and decoder on fitness and statistical descriptors,
    allowing for fitness-guided generation during sampling.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_binary_vae, plus):
        - 'condition_dim': dimension of conditioning vector (default: 3 for fitness, mean, std)
        - 'normalize_conditions': normalize conditioning values (default: True)

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)

    # Dynamic hidden layer computation
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]

    # Extract parameters
    condition_dim = params.get('condition_dim', 3)
    normalize_conditions = params.get('normalize_conditions', True)
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)

    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Compute conditioning vectors (fitness, mean, std per individual)
    fitness_1d = fitness.flatten()
    conditions = np.zeros((pop_size, 3))
    conditions[:, 0] = fitness_1d
    conditions[:, 1] = np.mean(population, axis=1)  # Mean bit density
    conditions[:, 2] = np.std(population, axis=1)   # Std of bits

    # Normalize conditions
    if normalize_conditions:
        condition_means = np.mean(conditions, axis=0)
        condition_stds = np.std(conditions, axis=0) + 1e-8
        conditions = (conditions - condition_means) / condition_stds
    else:
        condition_means = np.zeros(3)
        condition_stds = np.ones(3)

    # Convert to tensors
    data = torch.FloatTensor(population)
    conditions_tensor = torch.FloatTensor(conditions)

    # Create networks
    encoder = ConditionalBinaryVAEEncoder(n_vars, latent_dim, condition_dim, hidden_dims_enc,
                                          list_act_functs_enc, list_init_functs_enc,
                                          mi_layer=mi_layer, g_matrix=g_matrix,
                                          activation_func=activation_func if mi_layer else 'relu')
    decoder = ConditionalBinaryVAEDecoder(latent_dim, n_vars, condition_dim, hidden_dims_dec,
                                          list_act_functs_dec, list_init_functs_dec)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end

        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_cond = conditions_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch, batch_cond)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z, batch_cond)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)

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

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Beta={beta:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'condition_dim': condition_dim,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'condition_means': condition_means,
        'condition_stds': condition_stds,
        'type': 'binary_cvae'
    }

    return model


def learn_binary_descvae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Descriptor-augmented Binary VAE (Desc-VAE) model

    This variant augments the encoder input with descriptors (fitness, mean, std),
    allowing the latent space to better capture search landscape information.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_binary_vae, plus):
        - 'n_descriptors': number of descriptors to use (default: 3)

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)

    # Dynamic hidden layer computation
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]

    # Extract parameters
    n_descriptors = params.get('n_descriptors', 3)
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)

    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Compute descriptors (fitness, mean, std per individual)
    fitness_1d = fitness.flatten()
    descriptors = np.zeros((pop_size, 3))
    descriptors[:, 0] = fitness_1d
    descriptors[:, 1] = np.mean(population, axis=1)
    descriptors[:, 2] = np.std(population, axis=1)

    # Normalize descriptors
    descriptor_means = np.mean(descriptors, axis=0)
    descriptor_stds = np.std(descriptors, axis=0) + 1e-8
    descriptors = (descriptors - descriptor_means) / descriptor_stds

    # Convert to tensors
    data = torch.FloatTensor(population)
    descriptors_tensor = torch.FloatTensor(descriptors)

    # Create networks
    encoder = DescriptorBinaryVAEEncoder(n_vars, latent_dim, n_descriptors, hidden_dims_enc,
                                         list_act_functs_enc, list_init_functs_enc,
                                         mi_layer=mi_layer, g_matrix=g_matrix,
                                         activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end

        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_desc = descriptors_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch, batch_desc)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)

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

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Beta={beta:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'n_descriptors': n_descriptors,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'descriptor_means': descriptor_means,
        'descriptor_stds': descriptor_stds,
        'type': 'binary_descvae'
    }

    return model


def learn_binary_regvae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Regression-focused Binary VAE (Reg-VAE) model

    This variant uses fitness-weighted reconstruction loss to prioritize high-fitness solutions
    and includes a fitness predictor for multi-task learning.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_binary_vae, plus):
        - 'fitness_weight': weight for fitness prediction loss (default: 0.1)
        - 'use_fitness_weighting': use fitness-weighted reconstruction loss (default: True)
        - 'weighting_temperature': temperature for fitness weighting (default: 1.0)

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)

    # Dynamic hidden layer computation
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]

    # Extract parameters
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)
    fitness_weight = params.get('fitness_weight', 0.1)
    use_fitness_weighting = params.get('use_fitness_weighting', True)
    weighting_temperature = params.get('weighting_temperature', 1.0)

    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness.reshape(-1, 1))

    # Compute fitness weights for weighted reconstruction
    if use_fitness_weighting:
        # Normalize fitness to [0, 1] range
        fitness_1d = fitness.flatten()
        fitness_normalized = (fitness_1d - fitness_1d.min()) / (fitness_1d.max() - fitness_1d.min() + 1e-8)
        # Apply softmax-like weighting
        fitness_weights = np.exp(fitness_normalized / weighting_temperature)
        fitness_weights = fitness_weights / fitness_weights.sum() * len(fitness_weights)
        fitness_weights_tensor = torch.FloatTensor(fitness_weights)
    else:
        fitness_weights_tensor = torch.ones(pop_size)

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)
    fitness_predictor = FitnessPredictor(latent_dim, 1, 32)

    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(fitness_predictor.parameters()),
        lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()
    fitness_predictor.train()

    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end

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
            batch_weights = fitness_weights_tensor[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)

            # Weighted reconstruction loss
            if use_fitness_weighting:
                recon_loss_per_sample = F.binary_cross_entropy_with_logits(
                    recon_logits, batch, reduction='none'
                ).sum(dim=1)
                recon_loss = (recon_loss_per_sample * batch_weights).mean()
            else:
                recon_loss = F.binary_cross_entropy_with_logits(
                    recon_logits, batch, reduction='sum'
                ) / len(batch)

            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()

            # Fitness prediction loss
            pred_fitness = fitness_predictor(z)
            fit_loss = F.mse_loss(pred_fitness, batch_fitness)

            # Total loss
            loss = recon_loss + beta * kl_loss + fitness_weight * fit_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_fit_loss += fit_loss.item()
            n_batches += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            avg_fit = epoch_fit_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Fit={avg_fit:.4f}, Beta={beta:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'fitness_predictor_state': fitness_predictor.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'type': 'binary_regvae'
    }

    return model


def learn_binary_momvae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Moment-matching Binary VAE (Mom-VAE) model

    This variant adds a statistical alignment loss to ensure generated samples
    match the global statistics (mean and std) of the elite population.

    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars) with values in {0, 1}
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    params : dict, optional
        Training parameters (same as learn_binary_vae, plus):
        - 'moment_weight': weight for moment matching loss (default: 0.1)

    Returns
    -------
    model : dict
        Dictionary containing model state and parameters
    """
    if params is None:
        params = {}

    pop_size = population.shape[0]
    n_vars = population.shape[1]

    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)

    # Dynamic hidden layer computation
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]

    # Extract parameters
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)
    moment_weight = params.get('moment_weight', 0.1)

    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)

    # Compute target moments (for the whole population)
    target_mean = torch.mean(data, dim=0)
    target_std = torch.std(data, dim=0)

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end

        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_moment_loss = 0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)
            recon_probs = torch.sigmoid(recon_logits)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)

            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()

            # Moment matching loss (on probabilities, not hard samples)
            batch_mean = torch.mean(recon_probs, dim=0)
            batch_std = torch.std(recon_probs, dim=0)
            moment_loss = F.mse_loss(batch_mean, target_mean) + F.mse_loss(batch_std, target_std)

            # Total loss
            loss = recon_loss + beta * kl_loss + moment_weight * moment_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_moment_loss += moment_loss.item()
            n_batches += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches
            avg_moment = epoch_moment_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Moment={avg_moment:.4f}, Beta={beta:.4f}")

    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'target_mean': target_mean.numpy(),
        'target_std': target_std.numpy(),
        'type': 'binary_momvae'
    }

    return model


# ==============================================================================
# Enhanced VAE Variants (Based on DISCRETE_VAE_ANALYSIS.md)
# ==============================================================================

def learn_binary_bavae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Beta-Annealed VAE (BA-VAE) - Addresses posterior collapse
    
    This variant implements cyclical beta annealing to prevent KL divergence
    vanishing (posterior collapse). Uses a cyclical schedule where beta starts
    low, increases to allow reconstruction learning, then decreases again.
    
    Key Features:
    - Cyclical beta annealing schedule (Fu et al. 2019)
    - Prevents posterior collapse by giving reconstruction priority early
    - Multiple cycles allow learning of both reconstruction and latent structure
    
    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values of shape (pop_size,)
    params : dict, optional
        - 'n_cycles': number of beta annealing cycles (default: 4)
        - 'beta_max': maximum beta value (default: 1.0)
        - Other standard VAE parameters
    
    Returns
    -------
    model : dict
        Model dictionary with trained networks
    """
    if params is None:
        params = {}
    
    # Override beta annealing to use cyclical schedule
    n_cycles = params.get('n_cycles', 4)
    beta_max = params.get('beta_max', 1.0)
    epochs = params.get('epochs', 100)
    
    # Compute cycle length
    cycle_length = epochs // n_cycles
    
    # Use standard VAE learning but with modified parameters
    modified_params = params.copy()
    modified_params['beta_start'] = 0.0
    modified_params['beta_end'] = beta_max
    modified_params['beta_annealing_epochs'] = cycle_length // 2
    modified_params['use_cyclical_beta'] = True
    modified_params['n_cycles'] = n_cycles
    
    # Learn using standard binary VAE with cyclical beta
    return _learn_binary_vae_with_cyclical_beta(population, fitness, modified_params)


def _learn_binary_vae_with_cyclical_beta(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Internal implementation of VAE with cyclical beta annealing
    """
    pop_size = population.shape[0]
    n_vars = population.shape[1]
    
    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)
    
    # Dynamic architecture
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]
    
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    
    # Cyclical beta parameters
    n_cycles = params.get('n_cycles', 4)
    beta_max = params.get('beta_max', 1.0)
    cycle_length = epochs // n_cycles
    
    # Activation functions
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    # Training loop with cyclical beta
    encoder.train()
    decoder.train()
    
    for epoch in range(epochs):
        # Cyclical beta annealing (sawtooth pattern)
        # Beta goes from 0 to beta_max within each cycle, then resets
        cycle_pos = (epoch % cycle_length) / cycle_length
        beta = beta_max * cycle_pos  # Linear increase within each cycle (resets at cycle end)
        
        # Shuffle data
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            
            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)
            
            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()
            
            # Total loss with cyclical beta
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'type': 'binary_bavae'
    }
    
    return model


def learn_binary_aavae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Adaptive-Architecture VAE (AA-VAE) - Addresses overfitting
    
    This variant uses very conservative architecture sizing to prevent overfitting.
    It targets ~1-2 parameters per training sample, much smaller than standard VAE.
    
    Key Features:
    - Ultra-small hidden layers (sqrt(n_vars * pop_size))
    - Very small latent dimension (n_vars // 10)
    - Dropout regularization (0.3)
    - L2 weight decay
    
    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values of shape (pop_size,)
    params : dict, optional
        - 'dropout': dropout rate (default: 0.3)
        - 'weight_decay': L2 regularization (default: 0.0001)
        - Other standard VAE parameters
    
    Returns
    -------
    model : dict
        Model dictionary with trained networks
    """
    if params is None:
        params = {}
    
    pop_size = population.shape[0]
    n_vars = population.shape[1]
    
    # Ultra-conservative architecture to prevent overfitting
    # Target: ~1-2 params per sample
    max_hidden = int(np.sqrt(n_vars * pop_size))
    h1 = min(max_hidden, max(8, n_vars // 2))
    h2 = min(max_hidden // 2, max(4, n_vars // 4))
    
    # Very small latent dimension
    latent_dim = max(2, min(n_vars // 10, pop_size // 30))
    
    # Override parameters
    modified_params = params.copy()
    modified_params['hidden_dims_enc'] = params.get('hidden_dims_enc', [h1, h2])
    modified_params['hidden_dims_dec'] = params.get('hidden_dims_dec', [h2, h1])
    modified_params['latent_dim'] = params.get('latent_dim', latent_dim)
    modified_params['dropout'] = params.get('dropout', 0.3)
    modified_params['weight_decay'] = params.get('weight_decay', 0.0001)
    
    # Use regularized learning
    return _learn_binary_vae_with_regularization(population, fitness, modified_params)


def _learn_binary_vae_with_regularization(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Internal implementation of VAE with dropout and L2 regularization
    """
    pop_size = population.shape[0]
    n_vars = population.shape[1]
    
    # Extract parameters
    hidden_dims_enc = params['hidden_dims_enc']
    hidden_dims_dec = params['hidden_dims_dec']
    latent_dim = params['latent_dim']
    dropout = params.get('dropout', 0.3)
    weight_decay = params.get('weight_decay', 0.0001)
    
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    
    # Beta annealing
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)
    
    # Activation functions
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)

    # Create networks with dropout
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)
    
    # Note: Dropout is applied to the latent space during training
    # For more comprehensive dropout, the network classes would need to be modified
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    encoder.train()
    decoder.train()
    
    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end
        
        # Shuffle data
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            
            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            
            # Apply dropout to latent space during training
            if dropout > 0:
                z = F.dropout(z, p=dropout, training=True)
            
            recon_logits = decoder(z)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='sum'
            ) / len(batch)
            
            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'dropout': dropout,
        'type': 'binary_aavae'
    }
    
    return model


def learn_binary_fwvae(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn Fitness-Weighted VAE (FW-VAE) - Better fitness guidance
    
    This variant weights the reconstruction loss by fitness values, prioritizing
    accurate reconstruction of high-fitness solutions over low-fitness ones.
    
    Key Features:
    - Fitness-weighted reconstruction loss
    - Learns to reconstruct good solutions more accurately
    - Biases latent space toward high-fitness regions
    
    Parameters
    ----------
    population : np.ndarray
        Binary population of shape (pop_size, n_vars)
    fitness : np.ndarray
        Fitness values of shape (pop_size,)
    params : dict, optional
        - 'fitness_weight_strength': how much to weight by fitness (default: 2.0)
        - Other standard VAE parameters
    
    Returns
    -------
    model : dict
        Model dictionary with trained networks
    """
    if params is None:
        params = {}
    
    pop_size = population.shape[0]
    n_vars = population.shape[1]
    
    # Compute defaults
    default_batch_size = compute_default_batch_size(n_vars, pop_size)
    default_latent_dim = compute_default_latent_dim(n_vars)
    
    # Dynamic architecture
    h1 = min(n_vars, pop_size)
    latent_dim = params.get('latent_dim', default_latent_dim)
    target_params = 4.5 * pop_size
    h2 = max(4, int((target_params - n_vars * h1) / (h1 + latent_dim)))
    default_hidden_dims = [h1, h2]
    
    hidden_dims_enc = params.get('hidden_dims_enc', default_hidden_dims)
    hidden_dims_dec = params.get('hidden_dims_dec', list(reversed(default_hidden_dims)))
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    
    # Beta annealing
    beta_start = params.get('beta_start', 0.0)
    beta_end = params.get('beta_end', 1.0)
    beta_annealing_epochs = params.get('beta_annealing_epochs', epochs // 2)
    
    # Fitness weighting
    fitness_weight_strength = params.get('fitness_weight_strength', 2.0)
    
    # Activation functions
    list_act_functs_enc = params.get('list_act_functs_enc', None)
    list_act_functs_dec = params.get('list_act_functs_dec', None)
    list_init_functs_enc = params.get('list_init_functs_enc', None)
    list_init_functs_dec = params.get('list_init_functs_dec', None)

    # MI layer parameters
    mi_layer = params.get('mi_layer', False)
    g_matrix = None
    if mi_layer:
        # Compute mutual information matrix
        mi_matrix = compute_mutual_information_matrix(population)
        # Compute G-Test independence matrix
        g_matrix = compute_gtest_independence_matrix(population, alpha=0.05)
        # Get activation function for encoder (use first one)
        activation_func = list_act_functs_enc[0] if list_act_functs_enc else 'relu'

    # Convert to tensors
    data = torch.FloatTensor(population)
    fitness_tensor = torch.FloatTensor(fitness.reshape(-1, 1))

    # Normalize fitness to [0, 1] for weighting
    fitness_min = fitness_tensor.min()
    fitness_max = fitness_tensor.max()
    fitness_range = fitness_max - fitness_min
    if fitness_range > 1e-10:
        norm_fitness = (fitness_tensor - fitness_min) / fitness_range
    else:
        # All fitness values are the same
        norm_fitness = torch.ones_like(fitness_tensor)

    # Compute weights: 1.0 + fitness_weight_strength * norm_fitness
    weights = 1.0 + fitness_weight_strength * norm_fitness

    # Create networks
    encoder = BinaryVAEEncoder(n_vars, latent_dim, hidden_dims_enc,
                               list_act_functs_enc, list_init_functs_enc,
                               mi_layer=mi_layer, g_matrix=g_matrix,
                               activation_func=activation_func if mi_layer else 'relu')
    decoder = BinaryVAEDecoder(latent_dim, n_vars, hidden_dims_dec,
                               list_act_functs_dec, list_init_functs_dec)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    encoder.train()
    decoder.train()
    
    for epoch in range(epochs):
        # Beta annealing
        if epoch < beta_annealing_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_annealing_epochs)
        else:
            beta = beta_end
        
        # Shuffle data
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]
            batch_weights = weights[idx]
            
            # Forward pass
            mean, logvar = encoder(batch)
            z = reparameterize(mean, logvar)
            recon_logits = decoder(z)
            
            # Fitness-weighted reconstruction loss
            recon_loss_per_sample = F.binary_cross_entropy_with_logits(
                recon_logits, batch, reduction='none'
            ).sum(dim=1)  # Sum over variables
            
            weighted_recon_loss = (recon_loss_per_sample * batch_weights.squeeze()).mean()
            
            # KL divergence
            kl_loss = kl_divergence(mean, logvar).mean()
            
            # Total loss
            loss = weighted_recon_loss + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Return model
    model = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'latent_dim': latent_dim,
        'n_vars': n_vars,
        'hidden_dims_enc': hidden_dims_enc,
        'hidden_dims_dec': hidden_dims_dec,
        'type': 'binary_fwvae'
    }
    
    return model
