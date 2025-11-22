"""
Neural Network Utilities for EDAs

This module provides shared utilities for neural network-based EDAs including:
- 15 activation functions supported by PyTorch
- 15 weight initialization functions
- Helper functions for computing default architecture parameters

These utilities are used by gan.py, vae.py, dbd.py, backdrive.py, dendiff.py,
and their discrete variants.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Callable


# =============================================================================
# SUPPORTED ACTIVATION FUNCTIONS (15 total)
# =============================================================================

SUPPORTED_ACTIVATIONS = [
    'relu',           # Rectified Linear Unit
    'tanh',           # Hyperbolic tangent
    'sigmoid',        # Sigmoid function
    'linear',         # No activation (identity)
    'leaky_relu',     # Leaky ReLU with negative slope
    'elu',            # Exponential Linear Unit
    'selu',           # Scaled Exponential Linear Unit
    'gelu',           # Gaussian Error Linear Unit
    'silu',           # SiLU/Swish (x * sigmoid(x))
    'softplus',       # Softplus function
    'softsign',       # Softsign function
    'mish',           # Mish activation
    'hardswish',      # Hard Swish
    'hardsigmoid',    # Hard Sigmoid
    'prelu',          # Parametric ReLU
]


def get_activation(name: str, in_features: int = None) -> nn.Module:
    """
    Get an activation function module by name.

    Parameters
    ----------
    name : str
        Name of the activation function. Must be one of SUPPORTED_ACTIVATIONS.
    in_features : int, optional
        Number of input features (required for PReLU).

    Returns
    -------
    nn.Module
        The activation function module.

    Raises
    ------
    ValueError
        If the activation name is not supported.
    """
    name = name.lower()

    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'linear' or name == 'identity':
        return nn.Identity()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'softsign':
        return nn.Softsign()
    elif name == 'mish':
        return nn.Mish()
    elif name == 'hardswish':
        return nn.Hardswish()
    elif name == 'hardsigmoid':
        return nn.Hardsigmoid()
    elif name == 'prelu':
        # PReLU has learnable parameters, needs in_features
        num_params = in_features if in_features is not None else 1
        return nn.PReLU(num_parameters=num_params)
    else:
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Supported activations: {SUPPORTED_ACTIVATIONS}"
        )


# =============================================================================
# SUPPORTED INITIALIZATION FUNCTIONS (15 total)
# =============================================================================

SUPPORTED_INITIALIZATIONS = [
    'default',          # PyTorch default (Kaiming uniform for Linear)
    'xavier_uniform',   # Xavier/Glorot uniform initialization
    'xavier_normal',    # Xavier/Glorot normal initialization
    'kaiming_uniform',  # Kaiming/He uniform initialization
    'kaiming_normal',   # Kaiming/He normal initialization
    'orthogonal',       # Orthogonal initialization
    'normal',           # Normal distribution
    'uniform',          # Uniform distribution
    'zeros',            # Initialize to zeros
    'ones',             # Initialize to ones
    'eye',              # Identity matrix (for square matrices)
    'dirac',            # Dirac delta initialization (for conv layers)
    'sparse',           # Sparse initialization
    'trunc_normal',     # Truncated normal distribution
    'constant',         # Constant value initialization
]


def apply_weight_init(module: nn.Module, init_name: str, **kwargs) -> None:
    """
    Apply weight initialization to a module.

    Parameters
    ----------
    module : nn.Module
        The module to initialize (typically nn.Linear).
    init_name : str
        Name of the initialization method. Must be one of SUPPORTED_INITIALIZATIONS.
    **kwargs : dict
        Additional keyword arguments for initialization functions.
        - For 'normal': mean (default 0.0), std (default 0.02)
        - For 'uniform': a (default -0.1), b (default 0.1)
        - For 'constant': val (default 0.0)
        - For 'sparse': sparsity (default 0.1)
        - For 'trunc_normal': mean (default 0.0), std (default 0.02), a (default -2), b (default 2)

    Raises
    ------
    ValueError
        If the initialization name is not supported.
    """
    init_name = init_name.lower()

    if not hasattr(module, 'weight'):
        return

    weight = module.weight

    if init_name == 'default':
        # Keep PyTorch default initialization
        pass
    elif init_name == 'xavier_uniform':
        nn.init.xavier_uniform_(weight, gain=kwargs.get('gain', 1.0))
    elif init_name == 'xavier_normal':
        nn.init.xavier_normal_(weight, gain=kwargs.get('gain', 1.0))
    elif init_name == 'kaiming_uniform':
        nn.init.kaiming_uniform_(
            weight,
            a=kwargs.get('a', 0),
            mode=kwargs.get('mode', 'fan_in'),
            nonlinearity=kwargs.get('nonlinearity', 'relu')
        )
    elif init_name == 'kaiming_normal':
        nn.init.kaiming_normal_(
            weight,
            a=kwargs.get('a', 0),
            mode=kwargs.get('mode', 'fan_in'),
            nonlinearity=kwargs.get('nonlinearity', 'relu')
        )
    elif init_name == 'orthogonal':
        nn.init.orthogonal_(weight, gain=kwargs.get('gain', 1.0))
    elif init_name == 'normal':
        nn.init.normal_(weight, mean=kwargs.get('mean', 0.0), std=kwargs.get('std', 0.02))
    elif init_name == 'uniform':
        nn.init.uniform_(weight, a=kwargs.get('a', -0.1), b=kwargs.get('b', 0.1))
    elif init_name == 'zeros':
        nn.init.zeros_(weight)
    elif init_name == 'ones':
        nn.init.ones_(weight)
    elif init_name == 'eye':
        if weight.dim() == 2 and weight.size(0) == weight.size(1):
            nn.init.eye_(weight)
        else:
            # For non-square matrices, use identity-like initialization
            with torch.no_grad():
                weight.zero_()
                min_dim = min(weight.size(0), weight.size(1))
                for i in range(min_dim):
                    weight[i, i] = 1.0
    elif init_name == 'dirac':
        if weight.dim() >= 3:
            nn.init.dirac_(weight, groups=kwargs.get('groups', 1))
        else:
            # For non-conv layers, use identity-like
            nn.init.eye_(weight) if weight.dim() == 2 and weight.size(0) == weight.size(1) else nn.init.xavier_uniform_(weight)
    elif init_name == 'sparse':
        if weight.dim() == 2:
            nn.init.sparse_(weight, sparsity=kwargs.get('sparsity', 0.1), std=kwargs.get('std', 0.01))
        else:
            nn.init.xavier_uniform_(weight)
    elif init_name == 'trunc_normal':
        nn.init.trunc_normal_(
            weight,
            mean=kwargs.get('mean', 0.0),
            std=kwargs.get('std', 0.02),
            a=kwargs.get('a', -2.0),
            b=kwargs.get('b', 2.0)
        )
    elif init_name == 'constant':
        nn.init.constant_(weight, val=kwargs.get('val', 0.0))
    else:
        raise ValueError(
            f"Unknown initialization: '{init_name}'. "
            f"Supported initializations: {SUPPORTED_INITIALIZATIONS}"
        )

    # Initialize bias to zeros if present
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)


# =============================================================================
# HELPER FUNCTIONS FOR DEFAULT ARCHITECTURE PARAMETERS
# =============================================================================

def compute_default_hidden_dims(n_inputs: int, pop_size: int, n_layers: int = 2) -> List[int]:
    """
    Compute default hidden layer dimensions based on input size and population size.

    The default architecture uses:
    - First layer: max(5, n_inputs / 10) neurons
    - Second layer: max(10, pop_size / 10) neurons
    - For more layers, interpolate between these values

    Parameters
    ----------
    n_inputs : int
        Number of input features (variables) to the network.
    pop_size : int
        Size of the population (training dataset).
    n_layers : int, optional
        Number of hidden layers (default: 2).

    Returns
    -------
    List[int]
        List of hidden layer dimensions.
    """
    if n_layers < 1:
        return []

    first_layer = max(5, int(n_inputs / 10))
    second_layer = max(10, int(pop_size / 10))

    if n_layers == 1:
        return [first_layer]
    elif n_layers == 2:
        return [first_layer, second_layer]
    else:
        # For more than 2 layers, interpolate
        hidden_dims = [first_layer]
        for i in range(1, n_layers - 1):
            # Linear interpolation between first and second layer sizes
            ratio = i / (n_layers - 1)
            dim = int(first_layer + ratio * (second_layer - first_layer))
            hidden_dims.append(max(5, dim))
        hidden_dims.append(second_layer)
        return hidden_dims


def compute_default_batch_size(n_inputs: int, pop_size: int) -> int:
    """
    Compute default batch size based on input dimensions.

    Default: max(8, n_inputs / 50)

    Parameters
    ----------
    n_inputs : int
        Number of input features (variables).
    pop_size : int
        Size of the population (to ensure batch size doesn't exceed it).

    Returns
    -------
    int
        Default batch size.
    """
    batch_size = max(8, int(n_inputs / 50))
    # Ensure batch size doesn't exceed population size
    batch_size = min(batch_size, pop_size // 2) if pop_size > 1 else batch_size
    return max(1, batch_size)


def compute_default_latent_dim(n_inputs: int) -> int:
    """
    Compute default latent dimension for VAE models.

    Default: max(2, n_inputs / 50)

    Parameters
    ----------
    n_inputs : int
        Number of input features (variables).

    Returns
    -------
    int
        Default latent dimension.
    """
    return max(2, int(n_inputs / 50))


# =============================================================================
# LAYER BUILDING UTILITIES
# =============================================================================

def build_hidden_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    list_act_functs: Optional[List[str]] = None,
    list_init_functs: Optional[List[str]] = None,
    output_activation: Optional[str] = None,
    output_init: Optional[str] = None,
    dropout: float = 0.0,
    batch_norm: bool = False
) -> nn.Sequential:
    """
    Build a sequence of hidden layers with configurable activations and initializations.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output dimension.
    list_act_functs : List[str], optional
        List of activation functions, one for each hidden layer.
        If None, uses 'relu' for all layers.
        Length must match len(hidden_dims).
    list_init_functs : List[str], optional
        List of initialization functions, one for each hidden layer.
        If None, uses 'default' for all layers.
        Length must match len(hidden_dims).
    output_activation : str, optional
        Activation function for the output layer. If None, no activation.
    output_init : str, optional
        Initialization for the output layer. If None, uses 'default'.
    dropout : float, optional
        Dropout probability (default: 0.0, no dropout).
    batch_norm : bool, optional
        Whether to use batch normalization (default: False).

    Returns
    -------
    nn.Sequential
        The constructed neural network.

    Raises
    ------
    ValueError
        If list lengths don't match hidden_dims length.
    """
    n_hidden = len(hidden_dims)

    # Validate and set defaults for activation functions
    if list_act_functs is None:
        list_act_functs = ['relu'] * n_hidden
    elif len(list_act_functs) != n_hidden:
        raise ValueError(
            f"list_act_functs length ({len(list_act_functs)}) must match "
            f"hidden_dims length ({n_hidden})"
        )

    # Validate and set defaults for initialization functions
    if list_init_functs is None:
        list_init_functs = ['default'] * n_hidden
    elif len(list_init_functs) != n_hidden:
        raise ValueError(
            f"list_init_functs length ({len(list_init_functs)}) must match "
            f"hidden_dims length ({n_hidden})"
        )

    layers = []
    prev_dim = input_dim

    # Build hidden layers
    for i, hidden_dim in enumerate(hidden_dims):
        # Linear layer
        linear = nn.Linear(prev_dim, hidden_dim)
        apply_weight_init(linear, list_init_functs[i])
        layers.append(linear)

        # Optional batch normalization
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        # Activation function
        activation = get_activation(list_act_functs[i], in_features=hidden_dim)
        layers.append(activation)

        # Optional dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        prev_dim = hidden_dim

    # Output layer
    output_linear = nn.Linear(prev_dim, output_dim)
    if output_init:
        apply_weight_init(output_linear, output_init)
    layers.append(output_linear)

    # Optional output activation
    if output_activation:
        layers.append(get_activation(output_activation))

    return nn.Sequential(*layers)


def validate_list_params(
    hidden_dims: List[int],
    list_act_functs: Optional[List[str]],
    list_init_functs: Optional[List[str]]
) -> tuple:
    """
    Validate and return list parameters for network construction.

    Parameters
    ----------
    hidden_dims : List[int]
        List of hidden layer dimensions.
    list_act_functs : List[str], optional
        List of activation functions.
    list_init_functs : List[str], optional
        List of initialization functions.

    Returns
    -------
    tuple
        (validated_act_functs, validated_init_functs)
    """
    n_hidden = len(hidden_dims)

    if list_act_functs is None:
        list_act_functs = ['relu'] * n_hidden
    elif len(list_act_functs) != n_hidden:
        raise ValueError(
            f"list_act_functs length ({len(list_act_functs)}) must match "
            f"hidden_dims length ({n_hidden})"
        )

    if list_init_functs is None:
        list_init_functs = ['default'] * n_hidden
    elif len(list_init_functs) != n_hidden:
        raise ValueError(
            f"list_init_functs length ({len(list_init_functs)}) must match "
            f"hidden_dims length ({n_hidden})"
        )

    # Validate activation names
    for act in list_act_functs:
        if act.lower() not in [a.lower() for a in SUPPORTED_ACTIVATIONS]:
            raise ValueError(f"Unknown activation: '{act}'. Supported: {SUPPORTED_ACTIVATIONS}")

    # Validate initialization names
    for init in list_init_functs:
        if init.lower() not in [i.lower() for i in SUPPORTED_INITIALIZATIONS]:
            raise ValueError(f"Unknown initialization: '{init}'. Supported: {SUPPORTED_INITIALIZATIONS}")

    return list_act_functs, list_init_functs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_network_params(n_vars: int, pop_size: int) -> dict:
    """
    Get default network parameters based on problem dimensions.

    Parameters
    ----------
    n_vars : int
        Number of variables (network input dimension).
    pop_size : int
        Population size (training dataset size).

    Returns
    -------
    dict
        Dictionary containing default parameters:
        - 'hidden_dims': default hidden layer dimensions
        - 'batch_size': default batch size
        - 'list_act_functs': default activation functions
        - 'list_init_functs': default initialization functions
    """
    hidden_dims = compute_default_hidden_dims(n_vars, pop_size)
    batch_size = compute_default_batch_size(n_vars, pop_size)

    return {
        'hidden_dims': hidden_dims,
        'batch_size': batch_size,
        'list_act_functs': ['relu'] * len(hidden_dims),
        'list_init_functs': ['default'] * len(hidden_dims),
    }


def print_supported_options():
    """Print all supported activation and initialization functions."""
    print("=" * 60)
    print("SUPPORTED ACTIVATION FUNCTIONS (15 total)")
    print("=" * 60)
    for i, act in enumerate(SUPPORTED_ACTIVATIONS, 1):
        print(f"  {i:2d}. {act}")

    print("\n" + "=" * 60)
    print("SUPPORTED INITIALIZATION FUNCTIONS (15 total)")
    print("=" * 60)
    for i, init in enumerate(SUPPORTED_INITIALIZATIONS, 1):
        print(f"  {i:2d}. {init}")
