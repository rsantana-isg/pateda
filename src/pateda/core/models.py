"""
Probabilistic model representations for EDAs

This module defines the data structures for representing probabilistic models
learned during EDA execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Model:
    """
    Base class for probabilistic models

    Attributes:
        structure: Graph structure (varies by model type)
        parameters: Model parameters (tables, means, covariances, etc.)
        metadata: Additional model information
    """

    structure: Any
    parameters: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorizedModel(Model):
    """
    Factorized Distribution Algorithm model

    Equivalent to MATEDA's FDA models. Represents a model as a set of cliques
    (factors) with associated probability tables.

    Attributes:
        structure: Cliques matrix (numpy array)
                  Each row represents a clique with format:
                  [n_neighbors, n_new_vars, neighbor_indices..., new_var_indices...]
        parameters: List of probability tables, one per clique
        metadata: Additional information (generation learned, etc.)
    """

    structure: np.ndarray  # Cliques matrix
    parameters: List[np.ndarray]  # Probability tables per clique


@dataclass
class TreeModel(Model):
    """
    Tree-based probabilistic model

    Represents dependencies as a tree structure where each variable has at most
    one parent.

    Attributes:
        structure: Parent-child relationships as numpy array
                  Each row: [n_parents, parent_idx, child_idx]
        parameters: List of conditional probability tables
                   For root nodes: marginal probabilities
                   For other nodes: P(child|parent)
        metadata: Additional information (MI matrix used, etc.)
    """

    structure: np.ndarray  # Parent-child relationships
    parameters: List[np.ndarray]  # Conditional probability tables


@dataclass
class BayesianNetworkModel(Model):
    """
    Bayesian Network model

    Uses the ``bayes_nets`` library for structure and parameter learning
    (see :class:`bayes_nets.BayesianNetwork`).

    Attributes:
        structure: DAG adjacency matrix (numpy array), where
            ``structure[i, j] == 1`` means an edge i -> j (i is a parent of j).
        parameters: Conditional Probability Distributions, a dict mapping each
            variable index to ``{"parents": [...], "cpd": np.ndarray}``.
            For a root variable the CPD is a 1-D probability vector; for a
            variable with parents it is a 2-D array of shape
            ``(n_parent_configs, cardinality[var])``.
        metadata: Learning algorithm info, scores, etc.
    """

    structure: Any  # DAG adjacency matrix (numpy array)
    parameters: Any  # dict mapping var -> {"parents": [...], "cpd": np.ndarray}


@dataclass
class GaussianModel(Model):
    """
    Gaussian model for continuous variables

    Can represent univariate, multivariate, or network-structured Gaussian models.

    Attributes:
        structure: Dependency structure (None for univariate, network for structured)
        parameters: Dictionary with 'means', 'covariances', and optionally 'weights'
        metadata: Model type (univariate/multivariate/network), etc.
    """

    structure: Optional[np.ndarray] = None  # Dependency structure (if any)
    parameters: Dict[str, np.ndarray] = field(
        default_factory=dict
    )  # means, covariances, weights


@dataclass
class MarkovNetworkModel(Model):
    """
    Markov Network (Undirected graphical model)

    Used for MOA and other undirected model-based EDAs.

    Attributes:
        structure: Cliques defining the Markov network
        parameters: Potential tables for each clique
        metadata: Learning parameters, etc.
    """

    structure: np.ndarray  # Cliques
    parameters: List[np.ndarray]  # Potential tables


@dataclass
class MixtureModel(Model):
    """
    Mixture of probabilistic models

    Used for mixture of Gaussians and other mixture-based EDAs.

    Attributes:
        structure: Component structures (list of structures)
        parameters: Component parameters and mixture weights
        metadata: Number of components, EM iterations, etc.
    """

    structure: List[Any]  # One structure per component
    parameters: Dict[str, Any]  # 'components': list of params, 'weights': mixture weights


@dataclass
class NeuralNetworkModel(Model):
    """
    Neural Network model for regression-based EDAs

    Used for Backdrive-EDA and other neural network-based approaches.
    The network can learn a mapping from solutions to fitness (for backdrive)
    or from latent representations to solutions (for VAE, etc.)

    Attributes:
        structure: Neural network architecture (layer sizes, activation functions)
        parameters: Trained network weights and biases
        metadata: Training information (epochs, loss, etc.)
    """

    structure: Dict[str, Any]  # Network architecture details
    parameters: Any  # Trained network (PyTorch model, Keras model, or weights dict)
