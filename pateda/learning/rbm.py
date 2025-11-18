"""
RBM Model Learning for Discrete EDAs

This module provides learning algorithms for Restricted Boltzmann Machine (RBM) based
probabilistic models used in combinatorial optimization. Implementation based on the paper:
"Restricted Boltzmann Machine-Assisted Estimation of Distribution Algorithm for Complex Problems"
(Bao et al., Complexity 2018).

The module implements:
1. Softmax RBM: Uses softmax units for discrete variables
2. Energy-based surrogate model for fitness estimation
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRBM(nn.Module):
    """
    Restricted Boltzmann Machine with Softmax visible units.

    Suitable for discrete variables where each variable can take K possible values.
    The visible layer uses softmax units (one-hot encoding), and the hidden layer
    uses binary stochastic units.
    """

    def __init__(self, n_vars: int, cardinality: np.ndarray, n_hidden: int):
        """
        Initialize Softmax RBM.

        Parameters
        ----------
        n_vars : int
            Number of decision variables
        cardinality : np.ndarray
            Array of shape (n_vars,) with cardinality for each variable
        n_hidden : int
            Number of hidden units
        """
        super(SoftmaxRBM, self).__init__()

        self.n_vars = n_vars
        self.cardinality = cardinality
        self.n_hidden = n_hidden

        # Total number of visible softmax units
        self.n_visible = int(np.sum(cardinality))

        # Weights connecting visible to hidden layer
        # Shape: (n_visible, n_hidden)
        self.W = nn.Parameter(torch.randn(self.n_visible, n_hidden) * 0.01)

        # Biases for visible units (one bias per softmax unit)
        self.a = nn.Parameter(torch.zeros(self.n_visible))

        # Biases for hidden units
        self.b = nn.Parameter(torch.zeros(n_hidden))

        # Store cumulative indices for variable grouping
        self.cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    def _encode_population(self, population: np.ndarray) -> torch.Tensor:
        """
        Encode population into one-hot softmax representation.

        Parameters
        ----------
        population : np.ndarray
            Population array of shape (pop_size, n_vars) with discrete values

        Returns
        -------
        encoded : torch.Tensor
            One-hot encoded tensor of shape (pop_size, n_visible)
        """
        pop_size = population.shape[0]
        encoded = torch.zeros(pop_size, self.n_visible)

        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            for j in range(pop_size):
                value = int(population[j, i])
                encoded[j, start_idx + value] = 1.0

        return encoded

    def _decode_visible(self, visible: torch.Tensor) -> np.ndarray:
        """
        Decode one-hot softmax representation back to discrete values.

        Parameters
        ----------
        visible : torch.Tensor
            One-hot encoded tensor of shape (pop_size, n_visible)

        Returns
        -------
        population : np.ndarray
            Decoded population of shape (pop_size, n_vars)
        """
        pop_size = visible.shape[0]
        population = np.zeros((pop_size, self.n_vars), dtype=int)

        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            end_idx = self.cum_card[i + 1]
            # Get argmax within each variable's softmax group
            values = torch.argmax(visible[:, start_idx:end_idx], dim=1)
            population[:, i] = values.cpu().numpy()

        return population

    def sample_hidden(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units given visible units.

        P(h_j = 1 | v) = sigmoid(b_j + sum_i v_i * W_ij)

        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations (one-hot encoded)

        Returns
        -------
        hidden : torch.Tensor
            Binary hidden layer samples
        """
        activation = F.linear(visible, self.W.t(), self.b)
        prob = torch.sigmoid(activation)
        hidden = torch.bernoulli(prob)
        return hidden

    def prob_hidden_given_visible(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute P(h | v) for all hidden units.

        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations

        Returns
        -------
        prob : torch.Tensor
            Probabilities for hidden units
        """
        activation = F.linear(visible, self.W.t(), self.b)
        return torch.sigmoid(activation)

    def sample_visible(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units.

        For softmax visible units, P(v^k_i = 1 | h) uses softmax over K values.

        Parameters
        ----------
        hidden : torch.Tensor
            Hidden layer activations

        Returns
        -------
        visible : torch.Tensor
            One-hot encoded visible layer samples
        """
        pop_size = hidden.shape[0]
        visible = torch.zeros(pop_size, self.n_visible)

        # Compute activations for all visible units
        activation = F.linear(hidden, self.W, self.a)

        # Apply softmax separately for each variable
        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            end_idx = self.cum_card[i + 1]

            # Softmax over the K values for variable i
            probs = F.softmax(activation[:, start_idx:end_idx], dim=1)

            # Sample from multinomial
            samples = torch.multinomial(probs, 1).squeeze()

            # One-hot encode
            visible[torch.arange(pop_size), start_idx + samples] = 1.0

        return visible

    def prob_visible_given_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute probability distribution P(v | h) for visible units.

        Returns probability matrix where each variable uses softmax.

        Parameters
        ----------
        hidden : torch.Tensor
            Hidden layer activations

        Returns
        -------
        prob : torch.Tensor
            Probability distribution over visible units
        """
        activation = F.linear(hidden, self.W, self.a)
        prob = torch.zeros_like(activation)

        # Apply softmax for each variable separately
        for i in range(self.n_vars):
            start_idx = self.cum_card[i]
            end_idx = self.cum_card[i + 1]
            prob[:, start_idx:end_idx] = F.softmax(activation[:, start_idx:end_idx], dim=1)

        return prob

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy of visible layer.

        F(v) = -sum_i a_i * v_i - sum_j log(1 + exp(b_j + sum_i v_i * W_ij))

        This is used for the energy-based surrogate model.

        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations

        Returns
        -------
        energy : torch.Tensor
            Free energy values for each sample
        """
        # Visible bias term
        vbias_term = F.linear(visible, self.a.unsqueeze(0))

        # Hidden bias term with softplus
        hidden_activation = F.linear(visible, self.W.t(), self.b)
        hidden_term = torch.sum(F.softplus(hidden_activation), dim=1)

        return -vbias_term.squeeze() - hidden_term


def learn_softmax_rbm(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a Softmax RBM model from selected population using Contrastive Divergence.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) with discrete values in [0, K-1]
    fitness : np.ndarray
        Fitness values of shape (pop_size,) or (pop_size, n_objectives)
    cardinality : np.ndarray
        Array of shape (n_vars,) with cardinality for each variable
    params : dict, optional
        Training parameters containing:
        - 'n_hidden': number of hidden units (default: 2 * n_vars)
        - 'epochs': number of training epochs (default: 20)
        - 'batch_size': mini-batch size (default: 32)
        - 'learning_rate': learning rate (default: 0.01)
        - 'k_cd': number of Gibbs steps for CD-k (default: 1)
        - 'momentum': momentum coefficient (default: 0.5->0.9)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'rbm_state': RBM network state dict
        - 'n_vars': number of variables
        - 'cardinality': variable cardinalities
        - 'n_hidden': number of hidden units
        - 'type': 'softmax_rbm'
    """
    if params is None:
        params = {}

    n_vars = population.shape[1]

    # Extract parameters
    n_hidden = params.get('n_hidden', 2 * n_vars)
    epochs = params.get('epochs', 20)
    batch_size = params.get('batch_size', min(32, len(population) // 2))
    learning_rate = params.get('learning_rate', 0.01)
    k_cd = params.get('k_cd', 1)
    momentum_init = params.get('momentum', 0.5)

    # Create RBM
    rbm = SoftmaxRBM(n_vars, cardinality, n_hidden)

    # Encode population to one-hot
    data = rbm._encode_population(population)

    # Initialize momentum velocities
    velocity_W = torch.zeros_like(rbm.W)
    velocity_a = torch.zeros_like(rbm.a)
    velocity_b = torch.zeros_like(rbm.b)

    # Training loop
    rbm.train()

    for epoch in range(epochs):
        # Momentum schedule
        momentum = min(momentum_init + (0.9 - momentum_init) * epoch / epochs, 0.9)

        # Shuffle data
        perm = torch.randperm(len(data))

        epoch_error = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            idx = perm[i:i+batch_size]
            batch = data[idx]

            # Positive phase: compute P(h|v_0)
            v0 = batch
            h0_prob = rbm.prob_hidden_given_visible(v0)
            h0 = torch.bernoulli(h0_prob)

            # Negative phase: k steps of Gibbs sampling
            vk = v0
            hk = h0
            for _ in range(k_cd):
                vk = rbm.sample_visible(hk)
                hk_prob = rbm.prob_hidden_given_visible(vk)
                hk = torch.bernoulli(hk_prob)

            # Compute gradients
            positive_grad_W = torch.mm(v0.t(), h0_prob)
            negative_grad_W = torch.mm(vk.t(), hk_prob)

            grad_W = (positive_grad_W - negative_grad_W) / batch.shape[0]
            grad_a = torch.mean(v0 - vk, dim=0)
            grad_b = torch.mean(h0_prob - hk_prob, dim=0)

            # Update velocities with momentum
            velocity_W = momentum * velocity_W + learning_rate * grad_W
            velocity_a = momentum * velocity_a + learning_rate * grad_a
            velocity_b = momentum * velocity_b + learning_rate * grad_b

            # Update parameters
            rbm.W.data += velocity_W
            rbm.a.data += velocity_a
            rbm.b.data += velocity_b

            # Track reconstruction error
            epoch_error += torch.mean((v0 - vk) ** 2).item()
            n_batches += 1

    # Return model
    return {
        'rbm_state': rbm.state_dict(),
        'n_vars': n_vars,
        'cardinality': cardinality.copy(),
        'n_hidden': n_hidden,
        'type': 'softmax_rbm'
    }
