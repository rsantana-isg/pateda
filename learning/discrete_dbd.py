"""
Discrete Diffusion-by-Deblending (DbD) for Discrete EDAs

==============================================================================
OVERVIEW
==============================================================================

This module implements discrete versions of Diffusion-by-Deblending (DbD)
for binary and categorical optimization problems. DbD is a minimalist diffusion
model that learns to denoise blended distributions.

For discrete problems, we adapt the continuous alpha-deblending approach using:
1. **Probabilistic blending**: Stochastic mixing of discrete distributions
2. **Categorical denoising**: Learning to predict transition probabilities
3. **Gumbel-Softmax**: Differentiable discrete sampling

==============================================================================
DISCRETE DEBLENDING APPROACH
==============================================================================

Continuous DbD: x_alpha = (1-alpha)*x0 + alpha*x1
Discrete DbD: For each variable, sample from mixture of two distributions

For Binary Variables:
- p(x_i = 1 | alpha) = (1-alpha)*p0(x_i=1) + alpha*p1(x_i=1)
- Network learns to predict p1 given x_alpha and alpha

For Categorical Variables:
- p(x_i = k | alpha) = (1-alpha)*p0(x_i=k) + alpha*p1(x_i=k)
- Use softmax over categories

==============================================================================
KEY TECHNIQUES
==============================================================================

1. **Probabilistic Blending**:
   - Instead of deterministic interpolation, use probability mixing
   - Sample from mixture distribution during training

2. **Categorical Prediction**:
   - Network outputs probabilities over discrete values
   - Use cross-entropy loss for categorical variables
   - Binary cross-entropy for binary variables

3. **Gumbel-Softmax for Generation**:
   - Differentiable sampling during denoising
   - Temperature annealing for better discretization

4. **Iterative Denoising**:
   - Start from random distribution (alpha=0)
   - Progressively denoise toward learned distribution (alpha=1)
   - Multiple denoising steps for better sample quality

==============================================================================
ARCHITECTURE
==============================================================================

Binary DbD:
- Input: [x_blended (binary), alpha (scalar)]
- Network: MLP
- Output: Probability logits for binary values
- Loss: Binary Cross-Entropy

Categorical DbD:
- Input: [x_blended (one-hot), alpha (scalar)]
- Network: MLP with categorical outputs
- Output: Logits for each category
- Loss: Categorical Cross-Entropy

==============================================================================
USAGE CONSIDERATIONS
==============================================================================

When to use Discrete DbD:
- Alternative to VAE/GAN for discrete problems
- Simpler architecture than VAE (no encoder needed)
- More stable training than GAN
- Good for iterative refinement

Advantages:
- Simpler than VAE (no encoder, no KL term)
- More stable than GAN (no adversarial training)
- Iterative refinement allows quality control
- Probabilistic framework

Disadvantages:
- Requires two distributions (source and target)
- Iterative sampling can be slow
- Less explored for discrete problems than VAE/GAN

==============================================================================
REFERENCES
==============================================================================

- Santana, R., et al. (2023). "Learning search distributions in estimation
  of distribution algorithms with minimalist diffusion models."
- Jang, E., et al. (2016). "Categorical reparameterization with Gumbel-Softmax."

==============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
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
    """Gumbel-Softmax sampling"""
    y = logits + sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y

    return y


class BinaryDeblendingNet(nn.Module):
    """
    Network for binary discrete deblending

    Predicts p1(x=1) given blended binary sample and alpha
    """

    def __init__(self, n_vars: int, hidden_dims: list = None):
        super(BinaryDeblendingNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_vars = n_vars

        # Input: binary variables + alpha (scalar)
        layers = []
        prev_dim = n_vars + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output: logits for binary values
        layers.append(nn.Linear(prev_dim, n_vars))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor):
        """
        Predict target distribution probabilities

        Args:
            x: Binary input [batch_size, n_vars]
            alpha: Blending parameter [batch_size, 1] or [batch_size]

        Returns:
            Logits for p1 [batch_size, n_vars]
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # Concatenate input with alpha
        h = torch.cat([x, alpha], dim=1)

        # Predict logits
        logits = self.network(h)

        return logits


class CategoricalDeblendingNet(nn.Module):
    """
    Network for categorical discrete deblending

    Predicts p1(x=k) for each category k given blended sample and alpha
    """

    def __init__(self, n_vars: int, cardinality: np.ndarray, hidden_dims: list = None):
        super(CategoricalDeblendingNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_vars = n_vars
        self.cardinality = cardinality
        self.total_categories = int(np.sum(cardinality))

        # Cumulative indices
        self.cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

        # Input: one-hot encoded + alpha
        layers = []
        prev_dim = self.total_categories + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output: logits for all categories
        layers.append(nn.Linear(prev_dim, self.total_categories))

        self.network = nn.Sequential(*layers)

    def forward(self, x_onehot: torch.Tensor, alpha: torch.Tensor):
        """
        Predict target distribution logits

        Args:
            x_onehot: One-hot input [batch_size, total_categories]
            alpha: Blending parameter [batch_size, 1] or [batch_size]

        Returns:
            Logits for p1 [batch_size, total_categories]
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # Concatenate with alpha
        h = torch.cat([x_onehot, alpha], dim=1)

        # Predict logits
        logits = self.network(h)

        return logits


def create_blended_binary_samples(
    p0: np.ndarray,
    p1: np.ndarray,
    num_alpha_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create blended binary samples for training

    For each pair (x0, x1), create multiple alpha-blended samples
    by stochastically sampling from mixture distribution

    Args:
        p0: Source population [n_samples, n_vars] (binary)
        p1: Target population [n_samples, n_vars] (binary)
        num_alpha_samples: Number of alpha values per pair

    Returns:
        alpha: Alpha values [n_total, 1]
        x_blended: Blended samples [n_total, n_vars]
        x1_target: Target samples [n_total, n_vars]
    """
    n, m = p0.shape

    # Repeat samples
    x0 = np.repeat(p0, num_alpha_samples, axis=0)
    x1 = np.repeat(p1, num_alpha_samples, axis=0)

    # Sample alpha values
    alpha = np.random.uniform(0, 1, n * num_alpha_samples).reshape(-1, 1)

    # Create blended samples stochastically
    # For each bit: prob(x=1) = (1-alpha)*x0 + alpha*x1
    prob_1 = (1 - alpha) * x0 + alpha * x1
    x_blended = (np.random.rand(n * num_alpha_samples, m) < prob_1).astype(float)

    # Convert to tensors
    alpha_tensor = torch.FloatTensor(alpha)
    x_blended_tensor = torch.FloatTensor(x_blended)
    x1_tensor = torch.FloatTensor(x1)

    return alpha_tensor, x_blended_tensor, x1_tensor


def create_blended_categorical_samples(
    p0: np.ndarray,
    p1: np.ndarray,
    cardinality: np.ndarray,
    num_alpha_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create blended categorical samples for training

    Args:
        p0: Source population [n_samples, n_vars]
        p1: Target population [n_samples, n_vars]
        cardinality: Cardinality of each variable
        num_alpha_samples: Number of alpha samples

    Returns:
        alpha: Alpha values
        x_blended_onehot: Blended one-hot samples
        x1_onehot: Target one-hot samples
    """
    n, m = p0.shape
    total_categories = int(np.sum(cardinality))
    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    # Repeat samples
    x0 = np.repeat(p0, num_alpha_samples, axis=0)
    x1 = np.repeat(p1, num_alpha_samples, axis=0)

    # Sample alpha
    alpha = np.random.uniform(0, 1, n * num_alpha_samples).reshape(-1, 1)

    # Create blended samples
    x_blended = np.zeros((n * num_alpha_samples, m), dtype=int)

    for i in range(m):
        card = int(cardinality[i])
        # For each variable, sample from blended categorical distribution
        for j in range(n * num_alpha_samples):
            # Compute mixture probabilities
            p_cat = np.zeros(card)
            if x0[j, i] < card:
                p_cat[int(x0[j, i])] += (1 - alpha[j, 0])
            if x1[j, i] < card:
                p_cat[int(x1[j, i])] += alpha[j, 0]

            # Normalize
            if p_cat.sum() > 0:
                p_cat /= p_cat.sum()
            else:
                p_cat = np.ones(card) / card

            # Sample
            x_blended[j, i] = np.random.choice(card, p=p_cat)

    # Convert to one-hot
    x_blended_onehot = np.zeros((n * num_alpha_samples, total_categories))
    x1_onehot = np.zeros((n * num_alpha_samples, total_categories))

    for i in range(m):
        for j in range(n * num_alpha_samples):
            blended_val = int(x_blended[j, i])
            target_val = int(x1[j, i])
            x_blended_onehot[j, cum_card[i] + blended_val] = 1.0
            x1_onehot[j, cum_card[i] + target_val] = 1.0

    # Convert to tensors
    alpha_tensor = torch.FloatTensor(alpha)
    x_blended_tensor = torch.FloatTensor(x_blended_onehot)
    x1_tensor = torch.FloatTensor(x1_onehot)

    return alpha_tensor, x_blended_tensor, x1_tensor


def learn_binary_dbd(
    p0: np.ndarray,
    p1: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn binary discrete deblending model

    Args:
        p0: Source distribution samples [n_samples, n_vars] (binary)
        p1: Target distribution samples [n_samples, n_vars] (binary)
        params: Training parameters:
            - 'hidden_dims': hidden layer dimensions
              (default: computed from n_vars and n_samples)
            - 'list_act_functs': list of activation functions for hidden layers
            - 'list_init_functs': list of initialization functions for hidden layers
            - 'epochs': training epochs (default: 100)
            - 'batch_size': batch size (default: max(8, n_vars/50))
            - 'learning_rate': learning rate (default: 0.001)
            - 'num_alpha_samples': alpha samples per pair (default: 10)

    Returns:
        model: Model dictionary
    """
    if params is None:
        params = {}

    n_samples = p1.shape[0]
    n_vars = p0.shape[1]

    # Compute defaults based on input dimensions
    default_hidden_dims = compute_default_hidden_dims(n_vars, n_samples)
    default_batch_size = compute_default_batch_size(n_vars, n_samples)

    # Extract parameters with new defaults
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    num_alpha_samples = params.get('num_alpha_samples', 10)

    # Extract activation and initialization function lists
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)

    # Create training dataset
    alpha, x_blended, x1_target = create_blended_binary_samples(
        p0, p1, num_alpha_samples
    )

    # Create network
    network = BinaryDeblendingNet(n_vars, hidden_dims)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    network.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(alpha))

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(alpha), batch_size):
            idx = perm[i:i+batch_size]
            batch_alpha = alpha[idx]
            batch_x = x_blended[idx]
            batch_target = x1_target[idx]

            # Predict target distribution
            logits = network(batch_x, batch_alpha)

            # Loss: predict x1 from blended sample
            loss = criterion(logits, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    # Return model
    model = {
        'network_state': network.state_dict(),
        'n_vars': n_vars,
        'hidden_dims': hidden_dims,
        'type': 'binary_dbd'
    }

    return model


def learn_categorical_dbd(
    p0: np.ndarray,
    p1: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn categorical discrete deblending model

    Args:
        p0: Source samples [n_samples, n_vars]
        p1: Target samples [n_samples, n_vars]
        cardinality: Cardinality of each variable
        params: Training parameters

    Returns:
        model: Model dictionary
    """
    if params is None:
        params = {}

    n_vars = p0.shape[1]

    # Extract parameters
    hidden_dims = params.get('hidden_dims', [128, 64])
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 0.001)
    num_alpha_samples = params.get('num_alpha_samples', 10)

    # Create training dataset
    alpha, x_blended_onehot, x1_onehot = create_blended_categorical_samples(
        p0, p1, cardinality, num_alpha_samples
    )

    # Create network
    network = CategoricalDeblendingNet(n_vars, cardinality, hidden_dims)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # For one-hot targets

    # Training
    network.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(alpha))

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(alpha), batch_size):
            idx = perm[i:i+batch_size]
            batch_alpha = alpha[idx]
            batch_x = x_blended_onehot[idx]
            batch_target = x1_onehot[idx]

            # Predict
            logits = network(batch_x, batch_alpha)

            # Loss
            loss = criterion(logits, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    # Return model
    model = {
        'network_state': network.state_dict(),
        'n_vars': n_vars,
        'cardinality': cardinality,
        'hidden_dims': hidden_dims,
        'type': 'categorical_dbd'
    }

    return model
