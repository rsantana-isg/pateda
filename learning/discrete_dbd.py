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
    Supports optional fitness guidance for conditional generation
    """

    def __init__(self, n_vars: int, hidden_dims: list = None, use_fitness_guidance: bool = False):
        super(BinaryDeblendingNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_vars = n_vars
        self.use_fitness_guidance = use_fitness_guidance

        # Input: binary variables + alpha (scalar) + optional fitness (scalar)
        input_dim = n_vars + 1
        if use_fitness_guidance:
            input_dim += 1  # Add fitness dimension

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output: logits for binary values
        layers.append(nn.Linear(prev_dim, n_vars))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, fitness: torch.Tensor = None):
        """
        Predict target distribution probabilities

        Args:
            x: Binary input [batch_size, n_vars]
            alpha: Blending parameter [batch_size, 1] or [batch_size]
            fitness: Optional fitness values [batch_size, 1] or [batch_size]

        Returns:
            Logits for p1 [batch_size, n_vars]
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # Concatenate input with alpha
        inputs = [x, alpha]
        
        # Add fitness if using fitness guidance
        if self.use_fitness_guidance:
            if fitness is None:
                raise ValueError("Fitness must be provided when use_fitness_guidance=True")
            if fitness.dim() == 1:
                fitness = fitness.unsqueeze(1)
            inputs.append(fitness)

        h = torch.cat(inputs, dim=1)

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
    num_alpha_samples: int,
    fitness0: np.ndarray = None,
    fitness1: np.ndarray = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create blended binary samples for training

    For each pair (x0, x1), create multiple alpha-blended samples
    by stochastically sampling from mixture distribution

    Args:
        p0: Source population [n_samples, n_vars] (binary)
        p1: Target population [n_samples, n_vars] (binary)
        num_alpha_samples: Number of alpha values per pair
        fitness0: Optional fitness values for p0 [n_samples]
        fitness1: Optional fitness values for p1 [n_samples]

    Returns:
        alpha: Alpha values [n_total, 1]
        x_blended: Blended samples [n_total, n_vars]
        diff_target: Difference (x1 - x0) [n_total, n_vars]
        fitness_blended: Blended fitness values [n_total, 1] (or None if not provided)
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

    # CRITICAL FIX: Target is now the DIFFERENCE (x1 - x0)
    # This matches continuous DbD formulation where network learns velocity/direction
    diff_target = x1 - x0  # Values in {-1, 0, 1} for binary variables

    # Blend fitness values if provided
    fitness_blended = None
    if fitness0 is not None and fitness1 is not None:
        # Repeat fitness values
        f0 = np.repeat(fitness0.reshape(-1, 1), num_alpha_samples, axis=0)
        f1 = np.repeat(fitness1.reshape(-1, 1), num_alpha_samples, axis=0)
        # Linear blend of fitness: f_blend = (1-alpha)*f0 + alpha*f1
        fitness_blended = (1 - alpha) * f0 + alpha * f1

    # Convert to tensors
    alpha_tensor = torch.FloatTensor(alpha)
    x_blended_tensor = torch.FloatTensor(x_blended)
    diff_tensor = torch.FloatTensor(diff_target)
    fitness_tensor = torch.FloatTensor(fitness_blended) if fitness_blended is not None else None

    return alpha_tensor, x_blended_tensor, diff_tensor, fitness_tensor


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


def compute_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_function: str = 'mse',
    fitness: torch.Tensor = None,
    fitness_weight: float = 0.1
) -> torch.Tensor:
    """
    Compute loss with different loss functions
    
    Args:
        predicted: Predicted values [batch_size, n_vars]
        target: Target values [batch_size, n_vars]
        loss_function: Type of loss ('mse', 'weighted_mse', 'ranking', 'huber')
        fitness: Fitness values for weighting [batch_size, 1] (required for weighted_mse and ranking)
        fitness_weight: Weight for fitness guidance (default: 0.1)
    
    Returns:
        loss: Computed loss value
    """
    if loss_function == 'mse':
        # Standard MSE loss
        criterion = nn.MSELoss()
        return criterion(predicted, target)
    
    elif loss_function == 'weighted_mse':
        # Fitness-weighted MSE: higher fitness solutions get more weight
        if fitness is None:
            raise ValueError("Fitness values required for weighted_mse loss")
        
        # Normalize fitness to [0, 1] range
        fitness_normalized = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)
        
        # Compute weights: higher fitness → higher weight
        # Add fitness_weight to ensure minimum weight
        weights = fitness_weight + (1 - fitness_weight) * fitness_normalized
        
        # Weighted MSE
        squared_errors = (predicted - target) ** 2
        weighted_errors = weights * squared_errors
        return weighted_errors.mean()
    
    elif loss_function == 'ranking':
        # Ranking loss: focus on relative ordering of solutions
        if fitness is None:
            raise ValueError("Fitness values required for ranking loss")
        
        # Standard MSE loss
        mse_loss = F.mse_loss(predicted, target)
        
        # Ranking loss component: encourage network to preserve fitness ordering
        # For pairs with different fitness, the difference in predictions should correlate with fitness difference
        batch_size = predicted.shape[0]
        if batch_size > 1:
            # Sample pairs randomly for ranking
            n_pairs = min(batch_size // 2, 10)  # Sample up to 10 pairs
            idx1 = torch.randint(0, batch_size, (n_pairs,))
            idx2 = torch.randint(0, batch_size, (n_pairs,))
            
            # Fitness differences
            fitness_diff = fitness[idx1] - fitness[idx2]
            
            # Prediction magnitude differences (L2 norm of predicted difference vectors)
            pred_norm1 = torch.norm(predicted[idx1], dim=1, keepdim=True)
            pred_norm2 = torch.norm(predicted[idx2], dim=1, keepdim=True)
            pred_diff = pred_norm1 - pred_norm2
            
            # Ranking loss: sign of prediction difference should match sign of fitness difference
            # Use Huber loss for robustness
            ranking_loss = F.smooth_l1_loss(pred_diff, fitness_diff.sign() * 0.1)
            
            # Combine MSE and ranking loss
            return mse_loss + fitness_weight * ranking_loss
        else:
            return mse_loss
    
    elif loss_function == 'huber':
        # Huber loss: robust to outliers
        criterion = nn.SmoothL1Loss()  # Huber loss in PyTorch
        return criterion(predicted, target)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_function}. Must be one of: mse, weighted_mse, ranking, huber")


def learn_binary_dbd(
    p0: np.ndarray,
    p1: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness0: np.ndarray = None,
    fitness1: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn binary discrete deblending model

    Args:
        p0: Source distribution samples [n_samples, n_vars] (binary)
        p1: Target distribution samples [n_samples, n_vars] (binary)
        params: Training parameters:
            - 'hidden_dims': hidden layer dimensions
              (default: [16, 8] - reduced to prevent overfitting)
            - 'list_act_functs': list of activation functions for hidden layers
            - 'list_init_functs': list of initialization functions for hidden layers
            - 'epochs': training epochs (default: 100)
            - 'batch_size': batch size (default: max(8, n_vars/50))
            - 'learning_rate': learning rate (default: 0.001)
            - 'num_alpha_samples': alpha samples per pair (default: 20)
            - 'loss_function': loss type ('mse', 'weighted_mse', 'ranking', 'huber')
            - 'use_fitness_guidance': use fitness as conditional input (default: False)
            - 'fitness_weight': weight for fitness-based losses (default: 0.1)
        fitness0: Optional fitness values for p0 [n_samples]
        fitness1: Optional fitness values for p1 [n_samples]

    Returns:
        model: Model dictionary
    """
    if params is None:
        params = {}

    n_samples = p1.shape[0]
    n_vars = p0.shape[1]

    # CRITICAL FIX: Smaller default network to prevent overfitting
    # Old default [64, 32] had ~5000 parameters for 30 vars
    # New default [16, 8] has ~700 parameters
    default_hidden_dims = [max(16, n_vars // 2), max(8, n_vars // 4)]
    default_batch_size = compute_default_batch_size(n_vars, n_samples)

    # Extract parameters with new defaults
    hidden_dims = params.get('hidden_dims', default_hidden_dims)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', default_batch_size)
    learning_rate = params.get('learning_rate', 0.001)
    # CRITICAL FIX: Increase training data from 10 to 20 alpha samples
    num_alpha_samples = params.get('num_alpha_samples', 20)
    
    # NEW: Loss function and fitness guidance parameters
    loss_function = params.get('loss_function', 'mse')
    use_fitness_guidance = params.get('use_fitness_guidance', False)
    fitness_weight = params.get('fitness_weight', 0.1)

    # Extract activation and initialization function lists
    list_act_functs = params.get('list_act_functs', None)
    list_init_functs = params.get('list_init_functs', None)

    # Create training dataset
    # CRITICAL FIX: Now returns difference (x1-x0) instead of x1
    alpha, x_blended, diff_target, fitness_blended = create_blended_binary_samples(
        p0, p1, num_alpha_samples, fitness0, fitness1
    )

    # Create network with optional fitness guidance
    network = BinaryDeblendingNet(n_vars, hidden_dims, use_fitness_guidance)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

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
            batch_diff_target = diff_target[idx]
            batch_fitness = fitness_blended[idx] if fitness_blended is not None else None

            # Predict difference (x1 - x0)
            if use_fitness_guidance:
                predicted_diff = network(batch_x, batch_alpha, batch_fitness)
            else:
                predicted_diff = network(batch_x, batch_alpha)

            # Compute loss using specified loss function
            loss = compute_loss(
                predicted_diff, 
                batch_diff_target, 
                loss_function,
                batch_fitness,
                fitness_weight
            )

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
        'use_fitness_guidance': use_fitness_guidance,
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


# ==============================================================================
# Helper Functions for DbD Variants
# ==============================================================================

def find_closest_neighbors_binary(
    source_matrix: np.ndarray,
    reference_matrix: np.ndarray
) -> np.ndarray:
    """
    Find closest row in reference_matrix for each row in source_matrix
    based on Hamming distance (for binary variables)

    Parameters:
    -----------
    source_matrix : np.ndarray
        Source matrix of shape (N, m)
    reference_matrix : np.ndarray
        Reference matrix of shape (M, m)

    Returns:
    --------
    np.ndarray
        Closest neighbors matrix of shape (N, m)
    """
    N, m = source_matrix.shape
    M = reference_matrix.shape[0]

    # Compute pairwise Hamming distances
    # For binary: distance = sum of XOR
    distances = np.zeros((N, M))
    for i in range(N):
        distances[i] = np.sum(source_matrix[i] != reference_matrix, axis=1)

    # Find the index of the closest reference row for each source row
    closest_indices = np.argmin(distances, axis=1)

    # Retrieve the closest rows from reference_matrix
    closest_neighbor_matrix = reference_matrix[closest_indices]

    return closest_neighbor_matrix


def learn_univariate_binary(population: np.ndarray) -> np.ndarray:
    """
    Learn univariate marginal probabilities for binary variables

    Parameters:
    -----------
    population : np.ndarray
        Binary population matrix [n_samples, n_vars]

    Returns:
    --------
    np.ndarray
        Marginal probabilities p(x_i=1) for each variable [n_vars]
    """
    return np.mean(population, axis=0)


def sample_from_univariate_binary(
    marginal_probs: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Sample from univariate marginal distribution

    Parameters:
    -----------
    marginal_probs : np.ndarray
        Marginal probabilities [n_vars]
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    np.ndarray
        Sampled binary matrix [n_samples, n_vars]
    """
    n_vars = len(marginal_probs)
    samples = np.random.rand(n_samples, n_vars) < marginal_probs
    return samples.astype(float)


# ==============================================================================
# DbD Variant Learning Functions
# ==============================================================================

def learn_binary_dbd_cs(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-CS (Current to Selected) model

    DbD-CS learns the transition from current population to selected population.
    This is the standard DbD approach.

    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples]
        fitness_selected: Optional fitness values for selected_pop [n_samples]

    Returns:
        model: Model dictionary with 'variant' field set to 'cs'
    """
    # DbD-CS is the same as the standard DbD
    model = learn_binary_dbd(current_pop, selected_pop, params, fitness_current, fitness_selected)
    model['variant'] = 'cs'
    model['marginal_probs'] = None  # Not used in CS variant
    return model


def learn_binary_dbd_cd(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-CD (Current to Closest in selected - Distance) model

    DbD-CD pairs each solution in current population with its closest
    neighbor in the selected population based on Hamming distance.

    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples]
        fitness_selected: Optional fitness values for selected_pop [n_samples]

    Returns:
        model: Model dictionary with 'variant' field set to 'cd'
    """
    # Find closest neighbors in selected population
    p1_closest = find_closest_neighbors_binary(current_pop, selected_pop)
    
    # If fitness is provided, also match fitness values to closest neighbors
    fitness1_matched = None
    if fitness_selected is not None:
        # Get indices of closest neighbors
        n_current = current_pop.shape[0]
        fitness1_matched = np.zeros(n_current)
        for i in range(n_current):
            # Find the index of the closest neighbor
            dists = np.sum(np.abs(current_pop[i] - selected_pop), axis=1)
            closest_idx = np.argmin(dists)
            fitness1_matched[i] = fitness_selected[closest_idx]

    # Learn model from current to closest selected
    model = learn_binary_dbd(current_pop, p1_closest, params, fitness_current, fitness1_matched)
    model['variant'] = 'cd'
    model['marginal_probs'] = None  # Not used in CD variant
    return model


def learn_binary_dbd_uc(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-UC (Univariate approximation to Current) model

    DbD-UC learns the transition from a univariate approximation of the
    current population to the current population. This helps the model
    learn to recover variable dependencies.

    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population (not used for p1, but needed for consistency)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples]
        fitness_selected: Optional fitness values for selected_pop [n_samples] (not used)

    Returns:
        model: Model dictionary with 'variant' field set to 'uc'
    """
    if params is None:
        params = {}

    n_samples = current_pop.shape[0]
    n_vars = current_pop.shape[1]

    # Learn univariate marginals from current population
    marginal_probs = learn_univariate_binary(current_pop)

    # Sample from univariate distribution
    to_take = params.get('to_take', n_samples * 2)
    p0_univariate = sample_from_univariate_binary(marginal_probs, to_take)

    # Sample from current population
    p1_indices = np.random.randint(0, n_samples, size=to_take)
    p1_samples = current_pop[p1_indices, :]
    
    # Match fitness if provided
    fitness1_matched = None
    if fitness_current is not None:
        fitness1_matched = fitness_current[p1_indices]

    # Learn model from univariate to current
    # Note: p0 (univariate) has no fitness, so we pass None for fitness0
    model = learn_binary_dbd(p0_univariate, p1_samples, params, None, fitness1_matched)
    model['variant'] = 'uc'
    model['marginal_probs'] = marginal_probs
    return model


def learn_binary_dbd_us(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-US (Univariate approximation to Selected) model

    DbD-US learns the transition from a univariate approximation of the
    current population to the selected population. This learns both
    variable dependencies and fitness-improving changes.

    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples] (not used)
        fitness_selected: Optional fitness values for selected_pop [n_samples]

    Returns:
        model: Model dictionary with 'variant' field set to 'us'
    """
    if params is None:
        params = {}

    n_current = current_pop.shape[0]
    n_selected = selected_pop.shape[0]

    # Learn univariate marginals from current population
    marginal_probs = learn_univariate_binary(current_pop)

    # Sample from univariate distribution
    to_take = params.get('to_take', n_current * 2)
    p0_univariate = sample_from_univariate_binary(marginal_probs, to_take)

    # Sample from selected population
    p1_indices = np.random.randint(0, n_selected, size=to_take)
    p1_samples = selected_pop[p1_indices, :]
    
    # Match fitness if provided
    fitness1_matched = None
    if fitness_selected is not None:
        fitness1_matched = fitness_selected[p1_indices]

    # Learn model from univariate to selected
    # Note: p0 (univariate) has no fitness, so we pass None for fitness0
    model = learn_binary_dbd(p0_univariate, p1_samples, params, None, fitness1_matched)
    model['variant'] = 'us'
    model['marginal_probs'] = marginal_probs
    return model


# ==============================================================================
# DbD-T: Binary DbD with Markov Transformation
# ==============================================================================

def _prepare_dbd_params_for_continuous(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to prepare parameters for continuous DbD learning.
    Removes parameters specific to the transformation step.
    
    Parameters:
    -----------
    params : dict
        Original parameters dictionary
        
    Returns:
    --------
    dict
        Parameters suitable for continuous DbD learning
    """
    dbd_params = params.copy()
    dbd_params.pop('k', None)  # Remove k as it's not used by continuous DbD
    dbd_params.pop('alpha', None)  # Remove alpha to avoid confusion with continuous learning rate
    return dbd_params


def compute_conditional_probabilities(
    population: np.ndarray,
    k: int,
    alpha: float = 0.1
) -> list:
    """
    Compute conditional probabilities P(X_i | X_{i-1}, ..., X_{i-k}) for binary variables
    
    Parameters:
    -----------
    population : np.ndarray
        Binary population matrix [n_samples, n_vars]
    k : int
        Number of previous variables to condition on (order of Markov chain)
    alpha : float
        Smoothing parameter (Laplace smoothing)
        
    Returns:
    --------
    list of np.ndarray
        List of conditional probability tables, one per variable
        For variable i with k parents: cpd[parent_config, var_value]
        where parent_config is an index from 0 to 2^k - 1
    """
    n_samples, n_vars = population.shape
    conditional_probs = []
    
    # For first k variables, use marginal probabilities (no conditioning)
    for var in range(min(k, n_vars)):
        # Compute marginal P(X_i = 1) with smoothing
        count_0 = np.sum(population[:, var] == 0) + alpha
        count_1 = np.sum(population[:, var] == 1) + alpha
        total = count_0 + count_1
        prob_0 = count_0 / total
        prob_1 = count_1 / total
        
        # Store as a simple array [P(X=0), P(X=1)]
        marginal = np.array([prob_0, prob_1])
        conditional_probs.append(marginal)
    
    # For remaining variables, compute conditional probabilities
    for var in range(k, n_vars):
        # Number of previous variables to condition on
        n_parents = min(k, var)
        parent_vars = list(range(var - n_parents, var))
        n_parent_configs = 2 ** n_parents  # For binary variables
        
        # Count conditional occurrences
        # cpd[parent_config, var_value] = count
        cpd = np.zeros((n_parent_configs, 2))
        
        for sample_idx in range(n_samples):
            # Calculate parent configuration index (binary encoding)
            config_idx = 0
            for i, parent_var in enumerate(parent_vars):
                config_idx += int(population[sample_idx, parent_var]) * (2 ** i)
            
            var_value = int(population[sample_idx, var])
            cpd[config_idx, var_value] += 1
        
        # Apply smoothing
        cpd += alpha
        
        # Normalize each row to get conditional probabilities
        row_sums = cpd.sum(axis=1, keepdims=True)
        cpd = cpd / row_sums
        
        conditional_probs.append(cpd)
    
    return conditional_probs


def transform_binary_to_continuous(
    population: np.ndarray,
    conditional_probs: list,
    k: int
) -> np.ndarray:
    """
    Transform binary population to continuous using conditional probabilities
    
    For each variable X_i with value x_i, the continuous value is
    P(X_i = x_i | X_{i-1} = x_{i-1}, ..., X_{i-k} = x_{i-k})
    
    Parameters:
    -----------
    population : np.ndarray
        Binary population matrix [n_samples, n_vars]
    conditional_probs : list
        List of conditional probability tables from compute_conditional_probabilities
    k : int
        Number of previous variables used for conditioning
        
    Returns:
    --------
    np.ndarray
        Continuous population [n_samples, n_vars] with values in [0, 1]
    """
    n_samples, n_vars = population.shape
    continuous_pop = np.zeros_like(population, dtype=float)
    
    for var in range(n_vars):
        cpd = conditional_probs[var]
        
        if var < k:
            # For first k variables, use marginal probabilities
            for sample_idx in range(n_samples):
                var_value = int(population[sample_idx, var])
                continuous_pop[sample_idx, var] = cpd[var_value]
        else:
            # For remaining variables, use conditional probabilities
            n_parents = min(k, var)
            parent_vars = list(range(var - n_parents, var))
            
            for sample_idx in range(n_samples):
                # Calculate parent configuration index
                config_idx = 0
                for i, parent_var in enumerate(parent_vars):
                    config_idx += int(population[sample_idx, parent_var]) * (2 ** i)
                
                var_value = int(population[sample_idx, var])
                continuous_pop[sample_idx, var] = cpd[config_idx, var_value]
    
    return continuous_pop


def learn_binary_dbd_cs_t(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-CS-T (Current to Selected with Transformation) model
    
    This variant transforms binary populations to continuous using conditional
    probabilities, then applies continuous DbD learning.
    
    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters including:
            - 'k': order of Markov chain for conditional probabilities (default: 1)
            - 'alpha': smoothing parameter (default: 0.1)
            - All other params from learn_dbd (continuous)
    
    Returns:
        model: Model dictionary with 'variant' field set to 'cs_t'
    """
    if params is None:
        params = {}
    
    k = params.get('k', 1)
    alpha_smooth = params.get('alpha', 0.1)
    
    # Step 1: Compute conditional probabilities for source population
    conditional_probs_p0 = compute_conditional_probabilities(current_pop, k, alpha_smooth)
    
    # Step 2: Compute conditional probabilities for target population
    conditional_probs_p1 = compute_conditional_probabilities(selected_pop, k, alpha_smooth)
    
    # Step 3: Transform binary populations to continuous
    p0_continuous = transform_binary_to_continuous(current_pop, conditional_probs_p0, k)
    p1_continuous = transform_binary_to_continuous(selected_pop, conditional_probs_p1, k)
    
    # Step 4: Learn continuous DbD model
    from pateda.learning.dbd import learn_dbd
    dbd_params = _prepare_dbd_params_for_continuous(params)
    
    model = learn_dbd(p0_continuous, p1_continuous, dbd_params)
    
    # Step 5: Add transformation info to model
    model['variant'] = 'cs_t'
    model['k'] = k
    model['alpha_smooth'] = alpha_smooth
    model['conditional_probs'] = conditional_probs_p1  # Store target conditional probs for sampling
    
    return model


def learn_binary_dbd_cd_t(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-CD-T (Current to Closest in selected - Distance with Transformation)
    
    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples] (not used in _t variants)
        fitness_selected: Optional fitness values for selected_pop [n_samples] (not used in _t variants)
    
    Returns:
        model: Model dictionary with 'variant' field set to 'cd_t'
    """
    if params is None:
        params = {}
    
    k = params.get('k', 1)
    alpha_smooth = params.get('alpha', 0.1)
    
    # Find closest neighbors in selected population
    p1_closest = find_closest_neighbors_binary(current_pop, selected_pop)
    
    # Compute conditional probabilities
    conditional_probs_p0 = compute_conditional_probabilities(current_pop, k, alpha_smooth)
    conditional_probs_p1 = compute_conditional_probabilities(p1_closest, k, alpha_smooth)
    
    # Transform to continuous
    p0_continuous = transform_binary_to_continuous(current_pop, conditional_probs_p0, k)
    p1_continuous = transform_binary_to_continuous(p1_closest, conditional_probs_p1, k)
    
    # Learn continuous DbD model
    from pateda.learning.dbd import learn_dbd
    dbd_params = _prepare_dbd_params_for_continuous(params)
    
    model = learn_dbd(p0_continuous, p1_continuous, dbd_params)
    
    model['variant'] = 'cd_t'
    model['k'] = k
    model['alpha_smooth'] = alpha_smooth
    model['conditional_probs'] = conditional_probs_p1
    
    return model


def learn_binary_dbd_uc_t(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-UC-T (Univariate to Current with Transformation)
    
    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population (not used for p1, but needed for consistency)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples] (not used in _t variants)
        fitness_selected: Optional fitness values for selected_pop [n_samples] (not used in _t variants)
    
    Returns:
        model: Model dictionary with 'variant' field set to 'uc_t'
    """
    if params is None:
        params = {}
    
    k = params.get('k', 1)
    alpha_smooth = params.get('alpha', 0.1)
    n_samples = current_pop.shape[0]
    n_vars = current_pop.shape[1]
    
    # Learn univariate marginals from current population
    marginal_probs = learn_univariate_binary(current_pop)
    
    # Sample from univariate distribution
    to_take = params.get('to_take', n_samples * 2)
    p0_univariate = sample_from_univariate_binary(marginal_probs, to_take)
    
    # Sample from current population
    p1_indices = np.random.randint(0, n_samples, size=to_take)
    p1_samples = current_pop[p1_indices, :]
    
    # Compute conditional probabilities
    conditional_probs_p0 = compute_conditional_probabilities(p0_univariate, k, alpha_smooth)
    conditional_probs_p1 = compute_conditional_probabilities(p1_samples, k, alpha_smooth)
    
    # Transform to continuous
    p0_continuous = transform_binary_to_continuous(p0_univariate, conditional_probs_p0, k)
    p1_continuous = transform_binary_to_continuous(p1_samples, conditional_probs_p1, k)
    
    # Learn continuous DbD model
    from pateda.learning.dbd import learn_dbd
    dbd_params = _prepare_dbd_params_for_continuous(params)
    
    model = learn_dbd(p0_continuous, p1_continuous, dbd_params)
    
    model['variant'] = 'uc_t'
    model['k'] = k
    model['alpha_smooth'] = alpha_smooth
    model['marginal_probs'] = marginal_probs
    model['conditional_probs'] = conditional_probs_p1
    
    return model


def learn_binary_dbd_us_t(
    current_pop: np.ndarray,
    selected_pop: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fitness_current: np.ndarray = None,
    fitness_selected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Learn DbD-US-T (Univariate to Selected with Transformation)
    
    Args:
        current_pop: Current population samples [n_samples, n_vars] (binary)
        selected_pop: Selected population samples [n_samples, n_vars] (binary)
        params: Training parameters
        fitness_current: Optional fitness values for current_pop [n_samples] (not used in _t variants)
        fitness_selected: Optional fitness values for selected_pop [n_samples] (not used in _t variants)
    
    Returns:
        model: Model dictionary with 'variant' field set to 'us_t'
    """
    if params is None:
        params = {}
    
    k = params.get('k', 1)
    alpha_smooth = params.get('alpha', 0.1)
    n_current = current_pop.shape[0]
    n_selected = selected_pop.shape[0]
    
    # Learn univariate marginals from current population
    marginal_probs = learn_univariate_binary(current_pop)
    
    # Sample from univariate distribution
    to_take = params.get('to_take', n_current * 2)
    p0_univariate = sample_from_univariate_binary(marginal_probs, to_take)
    
    # Sample from selected population
    p1_indices = np.random.randint(0, n_selected, size=to_take)
    p1_samples = selected_pop[p1_indices, :]
    
    # Compute conditional probabilities
    conditional_probs_p0 = compute_conditional_probabilities(p0_univariate, k, alpha_smooth)
    conditional_probs_p1 = compute_conditional_probabilities(p1_samples, k, alpha_smooth)
    
    # Transform to continuous
    p0_continuous = transform_binary_to_continuous(p0_univariate, conditional_probs_p0, k)
    p1_continuous = transform_binary_to_continuous(p1_samples, conditional_probs_p1, k)
    
    # Learn continuous DbD model
    from pateda.learning.dbd import learn_dbd
    dbd_params = _prepare_dbd_params_for_continuous(params)
    
    model = learn_dbd(p0_continuous, p1_continuous, dbd_params)
    
    model['variant'] = 'us_t'
    model['k'] = k
    model['alpha_smooth'] = alpha_smooth
    model['marginal_probs'] = marginal_probs
    model['conditional_probs'] = conditional_probs_p1
    
    return model
