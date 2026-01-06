# DbD Variants Implementation Summary

## Overview

This document describes the implementation of enhanced DbD (Diffusion-by-Deblending) variants in `examples/discrete_DbD_EDA.py`. These variants extend the basic DbD algorithm with different loss functions and fitness guidance capabilities.

## Implemented Variants

### 1. DbD-Weighted (Fitness-Weighted MSE Loss)

**Usage:** `loss_function='weighted_mse'`

**Description:** Uses fitness-weighted MSE loss where solutions with higher fitness receive higher weights during training. This encourages the model to focus more on learning transitions for high-quality solutions.

**Implementation:**
- Normalizes fitness values to [0, 1] range
- Computes weights: `weight = fitness_weight + (1 - fitness_weight) * normalized_fitness`
- Applies weighted MSE: `loss = mean(weight * (predicted - target)^2)`
- Handles edge case when all fitness values are identical

**Example:**
```bash
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu weighted_mse 20 20 0 0.1 0 0
```

### 2. DbD-Ranking (Ranking Loss)

**Usage:** `loss_function='ranking'`

**Description:** Combines MSE loss with a ranking component that encourages the network to preserve the relative ordering of solutions by fitness. This helps maintain the quality hierarchy during the denoising process.

**Implementation:**
- Computes standard MSE loss
- Samples pairs of solutions and compares their fitness differences
- Adds ranking loss that penalizes when predicted magnitude differences don't match fitness ordering
- Uses Huber loss for robustness: `ranking_loss = smooth_l1_loss(pred_diff, sign(fitness_diff) * scale)`

**Example:**
```bash
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu ranking 20 20 0 0.1 0 0
```

### 3. DbD-Huber (Huber Loss)

**Usage:** `loss_function='huber'`

**Description:** Uses Huber loss (SmoothL1Loss) which is robust to outliers. This is useful when the training data contains noisy or anomalous solutions.

**Implementation:**
- Uses PyTorch's `nn.SmoothL1Loss()` 
- Behaves like L2 loss for small errors and L1 loss for large errors
- More stable than MSE when dealing with outliers

**Example:**
```bash
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu huber 20 20 0 0.1 0 0
```

### 4. C-DbD (Conditional DbD with Fitness Guidance)

**Usage:** `fitness_guided=1`

**Description:** Conditional DbD inspired by Conditional VAE (C-VAE). The network receives fitness as an additional input, allowing it to generate solutions conditioned on target fitness values.

**Implementation:**
- Network input: `[binary_variables, alpha, fitness]`
- Fitness is concatenated as an additional dimension to the input
- During sampling, high fitness values can be provided to guide generation toward high-quality solutions
- Can be combined with any loss function

**Example:**
```bash
# C-DbD with MSE loss
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu mse 20 20 0 0.1 1 0

# C-DbD with weighted MSE loss
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu weighted_mse 20 20 0 0.1 1 0
```

### 5. M-DbD (DbD with Markov Model Initialization)

**Usage:** `use_markov_init=1`

**Description:** Initializes new populations using a Markov chain model learned from selected solutions, rather than random initialization. This helps preserve local dependencies between variables.

**Implementation:**
- Learns k-order Markov chain from selected population
- Samples initial population from this Markov model
- Can be combined with any loss function and fitness guidance

**Example:**
```bash
python examples/discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu mse 20 20 0 0.1 0 1
```

## Technical Details

### Loss Functions

All loss functions are implemented in `compute_loss()` function in `learning/discrete_dbd.py`:

```python
def compute_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_function: str = 'mse',
    fitness: torch.Tensor = None,
    fitness_weight: float = 0.1
) -> torch.Tensor
```

**Parameters:**
- `predicted`: Network predictions [batch_size, n_vars]
- `target`: Target values (difference x1 - x0) [batch_size, n_vars]
- `loss_function`: One of 'mse', 'weighted_mse', 'ranking', 'huber'
- `fitness`: Fitness values [batch_size, 1] (required for weighted_mse and ranking)
- `fitness_weight`: Weight parameter for fitness-based losses

### Fitness Guidance

Fitness guidance is implemented through:

1. **Network Architecture:**
   - `BinaryDeblendingNet` accepts optional `use_fitness_guidance` parameter
   - When enabled, input dimension increases by 1 to accommodate fitness

2. **Training:**
   - Fitness values are blended: `fitness_blend = (1-alpha)*fitness0 + alpha*fitness1`
   - Network learns to predict transitions conditioned on fitness

3. **Sampling:**
   - `sample_binary_dbd()` accepts optional `target_fitness` parameter
   - During denoising, fitness is passed to network at each step
   - If not provided, defaults to high fitness value (1.0) to encourage quality

### Fitness Evaluation

Fitness values are automatically evaluated when needed:
- Always computed for loss functions: `weighted_mse`, `ranking`
- Always computed when `fitness_guided=True`
- Computed for both source and target populations
- Properly matched when sampling/pairing solutions

## Combining Features

All variants can be combined freely:

```bash
# C-DbD with Weighted MSE loss and Markov initialization
python examples/discrete_DbD_EDA.py 0 FC3 30 150 40 0.5 dbd relu weighted_mse 20 20 0 0.1 1 1

# DbD-CS variant with Ranking loss and fitness guidance
python examples/discrete_DbD_EDA.py 0 HIFF 64 200 50 0.5 dbd_cs elu ranking 20 20 0 0.1 1 0
```

## Testing

A comprehensive test suite is provided in `test_dbd_variants.py`:

```bash
python test_dbd_variants.py
```

This tests:
- All 4 loss functions (MSE, weighted MSE, ranking, Huber)
- Fitness guidance (C-DbD)
- Markov initialization (M-DbD)
- Different DbD variants (dbd, dbd_cs, dbd_uc)
- Combinations of features

## Performance Considerations

1. **Weighted MSE and Ranking losses**: Require fitness evaluation at each generation, adding computational cost
2. **Fitness guidance**: Increases network input dimension by 1, negligible overhead
3. **Markov initialization**: Slight overhead from learning Markov model, but can improve convergence
4. **Loss function choice**: 
   - MSE: Fastest, good default
   - Weighted MSE: Slightly slower, better for imbalanced fitness distributions
   - Ranking: Slowest (pair sampling), best for preserving solution quality hierarchy
   - Huber: Similar to MSE, more robust to outliers

## References

- Santana, R., et al. (2023). "Learning search distributions in estimation of distribution algorithms with minimalist diffusion models."
- Sohn, K., et al. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models" (C-VAE inspiration)
