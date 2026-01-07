# Alternative Differentiable Discrete Sampling Strategies

This document describes the three new alternative sampling strategies implemented for differentiable discrete sampling within neural networks for the PATEDA library.

## Overview

This implementation adds three new strategies for differentiable discrete sampling, complementing the existing Gumbel-Softmax and Corruption/Denoising approaches:

1. **Straight-Through Estimator (STE)**
2. **Hard Concrete Distribution**
3. **Deterministic Softmax**

## Motivation

While Gumbel-Softmax is the standard method for differentiable discrete sampling in neural networks, it has limitations:
- **Biased gradients** from Gumbel noise
- **Temperature tuning** required
- **Soft relaxation** doesn't provide exact discrete values during training

The new strategies address these limitations with different approaches suited for specific use cases.

## Implementation Details

### 1. Straight-Through Estimator (STE)

**File**: `learning/discrete_dendiff_ste.py`

**Key Concept**: Uses hard binary values (0 or 1) in the forward pass but allows gradients to flow as if the operation was continuous (identity function) in the backward pass.

**Advantages**:
- Unbiased gradients (no Gumbel noise)
- No temperature tuning needed
- Simpler than Gumbel-Softmax
- Direct discrete values in forward pass

**Use Cases**:
- General differentiable discrete optimization
- When unbiased gradients are important
- Binarized neural networks

**Implementation Highlights**:
```python
class StraightThroughBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Hard binarization in forward pass
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient flows through unchanged
        return grad_output
```

### 2. Hard Concrete Distribution

**File**: `learning/discrete_dendiff_hard_concrete.py`

**Key Concept**: Uses "stretching and folding" mechanism to allow the model to produce exact 0s and 1s during training, not just at inference.

**Advantages**:
- Can produce exact discrete values during training
- Better for binary gating and regularization
- More flexible than standard Concrete distribution

**Use Cases**:
- Neural architecture search
- Binary gating mechanisms
- L0 regularization
- Structured sparsity

**Implementation Highlights**:
```python
def sample_hard_concrete(logits, temperature=0.1, stretch_limits=(-0.1, 1.1)):
    # 1. Sample from uniform distribution
    u = torch.rand_like(logits)
    
    # 2. Compute stretched Concrete samples
    s = torch.sigmoid((log(u) - log(1-u) + logits) / temperature)
    
    # 3. Stretch to [gamma, zeta]
    s_stretched = s * (zeta - gamma) + gamma
    
    # 4. Clip to [0, 1] to get exact boundaries
    s_hard = torch.clamp(s_stretched, 0, 1)
    
    return s_hard
```

### 3. Deterministic Softmax

**File**: `learning/discrete_dendiff_deterministic.py`

**Key Concept**: Uses clean softmax without Gumbel noise for deterministic optimization and network inversion.

**Advantages**:
- Cleaner, more stable gradients
- Faster convergence to local optima
- No stochastic sampling during optimization
- Better for gradient-based optimization

**Use Cases**:
- Optimization tasks
- Fitness surrogate inversion
- Network inversion for generating inputs
- When determinism is preferred

**Implementation Highlights**:
```python
def deterministic_softmax(logits, hard=False):
    # Clean softmax without Gumbel noise
    probs = F.softmax(logits, dim=-1)
    
    if hard:
        # Use straight-through for hard selection
        probs_hard = torch.zeros_like(probs)
        probs_hard.scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
        probs = (probs_hard - probs).detach() + probs
    
    return probs
```

## Shared Utilities

To reduce code duplication and ensure consistency, a shared utilities module was created:

**File**: `learning/discrete_dendiff_utils.py`

**Contains**:
- `TimeEmbedding`: Sinusoidal time step embedding for diffusion models
- `make_noise_schedule()`: Unified noise/beta schedule creation (linear/cosine)
- `add_noise_binary()`: Binary noise addition by bit flipping
- `compute_diffusion_params()`: Precompute diffusion parameters
- `binarize_samples()`: Consistent sample binarization

## Integration

### Main EDA Interface

**File**: `examples/discrete_Dendiff_EDA.py`

All three new strategies are integrated into the main Dendiff EDA interface:

```python
# Variant mapping
variant_map = {
    'dendiff_gumbel': ...,
    'dendiff_corruption': ...,
    'dendiff_ste': (learn_discrete_dendiff_ste, sample_discrete_dendiff_ste),
    'dendiff_hard_concrete': (learn_discrete_dendiff_hard_concrete, sample_discrete_dendiff_hard_concrete),
    'dendiff_deterministic': (learn_discrete_dendiff_deterministic, sample_discrete_dendiff_deterministic),
}
```

### Sampling Functions

**File**: `sampling/discrete_dendiff.py`

Each strategy has its own sampling function that implements the reverse diffusion process:
- `sample_discrete_dendiff_ste()`
- `sample_discrete_dendiff_hard_concrete()`
- `sample_discrete_dendiff_deterministic()`

### Experiment Launcher

**File**: `launch_dendiff_experiments.py`

The experiment launcher supports all three new variants with appropriate default parameters:

```python
variants = [
    'dendiff_gumbel',
    'dendiff_corruption',
    'dendiff_ste',           # New
    'dendiff_hard_concrete',  # New
    'dendiff_deterministic'   # New
]
```

## Usage Examples

### Command Line

```bash
# Straight-Through Estimator
python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 \
    dendiff_ste ste relu mse 50 20 0 0.5 0.01 0.5

# Hard Concrete
python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 \
    dendiff_hard_concrete hard_concrete relu mse 100 20 0 0.1 0.0001 0.3

# Deterministic Softmax
python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 \
    dendiff_deterministic deterministic relu mse 100 20 0 1.0 0.0001 0.3
```

### Python API

```python
from pateda.examples.discrete_Dendiff_EDA import DendiffEDA
import numpy as np

# Create EDA with STE variant
eda = DendiffEDA(
    variant='dendiff_ste',
    n_vars=20,
    cardinality=np.full(20, 2),
    pop_size=80,
    selection_ratio=0.5,
    max_generations=20,
    sampling_strategy='ste',
    activation='relu',
    loss_function='mse',
    n_timesteps=50,
    n_sampling_steps=20,
    fitness_guided=False,
    temperature=0.5,
    beta_start=0.01,
    beta_end=0.5,
    random_seed=42
)

# Run optimization
best_fitness, best_solution, history = eda.run(fitness_function)
```

## Parameters

### Common Parameters (all variants)

- `n_timesteps`: Number of diffusion timesteps during training
- `n_sampling_steps`: Number of denoising steps during sampling
- `hidden_dims`: Network hidden layer dimensions
- `time_emb_dim`: Time embedding dimension
- `epochs`: Training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `activation`: Activation function

### Variant-Specific Parameters

**STE**:
- `noise_start`: Starting noise level (default: 0.01)
- `noise_end`: Ending noise level (default: 0.5)
- `schedule`: 'linear' or 'cosine'

**Hard Concrete**:
- `beta_start`: Starting noise level (default: 0.0001)
- `beta_end`: Ending noise level (default: 0.3)
- `temperature`: Hard Concrete temperature (default: 0.1)
- `stretch_limits`: Stretching range (default: (-0.1, 1.1))

**Deterministic**:
- `beta_start`: Starting noise level (default: 0.0001)
- `beta_end`: Ending noise level (default: 0.3)
- `schedule`: 'linear' or 'cosine'

## Testing

All three strategies have been tested and validated:

### Syntax Validation
```bash
python -m py_compile learning/discrete_dendiff_ste.py
python -m py_compile learning/discrete_dendiff_hard_concrete.py
python -m py_compile learning/discrete_dendiff_deterministic.py
```

### End-to-End Testing
Each strategy was tested on the OneMax benchmark problem:
- ✓ STE: Working correctly
- ✓ Hard Concrete: Working correctly
- ✓ Deterministic: Working correctly
- ✓ Gumbel (regression test): Still working

### Test Script
```bash
python tests/test_new_sampling_strategies.py
```

## Comparison of Strategies

| Strategy | Gradient Type | Discrete in Training | Temperature | Best For |
|----------|--------------|---------------------|-------------|----------|
| Gumbel-Softmax | Biased (noisy) | No (soft) | Required | General use, well-tested |
| Corruption | Unbiased | Yes (hard) | Optional | BERT-style masking |
| **STE** | Unbiased | Yes (hard) | Not needed | Binarized networks |
| **Hard Concrete** | Biased (structured) | Yes (exact 0/1) | Required | Neural arch search |
| **Deterministic** | Unbiased | No (soft) | Not needed | Optimization tasks |

## References

### Straight-Through Estimator
- Bengio et al. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"

### Hard Concrete Distribution
- Louizos et al. "Learning Sparse Neural Networks through L0 Regularization"
- Maddison et al. "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"

### Deterministic Approaches
- Gradient-based optimization in neural network inversion
- Fitness surrogate inversion techniques

## Future Work

Potential extensions:
1. **Vector Quantization (VQ)**: Implement VQ-VAE style codebook approach
2. **Embedding Space Optimization**: Direct gradient application to learnable embeddings
3. **Dirichlet-Based Relaxations**: Explore Dirichlet distribution as alternative
4. **Hybrid Approaches**: Combine multiple strategies adaptively

## License

This implementation follows the license of the PATEDA library (MIT License).

## Contact

For questions or issues, please refer to the PATEDA repository:
https://github.com/rsantana-isg/pateda
