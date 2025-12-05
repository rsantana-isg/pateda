# Diffusion-by-Deblending EDA (DbD-EDA)

Implementation of Diffusion-by-Deblending Estimation of Distribution Algorithms for continuous optimization, based on the paper:

**"Learning search distributions in estimation of distribution algorithms with minimalist diffusion models"**

## Overview

DbD-EDA is a novel EDA approach that uses alpha-deblending diffusion models to learn and sample search distributions. Unlike traditional denoising diffusion models (DDPM, DDIM), DbD-EDA uses a minimalist deterministic diffusion approach that:

- **Does not require Gaussian noise** as the initial distribution
- **Uses deterministic iterative deblending** for sampling
- **Learns to map between arbitrary distributions** (not just noise to data)
- **Is computationally efficient** with fewer network evaluations

## Key Components

### 1. Alpha-Deblending Neural Network (`pateda/learning/dbd.py`)

The core component is an MLP that learns to predict the difference vector `(x1 - x0)` given:
- A blended sample `x_alpha = (1-alpha)*x0 + alpha*x1`
- The blending parameter `alpha`

**Training objective:**
```
min E_alpha,x0,x1 ||D_theta(x_alpha, alpha) - (x1 - x0)||^2
```

### 2. Iterative Deblending Sampling (`pateda/sampling/dbd.py`)

Generates new samples by iteratively updating from source distribution `p0` to target distribution `p1`:

```
for t = 0 to T-1:
    x_alpha_{t+1} = x_alpha_t + (alpha_{t+1} - alpha_t) * D_theta(x_alpha_t, alpha_t)
```

### 3. Four DbD-EDA Variants

All variants are implemented in `pateda/examples/dbd_eda_example.py`:

#### DbD-CS (Current to Selected)
- **p0**: Current population
- **p1**: Selected population (best solutions)
- **Sampling starts from**: Selected population

Learns to map from the current population to high-fitness solutions. Most similar to standard EDA approach.

#### DbD-CD (Current to Distance-matched)
- **p0**: Current population
- **p1**: Distance-matched selected solutions (closest neighbors)
- **Sampling starts from**: Selected population

Creates focused mappings between each current solution and its nearest high-fitness neighbor, potentially faster convergence.

#### DbD-UC (Univariate current to Current)
- **p0**: Univariate Gaussian approximation of current population
- **p1**: Current population
- **Sampling starts from**: Univariate approximation of selected

Learns to restore variable dependencies disrupted by univariate approximation. The DM learns interaction structures.

#### DbD-US (Univariate current to Selected)
- **p0**: Univariate Gaussian approximation of current population
- **p1**: Selected population
- **Sampling starts from**: Univariate approximation of selected

Combines dependency learning with selection pressure - learns interactions necessary for high fitness.

## Restart Mechanism

All variants include an adaptive restart mechanism to escape local optima:

**Restart is triggered when:**
- Diversity falls below threshold: `std(selected_fitness) < diversity_threshold`
- No improvement for N generations: `generations_without_improvement >= trigger_no_improvement`

**On restart:**
- Population is reinitialized randomly
- Best K solutions are preserved
- Counter is reset

## Installation

Requires PyTorch:

```bash
pip install torch>=1.9.0
```

The DbD-EDA implementation uses PyTorch (not TensorFlow) to align with the DenDiff-EDA implementation in PATEDA.

## Usage

### Basic Example

```python
from pateda.examples.dbd_eda_example import DbDEDA
import numpy as np

# Define problem
def sphere(x):
    return np.sum(x**2)

n_vars = 10
bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

# Create DbD-EDA instance
eda = DbDEDA(
    variant='CS',  # or 'CD', 'UC', 'US'
    pop_size=200,
    selection_ratio=0.3,
    dbd_params={
        'num_alpha_samples': 10,
        'hidden_dims': [64, 64],
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_iterations': 10
    },
    restart_params={
        'trigger_no_improvement': 5,
        'diversity_threshold': 1e-6,
        'keep_best': 2
    }
)

# Run optimization
result = eda.optimize(
    sphere,
    n_vars,
    bounds,
    n_generations=50,
    verbose=True
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best solution: {result['best_solution']}")
```

### Running Examples

```bash
# Quick validation test
python pateda/examples/dbd_quick_test.py

# Comprehensive examples (all variants + benchmarks)
python pateda/examples/dbd_eda_example.py
```

## Parameters

### DbD Model Parameters (`dbd_params`)

- **num_alpha_samples** (int, default=10): Number of alpha values to sample per (x0, x1) pair during training
- **hidden_dims** (list, default=[64, 64]): Hidden layer dimensions for MLP
- **epochs** (int, default=50): Training epochs per generation
- **batch_size** (int, default=32): Batch size for training
- **learning_rate** (float, default=1e-3): Learning rate for Adam optimizer
- **num_iterations** (int, default=10): Number of iterative deblending steps during sampling

### Restart Parameters (`restart_params`)

- **trigger_no_improvement** (int, default=5): Generations without improvement before restart
- **diversity_threshold** (float, default=1e-6): Minimum diversity threshold
- **keep_best** (int, default=2): Number of best solutions to preserve on restart

## Architecture

```
pateda/
├── learning/
│   └── dbd.py              # Alpha-deblending learning
│       ├── AlphaDeblendingMLP      # Neural network model
│       ├── learn_dbd()              # Main learning function
│       ├── find_closest_neighbors() # For DbD-CD variant
│       └── sample_univariate_gaussian() # For DbD-UC/US variants
│
├── sampling/
│   └── dbd.py              # Alpha-deblending sampling
│       ├── iterative_deblending_sampling() # Core sampling algorithm
│       ├── sample_dbd()                    # Main sampling function
│       └── sample_dbd_from_univariate()    # For DbD-UC/US variants
│
└── examples/
    ├── dbd_eda_example.py  # Comprehensive examples
    │   ├── DbDEDA class    # Main EDA implementation
    │   ├── test_single_variant()
    │   ├── compare_all_variants()
    │   └── test_benchmark_functions()
    │
    └── dbd_quick_test.py   # Quick validation tests
```

## Comparison with DenDiff-EDA

| Feature | DenDiff-EDA | DbD-EDA |
|---------|-------------|---------|
| Initial distribution | Gaussian noise | Arbitrary (current/selected/univariate) |
| Diffusion process | Stochastic (DDPM) | Deterministic (alpha-deblending) |
| Network input | (x_t, t) | (x_alpha, alpha) |
| Network predicts | Noise ε | Difference (x1 - x0) |
| Sampling | Reverse diffusion with noise | Deterministic iterative deblending |
| Variants | Single approach | 4 variants (CS, CD, UC, US) |
| Theory basis | Score matching | Blending/deblending |

## Implementation Notes

1. **PyTorch vs TensorFlow**: This implementation uses PyTorch to align with the DenDiff-EDA implementation in PATEDA, whereas the original `enhanced_edas/efficient_diffusion_models.py` uses TensorFlow.

2. **Normalization**: Data is normalized to facilitate learning. The normalization ranges are stored in the model for denormalization during sampling.

3. **Modular Design**: Following PATEDA's structure with separate `learning` and `sampling` modules for modularity and reusability.

4. **Variant Selection**: The choice of variant depends on the problem:
   - **DbD-CS**: Good general-purpose variant
   - **DbD-CD**: Faster convergence when fitness landscape is smooth
   - **DbD-UC**: Useful for learning variable interactions
   - **DbD-US**: Combines interaction learning with selection pressure

5. **Restart Strategy**: Essential for preventing premature convergence, especially for multimodal problems.

## References

1. Main paper: "Learning search distributions in estimation of distribution algorithms with minimalist diffusion models"

2. Alpha-deblending foundation: Heitz, E., Belcour, L., & Chambon, T. (2023). "Iterative α-(de)Blending: A Minimalist Deterministic Diffusion Model." ACM SIGGRAPH 2023.

3. Related work:
   - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
   - Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." ICLR.

## Citation

If you use this implementation in your research, please cite both the DbD-EDA paper and PATEDA:

```bibtex
@article{dbdeda2024,
  title={Learning search distributions in estimation of distribution algorithms with minimalist diffusion models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This implementation is part of the PATEDA package and follows the same license terms.

## Contact

For questions or issues related to this implementation, please open an issue on the PATEDA GitHub repository.
