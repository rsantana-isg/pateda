# GAN-EDA Implementation

This document describes the implementation of Generative Adversarial Network (GAN) based Estimation of Distribution Algorithms for continuous optimization in the pateda framework.

## Overview

This implementation is based on the paper:
**"Generative Adversarial Networks in Estimation of Distribution Algorithms for Combinatorial Optimization"** by Malte Probst (2016)

## Architecture

### Core Components

The GAN-EDA implementation follows the modular structure of pateda with separate learning and sampling functions:

1. **Learning Module** (`pateda/learning/gan.py`)
   - `GANGenerator`: Neural network that generates samples from random noise
   - `GANDiscriminator`: Neural network that classifies samples as real or fake
   - `learn_gan()`: Main function to train the GAN on selected population

2. **Sampling Module** (`pateda/sampling/gan.py`)
   - `sample_gan()`: Generates new candidate solutions from trained GAN

### GAN Components

#### Generator Network
- **Input**: Random noise vector z from latent space (dimension: `latent_dim`)
- **Architecture**: Multi-layer perceptron with ReLU activations
  - Default: [latent_dim → 32 → 64 → input_dim]
  - Output layer: Sigmoid activation (for normalized data)
- **Output**: Generated samples in data space

#### Discriminator Network
- **Input**: Samples (either real or generated)
- **Architecture**: Multi-layer perceptron with ReLU activations
  - Default: [input_dim → 64 → 32 → 1]
  - Output layer: Sigmoid activation (probability)
- **Output**: Probability that input is real (not generated)

## Training Process

The GAN is trained using adversarial optimization:

### Discriminator Training
For each batch:
1. Sample real data from selected population
2. Generate fake data using generator
3. Train discriminator to distinguish real from fake
4. Loss: Binary cross-entropy

```
L_D = -[t * log(D(x)) + (1-t) * log(1-D(x))]
where t=1 for real data, t=0 for fake data
```

### Generator Training
For each batch:
1. Generate fake samples
2. Feed through discriminator
3. Train generator to fool discriminator (label fake as real)
4. Gradient backpropagated through discriminator

```
L_G = -log(D(G(z)))
```

## Usage

### Basic Usage

```python
from pateda.learning.gan import learn_gan
from pateda.sampling.gan import sample_gan
import numpy as np

# Assume we have selected population from an EDA
selected_population = np.random.randn(100, 10)  # 100 samples, 10 dimensions
selected_fitness = np.sum(selected_population**2, axis=1)

# Learn GAN model
model = learn_gan(
    selected_population,
    selected_fitness,
    params={
        'latent_dim': 5,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.0002
    }
)

# Sample new population
new_population = sample_gan(model, n_samples=100)
```

### In an EDA Loop

```python
import numpy as np
from pateda.learning.gan import learn_gan
from pateda.sampling.gan import sample_gan

def sphere(x):
    return np.sum(x**2, axis=1)

# Initialize
n_vars = 10
pop_size = 100
population = np.random.uniform(-5, 5, (pop_size, n_vars))
bounds = np.array([[-5]*n_vars, [5]*n_vars])

# EDA loop
for generation in range(50):
    # Evaluate
    fitness = sphere(population)

    # Select best 30%
    selection_size = int(pop_size * 0.3)
    idx = np.argsort(fitness)[:selection_size]
    selected_pop = population[idx]
    selected_fit = fitness[idx]

    # Learn GAN
    model = learn_gan(selected_pop, selected_fit, params={'epochs': 100})

    # Sample new population
    population = sample_gan(model, n_samples=pop_size, bounds=bounds)
```

## Parameters

### Learning Parameters (`learn_gan`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `latent_dim` | Dimension of latent noise space | max(2, n_vars // 2) |
| `hidden_dims_g` | Hidden layer dimensions for generator | [32, 64] |
| `hidden_dims_d` | Hidden layer dimensions for discriminator | [64, 32] |
| `epochs` | Number of training epochs | 100 |
| `batch_size` | Batch size for training | min(32, pop_size // 2) |
| `learning_rate` | Learning rate for Adam optimizer | 0.0002 |
| `beta1` | Adam beta1 parameter | 0.5 |
| `k_discriminator` | Discriminator updates per generator update | 1 |

### Sampling Parameters (`sample_gan`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Number of samples to generate | Required |
| `bounds` | Min/max bounds for each variable | None |
| `temperature` | Scaling factor for latent noise (controls diversity) | 1.0 |

## Implementation Details

### Data Normalization
- Input data is normalized to [0, 1] range before training
- Normalization parameters (min/max) are stored in the model
- Samples are denormalized back to original range after generation

### Network Initialization
- Weights initialized using PyTorch defaults (Xavier/Glorot uniform)
- Bias initialized to zero

### Optimizer
- Adam optimizer used for both generator and discriminator
- Default learning rate: 0.0002 (as recommended in GAN literature)
- Default beta1: 0.5 (lower than standard 0.9 for GANs)

### Training Strategy
- Alternating optimization: discriminator → generator
- Support for k-discriminator steps per generator step
- Labels: 1.0 for real, 0.0 for fake

## Files Created

```
pateda/
├── learning/
│   ├── gan.py                    # GAN learning module
│   └── __init__.py               # Updated to export learn_gan
├── sampling/
│   ├── gan.py                    # GAN sampling module
│   └── __init__.py               # Updated to export sample_gan
├── examples/
│   └── gan_eda_example.py        # Complete working example
└── tests/
    └── test_gan.py               # Comprehensive test suite
```

## Test Suite

The test suite (`test_gan.py`) includes:

### Basic Functionality Tests
- Basic GAN learning and model structure validation
- Basic GAN sampling and shape verification
- Sampling with bounds enforcement
- Sampling with temperature control

### Architecture Tests
- Custom network architectures
- Different latent dimensions
- Network depth variations

### Training Tests
- Multiple discriminator steps per generator step
- Different learning rates
- Different Adam beta1 parameters

### Integration Tests
- Optimization on sphere function
- Optimization on Rosenbrock function
- Small population scenarios

### Edge Cases
- Constant (zero variance) population
- Single variable problems
- High-dimensional problems (20+ variables)

## Known Limitations

As noted in the Probst (2016) paper, GAN-EDAs face several challenges:

1. **Noisy Training Data**: Early EDA generations contain mostly random data, making it difficult for the GAN to learn meaningful patterns
2. **Hyperparameter Sensitivity**: GANs are notoriously difficult to tune, requiring careful selection of learning rates, architectures, and training epochs
3. **Convergence Difficulties**: The adversarial training can be unstable, especially with small population sizes
4. **Performance**: The paper showed that GAN-EDAs were not competitive with state-of-the-art EDAs like BOA or DAE-EDA

### Comparison to Other Neural Network EDAs

According to Probst (2016):
- **RBM-EDA**: Competitive performance, good computational efficiency
- **DAE-EDA**: Superior performance on factorizable problems, low computational effort
- **GAN-EDA**: Limited success due to training instability with noisy data

## Potential Improvements

Future work could explore:

1. **Stabilization Techniques**:
   - Spectral normalization
   - Gradient penalty (WGAN-GP)
   - Progressive growing

2. **Architecture Improvements**:
   - Residual connections
   - Batch normalization
   - Self-attention mechanisms

3. **Training Strategies**:
   - Curriculum learning (easier problems first)
   - Pre-training on synthetic data
   - Ensemble of GANs

4. **Conditional GANs**:
   - Condition on fitness values
   - Guide generation toward promising regions

## References

1. Probst, M. (2016). "Generative Adversarial Networks in Estimation of Distribution Algorithms for Combinatorial Optimization." arXiv:1509.09235v2

2. Goodfellow, I., et al. (2014). "Generative Adversarial Nets." NIPS 2014.

3. Probst, M., Rothlauf, F., & Grahl, J. (2016). "Scalability of using Restricted Boltzmann Machines for combinatorial optimization." European Journal of Operational Research.

4. Probst, M., & Rothlauf, F. (2016). "Model building and sampling in estimation of distribution algorithms using denoising autoencoders." Technical report.

## Contact

For questions or issues related to this implementation, please refer to the main pateda repository.
