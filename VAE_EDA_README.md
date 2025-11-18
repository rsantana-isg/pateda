# VAE-EDA Implementation for PATEDA

## Overview

This implementation adds Variational Autoencoder (VAE) based Estimation of Distribution Algorithms to the PATEDA library. The implementation is based on the paper:

**"Expanding variational autoencoders for learning and exploiting latent representations in search distributions"**
By Unai Garciarena, Roberto Santana, and Alexander Mendiburu
Published in GECCO '18: Proceedings of the Genetic and Evolutionary Computation Conference

## Three VAE Variants Implemented

### 1. VAE (Basic Variational Autoencoder)
- **Description**: Standard VAE with encoder-decoder architecture
- **Use Case**: Basic distribution learning and sampling
- **Components**:
  - Encoder: Maps solutions to latent space (mean and log variance)
  - Decoder: Reconstructs solutions from latent samples
- **Loss Function**: Reconstruction loss + KL divergence

### 2. E-VAE (Extended VAE)
- **Description**: VAE with an additional fitness predictor network
- **Use Case**: Distribution learning with fitness approximation
- **Components**:
  - Encoder: Maps solutions to latent space
  - Decoder: Reconstructs solutions from latent samples
  - Predictor: Estimates fitness from latent representation
- **Loss Function**: Reconstruction loss + KL divergence + MSE (fitness prediction)
- **Benefits**: Can be used for surrogate-based filtering of generated solutions

### 3. CE-VAE (Conditional Extended VAE)
- **Description**: VAE that explicitly conditions the decoder on fitness values
- **Use Case**: Fitness-directed sampling and multi-objective optimization
- **Components**:
  - Encoder: Maps solutions to latent space
  - Conditional Decoder: Takes concatenated [latent variable, target fitness]
  - Predictor: Estimates fitness from latent representation
- **Loss Function**: Reconstruction loss + KL divergence + MSE (fitness prediction)
- **Benefits**: Allows explicit control over the fitness level of generated solutions

## Implementation Structure

Following the modular architecture of PATEDA:

### Learning Module (`pateda/learning/vae.py`)
- `learn_vae()`: Learn basic VAE model
- `learn_extended_vae()`: Learn E-VAE model with predictor
- `learn_conditional_extended_vae()`: Learn CE-VAE model with conditioning

### Sampling Module (`pateda/sampling/vae.py`)
- `sample_vae()`: Sample from basic VAE
- `sample_extended_vae()`: Sample from E-VAE (with optional predictor filtering)
- `sample_conditional_extended_vae()`: Sample from CE-VAE with fitness conditioning

### Neural Network Components

All VAE variants are implemented using **PyTorch** with the following architecture classes:

- `VAEEncoder`: Encoder network (input → latent distribution parameters)
- `VAEDecoder`: Standard decoder network (latent → reconstructed input)
- `FitnessPredictor`: Fitness prediction network (latent → fitness estimate)
- `ConditionalDecoder`: Conditional decoder (latent + fitness → reconstructed input)

## Usage Examples

### Example 1: Basic VAE

```python
import numpy as np
from pateda.learning.vae import learn_vae
from pateda.sampling.vae import sample_vae

# Create population
population = np.random.randn(100, 10)  # 100 solutions, 10 variables
fitness = np.sum(population**2, axis=1)

# Learn model
model = learn_vae(
    population, fitness,
    params={
        'latent_dim': 5,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)

# Sample new solutions
bounds = np.array([[-5]*10, [5]*10])
new_population = sample_vae(model, n_samples=50, bounds=bounds)
```

### Example 2: Extended VAE with Surrogate Filtering

```python
from pateda.learning.vae import learn_extended_vae
from pateda.sampling.vae import sample_extended_vae

# Learn E-VAE
model = learn_extended_vae(population, fitness, params={'latent_dim': 5, 'epochs': 50})

# Sample with predictor filtering (keeps best predicted solutions)
new_population = sample_extended_vae(
    model, n_samples=50, bounds=bounds,
    params={
        'use_predictor': True,
        'predictor_percentile': 30  # Keep best 30% of predicted
    }
)
```

### Example 3: Conditional VAE for Fitness-Directed Sampling

```python
from pateda.learning.vae import learn_conditional_extended_vae
from pateda.sampling.vae import sample_conditional_extended_vae

# Learn CE-VAE
model = learn_conditional_extended_vae(population, fitness, params={'latent_dim': 5, 'epochs': 50})

# Sample conditioned on target fitness
target_fitness = np.min(fitness)  # Target the best fitness found
new_population = sample_conditional_extended_vae(
    model, n_samples=50, bounds=bounds,
    params={
        'target_fitness': target_fitness,
        'fitness_noise': 0.1  # Add noise for exploration
    }
)
```

### Example 4: Complete EDA Loop

```python
def sphere(x):
    return np.sum(x**2, axis=1)

# Initialize
n_vars = 10
pop_size = 100
population = np.random.uniform(-5, 5, (pop_size, n_vars))

# EDA loop
for generation in range(30):
    # Evaluate
    fitness = sphere(population)

    # Select best 30%
    idx = np.argsort(fitness)[:int(pop_size * 0.3)]
    selected_pop = population[idx]
    selected_fit = fitness[idx]

    # Learn model
    model = learn_vae(
        selected_pop, selected_fit,
        params={'latent_dim': 5, 'epochs': 30, 'batch_size': 16}
    )

    # Sample new population
    bounds = np.array([[-5]*n_vars, [5]*n_vars])
    population = sample_vae(model, n_samples=pop_size, bounds=bounds)

    print(f"Gen {generation}: Best = {np.min(fitness):.6f}")
```

## Parameters

### Learning Parameters
- `latent_dim` (int, default=5): Dimension of the latent space
- `hidden_dims` (list, default=[32, 16]): Hidden layer dimensions for encoder/decoder
- `epochs` (int, default=50): Number of training epochs
- `batch_size` (int, default=32): Batch size for training
- `learning_rate` (float, default=0.001): Learning rate for Adam optimizer

### Sampling Parameters

**For all variants:**
- `n_samples` (int): Number of solutions to generate
- `bounds` (np.ndarray, optional): Variable bounds [min, max] for each dimension

**For E-VAE:**
- `use_predictor` (bool, default=False): Whether to use fitness predictor for filtering
- `predictor_percentile` (float, default=50): Percentile of predicted fitness to keep

**For CE-VAE:**
- `target_fitness` (float or array): Target fitness value(s) to condition on
- `fitness_noise` (float, default=0.1): Noise added to target fitness for exploration

## Testing

Comprehensive test suite is available in `pateda/tests/test_vae.py`:

```bash
pytest pateda/tests/test_vae.py -v
```

Test coverage includes:
- Basic VAE learning and sampling
- Extended VAE with fitness predictor
- Conditional VAE with fitness conditioning
- Multi-objective optimization scenarios
- Integration tests on benchmark functions (Sphere, Rosenbrock, Ackley)

## Dependencies

The VAE implementation requires PyTorch:

```bash
pip install torch>=2.0.0
```

This has been added to `requirements.txt`.

## Key Features

1. **Modular Design**: Separate learning and sampling functions following PATEDA architecture
2. **PyTorch Implementation**: Modern deep learning framework for efficient training
3. **Flexible Architecture**: Configurable network architectures and hyperparameters
4. **Fitness Conditioning**: CE-VAE allows explicit control over fitness levels
5. **Surrogate Filtering**: E-VAE predictor can filter low-quality samples
6. **Multi-objective Support**: All variants support multiple objectives
7. **Bounded Optimization**: All sampling functions respect variable bounds

## Performance Considerations

- **Training Time**: VAE training requires multiple epochs; adjust `epochs` and `batch_size` based on problem
- **Latent Dimension**: Typically set to 1/2 to 2/3 of the problem dimension
- **Population Size**: Larger populations provide better training data but increase computational cost
- **GPU Support**: PyTorch automatically uses GPU if available for faster training

## Future Enhancements

Potential extensions:
1. Discrete VAE for combinatorial optimization
2. Convolutional VAE for structured problems
3. β-VAE for better disentanglement
4. Integration with other PATEDA selection strategies
5. Adaptive learning rate scheduling
6. Multi-modal VAE for multi-modal optimization

## References

1. Garciarena, U., Santana, R., & Mendiburu, A. (2018). Expanding variational autoencoders for learning and exploiting latent representations in search distributions. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '18), pages 849-856.

2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

3. Larrañaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution algorithms: A new tool for evolutionary computation. Springer Science & Business Media.

## Contact

For questions or issues related to the VAE-EDA implementation, please refer to the main PATEDA repository.
