# Complete Integer Neural EDA Comparison

## Overview

This example demonstrates all six neural network-based EDAs applied to integer (multi-valued discrete) optimization problems. This extends the binary comparison to handle problems where variables can take more than two values.

## Neural EDAs Supported

1. **VAE-EDA**: Variational Autoencoder with Gumbel-Softmax (categorical)
2. **GAN-EDA**: Generative Adversarial Network (categorical)
3. **Backdrive-EDA**: Network inversion approach (discrete)
4. **DAE-EDA**: Denoising Autoencoder (categorical - newly extended)
5. **RBM-EDA**: Restricted Boltzmann Machine with softmax units
6. **DbD-EDA**: Diffusion-by-Deblending (categorical)

## Integer Test Problems

The script tests on three integer benchmark functions with cardinality=4:

1. **Integer OneMax** (n=20): Simple separable function (sum of integer values)
   - Tests basic convergence
   - Optimal: n × (cardinality - 1)

2. **Integer Max Blocks** (n=30, k=3): Block dependencies with bonuses
   - Tests block-level dependency modeling
   - Has bonus rewards for blocks with all maximum values

3. **Integer Multi-Level Trap** (n=30, k=3): Deceptive multi-level function
   - Tests ability to escape local optima
   - Deceptive structure with multiple levels

## Usage

### Basic Usage

Run the complete comparison:

```bash
python examples/complete_integer_neural_eda_comparison.py
```

This will:
- Test all 6 neural EDAs on 3 integer problems
- Run 3 independent trials per configuration
- Display success rates, mean fitness, and timing
- Provide comprehensive summary and recommendations

### Custom Configuration

Use the `UnifiedIntegerNeuralEDA` class for custom experiments:

```python
from pateda.examples.complete_integer_neural_eda_comparison import (
    UnifiedIntegerNeuralEDA,
    wrap_integer_onemax,
)
import numpy as np

# Problem setup
n_vars = 20
cardinality = 4
card_array = np.full(n_vars, cardinality)
fitness_func = wrap_integer_onemax(cardinality)

# Create and run EDA
eda = UnifiedIntegerNeuralEDA(
    method='dae',  # or 'vae', 'gan', 'backdrive', 'rbm', 'dbd'
    n_vars=n_vars,
    cardinality=card_array,
    pop_size=80,
    selection_ratio=0.5,
    max_generations=20,
    learning_params={'epochs': 30, 'batch_size': 32},
    sampling_params={'n_refinement_steps': 10},  # for DAE
)

best_fitness, best_solution, history = eda.run(fitness_func, verbose=True)
```

## New: Categorical DAE Support

This release extends the DAE (Denoising Autoencoder) to support categorical (multi-valued discrete) variables:

### Key Features

- **One-hot encoding**: Categorical variables are encoded as one-hot vectors
- **Proper loss function**: Uses cross-entropy loss for categorical distributions
- **Categorical corruption**: Randomly changes category assignments during training
- **Iterative refinement**: Sampling uses corruption-reconstruction cycles

### Implementation Details

```python
from pateda.learning.dae import learn_categorical_dae
from pateda.sampling.dae import sample_categorical_dae
import numpy as np

# Setup
n_vars = 30
cardinality = np.full(n_vars, 4)  # 4-valued variables
population = np.random.randint(0, 4, (100, n_vars))
fitness = np.random.rand(100)

# Learn model
model = learn_categorical_dae(
    population, 
    fitness, 
    cardinality,
    params={
        'epochs': 30,
        'batch_size': 32,
        'hidden_dims': [64, 32],
        'corruption_level': 0.1,
    }
)

# Sample new solutions
new_population = sample_categorical_dae(
    model, 
    n_samples=100,
    params={'n_refinement_steps': 10}
)
```

## Method Characteristics

### VAE-EDA (Categorical)
- **Pros**: Good balance of quality and speed, handles dependencies well
- **Cons**: Requires hyperparameter tuning
- **Best for**: Most integer problems, general-purpose use

### GAN-EDA (Categorical)
- **Pros**: Can learn complex distributions
- **Cons**: Training instability, mode collapse issues
- **Best for**: Research exploration (not recommended for production)

### Backdrive-EDA (Discrete)
- **Pros**: Directly optimizes for fitness, fitness-guided
- **Cons**: Can be slow, requires smooth fitness landscape
- **Best for**: Problems with smooth fitness landscapes

### DAE-EDA (Categorical - NEW)
- **Pros**: Simple, effective, iterative refinement
- **Cons**: Requires tuning refinement steps
- **Best for**: General integer problems, proven approach

### RBM-EDA (Softmax)
- **Pros**: Naturally handles multi-valued variables, classical approach
- **Cons**: Training can be slow
- **Best for**: When you want a well-established energy-based model

### DbD-EDA (Categorical)
- **Pros**: Simpler than VAE/GAN, no encoder needed
- **Cons**: Requires two populations, iterative sampling
- **Best for**: Research on diffusion-based methods

## Recommendations

1. **For most integer problems**: Start with VAE (categorical) or DAE (categorical)
2. **For classical approach**: Use RBM with softmax units
3. **For research/exploration**: Try DbD (new diffusion method)
4. **Avoid**: GAN unless you have specific reasons and expertise

## Requirements

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- All other pateda dependencies

## Performance Notes

- Neural EDAs work best with population sizes > 50
- GPU acceleration significantly speeds up training (if available)
- Integer problems with higher cardinality may need larger hidden dimensions
- Adjust epochs and batch size based on problem complexity

## Related Examples

- `complete_discrete_neural_eda_comparison.py`: Binary version of this comparison
- `discrete_neural_eda_comparison.py`: Simpler comparison with fewer methods
- `integer_functions_benchmark.py`: Traditional EDAs on integer functions
