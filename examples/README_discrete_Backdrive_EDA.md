# Discrete Backdrive EDA

## Overview

`discrete_Backdrive_EDA.py` provides a unified command-line interface for running configurable Backdrive-EDA algorithm variants on discrete optimization problems.

## Features

The program supports the following configurable parameters for Backdrive-EDA:

### Algorithm Variants
- **backdrive**: Standard Backdrive-EDA with network inversion
- **backdrive_adaptive**: Adaptive sampling with multiple target fitness levels for better diversity

### Configuration Parameters

1. **Weight Transfer** (`--weight-transfer`)
   - Transfer neural network weights between generations
   - Can improve learning efficiency by building on previous knowledge
   - Default: disabled

2. **Early Stopping** (`--early-stopping`)
   - Enable early stopping during neural network training
   - Helps prevent overfitting and reduces training time
   - Default: disabled

3. **Initialization Method** (`--init`)
   - `random`: Random initialization (default)
   - `perturb_best`: Initialize from best solution with perturbations
   - `perturb_selected`: Initialize from selected solutions with perturbations

4. **Loss Function** (`--loss`)
   - `mse`: Mean Squared Error (default)
   - `weighted_mse`: Fitness-weighted MSE focusing on high-fitness solutions
   - `ranking`: Ranking-based loss preserving relative ordering
   - `huber`: Huber loss (robust to outliers)

5. **Activation Function** (`--activation`)
   - Supported: relu (default), tanh, sigmoid, leaky_relu, elu, selu, gelu, silu, softplus, softsign, mish, hardswish, hardsigmoid, prelu
   - **Note**: Currently passed as parameter but activation function modification in the core DiscreteBackdriveNet would require additional implementation

6. **Surrogate Filtering** (`--surrogate-filtering`)
   - Use surrogate model to pre-filter generated solutions
   - Generates 3x population size and selects top predicted solutions
   - Can improve solution quality at cost of additional computation

7. **Truncation Percent** (positional argument)
   - Selection ratio for truncation selection (e.g., 0.5 = 50%)
   - Determines what fraction of population is used for learning

## Usage

### Basic Syntax

```bash
python discrete_Backdrive_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> [OPTIONS]
```

### Positional Arguments

- `seed`: Random seed (integer)
- `obj_func`: Objective function name
- `n`: Number of variables (integer)
- `pop_size`: Population size (integer)
- `n_gen`: Number of generations (integer)
- `trunc`: Truncation percent/selection ratio (float between 0 and 1)

### Optional Arguments

```
--variant {backdrive,backdrive_adaptive}
                      Backdrive variant (default: backdrive)
--weight-transfer     Transfer neural network weights between generations
--early-stopping      Use early stopping during training
--init {random,perturb_best,perturb_selected}
                      Initialization method (default: random)
--loss {mse,weighted_mse,ranking,huber}
                      Loss function (default: mse)
--activation ACT      Activation function (default: relu)
--surrogate-filtering
                      Use surrogate model for pre-filtering solutions
```

## Examples

### Basic Usage
```bash
# Standard Backdrive with default settings
python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5
```

### With Early Stopping and Weight Transfer
```bash
python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --weight-transfer --early-stopping
```

### Custom Initialization and Loss Function
```bash
python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --init perturb_best --loss weighted_mse
```

### Adaptive Variant with Custom Settings
```bash
python discrete_Backdrive_EDA.py 0 HIFF 64 200 50 0.5 --variant backdrive_adaptive --activation tanh
```

### With Surrogate Filtering
```bash
python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --surrogate-filtering
```

### Different Truncation Percentages
```bash
# 30% selection (more selective)
python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.3

# 70% selection (more diverse)
python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.7
```

### Testing Different Loss Functions
```bash
# MSE (default)
python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss mse

# Weighted MSE (focus on high-fitness)
python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss weighted_mse

# Ranking loss
python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss ranking

# Huber loss
python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss huber
```

## Supported Benchmark Problems

- **OneMax**: Simple counting problem
- **KDeceptive3, KDeceptive5**: K-deceptive problems
- **Deceptive3, Deceptive3Overlap**: Deceptive-3 variants
- **DecepMarta3, DecepMarta3New**: Marta's deceptive-3 variants
- **Decep3MH, TwoPeaksDecep3, DecepVenturini**: Other deceptive variants
- **HardDecep5**: Hard deceptive-5
- **HIFF**: Hierarchical If and only If (requires power of 2)
- **FHTrap1**: Hierarchical Trap (requires power of 3)
- **Polytree3, Polytree3Overlap, Polytree5**: Polytree problems
- **FC2, FC3, FC4, FC5**: Cuban function problems

## Output

The program outputs:
1. Configuration summary
2. Generation-by-generation progress
3. Final results including:
   - Best fitness found
   - Gap from optimal
   - Success indicator
   - Elapsed time
   - Best solution (first 20 variables)

## Implementation Details

### Backdrive vs Backdrive-Adaptive

- **Backdrive**: Standard implementation samples all solutions targeting maximum fitness
- **Backdrive-Adaptive**: Samples solutions at different target fitness levels (100%, 90%, 80%) with fractions (50%, 30%, 20%) to maintain diversity

### Initialization Methods

- **random**: Pure random initialization in continuous relaxation space
- **perturb_best**: Initialize from best solution with noise (bias_strength=5.0, noise=0.1)
- **perturb_selected**: Initialize from top selected solutions with noise

### Loss Functions

- **mse**: Standard mean squared error - treats all errors equally
- **weighted_mse**: Exponentially weighted by fitness - focuses on high-fitness regions
- **ranking**: Pairwise ranking loss - preserves relative ordering
- **huber**: Robust loss - less sensitive to outliers than MSE

## Notes

1. The `activation` parameter is accepted and passed to the learning function, but the current implementation of DiscreteBackdriveNet has hardcoded ReLU activations. Full activation function support would require modifications to the core network class.

2. Weight transfer between generations stores the entire model state and uses it for initialization in the next generation, but the pretrained_model parameter support would need to be implemented in the learning functions.

3. Surrogate filtering increases computational cost (generates 3x population) but can improve solution quality by pre-screening with the fitness model.

## Comparison with discrete_EDA.py

This program differs from `discrete_EDA.py` in that it:
- Focuses exclusively on Backdrive variants
- Provides explicit configuration parameters rather than algorithm names
- Adds truncation percent as a required parameter
- Uses argparse for better command-line interface
- Removes all non-Backdrive algorithms (VAE, GAN, DAE, RBM, DbD, traditional EDAs)

## Future Extensions

Future Backdrive variants can be easily added by:
1. Implementing the learning function (if different loss)
2. Implementing the sampling function (if different sampling strategy)
3. Adding to the `loss_function_map` or `sampling_function_map` in BackdriveEDA class
4. Adding the option to argument parser

The modular design allows testing new Backdrive ideas without modifying the core framework.
