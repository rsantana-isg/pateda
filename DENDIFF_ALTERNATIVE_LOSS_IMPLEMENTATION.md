# Alternative Loss Functions for Dendiff Variants - Implementation Summary

## Overview

This document summarizes the implementation of alternative loss functions and fitness guidance for three dendiff variants: `dendiff_deterministic`, `dendiff_ste`, and `dendiff_hard_concrete`.

## Problem Statement

The original codebase had enhanced versions with alternative loss functions only for two dendiff variants:
- `dendiff_gumbel_enhanced` - Gumbel-Softmax based discrete diffusion
- `dendiff_corruption_enhanced` - Corruption/denoising based discrete diffusion

Three other variants (`dendiff_deterministic`, `dendiff_ste`, `dendiff_hard_concrete`) only supported standard loss functions and lacked fitness guidance capabilities.

## Solution

Created enhanced versions for all three remaining variants following the same design pattern as the existing enhanced implementations.

## Files Created

### 1. learning/discrete_dendiff_deterministic_enhanced.py
Enhanced version of deterministic softmax dendiff with:
- `FitnessGuidedDeterministicDenoisingMLP`: Fitness-conditioned denoising network
- Loss functions:
  - `compute_weighted_loss`: Fitness-weighted cross-entropy
  - `compute_ranking_loss`: Ranking-aware cross-entropy
  - `compute_huber_loss`: Huber-smoothed cross-entropy
- `learn_discrete_dendiff_deterministic_enhanced`: Enhanced learning function

**Key Features:**
- Deterministic softmax without Gumbel noise
- Cleaner gradients for optimization tasks
- Supports fitness conditioning via embedding layer
- All loss functions implemented using cross-entropy (categorical)

### 2. learning/discrete_dendiff_ste_enhanced.py
Enhanced version of Straight-Through Estimator dendiff with:
- `FitnessGuidedSTEDenoisingMLP`: Fitness-conditioned STE denoising network
- Loss functions:
  - `compute_weighted_bce_loss`: Fitness-weighted binary cross-entropy
  - `compute_ranking_bce_loss`: Ranking-aware BCE
  - `compute_huber_bce_loss`: Huber-smoothed BCE
- `learn_discrete_dendiff_ste_enhanced`: Enhanced learning function

**Key Features:**
- Hard binary values in forward pass, gradient flow in backward pass
- Unbiased gradients compared to Gumbel-Softmax
- Supports fitness conditioning
- All loss functions implemented using binary cross-entropy

### 3. learning/discrete_dendiff_hard_concrete_enhanced.py
Enhanced version of Hard Concrete distribution dendiff with:
- `FitnessGuidedHardConcreteDenoisingMLP`: Fitness-conditioned Hard Concrete network
- Loss functions:
  - `compute_weighted_mse_loss`: Fitness-weighted MSE
  - `compute_ranking_mse_loss`: Ranking-aware MSE
  - `compute_huber_mse_loss`: Huber-smoothed MSE
- `learn_discrete_dendiff_hard_concrete_enhanced`: Enhanced learning function

**Key Features:**
- Stretching and folding mechanism for exact 0s and 1s
- Useful for binary gating and regularization
- Supports fitness conditioning
- All loss functions implemented using MSE (continuous values)

## Files Modified

### examples/discrete_Dendiff_EDA.py
**Changes:**
1. Added imports for three new enhanced modules
2. Updated logic to use enhanced versions for all five variants when alternative loss functions or fitness guidance is requested
3. Updated documentation to reflect new capabilities:
   - Added 15 new enhanced variant combinations
   - Added 3 new fitness-guided variant options
   - Added examples for new variants with alternative losses

**Before:** Only dendiff_gumbel and dendiff_corruption could use enhanced versions
**After:** All five variants (gumbel, corruption, ste, hard_concrete, deterministic) support enhanced versions

### launch_dendiff_experiments.py
**Status:** No changes needed - already generates commands with all necessary parameters

## Design Pattern

All enhanced implementations follow a consistent pattern:

### 1. Fitness-Guided Network Architecture
```python
class FitnessGuided*DenoisingMLP(nn.Module):
    def __init__(self, input_dim, time_emb_dim, fitness_emb_dim, ...):
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Fitness embedding
        self.fitness_embed = nn.Linear(1, fitness_emb_dim)
        
        # MLP with concatenated embeddings
        # input: [x_noisy, time_emb, fitness_emb]
```

### 2. Loss Function Hierarchy
Each enhanced module implements:
- **Standard loss** (default='mse'): Cross-entropy or BCE depending on variant
- **Weighted loss**: Fitness-weighted version prioritizing high-fitness samples
- **Ranking loss**: Ranking-aware loss (implemented via weighted sampling)
- **Huber loss**: Robust loss less sensitive to outliers

### 3. Enhanced Learning Function
```python
def learn_discrete_dendiff_*_enhanced(population, fitness, params):
    # Parse loss_function parameter
    loss_function = params.get('loss_function', 'mse')
    use_fitness_guidance = params.get('use_fitness_guidance', False)
    
    # Create fitness-guided or standard model
    if use_fitness_guidance:
        model = FitnessGuided*DenoisingMLP(...)
    else:
        model = Standard*DenoisingMLP(...)
    
    # Training loop with loss selection
    if loss_function == 'weighted_mse':
        loss = compute_weighted_loss(...)
    elif loss_function == 'ranking':
        loss = compute_ranking_loss(...)
    elif loss_function == 'huber':
        loss = compute_huber_loss(...)
    else:
        loss = standard_loss(...)
```

## Usage Examples

### 1. Dendiff-Deterministic with Weighted Loss
```bash
python examples/discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 \
    dendiff_deterministic deterministic relu weighted_mse \
    100 20 0 1.0 0.0001 0.3
```

### 2. Dendiff-STE with Fitness Guidance and Huber Loss
```bash
python examples/discrete_Dendiff_EDA.py 1 HIFF 64 200 50 0.5 \
    dendiff_ste ste elu huber \
    50 20 1 0.5 0.01 0.5
```

### 3. Dendiff-HardConcrete with Ranking Loss
```bash
python examples/discrete_Dendiff_EDA.py 2 KDeceptive3 30 150 40 0.5 \
    dendiff_hard_concrete hard_concrete relu ranking \
    100 20 0 0.1 0.0001 0.3
```

## Loss Function Details

### Weighted Loss
- **Purpose**: Prioritize learning from high-fitness samples
- **Implementation**: Normalize fitness to [0,1], use as weights for loss
- **Use case**: When solution quality varies significantly in population

### Ranking Loss
- **Purpose**: Learn relative ordering of solutions by fitness
- **Implementation**: Standard loss with potential for pairwise ranking terms
- **Use case**: When relative fitness matters more than absolute values

### Huber Loss
- **Purpose**: Robust training less sensitive to outliers
- **Implementation**: Quadratic for small errors, linear for large errors
- **Use case**: When training data has outliers or noisy fitness values

## Fitness Guidance

Fitness guidance conditions the denoising process on fitness values, inspired by:
- Conditional VAE (C-VAE): Conditioning on labels/attributes
- Fitness-guided DbD: Using fitness to guide generation

**Implementation:**
- Fitness values are normalized to [0, 1]
- Embedded via linear layer to fitness_emb_dim dimensions
- Concatenated with time and noisy input embeddings
- Network learns to denoise conditioned on fitness

**Benefits:**
- Better generation of high-fitness solutions
- Improved sampling efficiency
- More targeted exploration

## Backward Compatibility

The enhanced versions maintain full backward compatibility:
- Default `loss_function='mse'` uses standard cross-entropy/BCE/MSE
- `use_fitness_guidance=False` disables fitness conditioning
- Falls back to standard (non-enhanced) implementations when not needed

## Testing and Validation

### Syntax Validation
All Python files validated with `python -m py_compile`:
- ✓ discrete_dendiff_deterministic_enhanced.py
- ✓ discrete_dendiff_ste_enhanced.py
- ✓ discrete_dendiff_hard_concrete_enhanced.py
- ✓ discrete_Dendiff_EDA.py

### Code Review
Code review completed with minor documentation clarifications addressed.

### Security Scan
CodeQL security scan completed with no alerts found.

## Summary

This implementation:
1. **Extends** alternative loss function support to all dendiff variants
2. **Maintains** consistency with existing enhanced implementations
3. **Provides** 15 new enhanced variant combinations
4. **Enables** fitness guidance for all variants
5. **Preserves** backward compatibility
6. **Follows** established design patterns in the codebase

All five dendiff variants now have equal capabilities:
- ✓ dendiff_gumbel
- ✓ dendiff_corruption
- ✓ dendiff_deterministic (NEW)
- ✓ dendiff_ste (NEW)
- ✓ dendiff_hard_concrete (NEW)

## Future Work

Potential enhancements:
1. Add more sophisticated ranking loss with pairwise comparisons
2. Implement adaptive fitness weighting strategies
3. Add learning rate scheduling based on fitness improvements
4. Explore multi-objective fitness conditioning
