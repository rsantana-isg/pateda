# New Enhanced VAE EDA Variants

This document describes the 6 new enhanced VAE EDA variants added based on the critical analysis in `DISCRETE_VAE_ANALYSIS.md`.

## Overview

The new variants address the major issues identified in the VAE-EDA analysis:
1. **Posterior collapse** (KL divergence vanishing)
2. **Architecture overfitting** (too many parameters per sample)
3. **Lack of fitness guidance**
4. **Bernoulli sampling variance** (excessive stochasticity)

## New Variants

### 1. BA-VAE (Beta-Annealed VAE)

**Purpose**: Addresses posterior collapse with cyclical beta annealing

**Key Features**:
- Cyclical beta annealing schedule (Fu et al. 2019)
- Beta starts low, increases within each cycle
- Multiple cycles allow learning both reconstruction and latent structure
- Prevents KL divergence from vanishing

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 OneMax 30 100 50 0.5 BA-VAE --epochs 100
```

**Implementation**:
- Learning: `pateda.learning.discrete_vae.learn_binary_bavae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_bavae()`

### 2. AA-VAE (Adaptive-Architecture VAE)

**Purpose**: Addresses overfitting with ultra-conservative architecture sizing

**Key Features**:
- Very small hidden layers: `sqrt(n_vars * pop_size)`
- Small latent dimension: `n_vars // 10`
- Dropout regularization (0.3)
- L2 weight decay
- Target: ~1-2 parameters per training sample

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 Deceptive3 30 100 50 0.5 AA-VAE --epochs 100
```

**Implementation**:
- Learning: `pateda.learning.discrete_vae.learn_binary_aavae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_aavae()`

### 3. FW-VAE (Fitness-Weighted VAE)

**Purpose**: Better fitness guidance via weighted reconstruction loss

**Key Features**:
- Weights reconstruction loss by fitness values
- Prioritizes accurate reconstruction of high-fitness solutions
- Biases latent space toward high-fitness regions
- Uses formula: `weight = 1.0 + fitness_weight_strength * normalized_fitness`

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 HIFF 64 200 50 0.5 FW-VAE --epochs 100
```

**Implementation**:
- Learning: `pateda.learning.discrete_vae.learn_binary_fwvae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_fwvae()`

### 4. GS-VAE (Greedy-Sampling VAE)

**Purpose**: Reduces Bernoulli sampling variance with deterministic sampling

**Key Features**:
- Uses deterministic (argmax) sampling instead of stochastic Bernoulli
- Takes most likely value: `x_i = 1 if p_i > 0.5 else 0`
- Reduces variance in generated samples
- Better for exploitation

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 OneMax 30 100 50 0.5 GS-VAE
```

**Implementation**:
- Learning: Uses standard `learn_binary_vae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_gsvae()`

### 5. HS-VAE (Hybrid-Sampling VAE)

**Purpose**: Balances exploration and exploitation with hybrid sampling

**Key Features**:
- Combines deterministic (greedy) and stochastic sampling
- Default: 70% exploitation (deterministic) + 30% exploration (stochastic)
- Configurable exploration ratio
- Best of both sampling strategies

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 Deceptive3 30 100 50 0.5 HS-VAE
```

**Implementation**:
- Learning: Uses standard `learn_binary_vae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_hsvae()`

### 6. TC-VAE (Temperature-Controlled VAE)

**Purpose**: Adaptive exploration-exploitation via temperature annealing

**Key Features**:
- Temperature decreases linearly over generations
- High temperature (1.0) at start → exploration
- Low temperature (0.1) at end → exploitation
- Automatic adaptation to optimization progress

**Usage**:
```bash
python examples/discrete_VAE_EDA.py 0 FHTrap1 81 200 50 0.5 TC-VAE
```

**Implementation**:
- Learning: Uses standard `learn_binary_vae()`
- Sampling: `pateda.sampling.discrete_neural.sample_binary_tcvae()`

## Comparison Matrix

| Variant | Addresses Issue | Learning Modification | Sampling Modification | Best For |
|---------|----------------|----------------------|----------------------|----------|
| BA-VAE  | Posterior collapse | Cyclical beta | Standard | All problems |
| AA-VAE  | Overfitting | Ultra-small network | Standard | Small populations |
| FW-VAE  | Fitness guidance | Weighted loss | Standard | Hard problems |
| GS-VAE  | Sampling variance | None | Deterministic | Exploitation-focused |
| HS-VAE  | Balance | None | Hybrid | Balanced search |
| TC-VAE  | Exploration-exploitation | None | Adaptive temp | Long runs |

## Recommended Usage

1. **For small populations (< 100)**: Use **AA-VAE** to avoid overfitting
2. **For hard deceptive problems**: Use **FW-VAE** for better fitness guidance
3. **For quick convergence**: Use **BA-VAE** + **GS-VAE** combination
4. **For robust optimization**: Use **HS-VAE** or **TC-VAE**
5. **General purpose**: Start with **BA-VAE** as the default

## Example Commands

```bash
# BA-VAE on OneMax
python examples/discrete_VAE_EDA.py 0 OneMax 30 100 50 0.5 BA-VAE --epochs 100

# AA-VAE on Deceptive3 with small population
python examples/discrete_VAE_EDA.py 1 Deceptive3 30 80 40 0.5 AA-VAE --epochs 150

# FW-VAE on HIFF
python examples/discrete_VAE_EDA.py 2 HIFF 64 200 50 0.5 FW-VAE --beta-start 0.0 --beta-end 1.0

# GS-VAE on FHTrap1
python examples/discrete_VAE_EDA.py 3 FHTrap1 81 150 40 0.5 GS-VAE

# HS-VAE on KDeceptive3
python examples/discrete_VAE_EDA.py 4 KDeceptive3 30 100 50 0.5 HS-VAE

# TC-VAE on Polytree3
python examples/discrete_VAE_EDA.py 5 Polytree3 30 120 60 0.5 TC-VAE
```

## Parameter Tuning

### BA-VAE
- `n_cycles`: Number of beta annealing cycles (default: 4)
- `beta_max`: Maximum beta value (default: 1.0)

### AA-VAE
- `dropout`: Dropout rate (default: 0.3)
- `weight_decay`: L2 regularization (default: 0.0001)

### FW-VAE
- `fitness_weight_strength`: Fitness weighting strength (default: 2.0)

### HS-VAE
- `exploration_ratio`: Fraction of stochastic samples (default: 0.3)

### TC-VAE
- `temp_start`: Starting temperature (default: 1.0)
- `temp_end`: Ending temperature (default: 0.1)

## References

- Fu et al. (2019). "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing." NAACL 2019.
- Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR 2017.
- Kingma & Welling (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.

See `DISCRETE_VAE_ANALYSIS.md` for detailed analysis and motivation.
