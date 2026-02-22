# Discrete Dendiff EDA: Complete Guide

## Overview

This guide provides comprehensive documentation for the Discrete Denoising Diffusion (Dendiff) Estimation of Distribution Algorithms (EDAs) implemented in `examples/discrete_Dendiff_EDA.py`.

The Dendiff EDAs adapt the denoising diffusion probabilistic model (DDPM) framework from continuous to discrete binary optimization, providing theoretically grounded alternatives to VAE and GAN-based EDAs.

---

## Table of Contents

1. [Implemented Variants](#implemented-variants)
2. [Architecture and Design](#architecture-and-design)
3. [Parameters Guide](#parameters-guide)
4. [Usage Examples](#usage-examples)
5. [Comparison with Other EDAs](#comparison-with-other-edas)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Implemented Variants

### Base Variants

#### 1. Dendiff-Gumbel (`dendiff_gumbel`)

**Description**: Principled DDPM adaptation using Gumbel-Softmax for differentiable discrete sampling.

**Key Features**:
- Treats each binary variable as a categorical distribution with 2 classes
- Uses Gumbel-Softmax reparameterization trick for differentiable sampling
- Maintains T-step forward/reverse diffusion process
- Cross-entropy loss for discrete variables

**When to Use**:
- Need theoretically grounded diffusion model
- Want flexible temperature and timestep control
- Quality more important than speed
- Research/exploration of diffusion approaches

**Default Parameters**:
```
n_timesteps: 100
beta_start: 0.0001
beta_end: 0.3
temperature: 1.0
```

#### 2. Dendiff-Corruption (`dendiff_corruption`)

**Description**: BERT-style masked/corruption denoising approach.

**Key Features**:
- Simpler corruption-based forward process
- Direct bit-flip corruption instead of diffusion schedule
- Binary cross-entropy loss
- Fewer timesteps needed (typically 50)
- More interpretable corruption mechanism

**When to Use**:
- Prefer simpler, more interpretable approach
- Speed matters (faster than Gumbel)
- Familiar with BERT/masked language modeling
- Want straightforward implementation

**Default Parameters**:
```
n_timesteps: 50
corruption_start: 0.01
corruption_end: 0.5
temperature: 0.5
```

---

### Enhanced Variants (with Alternative Loss Functions)

The enhanced variants support alternative loss functions inspired by Backdrive and DbD EDAs:

#### 3. Dendiff with Weighted Loss (`loss=weighted_mse` or `loss=weighted_bce`)

**Description**: Fitness-weighted loss that prioritizes learning from high-fitness solutions.

**Inspired by**: `discrete_backdrive_weighted_mse.py`

**Loss Function**:
```python
# Normalize fitness to [0,1]
weights = (fitness - min) / (max - min)
# Weight cross-entropy by fitness
loss = sum(weights * cross_entropy(pred, target))
```

**When to Use**:
- Solutions have widely varying fitness values
- Want to focus learning on high-quality solutions
- Population contains many low-fitness outliers

**Parameters**:
```bash
--loss weighted_mse
```

#### 4. Dendiff with Ranking Loss (`loss=ranking`)

**Description**: Ranking-based loss that preserves relative fitness ordering.

**Inspired by**: `discrete_backdrive_ranking.py`

**When to Use**:
- Relative fitness ordering is more important than absolute values
- Dealing with noisy or unreliable absolute fitness values
- Want model to learn fitness landscape structure

**Parameters**:
```bash
--loss ranking
```

#### 5. Dendiff with Huber Loss (`loss=huber`)

**Description**: Robust loss function less sensitive to outliers.

**Inspired by**: `discrete_backdrive_huber.py`

**Loss Function**:
```python
error = |predicted - target|
huber = {
    0.5 * error^2           if error < delta
    delta * (error - 0.5*delta)  otherwise
}
```

**When to Use**:
- Population contains outliers or noisy fitness evaluations
- Want robust training that handles extreme values
- Standard cross-entropy leads to instability

**Parameters**:
```bash
--loss huber
```

---

### Fitness-Guided Variants

#### 6. Fitness-Guided Dendiff (`fitness_guided=1`)

**Description**: Conditions denoising network on fitness information, similar to Conditional VAE (C-VAE).

**Inspired by**:
- Conditional VAE (C-VAE) from `discrete_vae_analysis.md`
- Fitness-guided blending from `discrete_dbd.py`

**Architecture**:
```
Input: [corrupted_solution, timestep, fitness]
       ↓
Network: [solution_embedding + time_embedding + fitness_embedding]
       ↓
Output: clean_solution_probabilities
```

**How It Works**:
1. Fitness is normalized and embedded into a low-dimensional vector
2. Fitness embedding is concatenated with solution and timestep
3. Network learns to denoise conditioned on target fitness level
4. During sampling, can specify desired fitness level

**When to Use**:
- Want to guide generation towards high-fitness regions
- Have clear fitness targets or objectives
- Need more control over solution quality
- Inspired by conditional generative models

**Parameters**:
```bash
--fitness_guided 1
```

**Benefits**:
- Directed exploration toward high-fitness regions
- Can sample at different fitness levels
- Leverages fitness information during training
- More sample-efficient on difficult problems

---

## Architecture and Design

### Network Architecture

Both Gumbel and Corruption variants use similar MLP architectures with key differences in input/output:

#### Standard Architecture (Base Variants)

```
Gumbel Variant:
Input: [x_t (binary corrupted), t (timestep)] → concat with time_emb
       ↓
Hidden: [max(10, n_vars//2), max(10, n_vars//4)] with ReLU + Dropout(0.1)
       ↓
Output: [n_vars × 2] logits for binary classification

Corruption Variant:
Input: [x_t (binary corrupted), t (timestep)] → concat with time_emb
       ↓
Hidden: [max(10, n_vars//2), max(10, n_vars//4)] with ReLU + Dropout(0.1)
       ↓
Output: [n_vars] logits for bit probabilities
```

#### Fitness-Guided Architecture (Enhanced Variants)

```
Input: [x_t (binary), t (timestep), fitness (scalar)]
       ↓
Embeddings: [time_emb (16-32 dim), fitness_emb (8 dim)]
       ↓
Concat: [x_t, time_emb, fitness_emb]
       ↓
Hidden: [adaptive dims] with configurable activation + Dropout
       ↓
Output: [binary predictions]
```

### Adaptive Dimension Calculation

Following recommendations from `DISCRETE_DENDIFF_ANALYSIS.md`:

```python
# Hidden dimensions adapt to problem size and population
hidden_dims = [
    max(10, n_vars // 2),
    max(10, n_vars // 4)
]

# Batch size adapts to selected population
batch_size = max(10, selected_pop_size // 20)

# Time embedding dimension
time_emb_dim = min(32, max(4, n_vars // 8))
```

**Rationale**:
- Prevents overfitting on small populations
- Scales appropriately with problem size
- Maintains reasonable parameter/sample ratio
- Enables more weight updates with smaller batches

---

## Parameters Guide

### Complete Parameter List

```bash
python discrete_Dendiff_EDA.py \
    <seed>              # Random seed for reproducibility
    <obj_func>          # Objective function (OneMax, HIFF, Deceptive3, etc.)
    <n>                 # Number of variables
    <pop_size>          # Population size
    <n_gen>             # Number of generations
    <trunc>             # Selection ratio (0.0-1.0, e.g., 0.5 for 50%)
    <variant>           # dendiff_gumbel or dendiff_corruption
    <sampling_strategy> # gumbel or corruption
    <activation>        # relu, tanh, sigmoid, elu, selu, gelu, leakyrelu
    <loss>              # mse, weighted_mse, ranking, huber
    <n_timesteps>       # Training timesteps (50-100)
    <n_sampling_steps>  # Sampling steps (10-50)
    <fitness_guided>    # 0=no, 1=yes
    <temperature>       # Gumbel temperature or sampling temp (0.5-2.0)
    <beta_start>        # Starting corruption (0.0001-0.01)
    <beta_end>          # Ending corruption (0.3-0.5)
```

### Parameter Details

#### Core Parameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `variant` | Dendiff variant | `dendiff_gumbel`, `dendiff_corruption` | - |
| `n_timesteps` | Training diffusion steps | 50-100 | Gumbel:100, Corruption:50 |
| `n_sampling_steps` | Denoising steps during sampling | 10-50 | 20 |
| `temperature` | Sampling temperature | 0.5-2.0 | Gumbel:1.0, Corruption:0.5 |
| `beta_start` | Initial corruption level | 0.0001-0.01 | Gumbel:0.0001, Corruption:0.01 |
| `beta_end` | Maximum corruption level | 0.3-0.5 | Gumbel:0.3, Corruption:0.5 |

#### Enhanced Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `activation` | Hidden layer activation | relu, tanh, elu, selu, gelu, sigmoid | relu |
| `loss` | Loss function | mse, weighted_mse, ranking, huber | mse |
| `fitness_guided` | Fitness conditioning | 0, 1 | 0 |
| `sampling_strategy` | Sampling method | gumbel, corruption | (matches variant) |

### Parameter Recommendations

#### For Small Problems (n_vars ≤ 30)

```bash
--variant dendiff_corruption \
--n_timesteps 50 \
--n_sampling_steps 20 \
--temperature 0.5 \
--beta_start 0.01 \
--beta_end 0.5 \
--activation relu \
--loss mse
```

#### For Medium Problems (30 < n_vars ≤ 100)

```bash
--variant dendiff_gumbel \
--n_timesteps 100 \
--n_sampling_steps 20 \
--temperature 1.0 \
--beta_start 0.0001 \
--beta_end 0.3 \
--activation elu \
--loss weighted_mse \
--fitness_guided 1
```

#### For Deceptive/Hard Problems

```bash
--variant dendiff_gumbel \
--n_timesteps 100 \
--n_sampling_steps 30 \
--temperature 1.5 \
--fitness_guided 1 \
--activation tanh \
--loss weighted_mse
```

---

## Usage Examples

### Example 1: Basic Dendiff-Gumbel on OneMax

```bash
python examples/discrete_Dendiff_EDA.py \
    0 OneMax 20 80 20 0.5 \
    dendiff_gumbel gumbel relu mse \
    100 20 0 1.0 0.0001 0.3
```

**Explanation**:
- Seed: 0
- Problem: OneMax with 20 variables
- Population: 80, Generations: 20, Selection: 50%
- Standard Gumbel variant with MSE loss
- 100 training timesteps, 20 sampling steps
- No fitness guidance
- Temperature 1.0, beta range [0.0001, 0.3]

### Example 2: Dendiff-Corruption with Fitness Guidance on Deceptive3

```bash
python examples/discrete_Dendiff_EDA.py \
    42 Deceptive3 30 150 40 0.5 \
    dendiff_corruption corruption tanh weighted_mse \
    50 20 1 0.5 0.01 0.5
```

**Explanation**:
- Seed: 42
- Problem: Deceptive3 with 30 variables (requires multiple of 3)
- Population: 150, Generations: 40, Selection: 50%
- Corruption variant with tanh activation and weighted MSE loss
- 50 training timesteps, 20 sampling steps
- **Fitness guidance enabled**
- Temperature 0.5, corruption range [0.01, 0.5]

### Example 3: Dendiff-Gumbel with Huber Loss on HIFF

```bash
python examples/discrete_Dendiff_EDA.py \
    7 HIFF 64 200 50 0.5 \
    dendiff_gumbel gumbel elu huber \
    100 25 0 1.0 0.0001 0.3
```

**Explanation**:
- Seed: 7
- Problem: HIFF with 64 variables (must be power of 2)
- Population: 200, Generations: 50, Selection: 50%
- Gumbel variant with ELU activation and Huber loss (robust)
- 100 training timesteps, 25 sampling steps
- No fitness guidance
- Temperature 1.0, beta range [0.0001, 0.3]

### Example 4: Fast Dendiff-Corruption for Quick Experiments

```bash
python examples/discrete_Dendiff_EDA.py \
    0 FC3 30 100 30 0.5 \
    dendiff_corruption corruption relu mse \
    30 10 0 0.5 0.02 0.4
```

**Explanation**:
- Faster variant: fewer timesteps and sampling steps
- 30 training timesteps instead of 50
- Only 10 sampling steps instead of 20
- Good for quick experiments or debugging

---

## Comparison with Other EDAs

### vs. Discrete VAE

| Aspect | Dendiff | VAE |
|--------|---------|-----|
| **Architecture** | Simpler (no encoder) | More complex (encoder-decoder) |
| **Training Stability** | More stable | Risk of posterior collapse |
| **Sampling Speed** | Slower (T iterations) | Faster (1 forward pass) |
| **Sample Quality** | Progressive refinement | Direct generation |
| **Hyperparameters** | More (timesteps, schedule) | Fewer but sensitive (beta) |
| **Fitness Guidance** | Optional conditioning | E-VAE, CE-VAE variants |
| **Best For** | Small pops, stability | Large pops, speed |

### vs. Discrete DbD

| Aspect | Dendiff | DbD |
|--------|---------|-----|
| **Population Requirement** | Single population | Two populations (source, target) |
| **Training Objective** | Denoising | Deblending/interpolation |
| **Theoretical Foundation** | DDPM (strong) | Heuristic blending |
| **Complexity** | Medium | Medium-High |
| **Interpretability** | Corruption metaphor | Blending metaphor |
| **Best For** | Single-pop scenarios | Multi-pop available |

### vs. Discrete GAN

| Aspect | Dendiff | GAN |
|--------|---------|-----|
| **Training Stability** | Stable | Adversarial (unstable) |
| **Mode Collapse** | No risk | High risk |
| **Sample Quality** | Good, progressive | Excellent when trained |
| **Training Difficulty** | Moderate | High (requires expertise) |
| **Sampling Speed** | Slow | Fast |
| **Best For** | Stable, reliable | High-quality when tuned |

### vs. Discrete Backdrive

| Aspect | Dendiff | Backdrive |
|--------|---------|-----------|
| **Training Paradigm** | Forward denoising | Inverse mapping |
| **Architecture** | MLP + time embedding | MLP (simpler) |
| **Theoretical Basis** | DDPM framework | Network inversion |
| **Speed** | Slower | Faster |
| **Fitness Mapping** | Optional conditioning | Direct fitness→solution |
| **Best For** | Generative modeling | Direct optimization |

### vs. Traditional EDAs (UMDA, TreeEDA)

| Aspect | Dendiff | Traditional EDAs |
|--------|---------|------------------|
| **Expressiveness** | High (neural network) | Limited (factorized models) |
| **Computational Cost** | High (GPU beneficial) | Low (CPU sufficient) |
| **Scalability** | Good to n~100 | Excellent (n>1000) |
| **Interpretability** | Medium (corruption) | High (probabilistic model) |
| **Sample Efficiency** | Medium | High (on simple problems) |
| **Best For** | Complex landscapes | Simple/medium complexity |

---

## Best Practices

### 1. Choosing a Variant

**Use Dendiff-Gumbel when**:
- Need theoretically grounded approach
- Want maximum flexibility (temperature, schedule tuning)
- Quality is priority over speed
- Conducting research on diffusion models
- Problem size: n_vars ≤ 100

**Use Dendiff-Corruption when**:
- Prefer simpler, more interpretable approach
- Speed is important
- Familiar with BERT/corruption paradigms
- Want straightforward implementation
- Problem size: n_vars ≤ 100

### 2. Activation Function Selection

**ReLU** (default):
- Good general-purpose choice
- Fast, simple, works well in most cases

**ELU/SELU**:
- Better for deeper networks
- Self-normalizing properties
- Recommended for n_vars > 50

**Tanh**:
- Bounded outputs helpful for some problems
- Good for deceptive landscapes
- Can help with gradient flow

**GELU**:
- Modern activation, used in transformers
- Smooth approximation to ReLU
- Experimental, may improve performance

### 3. Loss Function Selection

**MSE/Standard** (default):
- Start here for most problems
- Well-understood, stable training

**Weighted MSE/BCE**:
- When fitness values vary widely
- Want to prioritize high-quality solutions
- Population has many outliers

**Ranking**:
- When absolute fitness is unreliable
- Relative ordering is what matters
- Noisy fitness evaluations

**Huber**:
- Robust to outliers and extreme values
- Training instability with standard loss
- Noisy or unreliable evaluations

### 4. Fitness Guidance

**Enable (fitness_guided=1) when**:
- Clear fitness targets or objectives
- Want directed exploration
- Difficult or deceptive problems
- Have sufficient population (>100)

**Disable (fitness_guided=0) when**:
- Exploratory optimization
- Very small populations (<100)
- Want simpler, faster training
- Baseline comparison needed

### 5. Hyperparameter Tuning

**Timesteps (n_timesteps)**:
- More timesteps = finer corruption control
- Gumbel: 50-100, Corruption: 30-50
- Increase for difficult problems
- Decrease for faster training

**Sampling Steps (n_sampling_steps)**:
- More steps = higher quality, slower sampling
- Typical range: 10-30
- Can reduce to 10 for speed
- Increase to 30-50 for quality

**Temperature**:
- Lower (0.5) = more discrete, less exploration
- Higher (1.5-2.0) = more stochastic, more exploration
- Anneal during run: start high, end low
- Gumbel typical: 0.8-1.2, Corruption: 0.4-0.6

**Beta/Corruption Range**:
- Start very low (0.0001-0.01) for gradual corruption
- End moderate (0.3-0.5) - don't fully destroy signal
- Gumbel: [0.0001, 0.3]
- Corruption: [0.01, 0.5]

### 6. Population Size Guidelines

| Problem Size | Min Pop Size | Recommended | Selection Ratio |
|--------------|--------------|-------------|-----------------|
| n ≤ 30 | 80 | 100-150 | 0.5 |
| 30 < n ≤ 60 | 150 | 200-300 | 0.5 |
| 60 < n ≤ 100 | 200 | 300-400 | 0.5 |

**Rule of thumb**: `pop_size ≥ 3 * n_vars`

### 7. Performance Optimization

**For Faster Execution**:
- Use Dendiff-Corruption instead of Gumbel
- Reduce n_timesteps (e.g., 30-50)
- Reduce n_sampling_steps (e.g., 10-15)
- Smaller hidden dimensions
- Disable fitness guidance

**For Better Quality**:
- Use Dendiff-Gumbel with more timesteps (100+)
- More sampling steps (25-50)
- Enable fitness guidance
- Larger hidden dimensions
- Higher temperature for exploration

---

## Troubleshooting

### Issue: Poor Solution Quality

**Possible Causes & Solutions**:

1. **Underfitting (too few parameters)**
   - Increase hidden dimensions
   - More training epochs
   - Reduce dropout if using

2. **Overfitting (too many parameters)**
   - Decrease hidden dimensions
   - Smaller batch size for more updates
   - Increase dropout
   - More training data (larger population)

3. **Insufficient denoising**
   - Increase n_sampling_steps
   - Lower temperature for more deterministic sampling
   - Adjust beta_end (less extreme corruption)

4. **Loss function mismatch**
   - Try weighted_mse for variable fitness
   - Try huber for noisy fitness
   - Enable fitness_guided for directed search

### Issue: Training Instability

**Possible Causes & Solutions**:

1. **Gradient explosion**
   - Already using gradient clipping (max_norm=1.0)
   - Reduce learning rate
   - Use ELU or SELU activation instead of ReLU

2. **Beta schedule too aggressive**
   - Reduce beta_end (e.g., from 0.5 to 0.3)
   - Use gentler schedule (cosine instead of linear)

3. **Batch size issues**
   - Ensure batch_size is reasonable (10-32)
   - Check adaptive batch calculation
   - Manually set if needed

### Issue: Slow Performance

**Possible Causes & Solutions**:

1. **Too many timesteps**
   - Reduce n_timesteps (try 50 instead of 100)
   - Reduce n_sampling_steps (try 10-15)

2. **Network too large**
   - Check adaptive hidden dims calculation
   - Manually reduce if needed
   - Use smaller time_emb_dim

3. **Fitness guidance overhead**
   - Disable if not needed
   - Reduces computational cost

4. **CPU vs GPU**
   - PyTorch will use GPU if available
   - For very small problems, CPU may be faster
   - Check CUDA availability

### Issue: No Improvement Over Generations

**Possible Causes & Solutions**:

1. **Model not learning**
   - Check if loss is decreasing during training
   - Increase epochs or learning rate
   - Verify population has fitness variation

2. **Sampling not using learned model**
   - Verify model is being loaded correctly
   - Check for exceptions in sampling
   - Try deterministic sampling (temperature→0)

3. **Selection too weak**
   - Increase selection ratio (e.g., 0.3 instead of 0.5)
   - Larger population size
   - More generations

### Issue: Variant Selection Confusion

**Guidelines**:

| Use Case | Recommended Variant |
|----------|---------------------|
| First time using Dendiff | dendiff_corruption with default params |
| Research/comparison | dendiff_gumbel (more standard DDPM) |
| Speed critical | dendiff_corruption with reduced timesteps |
| Quality critical | dendiff_gumbel with fitness guidance |
| Deceptive problem | Either with fitness_guided=1 |
| Simple problem | dendiff_corruption (faster, sufficient) |

---

## Advanced Topics

### Combining with Other Techniques

#### Hybrid Dendiff-UMDA

Start with UMDA for quick rough solutions, then refine with Dendiff:

1. Run UMDA for initial generations
2. Switch to Dendiff when convergence slows
3. Use Dendiff for fine-grained search

#### Ensemble Dendiff

Train multiple Dendiff models and combine:

1. Train 3-5 Dendiff models with different seeds
2. Sample from each independently
3. Combine populations and select best
4. Improves robustness and quality

### Experimental Variants

Future enhancements being considered (see `DISCRETE_DENDIFF_ANALYSIS.md` Section 10):

1. **Adaptive Timesteps**: Dynamic adjustment based on convergence
2. **Hierarchical Dendiff**: Multi-scale denoising for structured problems
3. **Variance Reduction**: Multiple denoising chains with best selection
4. **Schedule Learning**: Learn optimal beta/corruption schedules

---

## Computational Complexity

### Time Complexity

**Training (per generation)**:
```
O(epochs × pop_size × n_timesteps × n_vars × hidden_dim)
```

**Sampling (per generation)**:
```
O(pop_size × n_sampling_steps × n_vars × hidden_dim)
```

**Example** (n=30, pop=150, selected=75, hidden=32):
- Training: ~50 epochs × 75 × 30 × 32 ≈ 3.6M ops
- Sampling: 75 × 20 × 30 × 32 ≈ 1.4M ops
- Total per generation: ~5M ops

### Memory Complexity

```
O(n_vars × hidden_dim + batch_size × n_vars)
```

**Example** (n=30, hidden=32, batch=16):
- Parameters: ~3KB
- Activations: ~2KB
- Total: ~5KB (very small, GPU not required)

---

## References

For more details, see:

1. **Implementation Files**:
   - `examples/discrete_Dendiff_EDA.py`: Main program
   - `learning/discrete_dendiff_gumbel.py`: Base Gumbel variant
   - `learning/discrete_dendiff_corruption.py`: Base Corruption variant
   - `learning/discrete_dendiff_gumbel_enhanced.py`: Enhanced Gumbel with loss/fitness
   - `learning/discrete_dendiff_corruption_enhanced.py`: Enhanced Corruption with loss/fitness

2. **Analysis Documents**:
   - `DISCRETE_DENDIFF_ANALYSIS.md`: Detailed design and analysis
   - `DISCRETE_VAE_ANALYSIS.md`: VAE comparison and insights
   - `DISCRETE_DBD_ANALYSIS.md`: DbD comparison and insights

3. **Related Examples**:
   - `examples/discrete_EDA.py`: Unified interface for all EDAs
   - `examples/discrete_DbD_EDA.py`: DbD variants reference
   - `examples/discrete_Backdrive_EDA.py`: Backdrive variants reference

---

## Quick Start Checklist

- [ ] Choose variant (Gumbel for quality, Corruption for speed)
- [ ] Set appropriate n_timesteps (Gumbel: 100, Corruption: 50)
- [ ] Set n_sampling_steps (start with 20)
- [ ] Choose activation (relu is good default)
- [ ] Choose loss function (mse to start, weighted_mse for varied fitness)
- [ ] Decide on fitness guidance (0 for baseline, 1 for directed search)
- [ ] Set population size (at least 3× number of variables)
- [ ] Set appropriate beta/corruption range for your variant
- [ ] Run and monitor best fitness progress
- [ ] Adjust hyperparameters based on results

---

## Contributing

To add new Dendiff variants or improvements:

1. Study existing implementations in `learning/discrete_dendiff_*.py`
2. Follow the established architecture patterns
3. Add corresponding sampling functions in `sampling/discrete_dendiff.py`
4. Update `examples/discrete_Dendiff_EDA.py` to support new variant
5. Add tests and documentation
6. Update this guide with new variant information

---

**Version**: 1.0
**Last Updated**: 2026-01-06
**Author**: PATEDA Development Team
