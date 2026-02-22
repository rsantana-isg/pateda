# EDA Multi-value Representations Analysis

This document analyzes the support for **non-binary discrete representations** (cardinality > 2)
in the EDA implementations of this repository. Algorithms are categorized as having full support,
partial support, or binary-only support, and the required extensions are described for those that
lack full support.

---

## Summary Table

| Algorithm / File | Class | Non-binary Support | Notes |
|---|---|---|---|
| UMDA (`learning/umda.py`) | `LearnUMDA` | ✅ Full | Uses `cardinality` array throughout |
| PBIL (`learning/pbil.py`) | `LearnPBIL` | ✅ Full | Initialises uniform tables from `cardinality` |
| MIMIC (`learning/mimic.py`) | `LearnMIMIC` | ✅ Full | Uses `cardinality` for mutual-information ordering |
| BMDA (`learning/bmda.py`) | `LearnBMDA` | ✅ Full | Bivariate counts respect `cardinality` |
| EBNA (`learning/ebna.py`) | `LearnEBNA` | ✅ Full | BN structure/CPTs sized by `cardinality` |
| BOA (`learning/boa.py`) | `LearnBOA` | ✅ Full | Same as EBNA |
| FDA (`learning/fda.py`) | `LearnFDA` | ✅ Full | Delegates to `learn_fda_parameters` which uses `cardinality` |
| Tree-EDA (`learning/tree.py`) | `LearnTreeModel` | ✅ Full (fixed) | Divide-by-zero bug fixed; Laplace prior added |
| MN-FDA (`learning/mnfda.py`) | `LearnMNFDA` | ✅ Full | Uses MI matrix with `cardinality` |
| MN-FDAG (`learning/mnfdag.py`) | `LearnMNFDAG` | ✅ Full | G-test degrees of freedom account for `cardinality` |
| MOA (`learning/moa.py`) | `LearnMOA` | ✅ Full | Markov Overlapping-clique model; cardinality-aware |
| BSC (`learning/bsc.py`) | `LearnBSC` | ✅ Full | Sparse conditional counts sized by `cardinality` |
| Softmax RBM (`learning/rbm.py`) | `SoftmaxRBM` | ✅ Full | One-hot encoding covers any cardinality |
| Mixture Trees (`learning/mixture_trees.py`) | `LearnMixtureTrees` | ✅ Full | Delegates to `LearnTreeModel` per mixture |
| Affinity (`learning/affinity.py`) | `LearnAffinity` | ✅ Full | Cardinality-parameterised |
| Discrete VAE (`learning/discrete_vae.py`) | `CategoricalVAEDecoder` / `create_categorical_vae` | ✅ Full | Gumbel-Softmax relaxation; one-hot over each variable |
| Discrete GAN (`learning/discrete_gan.py`) | `CategoricalGANGenerator` / `create_categorical_gan` | ✅ Full | Gumbel-Softmax over full cardinality |
| Discrete Backdrive (`learning/discrete_backdrive.py`) | `DiscreteBackdriveNetwork` | ✅ Full (with embeddings) | Embedding layers activated automatically when `max(cardinality) > 2` |
| Binary VAE (`learning/discrete_vae.py`) | `BinaryVAE` / `create_binary_vae` | ⚠️ Binary-only | Bernoulli decoder; no multi-value path |
| Binary GAN (`learning/discrete_gan.py`) | `BinaryGANGenerator` / `create_binary_gan` | ⚠️ Binary-only | Sigmoid output; no multi-value path |
| CFDA (`learning/cfda.py`) | `LearnCFDA` | ❌ Binary-only | Explicitly raises `ValueError` when `cardinality != 2` |
| CUMDA (`learning/cumda.py`) | `LearnCUMDA` | ❌ Binary-only | Explicitly raises `ValueError` when `cardinality != 2` |
| DAE (`learning/dae.py`) | `DiscreteDAE` | ❌ Binary-only | Input dimension = `n_vars` (not `sum(cardinality)`); sigmoid output |
| Discrete DbD (`learning/discrete_dbd.py`) | `BinaryDbDNetwork` | ❌ Binary-only | Binary blending, binary cross-entropy |
| Discrete DenDiff – STE (`learning/discrete_dendiff_ste.py`) | `STEDenoisingMLP` | ❌ Binary-only | Hard-binary STE; binary BCE loss |
| Discrete DenDiff – Gumbel (`learning/discrete_dendiff_gumbel.py`) | (various) | ⚠️ Partial | Gumbel-Softmax is cardinality-aware but input encoding may assume binary |
| Discrete DenDiff – other variants | Multiple | ⚠️ Partial | Largely binary-oriented; see details below |

---

## Fully Supported Algorithms

The following algorithms **already handle non-binary discrete variables** with cardinality > 2
without any modifications:

### Classical probabilistic models
- **UMDA** – counts and normalises over arbitrary cardinality.
- **PBIL** – learning-rate update is independent of cardinality.
- **MIMIC** – chains variables in order of maximum conditional MI; works for any cardinality.
- **BMDA** – bivariate joint counts indexed by actual value pairs; cardinality-agnostic.
- **EBNA / BOA** – Bayesian Network CPTs are sized `card_parent × card_child`; fully general.
- **FDA** – factorised model; `learn_fda_parameters` uses `cardinality` throughout.
- **Tree-EDA (fixed)** – Chow–Liu tree; Laplace-smoothed conditional probabilities prevent
  divide-by-zero and ensure non-zero prior for absent configurations (see bug fix below).
- **MN-FDA / MN-FDAG / MOA** – Markov-network models; cardinality used in mutual-information
  and clique-table computations.

### Neural-network models with categorical support
- **Categorical VAE** (`create_categorical_vae`) – Gumbel-Softmax over each variable's values;
  one-hot input encoding respects arbitrary cardinality.
- **Categorical GAN** (`create_categorical_gan`) – same encoding strategy.
- **Softmax RBM** – one-hot inputs; softmax visible layer.
- **Discrete Backdrive** – embedding layers are inserted automatically for `cardinality > 2`.

---

## Bug Fix: Tree-EDA Divide-by-Zero

**Symptom:**
```
pateda/learning/tree.py:303: RuntimeWarning: divide by zero encountered in divide
cond_biv_prob_mle = aux_biv_prob / parent_probs
```

**Root cause:**
`parent_probs` was computed from raw (unsmoothed) univariate frequencies.  When a parent
variable's value never appears in the selected population its frequency is exactly 0, causing
division by zero.  The code then returned the MLE (unsmoothed) conditional table instead of
the already-computed Laplace-smoothed table.

**Fix (applied in `learning/tree.py`):**
1. Compute Laplace-smoothed parent marginals before division:
   ```python
   lap_parent_probs = (univ_prob[parent_idx] * n_samples + 1) / (n_samples + card_parent)
   ```
2. Use `lap_parent_probs` (not the raw `univ_prob`) as divisor.
3. Return the Laplace-smoothed conditional table instead of the MLE table.

This also satisfies the requirement that *configurations not present in the population have
a non-zero probability* (prior), which is essential for correct exploration in non-binary
problems where low population sizes may not cover all values.

---

## Algorithms Requiring Extension

### 1. CFDA (`learning/cfda.py`) – `LearnCFDA`

**Issue:** Explicitly rejects non-binary input:
```python
if not np.all(cardinality == 2):
    raise ValueError("CFDA only works with binary variables (cardinality=2)")
```

**What to implement:**
- Replace binary-specific constraint representation with a generalised version that
  operates on integer variables.
- The constraint satisfaction check must be extended to accept multi-value variables.
- Sampling (`sampling/cfda.py`) must be updated to use the generalised probability tables.

---

### 2. CUMDA (`learning/cumda.py`) – `LearnCUMDA`

**Issue:** Explicitly rejects non-binary input:
```python
if not np.all(cardinality == 2):
    raise ValueError("CUMDA only works with binary variables (cardinality=2)")
```

**What to implement:**
- Generalise the marginal representation from a single `p(x_i = 1)` scalar to a full
  probability vector of length `cardinality[i]`.
- Update the update rule (learning-rate blend) to work on probability vectors.
- Update the cumulative-constraint logic to handle multi-value variables.

---

### 3. DAE (`learning/dae.py`) – `DiscreteDAE`

**Issue:**
- Network input dimension is `n_vars` instead of `sum(cardinality)` (one-hot).
- Output activation is sigmoid, producing one value in `[0,1]` per variable (binary).

**What to implement:**
- Add one-hot encoding layer (input size `sum(cardinality)`) and softmax output per
  variable group (of size `cardinality[i]`).
- Replace binary cross-entropy loss with categorical cross-entropy.
- Provide a `corrupt_categorical` function analogous to `corrupt_binary`.
- Generalise the population-to-tensor conversion to use one-hot encoding.

---

### 4. Discrete DbD (`learning/discrete_dbd.py`) – `BinaryDbDNetwork`

**Issue:**
- Blending of two populations is defined as `x_blended = alpha * x0 + (1-alpha) * x1`
  for binary values; this is not meaningful for categorical values.
- Loss is binary cross-entropy.

**What to implement:**
- Define a **categorical blending** strategy, e.g. probabilistic interpolation: sample
  each position from `x0` with probability `alpha` and from `x1` with probability `1-alpha`.
- Replace `BinaryDbDNetwork` with a categorical variant:
  - Input: one-hot encoded blended sample + `alpha` scalar.
  - Output: softmax probabilities over `cardinality[i]` per variable.
  - Loss: categorical cross-entropy.
- Update `create_blended_binary_samples` to `create_blended_categorical_samples`.
- Update the sampling procedure to use the categorical output.

---

### 5. Discrete DenDiff variants (`learning/discrete_dendiff_*.py`)

The DenDiff family applies denoising in a diffusion-like process.  Most variants are
binary-oriented; the Gumbel variants are partially generalised.

#### 5a. STE variant (`discrete_dendiff_ste.py`) – `STEDenoisingMLP`
**Issue:** Hard-binary STE and binary BCE.  
**What to implement:**
- Replace binary STE with **categorical STE** using straight-through for argmax/one-hot.
- Change output to per-variable softmax, loss to categorical cross-entropy.

#### 5b. Corruption variant (`discrete_dendiff_corruption.py`)
**Issue:** `add_noise_binary` performs bit-flipping only valid for binary variables.  
**What to implement:**
- `add_noise_categorical`: randomly replace each position with a uniformly-chosen
  value from `{0, ..., cardinality[i]-1}` with probability equal to noise level.

#### 5c. Hard-concrete variant (`discrete_dendiff_hard_concrete.py`)
**Issue:** Hard-concrete relaxation targets Bernoulli (binary) gate.  
**What to implement:**
- Extend to categorical hard-concrete / straight-through Gumbel-Softmax.

#### 5d. Deterministic variant (`discrete_dendiff_deterministic.py`)
**Issue:** Deterministic binarisation step assumes two classes.  
**What to implement:**
- Replace deterministic binarisation with deterministic argmax over softmax outputs.

#### 5e. Gumbel variant (`discrete_dendiff_gumbel.py`) – *partially supported*
**Status:** Gumbel-Softmax is cardinality-aware in principle, but input encoding
  may still use flat binary representation.  
**What to implement:**
- Verify and update input encoding to one-hot per variable group.
- Ensure the forward pass reshapes correctly for arbitrary cardinality.

---

### 6. Binary VAE / Binary GAN

These classes (`BinaryVAE`, `BinaryGANGenerator`) are intentionally binary-only.
Their categorical counterparts (`CategoricalVAEDecoder`, `CategoricalGANGenerator`) already
provide full non-binary support.  No modification to the binary classes is required; users
should select the categorical classes for non-binary problems.

---

## Testing Multi-value EDA Implementations

A dedicated script `scripts/test_multivalue_eda.py` is provided to benchmark and validate
all supported discrete EDA implementations on non-binary benchmark functions.

It exercises the following benchmarks with configurable cardinality:
- **Integer OneMax** – simple convergence test.
- **Generalized k-deceptive** – deceptive trap requiring dependency modelling.
- **Integer Max Blocks** – block-level dependency.
- **Integer Multi-level Trap** – multi-level deceptive function.
- **Integer Dependency Chain** – sequential dependency.

Run with:
```bash
python scripts/test_multivalue_eda.py
```
or for a quick smoke test:
```bash
python scripts/test_multivalue_eda.py --quick
```

See also: `tests/test_tree_eda_multivalue.py` for unit/integration tests of the
Tree-EDA fix.
