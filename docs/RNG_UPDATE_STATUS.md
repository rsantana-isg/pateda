# RNG Parameter Update Status

## Overview
This document tracks the progress of adding `rng: Optional[np.random.Generator] = None` parameter support to all sampling method files in `/home/user/pateda/pateda/sampling/`.

## Update Pattern

For each file, the following changes are needed:

### 1. Add Optional import (if not present)
```python
from typing import Optional
```

### 2. Add rng parameter to sample() method signature
```python
def sample(
    self,
    n_vars: int,
    model: Model,
    cardinality: np.ndarray,
    aux_pop: Optional[np.ndarray] = None,
    aux_fitness: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,  # ADD THIS
    **params: Any,
) -> np.ndarray:
```

### 3. Initialize RNG at start of method
```python
if rng is None:
    rng = np.random.default_rng()
```

### 4. Replace np.random.* calls with rng.* calls
- `np.random.randint(a, b, size=n)` → `rng.integers(a, b, size=n)`
- `np.random.choice(...)` → `rng.choice(...)`
- `np.random.uniform(...)` → `rng.uniform(...)`
- `np.random.rand(m, n)` → `rng.random(size=(m, n))`
- `np.random.random()` → `rng.random()`
- `np.random.randn(m, n)` → `rng.standard_normal(size=(m, n))`
- `np.random.normal(mean, std, size=n)` → `rng.normal(mean, std, size=n)`
- `np.random.shuffle(arr)` → `rng.shuffle(arr)`
- `np.random.permutation(n)` → `rng.permutation(n)`
- `np.random.multivariate_normal(...)` → `rng.multivariate_normal(...)`

## Files Completed ✅

### Core Infrastructure
- ✅ **utils.py** - Updated `stochastic_universal_sampling()` function
  - Added rng parameter
  - Replaced `np.random.rand()` → `rng.random()`
  - Replaced `np.random.shuffle()` → `rng.shuffle()`

### SamplingMethod Classes
- ✅ **fda.py** - SampleFDA class
  - Added rng parameter to `sample()` method
  - Passes rng to `stochastic_universal_sampling()` calls

- ✅ **bayesian_network.py** - SampleBayesianNetwork class
  - Added rng parameter
  - Replaced `np.random.choice()` calls → `rng.choice()`

- ✅ **markov.py** - SampleMarkovChain and SampleMarkovChainForward classes
  - Added rng parameter to both classes
  - Replaced `np.random.choice()` calls → `rng.choice()`
  - Passes rng to `self._fda_sampler.sample()`

- ✅ **cumda.py** - SampleCUMDA and SampleCUMDARange classes
  - Added rng parameter to both classes
  - Replaced `np.random.randint()` → `rng.integers()`
  - Passes rng to `stochastic_universal_sampling()` calls

## Files Remaining ⏳

### SamplingMethod Classes
- ⏳ **histogram.py** - SampleEHM, SampleNHM (use `__call__` instead of `sample`)
  - Need to add rng to `__call__()` method
  - Replace `np.random.randint()` and `np.random.rand()`

- ⏳ **mallows.py** - Mallows model samplers (use `__call__` instead of `sample`)
  - SampleMallowsKendall, SampleMallowsCayley, etc.
  - Replace `np.random.rand()` calls

- ⏳ **mixture_gaussian.py** - SampleMixtureGaussian class + standalone functions
  - Update standalone functions: `sample_mixture_gaussian_univariate`, `sample_mixture_gaussian_full`, etc.
  - Update class `sample()` method
  - Replace `np.random.choice()`, `np.random.normal()`, `np.random.multivariate_normal()`

- ⏳ **mixture_trees.py** - SampleMixtureTrees, SampleMixtureTreesDirect
  - Replace `np.random.choice()` and `np.random.randint()`
  - Pass rng to FDA sampler calls

- ⏳ **basic_gaussian.py** - SampleGaussianUnivariate, SampleGaussianFull + standalone functions
  - Update standalone functions
  - Replace `np.random.normal()` and `np.random.multivariate_normal()`

- ⏳ **gibbs.py** - SampleGibbs class
  - Replace `np.random.randint()`, `np.random.permutation()`, `np.random.choice()`

- ⏳ **map_sampling.py** - SampleInsertMAP, SampleTemplateMAP, SampleHybridMAP
  - Replace `np.random.choice()`, `np.random.random()`, `np.random.randint()`

- ⏳ **cfda.py** - SampleCFDA, SampleCFDARange, SampleCFDAWeighted
  - Pass rng to `self.fda_sampler.sample()` calls

### Standalone Functions (No SamplingMethod Class)
- ⏳ **vine_copula.py** - Standalone functions
  - Add rng parameter to functions: `sample_vine_copula`, `sample_vine_copula_biased`, etc.
  - Replace `np.random.random()` and `np.random.randn()`

- ⏳ **vae.py** - Standalone functions (uses PyTorch)
  - Add rng parameter to functions
  - Handle `torch.randn()` - may need torch.Generator
  - Replace `np.random.randn()`

- ⏳ **dae.py** - Standalone functions (uses PyTorch)
  - Add rng parameter
  - Handle `torch.rand()` - may need torch.Generator

- ⏳ **backdrive.py** - Standalone functions (uses PyTorch)
  - Add rng parameter
  - Handle `torch.rand()`, `torch.randn()`

- ⏳ **dendiff.py** - Standalone functions (uses PyTorch)
  - Add rng parameter
  - Handle `torch.randn()`

- ⏳ **gan.py** - Standalone function (uses PyTorch)
  - Add rng parameter
  - Handle `torch.randn()`

- ⏳ **rbm.py** - Standalone functions
  - Add rng parameter
  - Replace `np.random.randint()`

- ⏳ **dbd.py** - Standalone functions
  - Add rng parameter
  - Replace `np.random.choice()` and `np.random.normal()`

## PyTorch Files Note

Files using PyTorch (`torch.randn`, `torch.rand`, etc.) may require special handling:
- PyTorch has its own `torch.Generator` for reproducibility
- Consider whether to:
  1. Add both `rng` (for numpy) and `torch_rng` (for torch) parameters
  2. Set PyTorch seed based on numpy rng state
  3. Use manual seeding approach

For now, these files can accept `rng` parameter for numpy operations but may not fully control PyTorch randomness without additional work.

## Quick Reference: Files by Category

**Core (with dependencies):**
- ✅ utils.py
- ✅ fda.py
- ⏳ cfda.py (delegates to fda)

**Discrete EDAs:**
- ✅ bayesian_network.py
- ✅ markov.py
- ⏳ gibbs.py
- ⏳ histogram.py
- ⏳ mallows.py
- ⏳ map_sampling.py

**Constraint Models:**
- ✅ cumda.py
- ⏳ cfda.py

**Continuous EDAs:**
- ⏳ basic_gaussian.py
- ⏳ mixture_gaussian.py
- ⏳ mixture_trees.py
- ⏳ vine_copula.py

**Neural Network Based:**
- ⏳ vae.py (PyTorch)
- ⏳ dae.py (PyTorch)
- ⏳ gan.py (PyTorch)
- ⏳ rbm.py
- ⏳ backdrive.py (PyTorch)
- ⏳ dendiff.py (PyTorch)
- ⏳ dbd.py

## Testing Recommendations

After updating all files:
1. Run existing test suite to ensure backward compatibility
2. Add tests for RNG reproducibility
3. Verify that passing same seed produces same results
4. Check that None parameter works (uses default_rng)

## Next Steps

1. Complete remaining SamplingMethod classes (gibbs, map_sampling, cfda, mixture_*, basic_gaussian)
2. Update standalone functions (vine_copula, dbd)
3. Handle PyTorch-based methods (vae, dae, gan, dendiff, backdrive)
4. Update histogram and mallows (use `__call__` pattern)
5. Run comprehensive tests
