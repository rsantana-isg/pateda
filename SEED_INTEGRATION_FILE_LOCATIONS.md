# Seed Parameter Integration - Detailed File Locations and Line Numbers

This document provides exact file paths and relevant line numbers for implementing seed parameter support.

---

## ABSOLUTE PATHS - All files

### Phase 1: Infrastructure (CRITICAL)

#### 1. Main EDA Class
**File**: `/home/user/pateda/pateda/core/eda.py`
- **Lines 127-134**: `__init__()` method - ADD random_seed parameter here
- **Lines 210-228**: `run()` method signature - verify it supports rng passing
- **Lines 255-260**: First seeding call in run() - pass rng here
- **Lines 288-295**: Sampling call in run() - pass rng here
- **Lines 362-366**: Selection call in run() - pass rng here
- **Lines 318-324**: Optional replacement call - pass rng here

#### 2. Component Abstract Base Classes
**File**: `/home/user/pateda/pateda/core/components.py`
- **Lines 17-40**: `SeedingMethod.seed()` - add rng parameter to signature and docstring
- **Lines 73-99**: `SamplingMethod.sample()` - add rng parameter to signature and docstring
- **Lines 101+**: Other abstract methods in SelectionMethod, ReplacementMethod, etc.

---

### Phase 2: Seeding Methods (CRITICAL)

#### 1. RandomInit
**File**: `/home/user/pateda/pateda/seeding/random_init.py` (61 lines total)
- **Lines 13-19**: Class definition
- **Lines 21-48**: `seed()` method definition
  - **Line 47**: `np.random.randint()` - REPLACE with `rng.integers()`
  - **Line 59**: `np.random.rand()` - REPLACE with `rng.random()`

#### 2. BiasInit  
**File**: `/home/user/pateda/pateda/seeding/bias_init.py` (53 lines total)
- Review for all `np.random.*()` calls
- Replace with `rng.*()` calls

#### 3. SeedingUnitationConstraint
**File**: `/home/user/pateda/pateda/seeding/seeding_unitation_constraint.py` (71 lines total)
- Review for all `np.random.*()` calls
- Replace with `rng.*()` calls

#### 4. SeedThisPop
**File**: `/home/user/pateda/pateda/seeding/seed_thispop.py` (72 lines total)
- Low priority - mostly uses provided population

---

### Phase 3: Sampling Methods (CRITICAL - 23 files)

#### Highest Priority (used in most EDAs)

##### 1. SampleFDA
**File**: `/home/user/pateda/pateda/sampling/fda.py` (241 lines total)
- **Lines 117-124**: `__init__()` method
- **Lines 126-149**: `sample()` method signature
- Search for all `np.random.*()` calls in sampling logic
- Common pattern: `np.random.rand()` for inverse transform sampling

##### 2. SampleBayesianNetwork
**File**: `/home/user/pateda/pateda/sampling/bayesian_network.py` (133 lines total)
- **Lines 22-29**: `__init__()` method
- **Lines 65-132**: `sample()` method
- **Lines 114, 130**: `np.random.choice()` - REPLACE with `rng.choice()`

##### 3. SampleHistogram
**File**: `/home/user/pateda/pateda/sampling/histogram.py`
- **Lines 52**: `np.random.randint()` - in SampleEHM
- **Line 83**: `np.random.rand()` - in _sample_categorical

##### 4. SampleMarkov
**File**: `/home/user/pateda/pateda/sampling/markov.py`
- Search for `np.random.choice()` calls
- Multiple occurrences throughout file

##### 5. SampleMallows
**File**: `/home/user/pateda/pateda/sampling/mallows.py` (404 lines total)
- Multiple `np.random.rand()` calls for sampling
- Perturb inversions probabilistically

#### Important (used frequently)

##### 6. SampleMixtureGaussian
**File**: `/home/user/pateda/pateda/sampling/mixture_gaussian.py` (274 lines total)
- `np.random.choice()` for component selection
- `np.random.randn()` for Gaussian sampling

##### 7. SampleMixtureTrees
**File**: `/home/user/pateda/pateda/sampling/mixture_trees.py` (273 lines total)
- `np.random.choice()` for tree selection
- `np.random.randint()` for value sampling

##### 8. SampleBasicGaussian
**File**: `/home/user/pateda/pateda/sampling/basic_gaussian.py` (334 lines total)
- `np.random.randn()` for continuous sampling
- Bounds checking with repairs

##### 9. SampleVineCopula
**File**: `/home/user/pateda/pateda/sampling/vine_copula.py` (304 lines total)
- `np.random.uniform()` for inverse transform
- `np.random.randn()` for copula sampling

##### 10. SampleVAE
**File**: `/home/user/pateda/pateda/sampling/vae.py` (246 lines total)
- `np.random.randn()` for latent noise
- TensorFlow operations may be involved

##### 11. SampleDAE
**File**: `/home/user/pateda/pateda/sampling/dae.py` (294 lines total)
- Neural network-based sampling
- Check for random operations

##### 12. SampleGibbs
**File**: `/home/user/pateda/pateda/sampling/gibbs.py` (338 lines total)
- `np.random.choice()` for Gibbs sampling
- Iterative sampling with conditioning

##### 13. SampleBackdrive
**File**: `/home/user/pateda/pateda/sampling/backdrive.py` (346 lines total)
- `np.random.normal()` and other random operations
- Backdrive model sampling

##### 14. SampleMAP
**File**: `/home/user/pateda/pateda/sampling/map_sampling.py` (568 lines total)
- **Lines 129-150**: `_sample_pls()` method - PLS sampling
- Uses `np.random.choice()` extensively

#### Other Sampling Methods

##### 15. SampleDenDiff
**File**: `/home/user/pateda/pateda/sampling/dendiff.py` (317 lines total)
- Diffusion-based sampling

##### 16. SampleGAN
**File**: `/home/user/pateda/pateda/sampling/gan.py`
- TensorFlow-based GAN sampling

##### 17. SampleRBM
**File**: `/home/user/pateda/pateda/sampling/rbm.py`
- TensorFlow-based RBM sampling

##### 18. SampleDBD
**File**: `/home/user/pateda/pateda/sampling/dbd.py`
- Dependency-Based Distribution sampling

##### 19. SampleCFDA
**File**: `/home/user/pateda/pateda/sampling/cfda.py` (356 lines total)
- Continuous FDA sampling

##### 20. SampleCUMDA
**File**: `/home/user/pateda/pateda/sampling/cumda.py` (238 lines total)
- Continuous UMDA sampling

##### 21. Additional Methods
**Files**:
- `/home/user/pateda/pateda/sampling/utils.py` - check for random functions

---

### Phase 4: Selection Methods (CRITICAL - 5 files)

#### 1. TournamentSelection
**File**: `/home/user/pateda/pateda/selection/tournament.py` (234 lines total)
- **Lines 50-234**: `select()` method
- Search for `np.random.choice()` calls for tournament

#### 2. ProportionalSelection
**File**: `/home/user/pateda/pateda/selection/proportional.py` (114 lines total)
- `np.random.choice()` for proportional selection

#### 3. RankingSelection
**File**: `/home/user/pateda/pateda/selection/ranking.py` (138 lines total)
- `np.random.choice()` for ranking-based selection

#### 4. BoltzmannSelection
**File**: `/home/user/pateda/pateda/selection/boltzmann.py` (136 lines total)
- `np.random.choice()` for Boltzmann selection

#### 5. SUSSelection
**File**: `/home/user/pateda/pateda/selection/sus.py` (123 lines total)
- **Lines 80-91**: `_stochastic_universal_sampling()` method
- **Line 87**: `np.random.uniform()` - REPLACE with `rng.uniform()`

#### NOT NEEDED (Deterministic)
- `/home/user/pateda/pateda/selection/truncation.py` - deterministic
- `/home/user/pateda/pateda/selection/non_dominated.py` - deterministic
- `/home/user/pateda/pateda/selection/pareto_front.py` - deterministic

---

### Phase 5: Learning Methods (LOW PRIORITY)

**Directory**: `/home/user/pateda/pateda/learning/`

Most learning methods are deterministic (compute statistics only). Review only if they use regularization:
- `/home/user/pateda/pateda/learning/vae.py` (686 lines) - may have random initialization
- `/home/user/pateda/pateda/learning/gan.py` - neural network training
- `/home/user/pateda/pateda/learning/rbm.py` - RBM training
- `/home/user/pateda/pateda/learning/dae.py` (374 lines) - DAE training

---

### Phase 6: Test Files (LOW PRIORITY - 29 files)

**Primary Location**: `/home/user/pateda/pateda/tests/`

Sample key test files:
- `test_discrete_eda.py` - Lines with `np.random.seed(42)`
- `test_gaussian_eda.py` - Lines with `np.random.seed(42)`
- `test_new_edas.py` - Lines with `np.random.seed()`
- All other test files in this directory

**Secondary Location**: `/home/user/pateda/tests/`
- Various test files

**Root Level**:
- `/home/user/pateda/test_*.py` - Various standalone tests

---

### Phase 7: Example Files (LOW PRIORITY - 40+ files)

**Primary Location**: `/home/user/pateda/examples/`
Examples using old-style `np.random.seed()` calls:
- `gaussian_eda_examples.py` - Line 73: `np.random.seed(42)`
- `default_eda_nk_landscape.py` - Check for seed usage
- All other example files

**Secondary Location**: `/home/user/pateda/pateda/examples/`
Examples using new EDA.run() API:
- `umda_onemax.py` - Update to use random_seed parameter
- `gaussian_eda_examples.py`
- `discrete_eda_examples.py`
- All other example files

---

## Quick File Count Summary

```
Total files to modify by priority:

PHASE 1 (Infrastructure): 2 files
  - pateda/core/eda.py
  - pateda/core/components.py

PHASE 2 (Seeding): 4 files
  - pateda/seeding/random_init.py
  - pateda/seeding/bias_init.py
  - pateda/seeding/seeding_unitation_constraint.py
  - pateda/seeding/seed_thispop.py (optional)

PHASE 3 (Sampling): 23 files
  - All files in pateda/sampling/

PHASE 4 (Selection): 5 files
  - pateda/selection/tournament.py
  - pateda/selection/proportional.py
  - pateda/selection/ranking.py
  - pateda/selection/boltzmann.py
  - pateda/selection/sus.py

PHASE 5 (Learning): 30 files
  - pateda/learning/*.py (review for random operations)

PHASE 6 (Tests): 29+ files
  - pateda/tests/*.py
  - tests/*.py
  - test_*.py (root level)

PHASE 7 (Examples): 40+ files
  - examples/*.py
  - pateda/examples/*.py

TOTAL: 100+ files
```

---

## Implementation Strategy by Directory

### 1. Core Changes (2 files, ~500 lines)
```
cd /home/user/pateda/pateda/core/
# Edit eda.py
# Edit components.py
```

### 2. Seeding Changes (3-4 files, ~270 lines)
```
cd /home/user/pateda/pateda/seeding/
# Edit random_init.py
# Edit bias_init.py
# Edit seeding_unitation_constraint.py
```

### 3. Sampling Changes (23 files, ~5700 lines)
```
cd /home/user/pateda/pateda/sampling/
# Edit all 23 *.py files
# Priority: fda.py, bayesian_network.py, mallows.py
```

### 4. Selection Changes (5 files, ~700 lines)
```
cd /home/user/pateda/pateda/selection/
# Edit tournament.py
# Edit proportional.py
# Edit ranking.py
# Edit boltzmann.py
# Edit sus.py
```

### 5. Learning Review (30 files, optional)
```
cd /home/user/pateda/pateda/learning/
# Review each file for random operations
# Low priority - most are deterministic
```

### 6. Test Updates (29+ files)
```
cd /home/user/pateda/pateda/tests/
# Update test files to use random_seed parameter
# Add reproducibility tests
```

### 7. Example Updates (40+ files)
```
cd /home/user/pateda/examples/
cd /home/user/pateda/pateda/examples/
# Update all examples to use random_seed parameter
```

---

## Verification Checklist

After each phase, verify:

```bash
# Run tests
python -m pytest /home/user/pateda/pateda/tests/ -v

# Check for remaining np.random.seed calls
grep -r "np\.random\.seed" /home/user/pateda/pateda/ --include="*.py"

# Check for remaining global np.random calls (in component files)
grep -r "np\.random\." /home/user/pateda/pateda/core/ --include="*.py"
grep -r "np\.random\." /home/user/pateda/pateda/sampling/ --include="*.py"

# Verify seed parameter works
python -c "from pateda import EDA; eda = EDA(..., random_seed=42)"
```

---

## Common Patterns by File Type

### Sampling Method Pattern
```python
# BEFORE (current)
def sample(self, n_vars, model, cardinality, aux_pop=None, aux_fitness=None, **params):
    new_pop = np.zeros((self.n_samples, n_vars))
    for i in range(self.n_samples):
        val = np.random.choice([0, 1], p=[0.3, 0.7])
        # ...
    return new_pop

# AFTER (with seed)
def sample(self, n_vars, model, cardinality, aux_pop=None, aux_fitness=None, rng=None, **params):
    if rng is None:
        rng = np.random.default_rng()
    new_pop = np.zeros((self.n_samples, n_vars))
    for i in range(self.n_samples):
        val = rng.choice([0, 1], p=[0.3, 0.7])
        # ...
    return new_pop
```

### Selection Method Pattern
```python
# BEFORE (current)
def select(self, population, fitness, **params):
    indices = np.random.choice(len(population), size=self.n_select, replace=False)
    return population[indices], fitness[indices]

# AFTER (with seed)
def select(self, population, fitness, rng=None, **params):
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(population), size=self.n_select, replace=False)
    return population[indices], fitness[indices]
```

---

