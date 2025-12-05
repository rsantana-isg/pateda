# PATEDA Codebase Analysis: Seed Parameter Integration

## Executive Summary

The PATEDA (Parallel Transformative Estimation of Distribution Algorithm) codebase is a comprehensive Python framework for Estimation of Distribution Algorithms (EDAs). Currently, seed management relies on global `np.random.seed()` calls scattered throughout test files and examples. A comprehensive refactoring is needed to support reproducible, thread-safe random number generation through explicit seed parameters.

---

## 1. Test File Structure

### Location
- **Primary location**: `/home/user/pateda/pateda/tests/` (29 test files)
- **Secondary locations**: `/home/user/pateda/tests/` and root-level `/test_*.py` files

### Test File Categories

#### Core Component Tests (in `/home/user/pateda/pateda/tests/`)
1. **test_discrete_eda.py** - Tests for UMDA, BMDA, FDA, EBNA, BOA
2. **test_gaussian_eda.py** - Tests for Gaussian-based EDAs (univariate, full, mixture)
3. **test_new_edas.py** - Tests for newer EDA variants
4. **test_vae.py** - Tests for VAE-based sampling
5. **test_rbm.py** - Tests for RBM models
6. **test_vine_copula.py** - Tests for Vine Copula sampling
7. **test_dae.py** - Tests for DAE models
8. **test_gan.py** - Tests for GAN models
9. **test_backdrive.py** - Tests for backdrive methods
10. **test_crossover.py** - Tests for crossover operations

#### Benchmark Tests
1. **test_binary_benchmark.py** - Binary function benchmarks
2. **test_gnbg_benchmark.py** - GNBG (Generalized Numerical Benchmark Generator)
3. **test_permutation_benchmark.py** - Permutation problem benchmarks

#### Specialized Tests
1. **test_knowledge_extraction_discrete.py** - Knowledge extraction from EDAs
2. **test_knowledge_extraction_continuous.py** - Continuous optimization knowledge extraction
3. **test_new_gaussian_edas_minimal.py** - Minimal Gaussian EDA tests
4. **test_new_continuous_edas.py** - Continuous EDA variants
5. **test_generalized_mallows.py** - Generalized Mallows model
6. **test_gmallows_simple.py** - Simplified Mallows tests
7. **test_cumda_cfda.py** - CUMDA and CFDA tests

### Current Seed Usage Pattern in Tests
```python
def test_example():
    np.random.seed(42)  # Set global seed at test start
    # ... rest of test
```

**Problem**: Global seed is set in test methods but not propagated through component initialization.

---

## 2. Example Scripts Location

### Primary Examples (`/home/user/pateda/examples/`)
20+ example scripts demonstrating various EDA configurations:

1. **Binary Optimization Examples**:
   - `default_eda_trap.py` - UMDA on Trap function
   - `default_eda_nk_landscape.py` - UMDA on NK landscapes
   - `additive_decomposable_examples.py` - Additive decomposable problems

2. **Continuous Optimization Examples**:
   - `gaussian_umda_sphere.py` - Gaussian UMDA on Sphere
   - `gaussian_full_rastrigin.py` - Full Gaussian on Rastrigin
   - `gaussian_network_ackley.py` - Gaussian Network on Ackley
   - `mixture_gaussian_rosenbrock.py` - Mixture of Gaussians on Rosenbrock

3. **Permutation/TSP Examples**:
   - `ehm_tsp_example.py` - Edge Histogram Model
   - `mallows_tsp_example.py` - Mallows model

4. **Complex Problem Examples**:
   - `umda_sat.py` - UMDA for SAT problems
   - `markov_eda_example.py` - Markov-based EDAs
   - `tree_eda_deceptive.py` - Tree-based EDA
   - `tree_eda_ising.py` - Ising model optimization
   - `mixture_trees_eda_example.py` - Mixture of trees

5. **Comparison Examples**:
   - `affinity_comparison_trap.py` - Affinity-based EDA comparison
   - `affinity_eda_deceptive.py` - Affinity EDA on deceptive problems

### Secondary Examples (`/home/user/pateda/pateda/examples/`)
Refined examples with EDA.run() API:

1. **Discrete EDAs**:
   - `discrete_eda_examples.py` - Various discrete EDAs
   - `bmda_onemax.py` - BMDA on OneMax

2. **Continuous EDAs**:
   - `gaussian_eda_examples.py` - Gaussian EDA showcase

3. **Advanced Models**:
   - `vae_eda_example.py` - VAE-based EDA
   - `gan_eda_example.py` - GAN-based EDA
   - `dbd_eda_example.py` - Dependency-Based model
   - `dendiff_eda_example.py` - Diffusion-based model

4. **Multi-objective Optimization**:
   - `selection_comparison.py` - Selection operator comparison

5. **Unified Examples**:
   - `umda_onemax.py` - Complete UMDA on OneMax using EDA.run()
   - `backdrive_eda_examples.py` - Backdrive approach examples

### Current Seed Usage in Examples
```python
# Style 1: Global seed at function start
np.random.seed(42)
population = np.random.randint(0, 2, (pop_size, n_vars))

# Style 2: During initialization
objective = create_nk_objective_function(n_vars=n_vars, k=k, random_seed=42)

# Style 3: No explicit seed (relies on inherited state)
population = np.random.uniform(lower, upper, (pop_size, n_vars))
```

---

## 3. EDA Instantiation and Calling Patterns

### Main EDA Class
**File**: `/home/user/pateda/pateda/core/eda.py`

### EDA Initialization Pattern
```python
class EDA:
    def __init__(
        self,
        pop_size: int,
        n_vars: int,
        fitness_func: Union[Callable, str],
        cardinality: Union[np.ndarray, List],
        components: EDAComponents,
    ):
        # NO SEED PARAMETER - This is the key gap!
```

### EDA Execution Pattern
```python
eda = EDA(
    pop_size=pop_size,
    n_vars=n_vars,
    fitness_func=fitness_func,
    cardinality=cardinality,
    components=components,
)

statistics, cache = eda.run(
    cache_config=[...],
    verbose=True,
)
```

### EDA Components Configuration
**File**: `/home/user/pateda/pateda/core/components.py`

```python
@dataclass
class EDAComponents:
    seeding: SeedingMethod
    learning: LearningMethod
    sampling: SamplingMethod
    selection: SelectionMethod
    stop_condition: StopCondition
    replacement: Optional[ReplacementMethod]
    # ... params dicts for each component
```

**Components do NOT have seed parameters in their initialization or method signatures.**

---

## 4. Current Random Seed Usage

### Global Seed Setting (Current Pattern)
Test files set `np.random.seed(42)` but this:
- Only affects the global numpy random state
- Doesn't propagate through object initialization
- Creates dependency on test setup ordering
- Prevents concurrent test execution

### Files with Seed-Related Code
**Files using np.random.seed()** (100+ files found):
- All test files (29 test files)
- Most example files (20+ files)
- Some benchmark files

### Seed Parameter in Functions
Some functions accept `random_seed` or `seed` parameters:
- `create_nk_objective_function(n_vars=n_vars, k=k, random_seed=42)`
- Some learning methods have alpha parameters but no seed parameters

---

## 5. Main EDA Classes and Initialization

### Core Classes (No Seed Support Currently)

#### Seeding Methods (`/home/user/pateda/pateda/seeding/` - 4 methods)
1. **RandomInit** (61 lines)
   - Initializes population randomly
   - Uses: `np.random.randint()`, `np.random.rand()`
   - **Needs seed parameter**

2. **BiasInit** (53 lines)
   - Biased initialization toward specific values
   - Uses numpy random operations
   - **Needs seed parameter**

3. **SeedThisPop** (72 lines)
   - Uses provided population as seed
   - **Low priority for seed parameter**

4. **SeedingUnitationConstraint** (71 lines)
   - Initialization with unitation constraints
   - **Needs seed parameter**

#### Learning Methods (`/home/user/pateda/pateda/learning/` - 30 methods)
Key methods (deterministic, low seed priority):
- LearnUMDA, LearnBMDA, LearnFDA
- LearnEBNA, LearnBOA
- LearnTree, LearnMarkov, LearnMallows
- LearnHistogram
- LearnRBM, LearnVAE, LearnGAN, LearnDAE
- LearnVineCopula, LearnBackdrive
- LearnDenDiff, LearnAffinity
- Mixture variants of above

**Note**: Most learning methods are deterministic (just compute statistics). Some may need seeds for regularization or initialization.

#### Sampling Methods (`/home/user/pateda/pateda/sampling/` - 23 methods)
**These HEAVILY use random number generation and MUST have seed support**:

1. **SampleFDA** (241 lines)
   - Uses: `np.random.rand()` for sampling

2. **SampleBayesianNetwork** (133 lines)
   - Uses: `np.random.choice()` for conditional sampling

3. **SampleHistogram** (EHM, NHM)
   - Uses: `np.random.randint()`, `np.random.rand()`

4. **SampleMarkov** (200+ lines)
   - Uses: `np.random.choice()` extensively

5. **SampleMallows** (404 lines)
   - Uses: `np.random.rand()` for inversions

6. **SampleMixtureGaussian** (274 lines)
   - Uses: `np.random.choice()`, `np.random.randn()`

7. **SampleMixtureTrees** (273 lines)
   - Uses: `np.random.choice()`, `np.random.randint()`

8. **SampleBasicGaussian** (334 lines)
   - Uses: `np.random.randn()` for continuous sampling

9. **SampleVineCopula** (304 lines)
   - Uses: `np.random.uniform()`, `np.random.randn()`

10. **SampleVAE** (246 lines)
    - Uses: `np.random.randn()` for noise

11. **SampleGAN** (TensorFlow-based)
    - Uses TensorFlow's random operations

12. **SampleRBM** (TensorFlow-based)
    - Uses TensorFlow's random operations

13. **SampleDAE** (294 lines)
    - Uses neural network sampling

14. **SampleGibbs** (338 lines)
    - Uses: `np.random.choice()` for Gibbs sampling

15. **SampleBackdrive** (346 lines)
    - Uses: `np.random.normal()`, `np.random.choice()`

16. **SampleDenDiff** (317 lines)
    - Diffusion-based sampling

17. **SampleDBD**, **SampleCFDA**, **SampleCUMDA**
    - Various specialized sampling methods

18. **SampleInsertMAP**, **SampleTemplateMAP** (map_sampling.py, 568 lines)
    - Uses PLS (Probabilistic Logic Sampling)
    - Uses: `np.random.choice()` extensively

#### Selection Methods (`/home/user/pateda/pateda/selection/` - 8 methods)
**Use random sampling in some variants**:

1. **TruncationSelection** (157 lines) - Deterministic
2. **TournamentSelection** (234 lines)
   - Uses: `np.random.choice()` for tournament

3. **ProportionalSelection** (114 lines)
   - Uses: `np.random.choice()`

4. **RankingSelection** (138 lines)
   - Uses: `np.random.choice()`

5. **BoltzmannSelection** (136 lines)
   - Uses: `np.random.choice()`

6. **SUSSelection** (123 lines) - Stochastic Universal Sampling
   - Uses: `np.random.uniform()`

7. **NonDominatedSelection** (81 lines) - Deterministic
8. **ParetoFrontSelection** (97 lines) - Deterministic

#### Other Components Using Randomness
- **Replacement methods**: Some use random selection
- **Repairing methods**: Some use random modifications
- **Local optimization**: May use randomness
- **Crossover operations**: Use random mixing

---

## 6. Where Seeds Need to Be Added

### Tier 1: CRITICAL (Must implement immediately)
These components use randomness and are called in every generation:

1. **RandomInit.seed()** - Initial population creation
2. **All SamplingMethod.sample()** implementations (23 methods)
3. **RandomizedSelection.select()** (Tournament, Proportional, Ranking, Boltzmann, SUS)

### Tier 2: IMPORTANT (Should implement soon)
Components that sometimes use randomness:

1. **BiasInit.seed()** - If using random bias
2. **SeedingUnitationConstraint.seed()** - If using random repairs
3. **CrossoverOperations** - Any random mixing
4. **LocalOptimization** - If using random perturbations
5. **ReplacementMethods** - If using random selection

### Tier 3: NICE-TO-HAVE (Can implement later)
Components where randomness is optional or rarely used:

1. **LearningMethods** - Mostly deterministic, but some might benefit (e.g., regularization)
2. **VectorizedLearning** - If sampling from regularization distributions
3. **Repairing methods** - If using random repairs

### Tier 4: NOT NEEDED
- Deterministic selection methods (Truncation, NonDominated, ParetoFront)
- Learning methods that only compute statistics

---

## 7. Integration Points Needed

### 1. EDA Class (`/home/user/pateda/pateda/core/eda.py`)
```python
class EDA:
    def __init__(
        self,
        pop_size: int,
        n_vars: int,
        fitness_func: Union[Callable, str],
        cardinality: Union[np.ndarray, List],
        components: EDAComponents,
        random_seed: Optional[int] = None,  # ADD THIS
    ):
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()
```

### 2. Component Base Classes (`/home/user/pateda/pateda/core/components.py`)
```python
class SeedingMethod(ABC):
    @abstractmethod
    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        rng: Optional[np.random.Generator] = None,  # ADD THIS
        **params: Any,
    ) -> np.ndarray:
        pass

class SamplingMethod(ABC):
    @abstractmethod
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
        pass

# Similar for SelectionMethod, ReplacementMethod, etc.
```

### 3. All 23 Sampling Implementations
Replace `np.random.*()` calls with `rng.*()` where rng is passed as parameter

### 4. All 8 Selection Implementations (where they use randomness)
Replace `np.random.choice()`, `np.random.uniform()` calls with `rng` calls

### 5. EDA.run() Method
Pass rng to all components:
```python
def run(self, ...):
    # In seeding phase:
    self.population = self.components.seeding.seed(
        ...,
        rng=self.rng,  # ADD THIS
    )
    
    # In sampling phase:
    new_pop = self.components.sampling.sample(
        ...,
        rng=self.rng,  # ADD THIS
    )
    
    # In selection phase:
    selected = self.components.selection.select(
        ...,
        rng=self.rng,  # ADD THIS
    )
```

---

## 8. Files Summary Table

### Test Files (29 total)
```
/home/user/pateda/pateda/tests/
├── test_discrete_eda.py (150+ lines)
├── test_gaussian_eda.py (200+ lines)
├── test_new_edas.py
├── test_vae.py
├── test_rbm.py
├── test_vine_copula.py
├── test_dae.py
├── test_gan.py
├── test_backdrive.py
├── test_crossover.py
├── test_binary_benchmark.py
├── test_gnbg_benchmark.py
├── test_permutation_benchmark.py
└── ... 16 more test files
```

### Core Framework Files (need seed integration)
```
/home/user/pateda/pateda/core/
├── eda.py (429 lines) - CRITICAL: Add seed parameter to __init__ and run()
└── components.py (200+ lines) - CRITICAL: Add rng parameters to abstract methods

/home/user/pateda/pateda/seeding/
├── random_init.py (61 lines) - CRITICAL
├── bias_init.py (53 lines) - IMPORTANT
├── seed_thispop.py (72 lines) - LOW
└── seeding_unitation_constraint.py (71 lines) - IMPORTANT

/home/user/pateda/pateda/sampling/ (23 methods)
├── fda.py (241 lines) - CRITICAL
├── bayesian_network.py (133 lines) - CRITICAL
├── histogram.py - CRITICAL
├── markov.py - CRITICAL
├── mallows.py (404 lines) - CRITICAL
├── mixture_gaussian.py (274 lines) - CRITICAL
├── mixture_trees.py (273 lines) - CRITICAL
├── basic_gaussian.py (334 lines) - CRITICAL
├── vine_copula.py (304 lines) - CRITICAL
├── vae.py (246 lines) - CRITICAL
├── gan.py - CRITICAL
├── rbm.py - CRITICAL
├── dae.py (294 lines) - CRITICAL
├── gibbs.py (338 lines) - CRITICAL
├── backdrive.py (346 lines) - CRITICAL
├── dendiff.py (317 lines) - CRITICAL
├── dbd.py - CRITICAL
├── cfda.py (356 lines) - CRITICAL
├── cumda.py (238 lines) - CRITICAL
├── map_sampling.py (568 lines) - CRITICAL
└── utils.py - Review for random usage

/home/user/pateda/pateda/selection/ (8 methods)
├── tournament.py (234 lines) - CRITICAL
├── proportional.py (114 lines) - CRITICAL
├── ranking.py (138 lines) - CRITICAL
├── boltzmann.py (136 lines) - CRITICAL
├── sus.py (123 lines) - CRITICAL
├── truncation.py (157 lines) - NOT NEEDED
├── non_dominated.py (81 lines) - NOT NEEDED
└── pareto_front.py (97 lines) - NOT NEEDED
```

### Example Files (20+, all in `/home/user/pateda/examples/`)
- All need seed parameter passed to EDA constructor

---

## 9. Key Statistics

| Category | Count | Files |
|----------|-------|-------|
| **Test Files** | 29 | `/pateda/tests/` + root level |
| **Example Scripts** | 40+ | `/examples/` + `/pateda/examples/` |
| **Sampling Methods** | 23 | Need seed parameter |
| **Learning Methods** | 30 | Mostly don't need seeds |
| **Selection Methods** | 8 | 5 need seed parameters |
| **Seeding Methods** | 4 | All need seed parameters |
| **Total Modules** | 100+ | Python files in codebase |

---

## 10. Recommendations for Implementation

### Phase 1: Infrastructure (Foundation)
1. Add `random_seed` parameter to EDA.__init__()
2. Create RNG instance (np.random.default_rng)
3. Update abstract base classes in components.py
4. Update EDA.run() to pass rng to all components

### Phase 2: Core Components (Highest Impact)
1. Update all 23 SamplingMethod implementations
2. Update RandomInit seeding method
3. Update 5 randomized SelectionMethod implementations
4. Update EDA.run() method calls

### Phase 3: Extended Components (Medium Impact)
1. Update BiasInit seeding method
2. Update SeedingUnitationConstraint
3. Update any CrossoverMethods
4. Update LocalOptimization if it uses randomness
5. Update ReplacementMethods if they use randomness

### Phase 4: Testing and Examples
1. Update all test files to pass random_seed
2. Update all example scripts
3. Add tests for reproducibility with seed parameter
4. Add documentation on seed usage

### Phase 5: Nice-to-Have Enhancements
1. Support seed sequences for deterministic but varied runs
2. Add seed management utilities
3. Document reproducibility best practices

---

## 11. Code Pattern Examples

### Before (Current)
```python
def seed(self, n_vars, pop_size, cardinality, **params):
    new_pop = np.zeros((pop_size, n_vars), dtype=int)
    for i in range(n_vars):
        new_pop[:, i] = np.random.randint(0, cardinality[i], size=pop_size)  # GLOBAL STATE
    return new_pop
```

### After (With Seed Support)
```python
def seed(self, n_vars, pop_size, cardinality, rng=None, **params):
    if rng is None:
        rng = np.random.default_rng()
    new_pop = np.zeros((pop_size, n_vars), dtype=int)
    for i in range(n_vars):
        new_pop[:, i] = rng.integers(0, cardinality[i], size=pop_size)  # RNG EXPLICIT
    return new_pop
```

### In EDA Class
```python
def __init__(self, ..., random_seed=None):
    if random_seed is not None:
        self.rng = np.random.default_rng(random_seed)
    else:
        self.rng = np.random.default_rng()

def run(self, ...):
    # Generation 0: Seeding
    self.population = self.components.seeding.seed(
        self.n_vars,
        self.pop_size,
        self.cardinality,
        rng=self.rng,  # Pass RNG!
        **self.components.seeding_params,
    )
    
    # Generation > 0: Sampling
    new_pop = self.components.sampling.sample(
        self.n_vars,
        self.model,
        self.cardinality,
        self.population,
        self.fitness,
        rng=self.rng,  # Pass RNG!
        **self.components.sampling_params,
    )
```

---

## Conclusion

The PATEDA codebase needs comprehensive seed parameter integration across:
- **1 main EDA class** 
- **23 sampling methods** (CRITICAL)
- **5 randomized selection methods** (CRITICAL)
- **4 seeding methods** (CRITICAL)
- **40+ example scripts**
- **29 test files**

The refactoring should follow a phased approach, starting with infrastructure changes and high-impact components (sampling and selection), then extending to test and example files.

