# Seed Parameter Integration - Quick Reference Guide

## Critical Files to Modify (Phase 1-2)

### Infrastructure (CRITICAL - Must do first)
```
pateda/core/eda.py
  ├── Add random_seed parameter to __init__()
  ├── Create self.rng = np.random.default_rng(random_seed)
  └── Pass rng to seeding, sampling, selection in run()

pateda/core/components.py
  ├── Add rng parameter to SeedingMethod.seed()
  ├── Add rng parameter to SamplingMethod.sample()
  ├── Add rng parameter to SelectionMethod.select()
  └── Add rng parameter to ReplacementMethod.replace()
```

### Seeding Methods (CRITICAL)
```
pateda/seeding/random_init.py (61 lines)
  └── Replace np.random.randint(), np.random.rand() with rng calls

pateda/seeding/bias_init.py (53 lines)
  └── Replace random operations with rng calls

pateda/seeding/seeding_unitation_constraint.py (71 lines)
  └── Replace random operations with rng calls
```

### Sampling Methods (CRITICAL - 23 files)
```
HIGHEST PRIORITY (used every generation):
  pateda/sampling/fda.py (241 lines)
  pateda/sampling/bayesian_network.py (133 lines)
  pateda/sampling/histogram.py - EHM/NHM
  pateda/sampling/markov.py
  pateda/sampling/mallows.py (404 lines)
  pateda/sampling/mixture_gaussian.py (274 lines)
  pateda/sampling/mixture_trees.py (273 lines)
  pateda/sampling/basic_gaussian.py (334 lines)

IMPORTANT (used in many EDAs):
  pateda/sampling/vine_copula.py (304 lines)
  pateda/sampling/vae.py (246 lines)
  pateda/sampling/dae.py (294 lines)
  pateda/sampling/gibbs.py (338 lines)
  pateda/sampling/backdrive.py (346 lines)
  pateda/sampling/map_sampling.py (568 lines)
  pateda/sampling/dendiff.py (317 lines)

OTHER:
  pateda/sampling/gan.py
  pateda/sampling/rbm.py
  pateda/sampling/dbd.py
  pateda/sampling/cfda.py (356 lines)
  pateda/sampling/cumda.py (238 lines)
  pateda/sampling/utils.py (check for random usage)
```

### Selection Methods (CRITICAL - 5 files)
```
pateda/selection/tournament.py (234 lines)
  └── Replace np.random.choice() with rng.choice()

pateda/selection/proportional.py (114 lines)
  └── Replace np.random.choice() with rng.choice()

pateda/selection/ranking.py (138 lines)
  └── Replace np.random.choice() with rng.choice()

pateda/selection/boltzmann.py (136 lines)
  └── Replace np.random.choice() with rng.choice()

pateda/selection/sus.py (123 lines)
  └── Replace np.random.uniform() with rng.uniform()
```

---

## File Counts by Module

| Module | File Count | Status |
|--------|-----------|--------|
| Seeding | 4 | All need seed params |
| Sampling | 23 | All CRITICAL |
| Selection | 8 | 5 need seed params |
| Learning | 30 | Mostly don't need |
| Replacement | ~5 | Some need params |
| Repairing | ~5 | Some need params |
| **Total** | **100+** | See full analysis |

---

## Implementation Checklist

### Phase 1: Infrastructure
- [ ] Add `random_seed` parameter to EDA.__init__()
- [ ] Create RNG instance in EDA class
- [ ] Update abstract base classes in components.py
- [ ] Update EDA.run() method signature

### Phase 2: Seeding Methods  
- [ ] RandomInit.seed() - add rng parameter
- [ ] BiasInit.seed() - add rng parameter
- [ ] SeedingUnitationConstraint.seed() - add rng parameter

### Phase 3: Sampling Methods (23 total)
- [ ] Priority 1: fda.py, bayesian_network.py, histogram.py, markov.py, mallows.py
- [ ] Priority 2: mixture_gaussian.py, mixture_trees.py, basic_gaussian.py
- [ ] Priority 3: vine_copula.py, vae.py, dae.py, gibbs.py, backdrive.py, map_sampling.py
- [ ] Remaining: dendiff.py, gan.py, rbm.py, dbd.py, cfda.py, cumda.py, utils.py

### Phase 4: Selection Methods (5 randomized)
- [ ] TournamentSelection.select() - add rng parameter
- [ ] ProportionalSelection.select() - add rng parameter
- [ ] RankingSelection.select() - add rng parameter
- [ ] BoltzmannSelection.select() - add rng parameter
- [ ] SUSSelection.select() - add rng parameter

### Phase 5: Update EDA.run() method
- [ ] Pass rng to seeding.seed()
- [ ] Pass rng to sampling.sample()
- [ ] Pass rng to selection.select()
- [ ] Pass rng to replacement.replace() (if applicable)
- [ ] Pass rng to repair operations (if applicable)

### Phase 6: Tests & Examples (Low priority)
- [ ] Update 29 test files to use random_seed parameter
- [ ] Update 40+ example scripts to use random_seed parameter
- [ ] Add reproducibility tests

---

## Common Replacement Patterns

### Pattern 1: np.random.randint()
```python
# BEFORE
value = np.random.randint(0, max_val, size=n)

# AFTER
value = rng.integers(0, max_val, size=n)
```

### Pattern 2: np.random.choice()
```python
# BEFORE
indices = np.random.choice(pop_size, size=n_select, replace=False)

# AFTER
indices = rng.choice(pop_size, size=n_select, replace=False)
```

### Pattern 3: np.random.uniform()
```python
# BEFORE
values = np.random.uniform(0, 1, size=n)

# AFTER
values = rng.uniform(0, 1, size=n)
```

### Pattern 4: np.random.rand()
```python
# BEFORE
values = np.random.rand(pop_size, n_vars)

# AFTER
values = rng.standard_normal(size=(pop_size, n_vars))
# OR for uniform
values = rng.random(size=(pop_size, n_vars))
```

### Pattern 5: np.random.randn()
```python
# BEFORE
values = np.random.randn(pop_size, n_vars)

# AFTER
values = rng.standard_normal(size=(pop_size, n_vars))
```

### Pattern 6: np.random.normal()
```python
# BEFORE
values = np.random.normal(mean, std, size=n)

# AFTER
values = rng.normal(mean, std, size=n)
```

### Pattern 7: np.random.shuffle()
```python
# BEFORE
np.random.shuffle(array)

# AFTER
rng.shuffle(array)
```

---

## Component Method Signature Changes

### Before (SeedingMethod)
```python
def seed(self, n_vars: int, pop_size: int, cardinality: np.ndarray, **params) -> np.ndarray:
```

### After (SeedingMethod)
```python
def seed(
    self,
    n_vars: int,
    pop_size: int,
    cardinality: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    **params: Any,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    # ... rest of method using rng instead of np.random
```

### Before (SamplingMethod)
```python
def sample(
    self,
    n_vars: int,
    model: Model,
    cardinality: np.ndarray,
    aux_pop: Optional[np.ndarray] = None,
    aux_fitness: Optional[np.ndarray] = None,
    **params: Any,
) -> np.ndarray:
```

### After (SamplingMethod)
```python
def sample(
    self,
    n_vars: int,
    model: Model,
    cardinality: np.ndarray,
    aux_pop: Optional[np.ndarray] = None,
    aux_fitness: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    **params: Any,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    # ... rest of method using rng instead of np.random
```

---

## Example Usage After Implementation

```python
# Fixed seed for reproducibility
eda = EDA(
    pop_size=300,
    n_vars=30,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    random_seed=42,  # NEW: Explicit seed parameter
)

# Random seed (non-reproducible)
eda = EDA(
    pop_size=300,
    n_vars=30,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    # random_seed=None (default)  NEW: Optional, uses system time if not provided
)

statistics, cache = eda.run()
```

---

## Test Pattern Updates

### Before
```python
def test_example():
    np.random.seed(42)
    population = np.random.randint(0, 2, (100, 10))
    # ... test
```

### After
```python
def test_example():
    rng = np.random.default_rng(42)
    population = rng.integers(0, 2, (100, 10))
    # ... test
```

Or using EDA:
```python
def test_eda_reproducibility():
    eda1 = EDA(..., random_seed=42)
    stats1, _ = eda1.run()
    
    eda2 = EDA(..., random_seed=42)
    stats2, _ = eda2.run()
    
    assert np.allclose(stats1.best_fitness, stats2.best_fitness)
```

---

## Files That Are DONE (No Changes Needed)

- pateda/selection/truncation.py (deterministic)
- pateda/selection/non_dominated.py (deterministic)
- pateda/selection/pareto_front.py (deterministic)
- Most learning methods (deterministic, only compute statistics)
- All functions modules (fitness functions)
- All visualization modules
- All statistics modules

---

## Priority Order for Implementation

### Week 1
1. Infrastructure (EDA, components abstract classes)
2. RandomInit.seed()
3. SampleFDA (most common sampling method)
4. TournamentSelection

### Week 2
5. All other sampling methods (priority by usage frequency)
6. All randomized selection methods
7. Other seeding methods

### Week 3
8. Update all test files
9. Update all example scripts
10. Documentation and testing

---

