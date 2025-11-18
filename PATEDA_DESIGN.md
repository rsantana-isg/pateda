# PATEDA Design Document

## Overview
PATEDA (Python Algorithms for Estimation of Distribution Algorithms) is a port of MATEDA-3.0 from MATLAB to Python. This document outlines the architecture, design decisions, and implementation plan.

## Design Goals

1. **Maintain MATEDA's functionality**: Port as many EDAs and features as possible
2. **Pythonic design**: Use Python best practices, type hints, and modern libraries
3. **Modular architecture**: Keep MATEDA's component-based structure
4. **Extensible**: Easy to add new EDAs, learning methods, and sampling methods
5. **Well-tested**: Include comprehensive tests and validation
6. **Performance**: Leverage NumPy/SciPy for efficient numerical operations

## Core Architecture

### 1. Main EDA Execution Framework

Instead of MATLAB's `eval()`-based approach, we'll use a class-based architecture with dependency injection:

```python
class EDA:
    """Main EDA executor - equivalent to RunEDA.m"""

    def __init__(self,
                 pop_size: int,
                 n_vars: int,
                 fitness_func: Callable,
                 cardinality: Union[np.ndarray, List],
                 components: Optional[EDAComponents] = None):
        """
        Args:
            pop_size: Population size
            n_vars: Number of variables
            fitness_func: Fitness evaluation function
            cardinality: Variable cardinalities (discrete) or ranges (continuous)
            components: EDA components (seeding, learning, sampling, etc.)
        """
        self.pop_size = pop_size
        self.n_vars = n_vars
        self.fitness_func = fitness_func
        self.cardinality = cardinality
        self.components = components or EDAComponents()

    def run(self, cache_config: CacheConfig) -> Tuple[Statistics, Cache]:
        """Execute the EDA"""
        # Main loop implementation
        pass
```

### 2. Component Architecture

Each EDA component (seeding, learning, sampling, etc.) follows a common interface:

```python
from abc import ABC, abstractmethod

class SeedingMethod(ABC):
    @abstractmethod
    def seed(self, n_vars: int, pop_size: int,
             cardinality: np.ndarray, **params) -> np.ndarray:
        """Generate initial population"""
        pass

class LearningMethod(ABC):
    @abstractmethod
    def learn(self, generation: int, n_vars: int,
              cardinality: np.ndarray, population: np.ndarray,
              fitness: np.ndarray, **params) -> Model:
        """Learn probabilistic model from population"""
        pass

class SamplingMethod(ABC):
    @abstractmethod
    def sample(self, n_vars: int, model: Model,
               cardinality: np.ndarray, aux_pop: np.ndarray,
               aux_fitness: np.ndarray, **params) -> np.ndarray:
        """Sample new population from model"""
        pass

class SelectionMethod(ABC):
    @abstractmethod
    def select(self, population: np.ndarray, fitness: np.ndarray,
               n_select: int, **params) -> Tuple[np.ndarray, np.ndarray]:
        """Select individuals for learning"""
        pass

class ReplacementMethod(ABC):
    @abstractmethod
    def replace(self, old_pop: np.ndarray, old_fitness: np.ndarray,
                new_pop: np.ndarray, new_fitness: np.ndarray,
                **params) -> Tuple[np.ndarray, np.ndarray]:
        """Combine old and new populations"""
        pass

class StopCondition(ABC):
    @abstractmethod
    def should_stop(self, generation: int, population: np.ndarray,
                    fitness: np.ndarray, **params) -> bool:
        """Check if evolution should stop"""
        pass
```

### 3. Model Representation

Models are represented as dataclasses for clarity and type safety:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Model:
    """Base class for probabilistic models"""
    structure: Any  # Graph structure (varies by model type)
    parameters: Any  # Model parameters (tables, means, covariances, etc.)
    metadata: Dict[str, Any] = None

@dataclass
class FactorizedModel(Model):
    """Factorized Distribution Algorithm model"""
    structure: np.ndarray  # Cliques matrix
    parameters: List[np.ndarray]  # Probability tables per clique

@dataclass
class TreeModel(Model):
    """Tree-based model"""
    structure: np.ndarray  # Parent-child relationships
    parameters: List[np.ndarray]  # Conditional probability tables

@dataclass
class BayesianNetworkModel(Model):
    """Bayesian Network model (using pgmpy)"""
    structure: Any  # pgmpy BayesianNetwork object
    parameters: Any  # CPD tables

@dataclass
class GaussianModel(Model):
    """Gaussian model for continuous variables"""
    structure: Optional[np.ndarray]  # Dependency structure (if any)
    parameters: Dict[str, np.ndarray]  # means, covariances
```

### 4. Component Registry

Use a registry pattern to avoid eval() and enable dynamic component selection:

```python
class ComponentRegistry:
    """Registry for EDA components"""

    def __init__(self):
        self._seeding = {}
        self._learning = {}
        self._sampling = {}
        self._selection = {}
        self._replacement = {}
        self._stop_conditions = {}

    def register_seeding(self, name: str, method: SeedingMethod):
        self._seeding[name] = method

    def get_seeding(self, name: str) -> SeedingMethod:
        return self._seeding[name]

    # Similar methods for other component types

# Global registry
registry = ComponentRegistry()
```

## Python Libraries

### Core Dependencies

1. **NumPy**: Numerical operations, arrays
2. **SciPy**: Statistical distributions, optimization
3. **pgmpy**: Bayesian network learning and inference (replaces BNT)
4. **NetworkX**: Graph operations for structure learning
5. **Matplotlib/Seaborn**: Visualization
6. **pandas**: Data analysis and statistics
7. **scikit-learn**: Mixture models, clustering (for Gaussian mixtures)
8. **typing_extensions**: Advanced type hints

### Optional Dependencies

1. **pomegranate**: Alternative for probabilistic models
2. **plotly**: Interactive visualizations
3. **numba**: JIT compilation for performance
4. **pytest**: Testing framework
5. **sphinx**: Documentation generation

## Folder Structure

Maintain MATEDA's organization with Python modules:

```
pateda/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── eda.py              # Main EDA class
│   ├── components.py       # Component interfaces
│   ├── models.py           # Model classes
│   └── registry.py         # Component registry
├── seeding/
│   ├── __init__.py
│   ├── random_init.py
│   └── bias_init.py
├── learning/
│   ├── __init__.py
│   ├── fda.py              # LearnFDA
│   ├── tree_model.py       # LearnTreeModel
│   ├── bayesian_network.py # LearnBN
│   ├── gaussian_univ.py
│   └── utils/
│       ├── marginal_prob.py
│       └── mutual_info.py
├── sampling/
│   ├── __init__.py
│   ├── fda.py              # SampleFDA
│   ├── bayesian_network.py # SampleBN
│   └── gaussian.py
├── selection/
│   ├── __init__.py
│   ├── truncation.py
│   ├── proportional.py
│   └── tournament.py
├── replacement/
│   ├── __init__.py
│   ├── elitism.py
│   └── best_elitism.py
├── stop_conditions/
│   ├── __init__.py
│   ├── max_generations.py
│   └── fitness_threshold.py
├── statistics/
│   ├── __init__.py
│   └── simple_statistics.py
├── visualization/
│   ├── __init__.py
│   ├── parallel_coords.py
│   └── convergence.py
├── functions/
│   ├── __init__.py
│   ├── discrete/
│   │   ├── onemax.py
│   │   ├── deceptive.py
│   │   └── nk_landscapes.py
│   └── continuous/
│       └── sphere.py
├── examples/
│   ├── umda_onemax.py
│   ├── tree_eda_deceptive.py
│   └── gaussian_continuous.py
├── tests/
│   ├── __init__.py
│   ├── test_seeding.py
│   ├── test_learning.py
│   └── test_sampling.py
└── docs/
    └── user_guide.md
```

## Initial Implementation Plan

### Phase 1: Core Framework (Week 1-2)

1. Implement base classes and interfaces
2. Component registry system
3. Main EDA execution loop
4. Basic statistics and caching

### Phase 2: Simple EDAs (Week 3-4)

Start with 3 fundamental EDAs:

1. **UMDA (Univariate Marginal Distribution Algorithm)**
   - Seeding: RandomInit
   - Learning: LearnFDA with independent variables
   - Sampling: SampleFDA
   - Selection: Truncation
   - Stop: Max generations

2. **Tree-EDA**
   - Learning: LearnTreeModel (mutual information-based tree)
   - Sampling: SampleFDA (with tree structure)
   - Rest same as UMDA

3. **Gaussian-UMDA (for continuous optimization)**
   - Learning: LearnGaussianUnivModel
   - Sampling: SampleGaussianUnivModel
   - Rest same as UMDA

### Phase 3: Test Functions (Week 4)

Implement basic test functions:
- OneMax (discrete)
- Deceptive-3 (discrete)
- Sphere function (continuous)

### Phase 4: Testing & Validation (Week 5)

1. Unit tests for all components
2. Integration tests for complete EDAs
3. Validation against MATEDA results
4. Performance benchmarks

### Phase 5: Extended EDAs (Week 6+)

Add more sophisticated EDAs:
- BOA-like algorithms (Bayesian networks)
- MOA (Markov networks)
- Gaussian networks
- Multi-objective EDAs

## API Design

### Simple Usage Example

```python
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import BestElitism
from pateda.stop_conditions import MaxGenerations
from pateda.functions import onemax
import numpy as np

# Define problem
n_vars = 30
pop_size = 300
cardinality = np.full(n_vars, 2)  # Binary variables

# Configure EDA components
components = EDAComponents(
    seeding=RandomInit(),
    learning=LearnFDA(cliques=None),  # Will create univariate
    sampling=SampleFDA(n_samples=pop_size),
    selection=TruncationSelection(ratio=0.5),
    replacement=BestElitism(),
    stop_condition=MaxGenerations(max_gen=10)
)

# Create and run EDA
eda = EDA(
    pop_size=pop_size,
    n_vars=n_vars,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components
)

statistics, cache = eda.run(cache_models=True)

# Access results
print(f"Best fitness: {statistics.best_fitness[-1]}")
print(f"Best solution: {statistics.best_individual}")
```

### Advanced Configuration

```python
from pateda import EDA
from pateda.learning import LearnTreeModel

# Tree-EDA with custom parameters
eda = EDA.from_config({
    'pop_size': 500,
    'n_vars': 60,
    'fitness_func': 'deceptive3',
    'cardinality': [2] * 60,
    'components': {
        'learning': {'method': 'tree_model', 'params': {}},
        'sampling': {'method': 'fda', 'params': {'n_samples': 500}},
        'selection': {'method': 'proportional', 'params': {}},
        'replacement': {'method': 'elitism', 'params': {'n_elite': 10}},
        'stop_condition': {'method': 'max_gen', 'params': {'max_gen': 50}}
    }
})

result = eda.run()
```

## Design Decisions

### 1. Classes vs Functions

**Decision**: Use classes for components, functions for utilities
- **Rationale**: Classes allow state and configuration, better match Python idioms
- **MATLAB equivalent**: MATLAB functions → Python classes with __call__

### 2. Model Representation

**Decision**: Use dataclasses for models
- **Rationale**: Type safety, clarity, easy serialization
- **MATLAB equivalent**: Cell arrays → dataclasses

### 3. Dynamic Dispatch

**Decision**: Use registry pattern instead of eval()
- **Rationale**: Type-safe, no security issues, better IDE support
- **MATLAB equivalent**: eval('method(...)') → registry.get('method')(...)

### 4. Bayesian Networks

**Decision**: Use pgmpy as primary BN library
- **Rationale**: Active development, comprehensive, pythonic API
- **Alternative**: pomegranate (consider for performance)

### 5. Type Hints

**Decision**: Full type hints throughout
- **Rationale**: Better IDE support, catch bugs early, documentation

### 6. Backward Compatibility

**Decision**: Keep MATLAB file structure alongside Python
- **Rationale**: Easy comparison, incremental migration, reference

## Migration Strategy

1. **Parallel development**: Add .py files alongside .m files
2. **Incremental testing**: Compare Python vs MATLAB outputs
3. **Documentation**: Document differences and improvements
4. **Examples**: Port example scripts to demonstrate equivalence

## Next Steps

1. Set up Python project structure
2. Implement core framework (EDA class, component interfaces)
3. Implement first EDA (UMDA)
4. Create test suite
5. Validate against MATEDA
6. Iterate and expand

---

**Author**: Claude (AI Assistant)
**Date**: 2025-11-18
**Version**: 1.0
**Based on**: MATEDA-3.0 by Roberto Santana
