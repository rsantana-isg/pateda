# pateda — Python Algorithms for Estimation of Distribution Algorithms

[![PyPI version](https://img.shields.io/pypi/v/pateda)](https://pypi.org/project/pateda/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`pateda` is a comprehensive Python library for **Estimation of Distribution Algorithms (EDAs)** supporting discrete, integer, continuous, and permutation-based optimization.  It is a clean, PyPI-ready package extracted from the PATEDA research framework developed at the University of the Basque Country (EHU/UPV).

## Features

### Classical discrete EDAs
| Algorithm | Class / function | Problem type |
|-----------|-----------------|--------------|
| UMDA      | `LearnUMDA` / `SampleFDA` | Binary / integer |
| PBIL      | `LearnPBIL`               | Binary |
| FDA       | `LearnFDA` / `SampleFDA`  | Binary / integer |
| CFDA      | `LearnCFDA` / `SampleCFDA` | Binary |
| CUMDA     | `LearnCUMDA` / `SampleCUMDA` | Binary |
| BMDA      | `LearnBMDA`               | Binary |
| EBNA      | `LearnEBNA`               | Binary |
| BOA       | `LearnBOA`                | Binary |
| MN-FDA    | `LearnMNFDA` / `LearnMNFDAG` | Binary / integer |
| Tree-EDA  | `LearnTreeModel`          | Binary / integer |
| Mixture Trees | `LearnMixtureTrees`   | Binary |
| MIMIC     | `LearnMIMIC`              | Binary |
| BSC       | `LearnBSC`                | Binary |
| MOA       | `LearnMOA`                | Multi-objective |
| Affinity  | `LearnAffinityFactorization` | Binary |
| Markov    | `LearnMarkovChain`        | Sequences |

### Continuous EDAs
| Algorithm | Learning | Sampling |
|-----------|---------|---------|
| Gaussian Univariate (UMDA-G) | `learn_gaussian_univariate` | `sample_gaussian_univariate` |
| Gaussian Full (EMNA)         | `learn_gaussian_full`       | `sample_gaussian_full` |
| Mixture of Gaussians         | `learn_mixture_gaussian_*`  | `sample_mixture_gaussian_*` |
| GMRF-EDA                     | `learn_gmrf_eda*`           | `sample_gmrf_eda` |
| Vine Copula EDA *(optional)* | learned automatically       | `sample_vine_copula` |

### Permutation EDAs
| Algorithm | Learning | Sampling |
|-----------|---------|---------|
| Mallows EDA (Kendall) | `LearnMallowsKendall` | `SampleMallowsKendall` |
| EHM / NHM             | `LearnEHM` / `LearnNHM` | `SampleEHM` / `SampleNHM` |

### Supporting components

| Component | Subpackage | Notes |
|-----------|-----------|-------|
| Selection | `pateda.selection` | Truncation, tournament, Boltzmann, SUS, proportional, ranking, non-dominated |
| Mutation  | `pateda.mutation`  | Bit-flip, frequency-balance |
| Crossover | `pateda.crossover` | Block, two-point, transposition |
| Seeding   | `pateda.seeding`   | Random, biased, constrained |
| Replacement | `pateda.replacement` | Elitist, generational |
| Repairing | `pateda.repairing` | Unitation, trigonometric, bounds |
| Local Opt | `pateda.local_optimization` | Greedy search, simulated annealing, scipy wrapper |
| Stop cond.| `pateda.stop_conditions` | Max generations, optimum found |
| Statistics | `pateda.statistics` | Per-generation tracking, population stats |
| Knowledge  | `pateda.knowledge_extraction` | Dependency analysis, MI, model visualization |

### Benchmark functions

Discrete: OneMax, Deceptive-3, Trap, NK-landscape, SAT, UBQP, Ising, HP-Protein, Additive Decomposable, Contiguous Block  
Continuous: Sphere, Rosenbrock, Rastrigin, Ackley, GNBG instances  
Permutation: TSP, LOP, QAP

## Installation

```bash
pip install pateda
```

**Optional extras:**

```bash
pip install "pateda[copula]"   # vine-copula EDAs (pyvinecopulib)
pip install "pateda[dev]"      # development tools
pip install "pateda[all]"      # everything
```

For neural-network-based EDAs (VAE, GAN, Diffusion, etc.) install the companion package:

```bash
pip install pateda-nn
```

## Quick start

```python
import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.onemax import onemax

n_vars = 50
cardinality = np.full(n_vars, 2)

components = EDAComponents(
    seeding=RandomInit(),
    learning=LearnUMDA(),
    sampling=SampleFDA(),
    selection=TruncationSelection(),
    stop_condition=MaxGenerations(max_gen=100),
    selection_params={"selection_size": 150},
)

eda = EDA(
    pop_size=300,
    n_vars=n_vars,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    random_seed=42,
)
stats, cache = eda.run()
print(f"Best: {stats.best_fitness_overall}")
```

See `examples/` for more complete demos covering discrete, continuous, permutation, and multi-objective problems.

## Project structure

```
pateda/
├── src/
│   └── pateda/
│       ├── core/           # EDA engine (EDA, EDAComponents, Model)
│       ├── learning/       # Probabilistic model learning
│       ├── sampling/       # Model-based sampling
│       ├── selection/      # Selection operators
│       ├── seeding/        # Population initialisation
│       ├── mutation/       # Mutation operators
│       ├── crossover/      # Crossover operators
│       ├── replacement/    # Replacement strategies
│       ├── repairing/      # Constraint repair
│       ├── stop_conditions/# Termination criteria
│       ├── local_optimization/ # Local search wrappers
│       ├── functions/      # Benchmark objective functions
│       ├── inference/      # MAP inference
│       ├── statistics/     # Run-time statistics
│       └── knowledge_extraction/ # Model analysis tools
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## Citation

If you use `pateda` in academic work, please cite:

```bibtex
@misc{pateda,
  author  = {Roberto Santana},
  title   = {pateda: Python Algorithms for Estimation of Distribution Algorithms},
  year    = {2024},
  url     = {https://github.com/rsantana-isg/pateda},
}
```

## License

MIT — see [LICENSE](LICENSE).
