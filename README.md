# PATEDA - Python Algorithms for Estimation of Distribution Algorithms

**A comprehensive Python framework for Estimation of Distribution Algorithms (EDAs)**

PATEDA is a modern Python implementation based on MATEDA-3.0, providing a rich collection of EDAs for discrete, continuous, and permutation optimization problems. The framework features advanced probabilistic models including neural networks, diffusion models, and Bayesian structures.

[![License](https://img.shields.io/badge/License-See%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [EDA Implementations](#eda-implementations)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

## Features

### Core Capabilities

- **60+ EDA implementations** covering classical, modern, and cutting-edge algorithms
- **Three optimization domains**: Discrete/Binary, Continuous, and Permutation problems
- **Advanced probabilistic models**: Bayesian networks, Markov networks, neural networks, diffusion models
- **Deep learning EDAs**: VAE, GAN, Denoising Diffusion, Backdrive, Neural EDAs
- **Modular architecture**: Composable components (seeding, learning, sampling, selection, replacement)
- **Multi-objective optimization**: Pareto-based selection and non-dominated sorting
- **Comprehensive benchmarks**: GNBG, binary functions, integer problems, permutation problems

### Modern Python Features

- **Type-safe**: Full type hints for better IDE support
- **Extensible**: Easy to add custom components
- **Well-tested**: Extensive test suite with 30+ test modules
- **Performance**: Optimized with NumPy, SciPy, and PyTorch
- **Visualization**: Built-in tools for analyzing EDA behavior

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### From Source

```bash
# Clone the repository
git clone https://github.com/rsantana-isg/pateda.git
cd pateda

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies include:
- NumPy, SciPy, scikit-learn (scientific computing)
- PyTorch (neural network-based EDAs)
- pgmpy (Bayesian network learning)
- networkx (graph algorithms)
- matplotlib (visualization)

## Quick Start

### Example 1: Binary Optimization with UMDA

```python
import numpy as np
from pateda.core import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning.umda import LearnUMDA
from pateda.sampling.histogram import SampleHistogram
from pateda.selection.truncation import TruncationSelection
from pateda.replacement.generational import GenerationalReplacement
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.functions.discrete.onemax import onemax

# Problem configuration
n_vars = 50
pop_size = 200
cardinality = np.full(n_vars, 2)  # Binary variables

# Configure EDA components
components = EDAComponents(
    seeding=RandomInit(),
    learning=LearnUMDA(n_vars=n_vars),
    sampling=SampleHistogram(n_samples=pop_size),
    selection=TruncationSelection(ratio=0.5),
    replacement=GenerationalReplacement(),
    stop_condition=MaxGenerations(max_gen=50)
)

# Create and run EDA
eda = EDA(
    pop_size=pop_size,
    n_vars=n_vars,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    minimize=False  # Maximization problem
)

statistics, cache = eda.run()
print(f"Best fitness: {statistics.best_fitness[-1]}")
print(f"Generations: {len(statistics.best_fitness)}")
```

### Example 2: Continuous Optimization with Gaussian EDA

```python
import numpy as np
from pateda.learning.basic_gaussian import learn_gaussian_univariate
from pateda.sampling.basic_gaussian import sample_gaussian_univariate

# Problem setup
n_vars = 10
pop_size = 100
bounds = np.array([[-5.0, 5.0]] * n_vars)

# Initialize population
population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, n_vars))
fitness = np.sum(population**2, axis=1)  # Sphere function

# Select best individuals
n_selected = pop_size // 2
selected_idx = np.argsort(fitness)[:n_selected]
selected_pop = population[selected_idx]

# Learn and sample
model = learn_gaussian_univariate(selected_pop, {})
new_population = sample_gaussian_univariate(model, pop_size, bounds, {})
```

### Example 3: Neural Network EDA

```python
from pateda.learning.nn_eda import learn_nn_eda
from pateda.sampling.nn_eda import sample_nn_eda_hybrid
import torch

# Configure neural EDA
params = {
    'hidden_dims': [128, 64],
    'latent_dim': 20,
    'n_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Learn and sample
model = learn_nn_eda(selected_pop, selected_fitness, params)
new_pop, new_fitness = sample_nn_eda_hybrid(
    model, pop_size, bounds, fitness_func, params
)
```

## Project Structure

```
pateda/
├── benchmarks/          # Benchmark scripts and analysis
│   ├── binary_functions_benchmark.py
│   ├── gnbg_benchmark.py
│   ├── integer_functions_benchmark.py
│   └── benchmark_dendiff_*.py
├── core/                # Core EDA framework
│   ├── eda.py          # Main EDA class
│   ├── components.py   # Component definitions
│   └── models.py       # Model structures
├── docs/                # Documentation and guides
│   ├── README_PATEDA.md
│   ├── IMPLEMENTATION_DESIGN.md
│   ├── EXAMPLES_GUIDE.md
│   └── [40+ documentation files]
├── enhanced_edas/       # Advanced EDA implementations
│   ├── diffusion_eda.py
│   ├── vae_models.py
│   ├── gaussian_models.py
│   └── GNBG_class.py
├── examples/            # Example scripts (50+ examples)
│   ├── umda_onemax.py
│   ├── dendiff_eda_example.py
│   ├── neural_eda_comparison.py
│   └── gaussian_eda_examples.py
├── experiments/         # Experimental comparisons
├── functions/           # Benchmark functions
│   ├── continuous/     # Continuous benchmarks
│   ├── discrete/       # Binary and integer problems
│   └── permutation/    # TSP, QAP, LOP
├── inference/           # MAP inference algorithms
├── knowledge_extraction/# Dependency analysis tools
├── learning/            # Learning algorithms (37 modules)
│   ├── umda.py         # Univariate models
│   ├── tree.py         # Tree-based models
│   ├── ebna.py, boa.py # Bayesian networks
│   ├── moa.py, markov.py # Markov networks
│   ├── dendiff.py      # Diffusion models
│   ├── nn_eda.py       # Neural networks
│   ├── vae.py, gan.py  # Deep generative models
│   └── [30+ more]
├── local_optimization/  # Local search methods
├── mutation/            # Mutation operators
├── permutation/         # Permutation utilities
├── replacement/         # Replacement strategies
├── repairing/           # Constraint handling
├── sampling/            # Sampling algorithms
├── seeding/             # Initialization methods
├── selection/           # Selection operators
├── statistics/          # Statistics tracking
├── stop_conditions/     # Stopping criteria
├── tests/               # Test suite (30+ test modules)
└── visualization/       # Plotting and analysis tools
```

## EDA Implementations

### Discrete/Binary EDAs

#### Classical Univariate
- **UMDA** (Univariate Marginal Distribution Algorithm) - `learning/umda.py`
- **PBIL** (Population-Based Incremental Learning) - `learning/pbil.py`
- **cGA** (Compact Genetic Algorithm) - `learning/histogram.py`

#### Bivariate Models
- **BMDA** (Bivariate Marginal Distribution Algorithm) - `learning/bmda.py`
- **MIMIC** (Mutual Information Maximizing Input Clustering) - `learning/mimic.py`

#### Factorized Distributions
- **FDA** (Factorized Distribution Algorithm) - `learning/fda.py`
- **CFDA** (Constrained FDA) - `learning/cfda.py`
- **cUMDA** (Constrained UMDA) - `learning/cumda.py`

#### Bayesian Network Models
- **EBNA** (Estimation of Bayesian Network Algorithm) - `learning/ebna.py`
- **BOA** (Bayesian Optimization Algorithm) - `learning/boa.py`
- **Tree-EDA** - `learning/tree.py`
- **Mixture of Trees** - `learning/mixture_trees.py`
- **BSC** (Bayesian Stochastic Classifier) - `learning/bsc.py`

#### Markov Network Models
- **MOA** (Markov Optimization Algorithm) - `learning/moa.py`
- **MN-FDA** (Markov Network FDA) - `learning/mnfda.py`
- **MN-FDAG** (Markov Network FDA Genetic) - `learning/mnfdag.py`
- **Markov EDA** - `learning/markov.py`
- **GMRF-EDA** (Gaussian Markov Random Field) - `learning/gmrf_eda.py`

#### Affinity-Based Models
- **Affinity Factorization** - `learning/affinity.py`

#### Neural Network Models
- **NN-EDA** (Neural Network EDA with fitness-weighted autoencoder) - `learning/nn_eda.py`
- **Discrete VAE** (Variational Autoencoder) - `learning/discrete_vae.py`
- **Discrete GAN** (Generative Adversarial Network) - `learning/discrete_gan.py`
- **Discrete DbD** (Denoising by Denoising) - `learning/discrete_dbd.py`
- **Discrete Backdrive** - `learning/discrete_backdrive.py`
- **RBM** (Restricted Boltzmann Machine) - `learning/rbm.py`
- **DAE** (Denoising Autoencoder) - `learning/dae.py`

### Continuous EDAs

#### Gaussian Models
- **Gaussian UMDA** (Univariate) - `learning/basic_gaussian.py`
- **Full Covariance Gaussian** - `learning/basic_gaussian.py`
- **Gaussian Network** - Various network-based implementations
- **Mixture of Gaussians** - `learning/mixture_gaussian.py`

#### Advanced Continuous Models
- **Dendiff** (Denoising Diffusion EDA) - `learning/dendiff.py`
- **Dendiff-ReLU** (Dendiff with ReLU activation) - `learning/dendiff_relu.py`
- **DbD-EDA** (Denoising by Denoising) - `learning/dbd.py`
- **Backdrive EDA** - `learning/backdrive.py`
- **VAE-EDA** (Variational Autoencoder) - `learning/vae.py`
- **GAN-EDA** (Generative Adversarial Network) - `learning/gan.py`
- **Vine Copula** - `learning/vine_copula.py`

### Permutation EDAs

- **Mallows Model** - `learning/mallows.py`
- **Generalized Mallows** with various distance metrics
- **Edge Histogram Model** - For TSP problems

### Key Features by Category

#### Deep Learning EDAs
All neural network-based EDAs support:
- GPU acceleration (PyTorch backend)
- Configurable architectures
- Fitness-weighted learning
- Elitism and hybrid sampling

#### Diffusion-Based EDAs
- **Dendiff**: Original denoising diffusion formulation
- **Dendiff-ReLU**: ReLU-based variant with improved stability
- **DbD**: Denoising by Denoising with direct sampling
- Configurable timesteps, network depth, and training epochs

#### Probabilistic Graphical Models
- Structure learning from data
- Various scoring metrics (BIC, AIC, K2)
- MAP inference support
- Gibbs sampling for Markov networks

## Documentation

### Main Documentation (in `docs/`)

- **[README_PATEDA.md](docs/README_PATEDA.md)** - Original project overview
- **[IMPLEMENTATION_DESIGN.md](docs/IMPLEMENTATION_DESIGN.md)** - Architecture and design
- **[EXAMPLES_GUIDE.md](docs/EXAMPLES_GUIDE.md)** - Comprehensive examples guide
- **[TESTING_QUICKSTART.md](docs/TESTING_QUICKSTART.md)** - Testing guide

### Algorithm-Specific Documentation

- **[DENDIFF_TESTING_README.md](docs/DENDIFF_TESTING_README.md)** - Diffusion EDA guide
- **[VAE_EDA_README.md](docs/VAE_EDA_README.md)** - VAE-EDA documentation
- **[GAN_EDA_IMPLEMENTATION.md](docs/GAN_EDA_IMPLEMENTATION.md)** - GAN-EDA guide
- **[GMRF_EDA_IMPLEMENTATION.md](docs/GMRF_EDA_IMPLEMENTATION.md)** - GMRF documentation
- **[PERMUTATION_EDA_IMPLEMENTATION_SUMMARY.md](docs/PERMUTATION_EDA_IMPLEMENTATION_SUMMARY.md)** - Permutation EDAs

### Benchmark Documentation

- **[README_BINARY.md](benchmarks/README_BINARY.md)** - Binary benchmarks
- **[README_INTEGER.md](benchmarks/README_INTEGER.md)** - Integer benchmarks
- **[README_GNBG.md](benchmarks/README_GNBG.md)** - GNBG continuous benchmarks
- **[PERMUTATION_BENCHMARK_README.md](docs/PERMUTATION_BENCHMARK_README.md)** - Permutation problems

### Additional Resources

- **[Mateda2.0-UserGuide.pdf](docs/Mateda2.0-UserGuide.pdf)** - Original MATLAB documentation
- **[MATLAB_PYTHON_MAPPING.md](docs/MATLAB_PYTHON_MAPPING.md)** - MATLAB to Python conversion guide
- **[PORTING_ROADMAP.md](docs/PORTING_ROADMAP.md)** - Development roadmap

## Examples

The `examples/` directory contains 50+ working examples organized by category:

### Binary/Discrete Examples
- `umda_onemax.py` - Simple UMDA on OneMax
- `bmda_onemax.py` - Bivariate model
- `tree_eda_deceptive.py` - Tree model on deceptive functions
- `ebna_nk_landscape.py` - Bayesian network on NK-landscapes
- `affinity_eda_deceptive.py` - Affinity-based factorization

### Continuous Examples
- `gaussian_umda_sphere.py` - Univariate Gaussian on Sphere
- `gaussian_full_rastrigin.py` - Full covariance on Rastrigin
- `mixture_gaussian_rosenbrock.py` - Mixture model on Rosenbrock
- `dendiff_eda_example.py` - Diffusion EDA
- `vae_eda_example.py` - VAE-based EDA
- `gan_eda_example.py` - GAN-based EDA

### Permutation Examples
- `mallows_tsp_example.py` - TSP with Mallows model
- `ehm_tsp_example.py` - Edge Histogram Model

### Advanced Examples
- `neural_eda_comparison.py` - Compare different neural EDAs
- `dendiff_relu_comparison.py` - Compare diffusion variants
- `comprehensive_eda_comparison.py` - Multi-algorithm benchmark

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_discrete_eda.py

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_gaussian_eda.py::TestGaussianUMDA
```

### Test Coverage

The test suite includes:
- **30+ test modules** covering all major components
- Unit tests for individual algorithms
- Integration tests for complete EDA runs
- Benchmark validation tests
- Performance regression tests

Key test modules:
- `test_discrete_eda.py` - Binary/discrete algorithms
- `test_gaussian_eda.py` - Continuous algorithms
- `test_dendiff_distributions.py` - Diffusion models
- `test_vae.py`, `test_gan.py` - Deep learning models
- `test_permutation_benchmark.py` - Permutation algorithms
- `test_knowledge_extraction_*.py` - Analysis tools

## Benchmarks

### Running Benchmarks

```bash
# Binary function benchmarks
python benchmarks/binary_functions_benchmark.py

# Integer function benchmarks
python benchmarks/integer_functions_benchmark.py

# GNBG continuous benchmarks
python benchmarks/gnbg_benchmark.py

# Dendiff parameter analysis
python benchmarks/benchmark_dendiff_parameter_analysis_gnbg.py

# Neural EDA comparison
python benchmarks/benchmark_nn_eda_vs_umda_gnbg.py
```

### Benchmark Functions

**Discrete/Binary:**
- OneMax, Trap, Deceptive functions
- NK-landscapes
- Ising model
- SAT problems
- HP Protein folding
- UBQP (Unconstrained Binary Quadratic Programming)

**Continuous:**
- GNBG suite (24 functions)
- Sphere, Rastrigin, Rosenbrock, Ackley
- Rotated and shifted variants

**Permutation:**
- TSP (Traveling Salesman Problem)
- QAP (Quadratic Assignment Problem)
- LOP (Linear Ordering Problem)

## Citation

If you use PATEDA in your research, please cite:

### PATEDA
```bibtex
@software{pateda2025,
  title={PATEDA: Python Algorithms for Estimation of Distribution Algorithms},
  author={Santana, Roberto},
  year={2025},
  url={https://github.com/rsantana-isg/pateda}
}
```

### Original MATEDA Papers

```bibtex
@article{santana2010mateda,
  title={Mateda-2.0: Estimation of distribution algorithms in MATLAB},
  author={Santana, Roberto and Bielza, Concha and Larra{\~n}aga, Pedro and
          Lozano, Jose A and Echegoyen, Carlos and Mendiburu, Alexander and
          Armananzas, Rub{\'e}n and Shakya, Siddartha},
  journal={Journal of Statistical Software},
  volume={35},
  number={7},
  pages={1--30},
  year={2010}
}

@article{irurozki2018algorithm,
  title={Algorithm 989: perm\_mateda: A Matlab Toolbox of Estimation of
         Distribution Algorithms for Permutation-based Combinatorial
         Optimization Problems},
  author={Irurozki, Ekhine and Ceberio, Josu and Santamaria, Jagoba and
          Santana, Roberto and Mendiburu, Alexander},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={44},
  number={4},
  pages={1--3},
  year={2018}
}
```

## License

This project maintains compatibility with MATEDA's licensing. See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas for contribution:
- New EDA algorithms
- Performance optimizations
- Additional benchmark functions
- Documentation improvements
- Bug fixes

Please ensure:
- Code follows existing style conventions
- Tests are included for new features
- Documentation is updated

## Authors and Acknowledgments

**Primary Author:**
- Roberto Santana (roberto.santana@ehu.es) - Original MATEDA and PATEDA development

**Acknowledgments:**
- Original MATEDA-3.0 development team
- BNT (Bayes Net Toolbox) by Kevin Murphy
- PMTK3 library contributors
- GNBG benchmark suite authors

## Project Status

**Current Version:** 0.2.0
**Status:** Active Development
**Last Updated:** December 2025

### Recent Updates
- Reorganized project structure (December 2025)
- Moved tests to `tests/` directory
- Moved benchmarks to `benchmarks/` directory
- Consolidated documentation in `docs/`
- Added 60+ EDA implementations
- Comprehensive test coverage (30+ test modules)
- 50+ working examples

### Roadmap
- [ ] Additional neural network architectures
- [ ] More efficient structure learning algorithms
- [ ] Parallel evaluation support
- [ ] Advanced visualization tools
- [ ] Web-based demo interface

## Contact

- **Issues:** Please use [GitHub Issues](https://github.com/rsantana-isg/pateda/issues)
- **Questions:** Roberto Santana (roberto.santana@ehu.es)
- **Repository:** https://github.com/rsantana-isg/pateda

---

**Keywords:** Estimation of Distribution Algorithms, Evolutionary Computation,
Probabilistic Models, Bayesian Networks, Neural Networks, Diffusion Models,
Black-box Optimization, Machine Learning
