# PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A comprehensive Python library for Estimation of Distribution Algorithms (EDAs), supporting discrete, continuous, and permutation-based optimization problems.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Estimation of Distribution Algorithms (EDAs) are evolutionary algorithms that learn probabilistic models of promising solutions and sample new candidates from these models. PATEDA provides a modern, extensible Python implementation with support for:

- **Multiple problem domains**: Binary, discrete, continuous, and permutation optimization
- **Advanced probabilistic models**: Bayesian networks, Markov networks, Gaussian models, neural networks, VAEs, GANs, and more
- **Flexible architecture**: Modular components for seeding, learning, sampling, selection, and replacement
- **Rich algorithm suite**: UMDA, Tree-EDA, EBNA, MOA, Gaussian EDAs, Mallows models, and neural-based approaches

## Features

### Core Capabilities
- **Discrete EDAs**: UMDA, BMDA, Tree-EDA, EBNA, MOA, Affinity Propagation-based EDAs
- **Continuous EDAs**: Gaussian UMDA, Gaussian networks, mixture models, GMRF, vine copulas
- **Neural Network-based EDAs**: VAE-EDA, GAN-EDA, Denoising Diffusion EDAs, Autoencoder-based approaches
- **Permutation EDAs**: Mallows models (Kendall, Cayley), Generalized Mallows, Markov chain models
- **Multi-objective optimization**: Support for Pareto-based optimization
- **Knowledge extraction**: Analysis and visualization of learned models

### Advanced Features
- **Hybrid approaches**: Crossover operators, local optimization, knowledge seeding
- **Adaptive mechanisms**: MAP sampling, backdrive, elitist replacement
- **Comprehensive benchmarks**: Standard test functions across all domains
- **Visualization tools**: Model structure analysis, convergence plots, distribution visualization

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/rsantana-isg/pateda.git
cd pateda

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy, SciPy, pandas
- pgmpy, networkx (for probabilistic models)
- scikit-learn (for utilities)
- PyTorch (for neural network-based EDAs)
- matplotlib, seaborn (for visualization)

See `requirements.txt` for complete dependency list.

## Quick Start

### Basic Binary Optimization

```python
from pateda.core.eda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import onemax
import numpy as np

# Problem configuration
n_vars = 30
pop_size = 300
cardinality = np.full(n_vars, 2)  # Binary variables

# Configure EDA components
components = EDAComponents(
    seeding=RandomInit(),
    learning=LearnFDA(cliques=None),  # Univariate model
    sampling=SampleFDA(n_samples=pop_size),
    selection=TruncationSelection(ratio=0.5),
    replacement=ElitistReplacement(n_elite=1),
    stop_condition=MaxGenerations(max_gen=50)
)

# Create and run EDA
eda = EDA(
    pop_size=pop_size,
    n_vars=n_vars,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    minimize=False
)

statistics, cache = eda.run(cache_models=True)
print(f"Best fitness: {statistics.best_fitness[-1]}")
```

### Continuous Optimization

```python
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnGaussianUMDA
from pateda.sampling import SampleGaussianUMDA
from pateda.functions.continuous import sphere

# Gaussian UMDA for continuous optimization
components = EDAComponents(
    learning=LearnGaussianUMDA(),
    sampling=SampleGaussianUMDA(n_samples=200),
    selection=TruncationSelection(ratio=0.5),
    stop_condition=MaxGenerations(max_gen=100)
)

eda = EDA(
    pop_size=200,
    n_vars=10,
    fitness_func=sphere,
    components=components,
    minimize=True
)

statistics, _ = eda.run()
```

### Neural Network-based EDA

```python
from pateda.enhanced_edas import DenoisingDiffusionEDA
from pateda.functions.discrete import trap

# Denoising Diffusion EDA with neural networks
eda = DenoisingDiffusionEDA(
    pop_size=500,
    n_vars=50,
    fitness_func=trap,
    cardinality=np.full(50, 2),
    hidden_dims=[128, 128],
    learning_rate=0.001,
    max_gen=100
)

statistics, _ = eda.run()
```

## Project Structure

```
pateda/
├── core/                   # Core EDA framework and components
├── seeding/                # Population initialization strategies
├── learning/               # Probabilistic model learning algorithms
├── sampling/               # Sampling methods from learned models
├── selection/              # Selection operators
├── replacement/            # Replacement strategies
├── mutation/               # Mutation operators
├── crossover/              # Crossover operators
├── local_optimization/     # Local search methods
├── stop_conditions/        # Stopping criteria
├── inference/              # Probabilistic inference methods
├── knowledge_extraction/   # Model analysis and interpretation
├── enhanced_edas/          # Advanced EDA implementations (VAE, GAN, etc.)
├── permutation/            # Permutation-specific components
├── functions/              # Benchmark test functions
│   ├── discrete/           # Binary and discrete problems
│   ├── continuous/         # Real-valued optimization
│   └── permutation/        # Permutation problems (TSP, QAP, etc.)
├── visualization/          # Visualization utilities
├── statistics/             # Statistics tracking and analysis
├── examples/               # Example scripts and use cases
├── tests/                  # Test suite
├── benchmarks/             # Benchmark scripts and results
├── scripts/                # Utility scripts
└── docs/                   # Documentation

```

## Implemented Algorithms

### Discrete EDAs
- **UMDA** (Univariate Marginal Distribution Algorithm)
- **BMDA** (Bivariate Marginal Distribution Algorithm)
- **Tree-EDA** (Tree-based EDA)
- **EBNA** (Estimation of Bayesian Network Algorithm)
- **MOA** (Mixture of Affinity-based EDAs)
- **FDA/cFDA** (Factorized/Conditional FDA)
- **Affinity Propagation EDA**

### Continuous EDAs
- **Gaussian UMDA**
- **Gaussian Full** (multivariate Gaussian)
- **Gaussian Network** (Gaussian Bayesian networks)
- **Mixture Gaussian**
- **GMRF-EDA** (Gaussian Markov Random Fields)
- **Vine Copula-based EDAs**

### Neural Network EDAs
- **VAE-EDA** (Variational Autoencoder)
- **GAN-EDA** (Generative Adversarial Network)
- **Denoising Diffusion EDA** (DbD-EDA)
- **Backdrive EDA** (Neural backdrive mechanisms)
- **Integer Neural EDA**

### Permutation EDAs
- **Mallows-EDA** (Kendall and Cayley distances)
- **Generalized Mallows**
- **Markov Chain Models**
- **Edge Histogram Models**

## Examples

The `examples/` directory contains over 50 example scripts demonstrating various EDAs and problem types:

```bash
# Discrete optimization
python examples/umda_onemax.py              # Basic UMDA
python examples/tree_eda_deceptive.py       # Tree-EDA on deceptive problems
python examples/ebna_nk_landscape.py        # EBNA on NK landscapes

# Continuous optimization
python examples/gaussian_umda_sphere.py     # Gaussian UMDA
python examples/gaussian_network_ackley.py  # Gaussian network EDA
python examples/mixture_gaussian_rosenbrock.py  # Mixture models

# Neural network-based
python examples/vae_eda_example.py          # VAE-EDA
python examples/gan_eda_example.py          # GAN-EDA
python examples/dendiff_eda_example.py      # Diffusion EDA

# Permutation problems
python examples/mallows_tsp_example.py      # TSP with Mallows
python examples/ehm_tsp_example.py          # TSP with edge histograms

# Advanced examples
python examples/comprehensive_eda_comparison.py  # Compare multiple EDAs
python examples/analysis_model_structure_visualization.py  # Visualize models
```

See `docs/EXAMPLES_GUIDE.md` for complete documentation of all examples.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_discrete_eda.py
pytest tests/test_gaussian_eda.py
pytest tests/test_permutation_benchmark.py

# Run with coverage
pytest --cov=pateda tests/
```

## Benchmarks

Performance benchmarks are available in the `benchmarks/` directory:

```bash
python benchmarks/benchmark_dendiff_distributions.py
python benchmarks/benchmark_nn_eda_vs_umda_gnbg.py
```

Results are saved in `benchmarks/results/`.

## Documentation

- **[Examples Guide](docs/EXAMPLES_GUIDE.md)** - Comprehensive guide to all example scripts
- **[PATEDA Design](docs/PATEDA_DESIGN.md)** - Architecture and design decisions
- **[Implementation Summaries](docs/)** - Detailed implementation documentation
- **[API Reference](docs/)** - Component and algorithm API documentation

## Citation

PATEDA builds upon the foundations of MATEDA (MATLAB toolbox for EDAs). If you use PATEDA in your research, please cite:

```bibtex
@article{santana2010mateda,
  title={Mateda-2.0: Estimation of distribution algorithms in MATLAB},
  author={Santana, Roberto and Bielza, Concha and Larra{\~n}aga, Pedro and Lozano, Jose A and Echegoyen, Carlos and Mendiburu, Alexander and Armananzas, Rub{\'e}n and Shakya, Siddartha},
  journal={Journal of Statistical Software},
  volume={35},
  number={7},
  pages={1--30},
  year={2010}
}

@article{irurozki2018algorithm,
  title={Algorithm 989: perm\_mateda: A Matlab Toolbox of Estimation of Distribution Algorithms for Permutation-based Combinatorial Optimization Problems},
  author={Irurozki, Ekhine and Ceberio, Josu and Santamaria, Jagoba and Santana, Roberto and Mendiburu, Alexander},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={44},
  number={4},
  pages={1--3},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas of interest:
- New EDA algorithms
- Additional benchmark problems
- Performance optimizations
- Documentation improvements
- Bug fixes

Please open an issue or submit a pull request.

## Authors and Acknowledgments

- **Roberto Santana** (roberto.santana@ehu.es) - Original MATEDA author and PATEDA contributor
- PATEDA Development Team

### Acknowledgments
- Original MATEDA development team
- pgmpy and PyTorch communities
- Contributors to the Python scientific computing ecosystem

## Contact

- For general questions: Roberto Santana (roberto.santana@ehu.es)
- For issues and bug reports: [GitHub Issues](https://github.com/rsantana-isg/pateda/issues)

---

**Status**: Active Development
**Version**: 0.1.0
**Last Updated**: January 2026
