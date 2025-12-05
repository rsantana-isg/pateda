# PATEDA - Python Algorithms for Estimation of Distribution Algorithms

PATEDA is a Python port of MATEDA-3.0, a comprehensive MATLAB toolbox for Estimation of Distribution Algorithms (EDAs).

## Overview

Estimation of Distribution Algorithms are evolutionary algorithms that learn probabilistic models of promising solutions and sample new candidates from these models. PATEDA brings MATEDA's extensive capabilities to the Python ecosystem with a modern, type-safe, and extensible architecture.

## Features

- **Multiple EDA implementations**: UMDA, Tree-EDA, BOA-like, Gaussian EDAs, and more
- **Flexible architecture**: Modular components (seeding, learning, sampling, selection, replacement)
- **Discrete and continuous optimization**: Support for both binary/discrete and real-valued problems
- **Probabilistic models**: Factorized distributions, Bayesian networks, Gaussian models, Markov networks
- **Multi-objective optimization**: Support for multi-objective problems
- **Visualization**: Tools for analyzing and visualizing EDAs behavior
- **Type-safe**: Full type hints for better IDE support and error detection

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/rsantana-isg/pateda.git
cd pateda

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using pip (when available)

```bash
pip install pateda
```

## Quick Start

```python
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
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
    learning=LearnFDA(cliques=None),
    sampling=SampleFDA(n_samples=pop_size),
    selection=TruncationSelection(ratio=0.5),
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
print(f"Best fitness: {statistics.best_fitness[-1]}")
```

## Project Structure

```
pateda/
├── core/           # Core EDA framework
├── seeding/        # Population initialization methods
├── learning/       # Probabilistic model learning
├── sampling/       # Sampling from models
├── selection/      # Selection methods
├── replacement/    # Replacement strategies
├── functions/      # Test functions
├── examples/       # Example scripts
└── tests/          # Test suite
```

## Implemented EDAs

### Current (v0.1.0)

- UMDA (Univariate Marginal Distribution Algorithm)
- Tree-EDA (Tree-based EDA)
- Gaussian-UMDA (for continuous optimization)

### Planned

- BOA (Bayesian Optimization Algorithm)
- MOA (Markov network EDA)
- EBNA (Estimation of Bayesian Network Algorithm)
- Gaussian networks
- Multi-objective EDAs
- Mixture models

## Documentation

- [Design Document](PATEDA_DESIGN.md) - Architecture and design decisions
- [User Guide](docs/user_guide.md) - Comprehensive usage guide
- [API Reference](docs/api/) - Detailed API documentation
- [Original MATEDA Guide](Mateda2.0-UserGuide.pdf) - Original MATLAB documentation

## Relationship to MATEDA

PATEDA is a faithful port of MATEDA-3.0 with the following improvements:

- **Modern Python**: Type hints, dataclasses, and Python best practices
- **Better APIs**: Replace MATLAB's eval() with type-safe component registry
- **Native libraries**: Use pgmpy, scikit-learn instead of MATLAB toolboxes
- **Testing**: Comprehensive test suite
- **Performance**: Leverage NumPy/SciPy for efficient computations

The original MATLAB code is preserved in this repository for reference and comparison.

## Citation

If you use PATEDA in your research, please cite the original MATEDA papers:

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

This project maintains compatibility with MATEDA's licensing. Please refer to the original MATEDA documentation for license details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Authors

- **Roberto Santana** - Original MATEDA author
- **Claude (AI Assistant)** - Python port

## Acknowledgments

- Original MATEDA development team
- BNT (Bayes Net Toolbox) by Kevin Murphy
- PMTK3 library contributors

## Contact

For questions about MATEDA: Roberto Santana (roberto.santana@ehu.es)
For PATEDA-specific issues: Please use GitHub issues

---

**Status**: Alpha - Under active development
**Version**: 0.1.0
**Last Updated**: 2025-11-18
