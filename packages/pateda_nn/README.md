# pateda-nn — Neural Network EDAs for pateda

[![PyPI version](https://img.shields.io/pypi/v/pateda-nn)](https://pypi.org/project/pateda-nn/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`pateda-nn` extends the [`pateda`](https://pypi.org/project/pateda/) package with **deep-learning-based EDA** algorithms using **PyTorch**.  It provides learning and sampling components that plug directly into the `pateda.EDA` framework.

## Implemented models

### Continuous optimisation (PyTorch)

| Algorithm | Learning | Sampling | Description |
|-----------|---------|---------|-------------|
| VAE-EDA       | `learn_vae`           | `sample_vae`           | Variational Autoencoder |
| E-VAE-EDA     | `learn_extended_vae`  | `sample_extended_vae`  | VAE + fitness predictor |
| CE-VAE-EDA    | `learn_conditional_extended_vae` | `sample_conditional_extended_vae` | Conditioned E-VAE |
| GAN-EDA       | `learn_gan`           | `sample_gan`           | Generative Adversarial Network |
| DBD-EDA       | `learn_dbd`           | `sample_dbd`           | Alpha-deblending diffusion |
| DAE-EDA       | `learn_dae`           | `sample_dae`           | Denoising Autoencoder |
| Dendiff-EDA   | `learn_dendiff`       | `sample_dendiff`       | Denoising diffusion (Gaussian) |
| Dendiff-ReLU  | `learn_dendiff` (relu variant) | `sample_dendiff` | Dendiff with ReLU activations |
| Backdrive-EDA | `learn_backdrive`     | `sample_backdrive`     | Backpropagation-guided EDA |
| Backdrive-Adaptive | `learn_backdrive` | `sample_backdrive_adaptive` | Adaptive learning-rate Backdrive |
| RBM-EDA       | `learn_rbm`           | `sample_rbm`           | Restricted Boltzmann Machine |
| NN-EDA        | `LearnNNEDA`          | `SampleNNEDA`          | Generic neural-network EDA |

### Discrete / binary optimisation (PyTorch)

| Algorithm | Module | Description |
|-----------|--------|-------------|
| Discrete VAE-EDA   | `learn_discrete_vae` / `sample_discrete_vae` | Gumbel-Softmax VAE |
| Discrete E-VAE-EDA | `learn_discrete_extended_vae`               | Extended discrete VAE |
| Discrete GAN-EDA   | `learn_discrete_gan` / `sample_discrete_gan` | Binary/categorical GAN |
| Discrete DBD-CS    | `learn_discrete_dbd_cs` / `sample_discrete_dbd_cs` | DBD cosine similarity |
| Discrete DBD-CD    | `learn_discrete_dbd_cd` / `sample_discrete_dbd_cd` | DBD cosine distance |
| Discrete Backdrive | `learn_discrete_backdrive` / `sample_discrete_backdrive` | Discrete backdrive |
| Discrete Dendiff (Gumbel, Corruption, STE, Deterministic, Hard Concrete) | `learning/discrete_dendiff_*.py` | Discrete diffusion variants |

#### Unified discrete sampling

`sample_discrete_nn(model, n_samples, cardinality=None, params=None, seed_pop=None)`
dispatches to the correct sampler based on `model['type']`, so you don't need to
remember which `sample_*` goes with which `learn_*`:

```python
from pateda_nn import sample_discrete_nn, supported_discrete_types
from pateda_nn.learning.discrete_vae import learn_binary_vae

model = learn_binary_vae(pop, fitness, params={"epochs": 20})
new_pop = sample_discrete_nn(model, n_samples=200)
# DBD CS/CD models additionally take a seed population:
#   sample_discrete_nn(dbd_model, 200, seed_pop=selected_pop)
print(supported_discrete_types())
```

### Legacy TF module (`pateda_nn.legacy`)

`pateda_nn.legacy` contains older **TensorFlow 2.x** implementations of VAE, GAN, DBD, and Diffusion EDAs for continuous problems, along with Gaussian models and GNBG benchmark utilities.  These are included for reproducibility of earlier experiments and are not actively maintained.  They require `pip install "pateda-nn[tensorflow]"`.

## Installation

```bash
pip install pateda-nn
```

This also installs `pateda` (the core package) and `torch`.

For the legacy TensorFlow module:

```bash
pip install "pateda-nn[tensorflow]"
```

## Requirements

- `pateda >= 0.1.0`
- `torch >= 2.0`
- `numpy >= 1.21`, `scipy >= 1.7`

## Quick start — Discrete VAE-EDA

```python
import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.onemax import onemax
from pateda_nn.learning.discrete_vae import learn_discrete_vae
from pateda_nn.sampling.discrete_neural import sample_discrete_vae

n_vars = 30
cardinality = np.full(n_vars, 2)

components = EDAComponents(
    seeding=RandomInit(),
    learning=learn_discrete_vae,
    sampling=sample_discrete_vae,
    selection=TruncationSelection(),
    stop_condition=MaxGenerations(max_gen=200),
    selection_params={"selection_size": 150},
    learning_params={"latent_dim": 10, "n_epochs": 50},
)

eda = EDA(
    pop_size=300,
    n_vars=n_vars,
    fitness_func=onemax,
    cardinality=cardinality,
    components=components,
    random_seed=42,
)
stats, _ = eda.run()
print(f"Best fitness: {stats.best_fitness_overall}")
```

## Project structure

```
pateda_nn/
├── src/
│   └── pateda_nn/
│       ├── __init__.py
│       ├── learning/       # PyTorch-based learning algorithms
│       ├── sampling/       # PyTorch-based sampling methods
│       └── legacy/         # TensorFlow legacy implementations
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## Citation

```bibtex
@misc{pateda_nn,
  author  = {Roberto Santana},
  title   = {pateda-nn: Neural Network EDA Implementations},
  year    = {2024},
  url     = {https://github.com/rsantana-isg/pateda},
}
```

## License

MIT — see [LICENSE](LICENSE).
