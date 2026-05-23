# pateda packages

This directory contains two independent, PyPI-ready Python packages extracted from the PATEDA research framework.

| Package | PyPI name | Description |
|---------|-----------|-------------|
| [`pateda/`](pateda/) | `pateda` | Classical EDA implementations (discrete, continuous, permutation) — no deep learning required |
| [`pateda_nn/`](pateda_nn/) | `pateda-nn` | Neural network EDA extensions (VAE, GAN, Diffusion, Backdrive, DBD, RBM) — requires PyTorch |

## Dependency graph

```
pateda          ← standalone (numpy, scipy, pgmpy, networkx, scikit-learn, …)
   ↑
pateda-nn       ← depends on pateda + torch
```

## Development install

```bash
# Install pateda in editable mode
pip install -e packages/pateda

# Install pateda-nn in editable mode (installs pateda automatically)
pip install -e packages/pateda_nn

# Or both at once
pip install -e packages/pateda -e packages/pateda_nn
```

## Building distribution packages

```bash
pip install build

cd packages/pateda
python -m build        # creates dist/pateda-*.whl and dist/pateda-*.tar.gz

cd ../pateda_nn
python -m build        # creates dist/pateda_nn-*.whl and dist/pateda_nn-*.tar.gz
```

## Publishing to PyPI

```bash
pip install twine
twine upload packages/pateda/dist/*
twine upload packages/pateda_nn/dist/*
```

## Regenerating the packages

If the upstream source in the repo root changes, re-run the reorganisation script to refresh both packages:

```bash
cd packages
python3 reorganize.py
```

This copies all source files and rewrites NN-specific imports.
