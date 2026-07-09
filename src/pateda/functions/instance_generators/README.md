# Multi-objective instance generators

Command-line scripts that build and save instances of three multi-objective
benchmark models, implemented from the reference code and papers in
`functions/Multi_Objective_Code/`. The models and their objective functions live
in `pateda.functions.discrete_binary.multiobjective`; these scripts drive them
and write instance files to `functions/MultiObjective_Instances/{mnm,mnk,mubqp}/`.

All three models produce **binary** multi-objective problems whose objective
functions return a vector to be **maximised** (pateda's MO convention), so the
generated instances can be fed directly to the MOEA-D / Pareto / indicator
drivers in `pateda.multiobjective` or to any EDA via the core `EDA` class.

## Models

| Model | Module | Source | Idea |
|-------|--------|--------|------|
| **mNM** | `multiobjective/mnm.py` | `mNM-Model/`, CAEPIA 2015 | Truncated Walsh / Markov-network expansion `f(x)=β₀+Σ_c β_c ∏_{i∈c} x_i`, `x∈{−1,+1}`. Objectives use different maximum interaction orders and a sign transform `x→±x`. |
| **MNK** | `multiobjective/mnk_landscape.py` | `Multi-NK/`, "Heterogeneous Objectives" | `m` independent random-neighbourhood NK landscapes over shared variables; supports a different `K` per objective (heterogeneous). Each objective lies in `[0,1)`. |
| **mUBQP** | `multiobjective/mubqp.py` | `mUBQP/`, "On the Design of Hard mUBQP Instances" | `f_k(x)=Σ_{ij} Q^k_{ij} x_i x_j`. Standard density+correlation generator, five artificial structured types, and hard instances composed from order-5 building blocks selected by difficulty metrics. |

## Usage

Each script takes positional arguments with the **seed first** and prints its
configuration:

```bash
# mNM: 12 vars, base order 3, sigma 5, objective orders 2 and 3
python generate_mnm_instances.py 111 12 3 5.0 2 3

# MNK: 20 vars, heterogeneous K = (2,4,3) -> 3 objectives
python generate_mnk_instances.py 111 20 2+4+3 3

# mUBQP: standard (density+correlation), rho=-0.7
python generate_mubqp_instances.py 111 standard 40 -0.7 0.4
# mUBQP: one of the 5 artificial structured types
python generate_mubqp_instances.py 111 artificial 100 3
# mUBQP: hard instance from order-5 building blocks (heavy or tiled)
python generate_mubqp_instances.py 111 hard 100 heavy
```

## Reloading an instance

```python
from pateda.functions.discrete_binary.multiobjective.mubqp import (
    MUBQPInstance, create_mubqp_objective_function)
inst = MUBQPInstance.load(path)          # .dat for mUBQP, .npz for mNM / MNK
objective = create_mubqp_objective_function(inst)  # objective(pop) -> (pop, n_obj)
```

`MNMInstance.load` / `MNKLandscape.load` and their
`create_mnm_objective_function` / `create_mnk_objective_function` work the same way.

## Hard mUBQP building blocks

The hard-instance path reproduces `SearchHarduBQPInstances.m`:

1. `enumerate_order5_chunks()` — all `2¹⁰` order-5 UBQP chunks (10 pairwise
   weights in `{−1,+1}`).
2. `chunk_pair_metrics(w1, w2)` — Pareto-set size, per-objective Boltzmann
   *deception*, and fitness-distance correlation over the `2⁵` search space.
3. `select_hard_chunk_pairs(...)` — rank sampled pairs by `chunk_pair_hardness`
   and keep the hardest.
4. `create_mubqp_from_chunk` (block-separable tiling) /
   `create_heavy_mubqp_from_chunks` (random overlapping placement) — compose a
   large instance from the hard chunks.
