# pateda test functions

Objective/fitness functions for optimization, organized by **representation**
(binary, non-binary discrete, continuous) and, within each, by **kind**
(*toy functions* vs *problems*).

> **Naming note.** The conceptual groups are *discrete-binary*,
> *discrete-non-binary*, *continuous*, each split into *toy-functions*,
> *problems* and (for the discrete representations) *multi-objective*. Python
> package names cannot contain hyphens, so the directories use underscores:
> `discrete_binary`, `discrete_non_binary`, `continuous`, `toy_functions`,
> `problems`, `multiobjective`.

## Layout

```
functions/
├── graph_utils.py            # shared graph-instance readers + graph_instances_dir()
├── instance_generators/      # CLI scripts to build multi-objective instances
│                             #   (mNM, MNK, mUBQP) -> MultiObjective_Instances/
├── discrete_binary/
│   ├── toy_functions/        # pseudo-boolean academic benchmarks
│   │   onemax, deceptive, trap, additive_decomposable, contiguous_block,
│   │   tree_max, four_peaks, six_peaks, continuous_peaks, plateau,
│   │   checkerboard, summation_cancellation
│   ├── problems/             # binary combinatorial problems
│   │   nk_landscape, ising, sat, ubqp, max_cut, max_clique,
│   │   max_independent_set, dominating_set, equal_products, linear_equations
│   └── multiobjective/       # multi-objective binary benchmarks
│       pseudoboolean (mo-onemax/zeromax, mo-deceptive, mUBQP),
│       mnm, mnk_landscape, mubqp (configurable instance families)
├── discrete_non_binary/
│   ├── toy_functions/        # integer / categorical benchmarks
│   │   integer_functions
│   ├── problems/             # non-binary combinatorial problems
│   │   hp_protein, graph_coloring, clique_covering, braid
│   └── multiobjective/       # multi-objective non-binary benchmarks
│       braid_biobjective (approximation error vs. braid length)
└── continuous/
    ├── toy_functions/        # real-valued benchmarks
    │   benchmarks (sphere, rastrigin, ...)
    └── problems/             # continuous problems
        ab_protein (AB off-lattice protein model)
```

Multi-objective functions return an objective **vector** to be maximised (the
convention used by `pateda.multiobjective` MOEA-D / Pareto / indicator drivers),
rather than the scalar returned by single-objective functions.

### Braid quantum-gate approximation (`discrete_non_binary/problems/braid.py`)

Approximates a target single-qubit gate `T` by an ordered product of `n`
elementary braid-generator matrices (a *braid*). A solution is a length-`n`
vector over `{0,...,2g-1}` selecting `g` generators and their `g` inverses; for
the Fibonacci-anyon set `g = 2` (cardinality 4: `sigma_1, sigma_2, sigma_1^{-1},
sigma_2^{-1}`). The error is the Frobenius distance `epsilon = |B - T|` and the
maximised fitness is `(1-lambda)/(1+epsilon) + lambda/l`. The **icosahedral
benchmark** uses any of the 60 icosahedral-group matrices as target
(`make_icosahedral_benchmark_problem`). A bi-objective variant
(`multiobjective/braid_biobjective.py`) trades error against braid length.
Sources analysed: `Braid_Optimization/papers/SOCOBraidPaper4` and
`Braid_Optimization/brading-icohesahedral/`.

Two functions are *binary* vs *non-binary* by their encoding: `max_cut`,
`max_clique`, `max_independent_set` and `dominating_set` assign one **binary**
variable per vertex, whereas `graph_coloring` and `clique_covering` assign a
**color / partition index** per vertex (cardinality > 2), so they live under
`discrete_non_binary`.

## Conventions

Every problem module exposes a factory `create_<name>_objective_function(...)`
returning `objective(population) -> fitness`, where `population` is a 2D array
`(pop_size, n_vars)` (a 1D vector is also accepted) and `fitness` is a 1D array.
This is the signature consumed by `pateda.core.eda.EDA`.

## Instances

Problems that read benchmark instances resolve their packaged directory
relative to this package:

| Problem | Directory | Loader |
|---|---|---|
| Ising | `Ising_Instances/`, `Ising_Instances_Sidon/` | `load_ising_benchmark_instance` |
| SAT | `SAT_instances/` | `load_sat_benchmark_instance` |
| UBQP | `UBQP_Instances/` | `load_ubqp_benchmark_instance` |
| Max-Cut | `graph_instances/max_cut/` | `read_max_cut_graph` |
| Max Clique | `graph_instances/maximum_clique/` | `read_dimacs_graph` |
| Graph Coloring | `graph_instances/graph_coloring/` | `read_dimacs_graph` |
| Max Independent Set | `graph_instances/max_independent_set/` | `read_dimacs_graph` |
| Dominating Set | `graph_instances/dominating_set/` | `read_dimacs_graph` |
| Clique Covering | `graph_instances/clique_covering/` | `read_dimacs_graph` |

Graph directories are located with
`pateda.functions.graph_utils.graph_instances_dir("<subdir>")`. `nk_landscape`,
`equal_products` and `linear_equations` generate their instances
programmatically and need no files.

The braid benchmark ships its instances in `Braid_Instances/`
(`fibonacci_anyon_generators.npz`, `icosahedral_group.npz`), regenerable with
`save_braid_instances()`; the definitions are deterministic so the loaders fall
back to generating them if the files are absent.

## Examples / tests

Graph problems:

* `examples/eda_graph_binary_problems.py` — UMDA and Tree-EDA on the four binary
  graph problems using packaged instances.
* `tests/test_graph_binary_problems.py` — objective correctness on
  hand-checkable graphs plus short EDA runs.

Braid quantum-gate approximation:

* `examples/braid_icosahedral_benchmark.py` — defines the icosahedral benchmark
  (any of the 60 group matrices as target) with a random-search baseline.
* `examples/eda_braid_icosahedral.py` — evaluates UMDA, Tree-EDA and MK-EDA on
  the icosahedral benchmark.
* `tests/test_braid.py` — generator/group algebra, error/length definitions,
  bi-objective vector, and a short EDA run.
