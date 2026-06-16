# perm_pateda

**Estimation of Distribution Algorithms (EDAs) for permutation-based combinatorial
optimization problems.**

`perm_pateda` is an independent, permutation-focused companion to
[`pateda`](../pateda). It contributes the model **learning** and **sampling**
methods that are specific to permutation spaces — histogram models and
distance-based **Mallows** / **Generalized Mallows** models — and reuses
`pateda` for everything common to all EDAs (the core EDA engine, selection,
replacement, statistics, and visualization utilities).

It is a Python port of the algorithms in **`perm_mateda`**
(Irurozki, Ceberio, Santamaria & Mendiburu, 2018,
*Algorithm 989: perm_mateda — A Matlab Toolbox of Estimation of Distribution
Algorithms for Permutation-based Combinatorial Optimization Problems*,
ACM TOMS 44(4), Article 47).

---

## Relationship to `pateda`

```
pateda  ─────────────────────────────►  perm_pateda
(core EDA engine, selection,            (permutation learning + sampling,
 replacement, statistics, viz,           Mallows / GMallows models,
 discrete & continuous EDAs)             histogram models, TSP/QAP/LOP)
```

`pateda` is a **dependency** of `perm_pateda`. All permutation-related code has
been removed from `pateda` and now lives here, so the two packages have a clean
separation of concerns:

| Concern | Package |
|---|---|
| Core EDA loop, components, models | `pateda` |
| Selection / replacement / statistics / visualization | `pateda` |
| Discrete & continuous EDAs (UMDA, EBNA, Gaussian, …) | `pateda` |
| Permutation distances (Kendall, Cayley, Ulam) | `perm_pateda` |
| Mallows & Generalized Mallows learning/sampling | `perm_pateda` |
| Edge / Node histogram models | `perm_pateda` |
| Permutation problems (TSP, QAP, LOP) | `perm_pateda` |

---

## Installation

`pateda` is an in-development package (not yet on PyPI), so install it first
from the local checkout, then install `perm_pateda`:

```bash
# from the repository root (…/github/pateda)
pip install -e packages/pateda
pip install -e packages/perm_pateda
```

For development tooling (pytest, ruff, …):

```bash
pip install -e "packages/perm_pateda[dev]"
```

Requires Python ≥ 3.9, `numpy`, `scipy`, and `pateda`.

---

## Quick start

```python
import numpy as np
from perm_pateda import MallowsKendallEDA
from perm_pateda.functions import create_random_lop

# A Linear Ordering Problem instance on 15 items
lop = create_random_lop(15, seed=0)

alg = MallowsKendallEDA(
    n_vars=15,
    fitness_func=lop,        # callable: permutation -> scalar (higher is better)
    pop_size=100,
    n_gen=50,
    selection_ratio=0.3,
    random_seed=0,
)
stats, _ = alg.run()
print("Best fitness:", stats.best_fitness_overall)
print("Best permutation:", stats.best_individual)
```

---

## Available algorithms

Plug-and-play wrappers (import from `perm_pateda`):

| Class | Model | Distance |
|---|---|---|
| `MallowsKendallEDA`   | Mallows               | Kendall's-τ |
| `MallowsCayleyEDA`    | Mallows               | Cayley |
| `GMallowsKendallEDA`  | Generalized Mallows   | Kendall's-τ |
| `GMallowsCayleyEDA`   | Generalized Mallows   | Cayley |
| `EHMEDA`              | Edge Histogram Model  | — |
| `NHMEDA`              | Node Histogram Model  | — |

See [`ROADMAP.md`](ROADMAP.md) for the planned additions (Mallows–Ulam,
`BestPermutation` consensus, the PFSP problem, and real-instance loaders) that
complete the feature set of the `perm_mateda` toolbox.

---

## Package layout

```
perm_pateda/
├── distances.py          # Kendall, Cayley, Ulam distances (+ helpers)
├── consensus.py          # central-permutation estimators (Borda, SetMedian)
├── learning/
│   ├── histogram.py      # LearnEHM, LearnNHM
│   └── mallows.py        # LearnMallows{Kendall,Cayley}, LearnGeneralizedMallows{Kendall,Cayley}
├── sampling/
│   ├── histogram.py      # SampleEHM, SampleNHM
│   └── mallows.py        # SampleMallows{Kendall,Cayley}, SampleGeneralizedMallows{Kendall,Cayley}
├── seeding/
│   └── permutation_init.py   # PermutationInit (random permutations)
├── functions/
│   ├── tsp.py            # Traveling Salesman Problem
│   ├── qap.py            # Quadratic Assignment Problem
│   └── lop.py            # Linear Ordering Problem
└── algorithms/
    └── permutation.py    # plug-and-play EDA wrappers
```

---

## Citation

If you use the distance-based permutation EDAs implemented here, please cite the
original toolbox:

> E. Irurozki, J. Ceberio, J. Santamaria, and A. Mendiburu (2018).
> *Algorithm 989: perm_mateda — A Matlab Toolbox of Estimation of Distribution
> Algorithms for Permutation-based Combinatorial Optimization Problems.*
> ACM Transactions on Mathematical Software, 44(4), Article 47.

## License

MIT — see [LICENSE](LICENSE).
