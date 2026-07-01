# Multi-Objective Optimisation in pateda: Software Design

This document describes how the three classical multi-objective optimisation
paradigms (see `multi_objective_optimization_paradigms.md`) are realised inside
the `pateda` library in a **general, model-agnostic** way: every paradigm works
with *any* pateda probabilistic model (discrete or continuous), exactly as the
single-objective EDA engine does. Permutation-based problems are deliberately
left to a separate derived package and are out of scope here.

---

## 1. Design principles

pateda's EDA engine (`pateda.core.eda.EDA`) is already multi-objective aware:
fitness is carried internally as a 2-D array `(pop_size, n_objectives)`, and the
evolutionary loop is a pipeline of **pluggable components** (seeding, learning,
sampling, selection, replacement, ...). The design exploits two extension
points so that nothing model-specific is ever hard-coded:

1. **Selection is the natural seam for Pareto- and indicator-based search.**
   Both paradigms differ from single-objective EDAs *only* in how they choose
   the promising subset that the model is learned from. They are therefore
   implemented as ordinary `SelectionMethod` objects and drop straight into the
   existing loop with any learning/sampling pair.

2. **Decomposition needs a different control flow, but not different models.**
   MOEA/D maintains one solution per weight vector and reproduces per
   sub-problem, which does not fit the single global learn/sample cycle. It is
   implemented as a separate *driver* (`MOEAD`) that **reuses the same
   `LearningMethod` and `SamplingMethod` components** the EDA engine uses. This
   is the EDA realisation of MOEA/D-GM (probabilistic graphical models inside
   MOEA/D).

A shared, representation-agnostic toolkit (`pateda.multiobjective`) provides the
common machinery: dominance, crowding, archive, weight vectors, scalarising
functions and quality indicators.

```
pateda/
├── multiobjective/            # model-agnostic toolkit (NEW)
│   ├── dominance.py           # dominance / non-dominated set helpers
│   ├── crowding.py            # NSGA-II crowding distance
│   ├── archive.py             # bounded external Pareto archive
│   ├── weights.py             # weight-vector designs (uniform, Das-Dennis)
│   ├── scalarization.py       # weighted sum, Tchebycheff, PBI
│   ├── indicators.py          # hypervolume, HV contributions, eps/IBEA, IGD
│   └── moead.py               # MOEA/D decomposition driver (NEW)
├── selection/
│   ├── pareto_front.py        # (existing) non-dominated sorting truncation
│   ├── non_dominated.py       # (existing) first-front selection
│   ├── crowding.py            # CrowdingDistanceSelection  (NEW, Pareto+diversity)
│   └── indicator_based.py     # IndicatorBasedSelection    (NEW, IBEA/SMS-EMOA)
└── functions/discrete/
    └── multiobjective.py      # MO discrete benchmarks (NEW)
```

---

## 2. Paradigm 1 — Pareto-based (verified + enhanced)

**Already present.** `ParetoFrontSelection` (non-dominated sorting truncation)
and `NonDominatedSelection` (first-front only), backed by
`selection/utils/pareto.py` (`pareto_dominates`, `find_pareto_set`,
`pareto_ranking`). Combined with any model they form a Pareto-based MOEDA.

**Gap found:** no diversity preservation — selection ordered by front only,
so it collapses onto a few clustered solutions.

**Enhancement:** `CrowdingDistanceSelection` implements full NSGA-II
environmental selection: accept complete fronts in order, and truncate the
overflowing front by **crowding distance** (keeping the most spread-out and the
boundary solutions). `crowding.py` houses the reusable distance metric, also
used to prune the archive.

```python
from pateda.selection import CrowdingDistanceSelection
components = EDAComponents(..., selection=CrowdingDistanceSelection(ratio=0.5))
```

---

## 3. Paradigm 2 — Indicator / metric-based (new)

Implemented as `IndicatorBasedSelection`, a `SelectionMethod` that scores
solutions with a quality indicator so a single scalar reflects both convergence
and diversity. Two indicators:

* `"epsilon"` — the **binary additive epsilon indicator** with the adaptive
  **IBEA** fitness assignment and iterative worst-removal environmental
  selection (Zitzler & Künzli, 2004).
* `"hypervolume"` — greedy removal by **exclusive hypervolume contribution**
  (SMS-EMOA style).

Because it is just a selection method, it works with any model and needs no
change to the engine:

```python
from pateda.selection import IndicatorBasedSelection
components = EDAComponents(..., selection=IndicatorBasedSelection(indicator="epsilon"))
```

The supporting indicators in `indicators.py` (exact hypervolume by recursive
slicing, per-point HV contributions, additive-epsilon matrix / IBEA fitness,
IGD) double as **evaluation metrics** for experiments.

---

## 4. Paradigm 3 — Decomposition-based (new)

`MOEAD` decomposes the problem into `N` scalar sub-problems (one weight vector
each) and optimises them cooperatively over neighbourhoods. The key design
decision is that **reproduction is delegated to pateda components**:

* `weights.py` builds the weight vectors — evenly spaced for two objectives,
  Das-Dennis simplex lattice for `m >= 3` — and the Euclidean neighbourhoods.
* `scalarization.py` provides weighted-sum / Tchebycheff / PBI, all returning a
  *cost* (lower is better) and direction-aware via the `maximize` flag, so the
  driver stays agnostic to the optimisation sense.
* Each generation, for every sub-problem, a model is learned from its mating
  pool and one offspring is sampled — using the injected `LearningMethod` /
  `SamplingMethod`. Two scopes are available:
  * `"neighbourhood"` (default): a model per sub-problem, specialised to its
    region of the front (true MOEA/D-GM);
  * `"global"`: one model per generation for the whole population (cheaper).
* An external `ParetoArchive` (optionally capacity-bounded, pruned by crowding
  distance) collects the non-dominated solutions found.

```python
from pateda.multiobjective import MOEAD
moead = MOEAD(n_vars, cardinality, fitness_func, components,
              n_obj=2, n_weights=100, scalarization="tchebycheff", maximize=True,
              n_gen=100, model_scope="neighbourhood", random_seed=42)
result = moead.run()
front_objs = result.pareto_objectives
```

`cardinality` is a 1-D array for discrete problems or a `(2, n_vars)` bounds
array for continuous ones, so the *same* driver serves both representations.

---

## 5. Benchmarks and scripts

`functions/discrete/multiobjective.py` adds discrete MO test problems
(maximisation):

* `mo_onemax_zeromax` — diversity-only problem with a known analytic front
  (`f1 + f2 = n`); good for measuring spread / IGD.
* `make_mo_deceptive` — two conflicting deceptive trap objectives; needs a model
  that captures building blocks.
* `make_mubqp` — multi-objective UBQP with tunable objective correlation `rho`
  (hard combinatorial benchmark; cf. Liefooghe et al. on hard mUBQP instances).

Scripts in `examples/`:

* `multiobjective_approaches_demo.py` — runs all three paradigms with a chosen
  model on a chosen problem, reports hypervolume (and IGD when the true front is
  known) and optionally saves a Pareto-front figure.
* `run_mo_eda.py` — positional, seed-first single-experiment runner suitable for
  SLURM launchers: `SEED APPROACH PROBLEM MODEL N_VARS POP_SIZE N_GEN [SCALARIZATION]`.

---

## 6. Summary

| Paradigm | pateda mechanism | Status | Works with any model |
|----------|------------------|--------|----------------------|
| Pareto-based | `ParetoFrontSelection`, `NonDominatedSelection`, **`CrowdingDistanceSelection`** | verified + enhanced (crowding) | yes (selection seam) |
| Indicator-based | **`IndicatorBasedSelection`** (IBEA / SMS-EMOA) + `indicators.py` | new | yes (selection seam) |
| Decomposition-based | **`MOEAD`** driver + `weights`/`scalarization`/`archive` | new | yes (reuses learning/sampling) |
```
