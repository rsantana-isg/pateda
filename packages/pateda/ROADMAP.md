# pateda ROADMAP

## Completed (v0.1.0)

### Core EDA engine
- [x] `EDA` class with full generational loop (seeding → evaluate → statistics → select → learn → sample → repair → mutate → replace)
- [x] `EDAComponents` dataclass for plug-and-play component substitution
- [x] Multi-objective support (NSGA-II-style non-dominated sorting in selection)
- [x] `Statistics` and `Cache` containers for run-time monitoring

### Learning (discrete)
- [x] UMDA, PBIL, BSC (univariate)
- [x] FDA, CFDA, CUMDA (factorised)
- [x] BMDA, EBNA, BOA (Bayesian network)
- [x] MN-FDA, MN-FDAG, MN-FDA-R, MN-FDAG-R (Markov network)
- [x] Tree-EDA, Tree-EDA-R (tree-structured)
- [x] Mixture-of-Trees
- [x] MIMIC (dependency chain)
- [x] MOA (multi-objective archive)
- [x] Affinity EDA, Affinity-Elim EDA
- [x] Markov chain (sequence / permutation)

### Learning (continuous)
- [x] Gaussian univariate / full
- [x] Mixture of Gaussians (univariate, full, EM)
- [x] GMRF-EDA (Lasso / ElasticNet / LARS structure learning)
- [x] Vine copula EDA (C-vine, D-vine, auto — optional `pyvinecopulib` dep)

### Learning (permutation)
- [x] Mallows EDA — Kendall metric
- [x] EHM / NHM edge histogram model

### Sampling
- [x] Ancestral sampling (FDA, CFDA, CUMDA, Bayesian network)
- [x] Gibbs sampling
- [x] MAP-based sampling (Insert, Template, Hybrid)
- [x] Partial sampling (FDA variant)
- [x] Markov chain forward sampling
- [x] Mixture-of-Trees direct / indirect sampling
- [x] Gaussian / Mixture-Gaussian sampling
- [x] GMRF sampling
- [x] Vine-copula sampling (biased, conditional)
- [x] Mallows / EHM / NHM sampling

### Selection
- [x] Truncation, tournament, proportional, Boltzmann, ranking, SUS
- [x] Non-dominated sorting (NSGA-II)
- [x] Pareto-front extraction

### Other components
- [x] Mutation: bit-flip, frequency-balance (binary + multi-value)
- [x] Crossover: block, two-point, transposition
- [x] Seeding: random, biased, unitation-constrained
- [x] Replacement: elitist, generational
- [x] Repairing: unitation, trigonometric, bounds
- [x] Stop conditions: max generations, optimum-found
- [x] Local optimisation: greedy search, simulated annealing, scipy wrappers
- [x] Statistics: per-generation tracker, population stats
- [x] Knowledge extraction: dependency analysis, MI, model visualisation

### Benchmark functions
- [x] Discrete: OneMax, Deceptive-3/4/5, Trap, NK-landscape, SAT, UBQP, Ising, HP-Protein, Additive Decomposable, Contiguous Block, integer functions
- [x] Continuous: Sphere, Rosenbrock, Rastrigin, Ackley, GNBG benchmark suite
- [x] Permutation: TSP, LOP, QAP

---

## Planned / In progress (v0.2.0)

### Code quality & packaging
- [ ] Add type annotations to all public APIs
- [ ] Replace `pgmpy` dependency in BOA/EBNA with a lighter graph structure (pgmpy is heavy)
- [ ] Unified `learn()` / `sample()` interface for all algorithms (some still use functional APIs)
- [ ] `EDA.from_config(dict)` factory (currently raises `NotImplementedError`)
- [ ] Sphinx API documentation with ReadTheDocs deployment
- [ ] 100 % unit-test coverage for core, selection, seeding, stop_conditions
- [ ] Integration tests: OneMax with UMDA (should converge in ≤ 100 gen for n≤50)

### New algorithms
- [ ] CGA (compact GA)
- [ ] BOA with MDL scoring
- [ ] Restricted Boltzmann Machine (RBM) for discrete problems — without PyTorch (NumPy only)
- [ ] Additive-tree EDA (MT-EDA variants)

### Continuous EDA improvements
- [ ] Population restart strategy (diversity detection + re-seed)
- [ ] Covariance Matrix Adaptation (CMA-ES) as a baseline comparison
- [ ] GNBG benchmark: full parameter sweep utilities

### Permutation
- [ ] Cayley metric Mallows model
- [ ] Generalised Mallows with learned spread parameters
- [ ] Integration with multi-objective permutation problems (LOP, QAP bi-objective)

### Difficult to incorporate (tracked here)
- **Vine-copula learning** — `pyvinecopulib` has no Windows conda package; installation is fragile. Consider replacing with a pure-NumPy bivariate copula implementation.
- **BOA structure learning** — `pgmpy`'s BDeu scorer is slow for > 30 variables; a C-extension would be needed for competitive performance.
- **Affinity EDA** — the elimination heuristic relies on a combinatorial structure that is hard to generalise beyond binary; needs a redesign for multi-value variables.

---

## v0.3.0 and beyond

- [ ] GPU-accelerated fitness evaluation (batch vectorisation)
- [ ] Parallel population evaluation (multiprocessing / joblib)
- [ ] Benchmark CLI: `pateda-bench --algo UMDA --problem onemax --n 50 --seeds 20`
- [ ] Weights & Biases / MLflow integration for experiment tracking
- [ ] Numba JIT for inner sampling loops (currently optional)
