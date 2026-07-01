# ROADMAP: pateda User Guide

## Overview

This roadmap organizes the development of the **pateda** user guide in phases.
The guide targets researchers and practitioners familiar with evolutionary computation
who want to use or extend pateda for single- and multi-objective optimization with EDAs.
It follows the structure of the MATEDA-2.0 guide (Santana et al., 2009) while reflecting
the Python architecture and extended algorithm set of pateda.

---

## Phase 1 — Foundation (current)

**Goal:** Produce a self-contained skeleton with all major sections drafted.

### 1.1 Introduction
- What are EDAs; motivation and historical context
- Position of pateda among existing EDA toolkits (MATEDA, BOA toolkit, etc.)
- Overview of pateda's three usage modes:
  1. Quick-start with plug-and-play algorithm classes
  2. Flexible EDA assembly via `EDAComponents`
  3. Custom component subclassing
- Summary of capabilities: discrete / continuous, single / multi-objective,
  knowledge extraction, a-priori structure injection

### 1.2 Installation and Package Structure
- Installation from source (`pip install -e packages/pateda`)
- Module tree (`core`, `algorithms`, `learning`, `sampling`, `selection`, ...)
- Running the examples in `packages/pateda/examples/`

### 1.3 Defining and Executing an EDA
- The generic EDA loop (Algorithm pseudocode)
- The `EDA` class (`core/eda.py`): constructor, `run()`, `Statistics`, `Cache`
- The `EDAComponents` dataclass (`core/components.py`)
- Step-by-step example: UMDA on OneMax (discrete, binary)
- Step-by-step example: GaussianUMDA on Sphere (continuous)
- Fitness convention (maximization; negation for minimization)
- Cardinality vs. bounds: discrete and continuous representations

### 1.4 Plug-and-Play Algorithm Classes
- `algorithms/discrete.py`: table of all discrete algorithms
- `algorithms/continuous.py`: table of all continuous algorithms
- Common parameters: `n_vars`, `cardinality`/`bounds`, `fitness_func`,
  `pop_size`, `n_gen`, `selection_ratio`, `elitism`, `random_seed`
- Worked example with `EBNA` and `GaussianNetwork`

---

## Phase 2 — Probabilistic Models and Algorithm Families

**Goal:** Document all model families and the learning/sampling component pairs.

### 2.1 Representations
- Discrete representation: binary, multi-valued, mixed cardinality
- Continuous representation: box-bounded real variables
- How `cardinality` encodes both cases

### 2.2 Factorized Distributions (discrete)
- Definition and MATEDA-style `Cliques`/`Tables` format
- `LearnFDA` / `SampleFDA`
- Markov chain as a special factorization: `LearnMarkovChain` / `SampleMarkovChain` (MK-EDA)
- Constrained factorizations: `LearnCFDA` / `SampleCFDA`

### 2.3 Algorithms Based on Univariate Models
- UMDA (`LearnUMDA`)
- PBIL (`LearnPBIL`): incremental learning rate
- BSC (`LearnBSC`): fitness-weighted marginals
- MIMIC (`LearnMIMIC`): chain of mutual-information-maximizing pairs

### 2.4 Algorithms Based on Trees and Forests
- Tree-EDA (`LearnTreeModel`): maximum-weight spanning tree of MI
- Tree-EDA_r (`LearnTreeModelR`): restricted tree with interaction matrix
- BMDA (`LearnBMDA`): bivariate chi-square forest
- AffEDA (`LearnAffinityFactorization`): affinity propagation cliques

### 2.5 Algorithms Based on Bayesian Networks
- Bayesian network representation; directed acyclic graphs
- EBNA (`LearnEBNA`): score-and-search with BIC
- BOA (`LearnBOA`): greedy MDL/BIC search
- Sampling Bayesian networks: `SampleBayesianNetwork`
- MAP/k-MPE sampling: `SampleGibbs`, `map_sampling`, `kmap_sampling`

### 2.6 Algorithms Based on Markov Networks
- Markov random fields; Gibbs distributions; Gibbs sampling
- MN-FDA (`LearnMNFDA`): chi-square Markov network + FDA sampling
- MN-FDA_r (`LearnMNFDAR`): random variable orderings
- MN-FDAg (`LearnMNFDAG`): augmented graph + Gibbs sampling
- MN-FDAg_r (`LearnMNFDAGR`): augmented graph + random Gibbs ordering
- MOA (`LearnMOA`): Markovianity-based k-neighborhood + Gibbs

### 2.7 Mixture Models (discrete)
- MT-EDA (`LearnMixtureTrees`): mixture of trees with EM or fitness-proportional weights
- Adaptive and prior-based extensions

### 2.8 Gaussian EDAs (continuous)
- Univariate Gaussian UMDA (`LearnGaussianUnivariate`)
- Full multivariate Gaussian (`LearnGaussianFull`)
- Variance scaling; stagnation avoidance

### 2.9 Gaussian Network EDA (continuous)
- `GaussianNetworkEDA` (`base.py`): GMRF-based continuous EDA
- Comparing Gaussian network to full multivariate Gaussian

### 2.10 Mixture of Gaussians (continuous)
- `LearnMixtureGaussian` / `SampleMixtureGaussian`
- Worked example: Rosenbrock / spacecraft trajectory

### 2.11 Vine Copula EDA (continuous)
- `learning/vine_copula.py` / `sampling/vine_copula.py`
- When to prefer copula models

---

## Phase 3 — EDA Components in Depth

**Goal:** Document each component family with its interface and available implementations.

### 3.1 Seeding Methods
- `RandomInit`: uniform random initialization
- `BiasInit`: biased binary initialization
- `SeedingUnitationConstraint`: fixed-unitation seeding
- `SeedThisPop`: seed with a given population

### 3.2 Selection Methods
- Truncation selection (`TruncationSelection`)
- Tournament selection (`TournamentSelection`)
- Proportional / SUS selection (`ProportionalSelection`, `SUSSelection`)
- Boltzmann selection (`BoltzmannSelection`)
- Non-dominated selection (`NonDominatedSelection`)
- Ranking-based selection (`RankingSelection`)
- Crowding-based selection (`CrowdingSelection`)
- Indicator-based selection (SMS-EMOA style)
- Pareto-front selection (`ParetoFrontSelection`)
- Selection weighting: uniform, proportional, Boltzmann (`selection_weighting`)

### 3.3 Replacement Methods
- Elitist replacement (`ElitistReplacement`)
- Generational replacement

### 3.4 Repairing Methods
- Bounds repairing (`bounds.py`)
- Trigonometric repairing (`trigonometric.py`)
- Unitation repairing (`unitation.py`, `unitation_method.py`)

### 3.5 Mutation Operators
- Bitflip mutation (`bitflip.py`)
- Frequency balance mutation (`frequency_balance*.py`)

### 3.6 Crossover Operators
- Two-point crossover (`two_point.py`)
- Block crossover (`block.py`)
- Transposition (`transposition.py`)

### 3.7 Local Optimization
- Greedy search (`greedy_search.py`, `discrete_greedy_search.py`)
- Simulated annealing (`discrete_simulated_annealing.py`)
- Contiguous block optimization (`contiguous_block_opt.py`)
- SciPy integration (`scipy_local_search.py`)

### 3.8 Stopping Conditions
- `MaxGenerations`
- `MaxGenerationsOrOptimum`

### 3.9 Statistics Tracking
- `Statistics` object: best, mean, std, worst per generation
- `Cache` object: populations, fitness values, models, selected populations
- `tracker.py` and `population_stats.py`

### 3.10 Inference Methods
- K-MAP inference (`kmpc.py`)
- MAP inference (`map_inference.py`)
- Partial sampling (`sampling/partial.py`)

---

## Phase 4 — Test Functions and Problem Classes

**Goal:** Catalog all built-in benchmark functions.

### 4.1 Discrete Functions
- OneMax (`onemax.py`)
- Deceptive trap functions (`deceptive.py`, `trap.py`)
- Additively decomposable functions (`additive_decomposable.py`)
- NK landscape (`nk_landscape.py`)
- Integer functions (`integer_functions.py`)
- Contiguous block functions (`contiguous_block.py`)
- Ising model (`ising.py`)
- HP protein model (`hp_protein.py`)
- SAT (`sat.py`)
- UBQP (`ubqp.py`)
- Multi-objective discrete functions (`multiobjective.py`)

### 4.2 Continuous Functions
- Standard benchmarks: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank (`benchmarks.py`)
- AB off-lattice protein model (`ab_protein.py`)
- GNBG instances (`GNBG_Instances.Python-main/`)

---

## Phase 5 — Multi-Objective Optimization

**Goal:** Document the full multi-objective infrastructure.

### 5.1 Multi-Objective Problem Definition
- Fitness functions returning vectors
- Pareto dominance (`multiobjective/dominance.py`)

### 5.2 Multi-Objective Selection Methods
- Non-dominated sorting + crowding (NSGA-II style)
- Indicator-based (SMS-EMOA style, `indicators.py`)
- Decomposition approach (MOEA/D): `multiobjective/moead.py`

### 5.3 Multi-Objective EDA Pipelines
- Assembling a multi-objective EDA with `EDAComponents`
- Scalarization methods (`scalarization.py`)
- Reference-point weights (`weights.py`)
- Archive management (`archive.py`)
- Example: multi-objective decomposable NK landscape

### 5.4 Pareto Front Approximation and Metrics
- Extracting Pareto sets from the cache
- Hypervolume, IGD, coverage metrics
- Visualization: parallel coordinates

---

## Phase 6 — Injecting A-Priori Knowledge

**Goal:** Document mechanisms for incorporating problem structure.

### 6.1 Fixed Factorizations (FDA)
- Specifying a known clique structure via `cliques` parameter in `FDA`
- Example: Markov-chain structure for HP protein folding

### 6.2 Interaction Matrices
- `interaction_matrix` parameter in `TreeEDAR`, `MNFDAR`, `MNFDAGR`
- Encoding domain knowledge as allowed/forbidden dependencies

### 6.3 Biased Initialization
- `BiasInit`: setting variable-wise priors for initial population
- `SeedThisPop`: seeding with a custom starting population

### 6.4 Constrained EDAs (Unitation Constraints)
- `CUMDA` and `CFDA`: enforcing fixed Hamming weight
- `SeedingUnitationConstraint`, `SampleCUMDA`, `SampleCFDA`

### 6.5 Selection Weighting
- Boltzmann and proportional weights in `EDA.__init__`
  (`selection_weighting`, `weighting_beta`)
- Effect on model accuracy vs. selection pressure

---

## Phase 7 — Knowledge Extraction and Visualization

**Goal:** Document the knowledge extraction module in full.

### 7.1 Dependency Analysis
- Computing correlation matrices from selected populations
- Learning Bayesian / Gaussian networks a posteriori (`dependency_analysis.py`)
- Mutual information between variables (`compute_mutual_information`)

### 7.2 Network Measures
- Edge frequency matrices across runs and generations
- Detecting frequent substructures (`network_measures.py`)

### 7.3 Network Visualizations
- Frequency matrix heat-maps
- Parallel coordinate view of edge presence
- Dendrogram clustering of edges (`network_visualizations.py`)
- Model visualization utilities (`model_visualizations.py`)

### 7.4 Fitness-Related Measures
- Response to selection $R(t)$, amount of selection $S(t)$, realized heritability $b(t)$
- Fitness distribution histograms (`fitness_measures.py`)

### 7.5 Continuous Variable Analysis
- Visualization of Gaussian model parameters across generations
- Vine copula structure visualization (`vine_analysis.py`)
- Gaussian network edge evolution (`gaussian_networks.py`)

### 7.6 EDA Strategy Comparison
- `eda_strategies.py`: running multiple EDA configurations and comparing
  convergence curves, final fitness distributions, and model complexity

---

## Phase 8 — Advanced Topics and Extensions

**Goal:** Describe advanced use cases.

### 8.1 Implementing Custom Components
- Subclassing `LearningMethod`, `SamplingMethod`, `SelectionMethod`, etc.
- Example: custom problem-specific learning method

### 8.2 Function Approximation with EDA Models
- Using the probabilistic model as a surrogate for fitness
- Correlation between model probabilities and fitness
- k-MPE configurations as candidate solutions

### 8.3 Combining EDAs with Local Search
- Plugging in `GreedySearch` or `DiscreteSimulatedAnnealing`
- Crossover hybridization

### 8.4 Parallelization and Large-Scale Use
- Population-level parallelism: evaluation loop can be parallelized
- SLURM-based experimental pipelines (multi-seed, multi-algorithm)

### 8.5 Comparing pateda to MATEDA-2.0
- Key architectural differences: Matlab → Python, eval() → abstract classes
- Equivalent function names and new capabilities

---

## Appendices (ongoing)

- **A.** Algorithm identifier table (matches CLAUDE.md naming convention)
- **B.** Parameter reference for all algorithm classes
- **C.** Full `EDAComponents` API reference
- **D.** pateda.bib: complete bibliography

---

## Completion Timeline (suggested)

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Introduction, installation, generic EDA, plug-and-play classes | **Draft** |
| 2 | Probabilistic model families, algorithm descriptions | Planned |
| 3 | Component API reference | Planned |
| 4 | Benchmark functions | Planned |
| 5 | Multi-objective infrastructure | Planned |
| 6 | A-priori knowledge injection | Planned |
| 7 | Knowledge extraction module | Planned |
| 8 | Advanced topics | Planned |
| App | Appendices | Planned |
