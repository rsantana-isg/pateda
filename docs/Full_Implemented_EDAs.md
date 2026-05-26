# Full Implemented EDAs in pateda

This document describes all Estimation of Distribution Algorithms (EDAs) available through the plug-and-play wrapper API introduced in `pateda/algorithms/`. Each entry gives the algorithm name, type, a brief description, key parameters, a usage example, and the source learning file.

---

## Discrete EDAs

### UMDA — Univariate Marginal Distribution Algorithm

**Type:** Discrete

**Description:** The simplest EDA, introduced by Mühlenbein & Paass (1996). UMDA assumes complete independence between variables and models each variable with its own univariate marginal distribution estimated by frequency counting from the selected population. It serves as the baseline against which all other EDAs are compared.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing to prevent zero probabilities.

**Source:** `learning/umda.py` (LearnUMDA), `sampling/fda.py` (SampleFDA)

**Example:**
```python
from pateda import UMDA
import numpy as np

def onemax(x): return float(np.sum(x))

alg = UMDA(n_vars=20, cardinality=2, fitness_func=onemax,
           pop_size=200, n_gen=50, random_seed=42)
stats, cache = alg.run()
print("Best:", stats.best_fitness_overall)
```

---

### BMDA — Bivariate Marginal Distribution Algorithm

**Type:** Discrete

**Description:** BMDA extends UMDA by detecting pairwise variable dependencies using chi-square independence tests. It builds a forest of bivariate marginals, capturing first-order interactions while remaining computationally efficient.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/bmda.py` (LearnBMDA)

**Example:**
```python
from pateda import BMDA

alg = BMDA(n_vars=20, cardinality=2, fitness_func=onemax,
           pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### TreeEDA — Tree-structured EDA

**Type:** Discrete

**Description:** Learns a maximum-weight spanning tree of the pairwise mutual information matrix. The tree captures the most important pairwise dependencies and samples using a probabilistic logic sampling (PLS) approach that respects the tree structure.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/tree.py` (LearnTreeModel)

**Example:**
```python
from pateda import TreeEDA

alg = TreeEDA(n_vars=20, cardinality=2, fitness_func=onemax,
              pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### TreeEDAR — Tree-EDA with Random Root

**Type:** Discrete

**Description:** Like TreeEDA but uses a randomized root selection strategy during structure learning, which can improve exploration by varying the dependency ordering across generations.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/tree_r.py` (LearnTreeModelR)

**Example:**
```python
from pateda import TreeEDAR

alg = TreeEDAR(n_vars=20, cardinality=2, fitness_func=onemax,
               pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### MIMIC — Mutual Information Maximizing Input Clustering

**Type:** Discrete

**Description:** De Bonet et al. (1997). MIMIC learns a chain-structured model where the variable ordering maximizes mutual information between consecutive variables. Each variable is conditioned on the preceding variable in the chain.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/mimic.py` (LearnMIMIC)

**Example:**
```python
from pateda import MIMIC

alg = MIMIC(n_vars=20, cardinality=2, fitness_func=onemax,
            pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### PBIL — Population-Based Incremental Learning

**Type:** Discrete

**Description:** Baluja (1994). PBIL maintains a probability vector that is updated incrementally using a learning rate (alpha). Unlike UMDA, the probability vector is not replaced each generation but blended with the current population's statistics, providing smoother transitions.

**Key parameters:**
- `alpha` (float, default 0.1): Learning rate for probability update.

**Source:** `learning/pbil.py` (LearnPBIL)

**Example:**
```python
from pateda import PBIL

alg = PBIL(n_vars=20, cardinality=2, fitness_func=onemax,
           pop_size=200, n_gen=50, alpha=0.1, random_seed=42)
stats, _ = alg.run()
```

---

### EBNA — Estimation of Bayesian Network Algorithm

**Type:** Discrete

**Description:** Etxeberria & Larrañaga (1999). EBNA learns a Bayesian network structure using a score-and-search approach guided by the BIC criterion. It captures arbitrary directed acyclic dependency structures and samples using ancestral (topological) sampling.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/ebna.py` (LearnEBNA), `sampling/bayesian_network.py` (SampleBayesianNetwork)

**Example:**
```python
from pateda import EBNA

alg = EBNA(n_vars=20, cardinality=2, fitness_func=onemax,
           pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### BOA — Bayesian Optimization Algorithm

**Type:** Discrete

**Description:** Pelikan, Goldberg & Cantú-Paz (1999). BOA learns a Bayesian network via greedy structure search with MDL/BIC scoring. It is one of the most widely studied EDAs and serves as the foundation for hierarchical BOA (hBOA).

**Key parameters:** None beyond the standard set.

**Source:** `learning/boa.py` (LearnBOA), `sampling/bayesian_network.py` (SampleBayesianNetwork)

**Example:**
```python
from pateda import BOA

alg = BOA(n_vars=20, cardinality=2, fitness_func=onemax,
          pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### AffEDA — Affinity-based EDA

**Type:** Discrete

**Description:** Uses affinity propagation clustering on the pairwise mutual information matrix to automatically discover groups of related variables. Each cluster becomes a clique in a factorized distribution model.

**Key parameters:**
- `max_clique_size` (int, default 5): Maximum number of variables per clique.
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/affinity.py` (LearnAffinityFactorization)

**Example:**
```python
from pateda import AffEDA

alg = AffEDA(n_vars=20, cardinality=2, fitness_func=onemax,
             pop_size=200, n_gen=50, max_clique_size=5, random_seed=42)
stats, _ = alg.run()
```

---

### MKEDA — k-order Markov Chain EDA

**Type:** Discrete

**Description:** Models variables as a k-order Markov chain following a fixed variable ordering. Each variable is conditioned on the k preceding variables, allowing capture of sequential dependencies while maintaining a linear number of parameters.

**Key parameters:**
- `k` (int, default 1): Markov order.
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/markov.py` (LearnMarkovChain), `sampling/markov.py` (SampleMarkovChain)

**Example:**
```python
from pateda import MKEDA

alg = MKEDA(n_vars=20, cardinality=2, fitness_func=onemax,
            pop_size=200, n_gen=50, k=1, random_seed=42)
stats, _ = alg.run()
```

---

### MTED — Mixture of Trees EDA

**Type:** Discrete

**Description:** Santana, Ochoa & Soto (2001). MTED combines multiple tree-structured models with mixture weights. This enables modelling of multimodal distributions, where different trees capture different solution patterns (building blocks).

**Key parameters:**
- `n_trees` (int, default 5): Number of tree components in the mixture.
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/mixture_trees.py` (LearnMixtureTrees), `sampling/mixture_trees.py` (SampleMixtureTrees)

**Example:**
```python
from pateda import MTED

alg = MTED(n_vars=20, cardinality=2, fitness_func=onemax,
           pop_size=200, n_gen=50, n_trees=5, random_seed=42)
stats, _ = alg.run()
```

---

### MNFDA — Markov Network Factorized Distribution Algorithm

**Type:** Discrete

**Description:** Santana (2013). MN-FDA learns a Markov network structure using pairwise chi-square independence tests, then finds maximal cliques to define the factorized distribution. Sampling uses FDA (probabilistic logic sampling) via the junction tree.

**Key parameters:**
- `max_clique_size` (int, default 3): Maximum clique size in the Markov network.

**Source:** `learning/mnfda.py` (LearnMNFDA), `sampling/fda.py` (SampleFDA)

**Example:**
```python
from pateda import MNFDA

alg = MNFDA(n_vars=20, cardinality=2, fitness_func=onemax,
            pop_size=200, n_gen=50, max_clique_size=3, random_seed=42)
stats, _ = alg.run()
```

---

### MNFDAR — MN-FDA with Random Ordering

**Type:** Discrete

**Description:** Variant of MN-FDA that uses Gibbs sampling with random variable orderings instead of deterministic PLS sampling, improving exploration of the search space.

**Key parameters:**
- `max_clique_size` (int, default 3): Maximum clique size.

**Source:** `learning/mnfda_r.py` (LearnMNFDAR), `sampling/gibbs.py` (SampleGibbs)

**Example:**
```python
from pateda import MNFDAR

alg = MNFDAR(n_vars=20, cardinality=2, fitness_func=onemax,
             pop_size=200, n_gen=50, max_clique_size=3, random_seed=42)
stats, _ = alg.run()
```

---

### MNFDAG — MN-FDA with Augmented Graph

**Type:** Discrete

**Description:** MN-FDAg augments the Markov network structure with additional edges based on mutual information scores before finding maximal cliques. This captures more dependencies at the cost of larger cliques. Sampling uses Gibbs.

**Key parameters:**
- `max_clique_size` (int, default 3): Maximum clique size.

**Source:** `learning/mnfdag.py` (LearnMNFDAG), `sampling/gibbs.py` (SampleGibbs)

**Example:**
```python
from pateda import MNFDAG

alg = MNFDAG(n_vars=20, cardinality=2, fitness_func=onemax,
             pop_size=200, n_gen=50, max_clique_size=3, random_seed=42)
stats, _ = alg.run()
```

---

### MNFDAGR — MN-FDAg with Random Ordering

**Type:** Discrete

**Description:** Variant of MN-FDAg using Gibbs sampling with random variable orderings, combining the augmented graph structure with randomized sampling.

**Key parameters:**
- `max_clique_size` (int, default 3): Maximum clique size.

**Source:** `learning/mnfdag_r.py` (LearnMNFDAGR), `sampling/gibbs.py` (SampleGibbs)

**Example:**
```python
from pateda import MNFDAGR

alg = MNFDAGR(n_vars=20, cardinality=2, fitness_func=onemax,
              pop_size=200, n_gen=50, max_clique_size=3, random_seed=42)
stats, _ = alg.run()
```

---

### MOA — Markovianity-Based Optimization Algorithm

**Type:** Discrete

**Description:** Santana (2013). MOA learns local Markov neighborhoods for each variable: for each variable Xi, it identifies the k nearest neighbors (by mutual information) and learns P(Xi | neighbors). Gibbs sampling then exploits these local conditional distributions.

**Key parameters:**
- `k_neighbors` (int, default 3): Number of Markov neighbors per variable.

**Source:** `learning/moa.py` (LearnMOA), `sampling/gibbs.py` (SampleGibbs)

**Example:**
```python
from pateda import MOA

alg = MOA(n_vars=20, cardinality=2, fitness_func=onemax,
          pop_size=200, n_gen=50, k_neighbors=3, random_seed=42)
stats, _ = alg.run()
```

---

### CUMDA — Constrained UMDA

**Type:** Discrete (binary only, unitation constraint)

**Description:** Santana & Ochoa. CUMDA enforces a fixed number of ones in every solution using Stochastic Universal Sampling (SUS). It is equivalent to UMDA but restricted to binary vectors with a fixed unitation value. Requires specifying `n_ones`.

**Key parameters:**
- `n_ones` (int, required): Exact number of ones in each solution.
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.

**Source:** `learning/cumda.py` (LearnCUMDA), `sampling/cumda.py` (SampleCUMDA)

**Example:**
```python
from pateda import CUMDA

def knapsack(x): return float(np.sum(x * weights))  # example

alg = CUMDA(n_vars=20, cardinality=2, fitness_func=knapsack,
            n_ones=10, pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### FDA — Factorized Distribution Algorithm

**Type:** Discrete

**Description:** Mühlenbein, Mahnig & Rodriguez (1999). FDA represents the joint distribution as a product of factors (cliques). The default wrapper uses a univariate factorization (equivalent to UMDA but routed through the general factorization machinery), but it accepts an explicit clique structure matrix when a problem-specific factorization is known (e.g. from a junction tree). Each row of the clique matrix has the MATEDA form `[n_overlap, n_new, overlap_indices..., new_indices...]`.

**Key parameters:**
- `cliques` (np.ndarray or None, default None): Optional clique structure matrix; `None` means univariate.
- `alpha` (float, default 1.0): Laplace smoothing pseudo-count for the probability tables; the default of 1.0 matches the original MATEDA-2.0 implementation, set to 0.0 to disable.

**Source:** `learning/fda.py` (LearnFDA), `sampling/fda.py` (SampleFDA)

**Example:**
```python
from pateda import FDA

alg = FDA(n_vars=20, cardinality=2, fitness_func=onemax,
          pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### BSC — Bisection (fitness-weighted univariate EDA)

**Type:** Discrete (originally listed as *Theoretical* in `Implemented_EDAs.md`)

**Description:** Inza et al. (2000); MATEDA-1.0. BSC estimates each variable's marginal using fitness-weighted counts rather than plain frequencies:
`P(X_i = k) = sum(fitness of individuals with X_i = k) / sum(all fitness)`.
The probability mass is biased toward values that appear in high-fitness individuals, which can accelerate convergence at the price of reduced diversity. Sampling reuses `SampleFDA` because the learned object is a `FactorizedModel` with univariate cliques.

**Key parameters:**
- `alpha` (float, default 0.0): Laplace smoothing pseudo-count.
- `normalize_fitness` (bool, default True): Min-max normalize the fitness before weighting.

**Source:** `learning/bsc.py` (LearnBSC), `sampling/fda.py` (SampleFDA)

**Example:**
```python
from pateda import BSC

alg = BSC(n_vars=20, cardinality=2, fitness_func=onemax,
          pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### CFDA — Constrained FDA

**Type:** Discrete (binary only, unitation constraint)

**Description:** CFDA extends FDA to handle unitation constraints using a sample-and-repair strategy: solutions are sampled from the factorized distribution and then repaired to satisfy the exact number of ones constraint.

**Key parameters:**
- `n_ones` (int, required): Exact number of ones in each solution.

**Source:** `learning/cfda.py` (LearnCFDA), `sampling/cfda.py` (SampleCFDA)

**Example:**
```python
from pateda import CFDA

alg = CFDA(n_vars=20, cardinality=2, fitness_func=knapsack,
           n_ones=10, pop_size=200, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

## Continuous EDAs

### GaussianUMDA — Gaussian UMDA

**Type:** Continuous

**Description:** The continuous analogue of UMDA. Each variable is modelled independently with a Gaussian distribution. Mean and standard deviation are estimated from the selected population. Simple and fast, suitable for problems with weak variable interactions.

**Key parameters:** None beyond the standard set (plus `bounds`).

**Source:** `learning/basic_gaussian.py` (LearnGaussianUnivariate), `sampling/basic_gaussian.py` (SampleGaussianUnivariate)

**Example:**
```python
from pateda import GaussianUMDA
import numpy as np

def sphere(x): return -float(np.sum(x**2))

alg = GaussianUMDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
                   pop_size=200, n_gen=100, random_seed=42)
stats, _ = alg.run()
```

---

### GaussianEDA — Full Multivariate Gaussian EDA

**Type:** Continuous

**Description:** Learns a full covariance matrix from the selected population, capturing all pairwise linear dependencies. Equivalent to fitting a multivariate Gaussian to the selected individuals. Suitable for problems with strong linear correlations.

**Key parameters:** None beyond the standard set.

**Source:** `learning/basic_gaussian.py` (LearnGaussianFull), `sampling/basic_gaussian.py` (SampleGaussianFull)

**Example:**
```python
from pateda import GaussianEDA

alg = GaussianEDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
                  pop_size=200, n_gen=100, random_seed=42)
stats, _ = alg.run()
```

---

### MixtureGaussianEDA — Mixture of Gaussians EDA

**Type:** Continuous

**Description:** Clusters the selected population into `n_components` groups using k-means and fits a Gaussian model to each cluster. The mixture enables modelling of multimodal landscapes with multiple separated optima.

**Key parameters:**
- `n_components` (int, default 3): Number of Gaussian components.

**Source:** `learning/mixture_gaussian.py` (LearnMixtureGaussian), `sampling/mixture_gaussian.py` (SampleMixtureGaussian)

**Example:**
```python
from pateda import MixtureGaussianEDA

alg = MixtureGaussianEDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
                         pop_size=200, n_gen=100, n_components=3, random_seed=42)
stats, _ = alg.run()
```

---

### GMRFEDA — Gaussian Markov Random Field EDA

**Type:** Continuous

**Description:** Karshenas et al. (2012). GMRF-EDA learns variable dependencies via regularized regression (LASSO, Elastic Net, etc.), clusters variables into disjoint cliques using affinity propagation, and fits a multivariate Gaussian to each clique. This creates a factorized Gaussian Markov network.

**Key parameters:**
- `regularization` (str, default 'lasso'): Regularization type: 'lasso' | 'elasticnet' | 'lars' | 'lassolars'.

**Source:** `learning/gmrf_eda.py` (learn_gmrf_eda), `sampling/gmrf_eda.py` (sample_gmrf_eda)

**Example:**
```python
from pateda import GMRFEDA

alg = GMRFEDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
              pop_size=200, n_gen=100, regularization='lasso', random_seed=42)
stats, _ = alg.run()
```

---

### VineEDA — Vine Copula EDA

**Type:** Continuous

**Description:** Soto et al. (2011). VineEDA models the joint distribution of variables using vine copulas, which decompose the multivariate distribution into bivariate building blocks (pair-copulas). Captures complex non-linear dependencies. Requires the optional `pyvinecopulib` package.

**Key parameters:** None beyond the standard set.

**Source:** `learning/vine_copula.py` (learn_vine_copula_auto), `sampling/vine_copula.py` (sample_vine_copula)

**Example:**
```python
from pateda import VineEDA  # requires: pip install pyvinecopulib

alg = VineEDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
              pop_size=200, n_gen=100, random_seed=42)
stats, _ = alg.run()
```

---

## Permutation EDAs

All permutation EDAs operate on permutations of `range(n_vars)` and use `PermutationInit` for seeding.

### EHMEDA — Edge Histogram Model EDA

**Type:** Permutation

**Description:** Ceberio et al. (2015). EHM-EDA learns an edge histogram matrix recording transition counts between consecutive items across all selected permutations. New permutations are generated by sequentially sampling next items based on edge probabilities.

**Key parameters:** None beyond the standard set.

**Source:** `learning/histogram.py` (LearnEHM), `sampling/histogram.py` (SampleEHM)

**Example:**
```python
from pateda import EHMEDA
import numpy as np

def lop(perm): return float(np.dot(perm, np.arange(len(perm))))

alg = EHMEDA(n_vars=12, fitness_func=lop,
             pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### NHMEDA — Node Histogram Model EDA

**Type:** Permutation

**Description:** Learns a node histogram matrix recording how often each item appears at each position in the selected permutations. Samples by sequentially selecting items for each position while avoiding repetition.

**Key parameters:** None beyond the standard set.

**Source:** `learning/histogram.py` (LearnNHM), `sampling/histogram.py` (SampleNHM)

**Example:**
```python
from pateda import NHMEDA

alg = NHMEDA(n_vars=12, fitness_func=lop,
             pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### MallowsKendallEDA — Mallows EDA (Kendall)

**Type:** Permutation

**Description:** Ceberio et al. (2011). Fits a Mallows model with the Kendall tau distance. The model is defined by a consensus ranking (central permutation) and a spread parameter theta. Sampling generates v-vectors (Lehmer codes) from the learned probability matrix.

**Key parameters:** None beyond the standard set.

**Source:** `learning/mallows.py` (LearnMallowsKendall), `sampling/mallows.py` (SampleMallowsKendall)

**Example:**
```python
from pateda import MallowsKendallEDA

alg = MallowsKendallEDA(n_vars=12, fitness_func=lop,
                        pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

## Additional EDAs (not in original list)

### BMDA — Bivariate Marginal Distribution Algorithm

**Type:** Discrete

**Description:** Learns pairwise dependencies using chi-square tests and builds a forest of bivariate marginal distributions. See the Discrete section above for the full entry.

**Source:** `learning/bmda.py` (LearnBMDA)

---

### MallowsCayleyEDA — Mallows EDA (Cayley)

**Type:** Permutation

**Description:** Irurozki, Calvo & Lozano (2013). Fits a Mallows model using the Cayley (transposition) distance. The spread is parameterized by the probability of each transposition position being active. Generally faster to learn than the Kendall version.

**Key parameters:** None beyond the standard set.

**Source:** `learning/mallows.py` (LearnMallowsCayley), `sampling/mallows.py` (SampleMallowsCayley)

**Example:**
```python
from pateda import MallowsCayleyEDA

alg = MallowsCayleyEDA(n_vars=12, fitness_func=lop,
                       pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### GMallowsKendallEDA — Generalized Mallows EDA (Kendall)

**Type:** Permutation

**Description:** Fligner & Verducci (1986); Ceberio et al. (2014). Uses position-dependent spread parameters (one theta per position) with the Kendall distance. More flexible than the standard Mallows model, capturing different levels of uncertainty at each position.

**Key parameters:** None beyond the standard set.

**Source:** `learning/mallows.py` (LearnGeneralizedMallowsKendall), `sampling/mallows.py` (SampleGeneralizedMallowsKendall)

**Example:**
```python
from pateda import GMallowsKendallEDA

alg = GMallowsKendallEDA(n_vars=12, fitness_func=lop,
                         pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

### GMallowsCayleyEDA — Generalized Mallows EDA (Cayley)

**Type:** Permutation

**Description:** Ceberio et al. (2014). Generalized Mallows model with position-dependent theta parameters using the Cayley distance. Combines the flexibility of generalized models with the computational advantages of the Cayley metric.

**Key parameters:** None beyond the standard set.

**Source:** `learning/mallows.py` (LearnGeneralizedMallowsCayley), `sampling/mallows.py` (SampleGeneralizedMallowsCayley)

**Example:**
```python
from pateda import GMallowsCayleyEDA

alg = GMallowsCayleyEDA(n_vars=12, fitness_func=lop,
                        pop_size=100, n_gen=50, random_seed=42)
stats, _ = alg.run()
```

---

## Common Parameters for All Wrapper Classes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_vars` | int | required | Number of variables (or permutation length) |
| `cardinality` | int or array | required | Value range per variable (discrete only) |
| `bounds` | tuple or ndarray | required | Search bounds (continuous only) |
| `fitness_func` | callable | required | Fitness function (higher is better) |
| `pop_size` | int | 100 | Population size |
| `n_gen` | int | 50 | Number of generations |
| `selection_ratio` | float | 0.5 | Fraction selected for learning |
| `elitism` | bool | True | Preserve best individual across generations |
| `random_seed` | int or None | None | RNG seed for reproducibility |

---

## Quick Import Reference

```python
# Discrete
from pateda import (UMDA, BMDA, TreeEDA, TreeEDAR, MIMIC, PBIL,
                    EBNA, BOA, AffEDA, MKEDA, MTED,
                    MNFDA, MNFDAR, MNFDAG, MNFDAGR, MOA,
                    CUMDA, CFDA, FDA, BSC)

# Continuous
from pateda import (GaussianUMDA, GaussianEDA, MixtureGaussianEDA,
                    GMRFEDA, VineEDA)

# Permutation
from pateda import (EHMEDA, NHMEDA,
                    MallowsKendallEDA, MallowsCayleyEDA,
                    GMallowsKendallEDA, GMallowsCayleyEDA)
```
