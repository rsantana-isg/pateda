# Knowledge Extraction from EDAs — Methods and Extensions

This document describes the knowledge-extraction methods available in
`pateda.knowledge_extraction`, with emphasis on the **network-theoretic
analysis of the graphical structures** of the probabilistic models learned by
Estimation of Distribution Algorithms (EDAs).  It also lists possible
extensions.

The methods follow two reference papers (in `paper/knowledge_extraction/`):

* **[NM]** R. Santana, R. Armañanzas, C. Bielza, P. Larrañaga.
  *Network measures for information extraction in evolutionary algorithms.*
  International Journal of Computational Intelligence Systems, 6(6):1163–1188,
  2013.
* **[MM]** R. Santana, C. Bielza, J. A. Lozano, P. Larrañaga.
  *Mining probabilistic models learned by EDAs in the optimization of
  multi-objective problems.* GECCO-2009, pp. 445–452.

---

## 1. Motivation

Beyond returning a solution, an EDA learns, at every generation, a probabilistic
graphical model (PGM) of the *selected* population.  The **graphical component**
of these models (the dependency structure) encodes information about the problem
and about the behaviour of the algorithm.  Two analysis stages are proposed
([NM], Algorithm 1; [MM], Section 3):

1. **Structure extraction** — map each learned model to a graph (adjacency
   matrix).  Bayesian networks give directed graphs; tree models give directed
   trees; factorized / Markov-network models give the undirected *interaction
   graph* induced by their cliques.
2. **Mining / measurement** — compute local and global network measures, motif
   statistics, frequency matrices, etc., and visualize their evolution.

These descriptors can be used to study problem difficulty, classify problem
instances, compare variation operators / learning strategies, and predict
algorithm behaviour.

---

## 2. Module map

| Module | Purpose |
|--------|---------|
| `fitness_measures` | Selection-response measures (response to selection, amount of selection, realized heritability, objective distribution, fitness evolution). |
| `dependency_analysis` | *A posteriori* dependency learning from populations: correlation matrices, mutual information, Bayesian / Gaussian network learning, structure scores. |
| `model_visualizations` | Multidimensional visualizations of structures: **dendrograms** and **glyphs** (star, circle, box, Chernoff). |
| `eda_strategies` | Per-generation extraction for the different EDA families (BN evolution, Gaussian-parameter evolution, probability-distribution evolution), comprehensive reports, run comparison. |
| **`network_measures`** | **Network-theoretic measures computed from the graphical structures of the learned PGMs** (the core of [NM]). |
| **`network_visualizations`** | Plots for the evolution of the network measures and the structure-mining artefacts ([NM] Figs. 2,10–13; [MM] Figs. 1–8). |
| **`gaussian_networks`** *(new)* | **Interaction networks extracted from the covariance / precision matrix of Gaussian models** learned by continuous EDAs; combinable with Bayesian-network networks. |
| **`vine_analysis`** *(new)* | **Structure and parameter analysis of the vine-copula models** learned by continuous EDAs (first-tree network, family composition, Kendall's-τ per tree, truncation). |
| **`continuous_visualizations`** *(new)* | Plots specific to the continuous case (Gaussian parameter / partial-correlation evolution, precision heat-maps, partial-correlation and vine first-tree networks, family composition, τ-by-tree, network comparison). |

The papers referenced by the continuous modules (in `paper/knowledge_extraction/`):

* **[SIC]** A. S. Sundaramoorthy et al., "Sparse Inverse Covariance Estimation
  for Causal Inference in Process Data Analytics", IEEE TCST 30(3), 2022.
* **[GL]** J. Friedman, T. Hastie, R. Tibshirani, "Sparse inverse covariance
  estimation with the graphical lasso", Biostatistics 9(3), 2008.
* **[SLGM]** M. Drton, M. H. Maathuis, "Structure Learning in Graphical
  Modeling", Annual Review of Statistics, 2017.
* **[VC]** D. Carrera, R. Santana, J. A. Lozano, "Vine copula classifiers for
  the mind reading problem", Progress in Artificial Intelligence 5, 2016; and
  the regular-vine "sand dunes on Mars" classification paper.
* **[PCC]** K. Aas, C. Czado, A. Frigessi, H. Bakken, "Pair-copula constructions
  of multiple dependence", 2009; T. Bedford, R. M. Cooke, "Vines", 2002.

---

## 3. Existing methods (verified)

### 3.1 Fitness measures (`fitness_measures`)
* `response_to_selection`, `amount_of_selection`, `realized_heritability` —
  classic selection-response statistics.
* `compute_objective_distribution`, `analyze_fitness_evolution`.

### 3.2 Dependency analysis (`dependency_analysis`)
* `compute_correlation_matrix`, `compute_mutual_information`.
* `learn_bayesian_network`, `learn_gaussian_network` — *a posteriori*
  structure learning from a population (the "structure extraction" step for GA
  populations or for re-deriving a model from data).
* `analyze_variable_dependencies`, `compute_structure_score`,
  `compute_local_score`, `has_cycle`.

### 3.3 Model visualizations (`model_visualizations`)
* `view_dendrogram_structure` — hierarchical clustering of edges by
  co-occurrence ([MM] Fig. 6).
* `view_glyph_structure` + `draw_{star,circle,box,chernoff}_glyph` — glyph
  representation of structure exemplars ([MM] Fig. 8).

### 3.4 EDA strategies (`eda_strategies`)
* `extract_bayesian_network_evolution` — per-generation adjacency matrices,
  **edge-frequency matrix**, stable / emerging / disappearing edges.
* `extract_gaussian_parameters_evolution`,
  `extract_probability_distribution_evolution`.
* `generate_comprehensive_report`, `compare_eda_runs`.

> The pre-existing code covered fitness measures, dependency learning, edge
> frequencies and the dendrogram/glyph visualizations, but **did not compute the
> topological network measures** of [NM].  This was the main gap.

---

## 4. New: network measures (`network_measures`)

### 4.1 Structure extraction
* `model_to_adjacency(model, n_vars)` → `(adjacency, is_directed)`.
  Handles `BayesianNetworkModel` (directed adjacency), `TreeModel`
  (`tree_to_adjacency`, directed parent→child), `FactorizedModel` /
  `MarkovNetworkModel` (`cliques_to_adjacency`, undirected interaction graph),
  `GaussianModel` and raw numpy arrays.
* `to_networkx(adjacency, directed)` — build a `networkx` (di)graph.

### 4.2 The collection of measures ([NM], Section 2.3)
`compute_network_measures(model, ...)` returns all of the following.  The first
column gives the descriptor name used in [NM, Table 1].

| [NM] name | Function / key | Notes |
|-----------|----------------|-------|
| `dagdif` | `dagdif`, key `dagdif` | # different arcs between consecutive generations (contrastive). |
| `Ndensity` | `network_density`, key `density` | edges / (n²−n). |
| `indegree` / `outdegree` | `degree_statistics` | mean in/out degree; full degree distribution. |
| `betw. cent.` | `betweenness` | mean vertex and **edge** betweenness centrality. |
| `pair dist.` | `distance_measures` | mean pairwise distance (disconnected → large value). |
| `reachability` | `distance_measures` | mean reachability. |
| `clust. coef.` | `clustering_coefficient` | mean clustering (Fagiolo for digraphs). |
| `shortcut prob.` | `shortcut_probability` | fraction of arcs with range *g<sub>ij</sub> > 2*. |
| `n. motifs Z=3` | `motif_number(G,3)` | from the directed **triad census**. |
| `n. motifs Z=4` | `motif_number(G,4)` | connected 4-node induced sub-graphs. |
| `max. modularity` | `max_modularity` | Louvain modularity on the skeleton. |
| `vert. part. coef.` | `participation_coefficient` | Guimerà–Amaral participation coefficient. |

Additional classic measures (Sections 2.1–2.2 of [NM] and the feature list of
[MM]): characteristic path length, **radius** / **diameter**
(`eccentricity_measures`), degree **assortativity** (`assortativity`), number /
average size of **connected components** (`connected_components_stats`),
**maximum clique** size and number of maximal cliques (`clique_stats`).

Motif tooling: `triad_census` (16 directed triad types), `motif_number`,
`motif_spectrum` (frequency per isomorphism class via a Weisfeiler–Lehman hash),
`_CONNECTED_TRIADS` (the 13 connected triad classes of [NM] Fig. 1).

### 4.3 Evolution / structure mining
* `compute_measures_evolution(models)` → `{per_generation, series, adjacencies}`,
  where `series[name]` is a per-generation array for each scalar measure
  (`SCALAR_MEASURE_KEYS`).
* `edge_frequency_matrix(adjacencies)` — arc coincidence/frequency matrix
  ([MM] frequency matrices; [NM] Fig. 10).
* `aggregate_degree_distribution(adjacencies)` — average number of vertices per
  degree ([MM] Fig. 2).
* `triad_census_series(adjacencies)` — per-generation triad frequencies
  ([NM] Fig. 13).

---

## 5. New: network visualizations (`network_visualizations`)

| Function | Reproduces |
|----------|-----------|
| `plot_measures_evolution` | per-generation curves of several measures ([NM] Fig. 12). |
| `compare_measure_evolution`, `compare_measures_grid` | the **same measure(s) across several EDAs** ([NM] Figs. 11–13). |
| `plot_edge_frequency_matrix` | arc-frequency heat-map ([NM] Fig. 10; [MM] frequency matrices). |
| `plot_degree_distribution` | average degree distribution ([MM] Fig. 2). |
| `plot_motif_evolution` | triad (Z=3) motif frequencies over generations ([NM] Fig. 13). |
| `plot_network_snapshots` | the learned network drawn at selected generations. |
| `plot_betweenness_two_approaches` | the *vertex* vs *generation* views of betweenness ([NM] Fig. 11). |

These complement the existing dendrogram / glyph / (parallel-coordinate)
visualizations in `model_visualizations`.

---

## 6. Demonstration script

`scripts/Test_Knowledge_Extraction.py` runs one EDA of each class on the
**Deceptive3** problem (whose variables interact in known blocks of three),
**caching the structure and parameters of every generation**, and then renders
the network analysis:

* **Factorization-based**: MN-FDA (`FactorizedModel`).
* **Bayesian-network** (three different learners): EBNA (BIC + local search),
  BOA, AffEDA (affinity-propagation clique discovery) — compared against each
  other with `compare_measures_grid`.
* **Tree-based**: Tree-EDA.

It writes, per EDA, a `*_measures.csv` (per-generation measures), a
`*_adjacencies.npz` (per-generation structure), and the figures above, plus a
`bn_learners_comparison.png` and a global `summary.csv`.

```bash
python scripts/Test_Knowledge_Extraction.py                 # defaults
python scripts/Test_Knowledge_Extraction.py --quick         # fast smoke run
python scripts/Test_Knowledge_Extraction.py --n-vars 18 --pop-size 700 --n-gen 15
```

Typical observations: MN-FDA and AffEDA recover order-3 cliques
(`max_clique_size = 3–4`), EBNA / Tree-EDA learn sparser, tree-like skeletons
(`clustering ≈ 0`), and every algorithm's structure first grows and then
collapses as the population converges (visible in `n_edges` and `dagdif`).

---

## 7. Continuous EDAs

The discrete EDAs above learn an explicit dependency graph (Bayesian network,
tree or clique factorization).  Continuous EDAs instead learn **Gaussian
models** (mean + covariance) or **vine copulas**, from which the interaction
structure has to be *recovered*.  Two new modules do this and feed the result to
the same `network_measures` machinery, so that continuous and discrete EDAs can
be analysed — and compared — within one framework.

### 7.1 Gaussian interaction networks (`gaussian_networks`)

For `x ~ N(mu, Sigma)`, the **inverse covariance** (precision) matrix
`Theta = Sigma^{-1}` is the Gaussian graphical model: `Theta_{ij}=0` **iff**
`x_i` and `x_j` are conditionally independent given the rest ([SIC], Prop. 1).
The precision support is therefore the *continuous analogue of the Bayesian
network* of discrete EDAs.

| Function | What it does | Refs |
|----------|--------------|------|
| `extract_gaussian_parameters(model)` | robustly read `(mean, cov)` from a `GaussianModel` / GMRF dict / attribute object | — |
| `covariance_to_precision(cov)` | `Theta = Sigma^{-1}` (ridge-regularised) | [SIC] |
| `partial_correlation_matrix(precision)` | `rho_{ij} = -Theta_{ij}/sqrt(Theta_{ii}Theta_{jj})` | [SLGM] |
| `glasso_precision(cov, alpha)` | sparse precision via the **graphical lasso** | [GL] |
| `gaussian_interaction_network(model, method)` | undirected GGM network (methods `partial_correlation` / `precision` / `glasso` / `correlation`) | [SIC],[GL] |
| `orient_edges_likelihood_score(adj, cov)` | orient the undirected GGM into a **directed (causal) graph** with the Gaussian likelihood score (comparable to a BN) | [SIC] §II-C |
| `compare_networks(a, b)` / `combine_networks(a, b, mode)` | **combine a Gaussian network with a Bayesian-network (or known) network** — common/unique edges, Jaccard, union/intersection/agreement | [NM] |
| `gaussian_network_evolution(models, method)` | the per-generation network sequence (→ `compute_measures_evolution`) | [NM] |

The output adjacency matrices are directly accepted by `compute_network_measures`
and `network_visualizations`, so all the topological measures of Section 4 apply
to Gaussian EDAs as well.

### 7.2 Vine-copula analysis (`vine_analysis`)

A vine copula factorizes the dependence into a sequence of `n-1` nested trees of
bivariate *pair-copulas* ([PCC]).  The first tree `T_1` holds the strongest
*unconditional* dependencies; higher trees hold conditional ones.  Each edge
carries a copula **family** (Gaussian, Clayton, Gumbel, Frank, t, independence,
…) and a dependence parameter summarised by **Kendall's τ** ([VC]).  These
analyses are meaningful **only when the structure and/or families are learned
during the search** (e.g. the auto `VineEDA`).

| Function | What it does |
|----------|--------------|
| `vine_structure(model)` | full structure: per-(tree, edge) conditioned/conditioning sets, family, τ, parameters, rotation (decoded from the `pyvinecopulib` R-vine matrix) |
| `first_tree_network(model, tau_threshold)` | `T_1` interaction network (strongest unconditional dependencies) — combinable with the Gaussian / BN networks |
| `family_composition(model)` | counts/frequencies of pair-copula families (overall and per tree): which *types* of dependence are selected |
| `tau_by_tree(model)` | mean/max `|τ|` per tree (dependence strength and its decay with tree level) |
| `effective_truncation(model)` | last tree with a non-independence pair-copula (model complexity) |
| `analyze_vine(model)` / `vine_evolution(models)` | one-call summary and the per-generation evolution (first-tree network, family frequencies, τ, truncation) |

`continuous_visualizations` adds the matching plots: `plot_gaussian_parameter_evolution`,
`plot_precision_heatmap`, `plot_partial_correlation_network`,
`plot_network_comparison` (Gaussian vs BN/vine), `plot_vine_first_tree`,
`plot_family_composition`, `plot_tau_by_tree`, `plot_vine_evolution`.

> The existing `eda_strategies.extract_gaussian_parameters_evolution` was also
> fixed to read the pateda `GaussianModel` (and GMRF) parameter layout.

### 7.3 Demonstration script

`scripts/Test_Continuous_EDA_Knowledge_Extraction.py` runs continuous EDAs on
the (negative) Rosenbrock function — whose terms couple *consecutive* variables,
giving a known chain interaction structure — caching the model of every
generation and analysing the learned structures:

* **Gaussian EDA**: per-generation Gaussian interaction networks (precision /
  partial correlation), network-measure evolution, parameter evolution, and a
  comparison/combination with the known chain structure.
* **Vine EDA (auto)**: first-tree network, pair-copula **family composition**,
  Kendall-τ-by-tree, effective truncation, and their evolution.
* **C-vine EDA**: vine structure learned with a fixed Gaussian family.
* **Combination**: Gaussian vs Vine-first-tree networks are compared and merged;
  the union of the two recovers most of the known chain (high Jaccard),
  illustrating how the continuous and copula views can be combined.

```bash
python scripts/Test_Continuous_EDA_Knowledge_Extraction.py
python scripts/Test_Continuous_EDA_Knowledge_Extraction.py --quick
python scripts/Test_Continuous_EDA_Knowledge_Extraction.py --n-vars 10 --pop-size 600 --n-gen 20
```

(`pateda.algorithms.continuous._BaseVineEDA.run` gained a `cache_models=True`
option so the vine EDAs expose their per-generation models for this analysis.)

---

## 8. Possible extensions

**Measures**
* Directed-specific modularity (Newman spectral generalized to digraphs, as in
  [NM]); the current implementation uses Louvain on the undirected skeleton.
* Weighted networks: use edge weights derived from CPD parameters / mutual
  information (the "Another possible development" of [NM] §5) instead of 0/1
  adjacencies.
* Full motif **spectra** for Z=4 with explicit class labelling (currently a
  hash-keyed spectrum + total motif number).
* Overlapping community detection (CPM / clique percolation) for problem
  decomposition.

**Mining / learning ([NM], Algorithm 1, steps 3–4)**
* Map the per-generation measure vectors to problem characteristics with a
  classifier (multinomial logistic regression / multivariate Gaussian) to
  **predict difficulty, convergence or the number of optima**, as in [NM].
* Univariate-probability + vertex-measure **regression** to predict the dynamics
  of subsequent generations ([NM], Section 3.3.3).
* Affinity-propagation clustering of the set of learned structures to obtain
  *exemplar* structures and their glyphs ([MM], Fig. 8).

**Spurious vs. original structure** ([MM], Figs. 4–5)
* Given a known interaction graph, classify learned edges/triples as *spurious*
  or *original*, and compare expected vs. observed appearance probabilities of
  triples — directly supported by `edge_frequency_matrix` and the motif tools.

**Visualization**
* Parallel-coordinate view of edges × generations ([MM], Fig. 7).
* Choropleth / contour frequency maps and per-class average degree distributions
  for multi-instance studies.

**Continuous models**
* Causal orientation beyond the bivariate Gaussian likelihood score (full
  two-step SIC + likelihood-score causal discovery of [SIC]) to obtain directed
  Gaussian networks that can be merged edge-by-edge with directed BNs.
* `GraphicalLassoCV` / stability selection to choose the sparsity level
  automatically instead of a fixed `alpha`/threshold.
* Weighted vine first-tree networks (use `|τ|` as edge weight) and conditional
  (higher-tree) interaction networks, not just the first tree.
* Tail-dependence summaries from the selected copula families (lower/upper tail
  coefficients) to characterise the search landscape ([VC], [PCC]).

**Integration**
* Online use: bias model building from the measures (e.g. discourage spurious
  shortcuts); transfer structural knowledge between related instances.
* Extend the extraction to the deep-generative EDAs of `pateda_nn` by deriving a
  dependency graph from the learned generative models.
* Combine the discrete (BN/tree/FDA), Gaussian and vine networks of mixed
  problems into a single multi-view structural descriptor.
