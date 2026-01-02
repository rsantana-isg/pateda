# PATEDA Examples Guide

**Complete guide to running and understanding PATEDA example scripts**

This document provides a comprehensive guide to all example scripts in the `examples/` directory, including those ported from MATLAB ScriptsMateda and new Python-specific examples.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Ported MATLAB Scripts](#ported-matlab-scripts)
3. [New Python Examples](#new-python-examples)
4. [Analysis and Visualization](#analysis-and-visualization)
5. [Script Organization](#script-organization)
6. [EDA Algorithms Reference](#eda-algorithms-reference)
7. [Problem Domains Reference](#problem-domains-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Running an Example

```bash
# From the repository root
python examples/moa_deceptive3.py

# Or use the Python module syntax
python -m examples.moa_deceptive3
```

### Basic Template

All examples follow a similar structure:

```python
from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import StopCriteriaMaxGen

# 1. Define or import fitness function
def fitness_func(x):
    return np.sum(x)  # OneMax

# 2. Configure EDA components
components = EDAComponents(
    seeding=RandomInit(),
    selection=TruncationSelection(proportion=0.5),
    learning=LearnHistogram(),
    sampling=SampleHistogram(pop_size),
    replacement=NoReplacement(),
    stop_condition=StopCriteriaMaxGen(100),
)

# 3. Create and run EDA
eda = EDA(
    pop_size=200,
    n_vars=30,
    fitness_func=fitness_func,
    cardinality=np.full(30, 2),
    components=components,
)

stats, cache = eda.run(verbose=True)
```

---

## Ported MATLAB Scripts

These scripts replicate functionality from `ScriptsMateda/` to ensure compatibility and validation.

### 1. MOA for Deceptive3 Function

**File:** `examples/moa_deceptive3.py`
**MATLAB Source:** `ScriptsMateda/OptimizationScripts/MOA_Deceptive3.m`

**Description:**
Demonstrates MOA (Markovianity Based Optimization Algorithm) with Gibbs sampling on Goldberg's Deceptive3 function.

**Key Features:**
- MOA learning with k=8 neighbors
- Gibbs sampling with Boltzmann temperature
- Elitism replacement (10 best)
- 30 variables (10 blocks of 3 bits)

**Usage:**
```bash
python examples/moa_deceptive3.py
```

**Expected Output:**
- Single run: Best fitness ≈ 10.0 (optimal)
- Comparison: 10 runs showing success rate and convergence statistics

**Configuration:**
- Population size: 500
- Max generations: 100
- Learning: `LearnMOA(k_neighbors=8, threshold_factor=1.5)`
- Sampling: `SampleGibbs(IT=10, temperature=1.0)`

---

### 2. Bayesian Tree EDA with MPE Sampling for Ising Model

**File:** `examples/bayesian_tree_ising_mpe.py`
**MATLAB Source:** `ScriptsMateda/OptimizationScripts/BayesianTree_IsingModel.m`

**Description:**
Bayesian Tree EDA with Most Probable Explanation (MPE/MAP) sampling for the Ising spin glass problem.

**Key Features:**
- Tree-based Bayesian network learning
- MPE/MAP sampling (inserts most probable configuration)
- Ising model energy minimization
- 64 variables (8×8 lattice)

**Usage:**
```bash
python examples/bayesian_tree_ising_mpe.py
```

**Expected Output:**
- Best energy value (optimal: 86.0 for instance 1)
- Comparison showing MPE advantage over standard PLS

**Configuration:**
- Population size: 500
- Max generations: 150 or until optimal found
- Learning: `LearnTreeModel(max_parents=1)`
- Sampling: `SampleInsertMAP(map_method='bp')`

**Note:** If Ising instance files are not available, a random instance is generated.

---

## New Python Examples

These examples demonstrate new combinations and capabilities not present in MATLAB ScriptsMateda.

### 3. Gaussian Network EDA for Ackley Function

**File:** `examples/gaussian_network_ackley.py`
**New:** ✅ Not in MATLAB ScriptsMateda

**Description:**
Demonstrates Gaussian Bayesian Network EDA on the Ackley function, a challenging multimodal continuous benchmark.

**Key Features:**
- Continuous Bayesian network learning
- Multimodal function optimization
- Comparison of Gaussian EDA variants (UMDA, Full, Network)

**Usage:**
```bash
python examples/gaussian_network_ackley.py
```

**Expected Output:**
- Best Ackley value close to 0.0 (global optimum)
- Comparison showing network advantages for dependencies

**Configuration:**
- Problem: 10-dimensional Ackley
- Bounds: [-5, 5] per dimension
- Learning: `LearnGaussianNetwork(max_parents=2)`
- Sampling: `SampleGaussianNetwork(sampling_noise=0.1)`

---

### 4. EBNA for NK Landscape

**File:** `examples/ebna_nk_landscape.py`
**New:** ✅ Extended NK Landscape analysis

**Description:**
Systematic study of EBNA performance on NK landscapes with varying epistasis (K parameter).

**Key Features:**
- NK landscape with tunable epistasis
- EBNA structure learning captures interactions
- Performance vs epistasis analysis
- EBNA vs UMDA comparison

**Usage:**
```bash
python examples/ebna_nk_landscape.py
```

**Expected Output:**
- Performance metrics across K = [1, 2, 4, 8]
- Demonstration that EBNA outperforms UMDA on epistatic problems

**Configuration:**
- N = 40 variables
- K = 1, 2, 4, 8 (epistatic neighbors)
- Learning: `LearnBayesianNetwork(max_parents=min(k+2, 5))`

---

### 5. Comprehensive EDA Algorithm Comparison

**File:** `examples/comprehensive_eda_comparison.py`
**New:** ✅ Comprehensive benchmarking framework

**Description:**
Systematic comparison of 5 EDA algorithms across 3 benchmark problems with detailed metrics.

**Algorithms Tested:**
1. UMDA (Univariate)
2. EBNA (Bayesian Network)
3. Tree EDA
4. Affinity EDA
5. MOA (Markov Networks)

**Problems Tested:**
1. OneMax-30 (separable)
2. Deceptive3-30 (deceptive, 3-bit blocks)
3. Trap5-25 (deceptive, 5-bit blocks)

**Metrics:**
- Success rate (reaching optimum)
- Mean best fitness
- Robustness (standard deviation)
- Efficiency (generations to convergence)

**Usage:**
```bash
python examples/comprehensive_eda_comparison.py
```

**Expected Output:**
- Summary table with all metrics
- Comparison plots saved as `eda_comparison_results.png`
- Observations on algorithm strengths/weaknesses

**Use Case:**
Use this script to determine which EDA is best suited for your problem characteristics.

---

## Analysis and Visualization

### 6. Model Structure Visualization and Analysis

**File:** `examples/analysis_model_structure_visualization.py`
**MATLAB Source:** Inspired by `ScriptsMateda/AnalysisScripts/BN_StructureVisualization.m` and `FitnessMeasuresComp.m`

**Description:**
Extract, visualize, and analyze probabilistic models learned by EDAs during evolution.

**Key Features:**
- Extract Bayesian network structures from cache
- Visualize structure evolution across generations
- Edge frequency analysis (which edges appear most often)
- Fitness evolution plots
- Population diversity tracking

**Usage:**
```bash
python examples/analysis_model_structure_visualization.py
```

**Expected Output:**
- Multiple matplotlib figures showing:
  - Fitness evolution (best, mean, ±std)
  - Population diversity over time
  - Network structures at different generations
  - Edge frequency heatmap
  - Aggregated structure (edges present in >30% of generations)

**Configuration:**
- Problem: Trap-5 (15 variables)
- Algorithm: EBNA
- Generations: 50
- Visualizations saved as interactive plots

**Dependencies:**
- matplotlib (required)
- networkx (optional, for better graph layouts)

---

## Script Organization

### Examples Directory Structure

```
examples/
├── moa_deceptive3.py                          # Ported from MATLAB
├── bayesian_tree_ising_mpe.py                 # Ported from MATLAB
├── gaussian_network_ackley.py                 # New continuous optimization
├── ebna_nk_landscape.py                       # New epistatic problem study
├── comprehensive_eda_comparison.py            # New benchmarking framework
├── analysis_model_structure_visualization.py  # New analysis tools
│
# Existing examples (from previous development)
├── umda_onemax.py                             # Basic UMDA
├── ebna_deceptive.py                          # EBNA on Deceptive
├── tree_eda_deceptive.py                      # Tree EDA on Deceptive
├── tree_eda_hp_protein.py                     # Tree EDA on HP Protein
├── markov_chain_hp_protein.py                 # Markov Chain FDA
├── gaussian_umda_sphere.py                    # Gaussian UMDA
├── gaussian_full_rastrigin.py                 # Gaussian Full Model
├── mixture_gaussian_rosenbrock.py             # Mixture of Gaussians
└── ... (see full list below)
```

### Complete Example List

#### Discrete Optimization
- `umda_onemax.py` - UMDA on OneMax
- `bmda_onemax.py` - BMDA on OneMax
- `ebna_deceptive.py` - EBNA on Deceptive3
- `tree_eda_deceptive.py` - Tree EDA on Deceptive3
- `affinity_eda_deceptive.py` - Affinity EDA on Deceptive3
- `default_eda_trap.py` - Default EDA on Trap function
- `default_eda_nk_landscape.py` - Default EDA on NK Landscape
- `tree_eda_ubqp.py` - Tree EDA on uBQP (multi-objective)
- `umda_sat.py` - UMDA on SAT problems
- `moa_deceptive3.py` - **NEW:** MOA on Deceptive3

#### Protein Folding (HP Model)
- `tree_eda_hp_protein.py` - Tree EDA on HP Protein
- `markov_chain_hp_protein.py` - Markov Chain FDA on HP Protein

#### Ising Model
- `tree_eda_ising.py` - Tree EDA on Ising
- `affinity_elim_eda_ising.py` - Affinity EDA on Ising
- `bayesian_tree_ising_mpe.py` - **NEW:** Bayesian Tree with MPE sampling

#### Continuous Optimization
- `gaussian_umda_sphere.py` - Gaussian UMDA on Sphere
- `gaussian_full_rastrigin.py` - Gaussian Full on Rastrigin
- `mixture_gaussian_rosenbrock.py` - Mixture Gaussian on Rosenbrock
- `gaussian_network_ackley.py` - **NEW:** Gaussian Network on Ackley

#### Permutation Problems (TSP)
- `mallows_tsp_example.py` - Mallows model for TSP
- `ehm_tsp_example.py` - Edge Histogram Model for TSP

#### Advanced / Specialized
- `ebna_nk_landscape.py` - **NEW:** EBNA epistatic study
- `comprehensive_eda_comparison.py` - **NEW:** Multi-algorithm benchmark
- `analysis_model_structure_visualization.py` - **NEW:** Model analysis

---

## EDA Algorithms Reference

### Discrete EDAs

| Algorithm | Learning Method | Sampling Method | Best For |
|-----------|----------------|-----------------|----------|
| **UMDA** | `LearnHistogram` | `SampleHistogram` | Separable problems (OneMax) |
| **EBNA** | `LearnBayesianNetwork` | `SampleBN` | Epistatic problems with dependencies |
| **Tree EDA** | `LearnTreeModel` | `SampleFDA` | Hierarchical dependencies |
| **Affinity EDA** | `LearnAffinityModel` | `SampleFDA` | Automatic structure discovery |
| **MOA** | `LearnMOA` | `SampleGibbs` | Local Markov dependencies |
| **MN-FDA** | `LearnMNFDA` | `SampleFDA` | Markov networks |

### Continuous EDAs

| Algorithm | Learning Method | Sampling Method | Best For |
|-----------|----------------|-----------------|----------|
| **Gaussian UMDA** | `LearnGaussianUnivariate` | `SampleGaussianUnivariate` | Separable continuous |
| **Gaussian Full** | `LearnGaussianFull` | `SampleGaussianFull` | Fully dependent continuous |
| **Gaussian Network** | `LearnGaussianNetwork` | `SampleGaussianNetwork` | Sparse dependencies |
| **Mixture Gaussian** | `LearnMixtureGaussian` | `SampleMixtureGaussian` | Multi-modal functions |

### Permutation EDAs

| Algorithm | Learning Method | Sampling Method | Best For |
|-----------|----------------|-----------------|----------|
| **Mallows** | `LearnMallowsModel` | `SampleMallowsModel` | TSP, permutations |
| **EHM** | `LearnEdgeHistogram` | `SampleEdgeHistogram` | TSP edge patterns |

---

## Problem Domains Reference

### Discrete Benchmarks

| Function | Description | Optimal | Difficulty |
|----------|-------------|---------|-----------|
| **OneMax** | Count 1s | N | Easy (separable) |
| **Deceptive3** | 3-bit deceptive blocks | N/3 | Medium (deceptive) |
| **Trap-k** | k-bit trap blocks | N/k × (k+1) | Medium (deceptive) |
| **NK Landscape** | Epistatic with K interactions | Varies | Hard (epistatic) |
| **Ising Model** | Spin glass energy | Instance-specific | Hard |
| **HP Protein** | Protein folding energy | Instance-specific | Hard |
| **uBQP** | Quadratic programming | Instance-specific | Hard |
| **SAT** | Boolean satisfiability | # satisfied clauses | Hard |

### Continuous Benchmarks

| Function | Description | Global Minimum | Characteristics |
|----------|-------------|----------------|-----------------|
| **Sphere** | Sum of squares | 0 at origin | Separable, unimodal |
| **Rastrigin** | Highly multimodal | 0 at origin | Multimodal, separable |
| **Rosenbrock** | Valley function | 0 at (1,...,1) | Non-separable, valley |
| **Ackley** | Multimodal with exponentials | 0 at origin | Multimodal, non-separable |

### Permutation Benchmarks

| Function | Description | Domain |
|----------|-------------|---------|
| **TSP** | Traveling Salesman | Route optimization |
| **QAP** | Quadratic Assignment | Facility location |

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'pateda'`

**Solution:**
```bash
# Install in development mode from repo root
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/pateda"
```

#### 2. Missing Dependencies

**Problem:** `ImportError: No module named 'networkx'`

**Solution:**
```bash
pip install networkx matplotlib numpy scipy
```

#### 3. Ising Model Instance Not Found

**Problem:** `FileNotFoundError: Ising instance file not found`

**Solution:**
The script automatically generates a random instance if files are missing. To use actual instances:

1. Place Ising instance files in `functions/ising-model/`
2. Format: `SG_{n}_{inst}.txt`

#### 4. Low Performance / Slow Convergence

**Problem:** Algorithm doesn't find optimum

**Solutions:**
- Increase population size
- Increase max generations
- Adjust selection pressure (try 0.3 - 0.5 proportion)
- For continuous: adjust bounds and noise parameters
- For MOA/Gibbs: increase IT parameter

#### 5. Visualization Not Showing

**Problem:** Plots don't display

**Solution:**
```python
# Add at end of script
plt.show()

# Or save to file
plt.savefig('my_plot.png')
```

---

## Advanced Usage

### Custom Fitness Functions

```python
def my_fitness(x: np.ndarray) -> float:
    """
    Custom fitness function

    Args:
        x: Solution vector (binary or continuous)

    Returns:
        Fitness value (higher is better)
    """
    # Your implementation
    return fitness_value
```

### Caching Models and Populations

```python
# Enable caching to extract models later
stats, cache = eda.run(verbose=True)

# Access cached data
if 'models' in cache:
    final_model = cache['models'][-1]

if 'populations' in cache:
    all_populations = cache['populations']
```

### Custom Stop Conditions

```python
from pateda.core.components import StopCriteria

class StopOnOptimum(StopCriteria):
    def __init__(self, optimal_value, tolerance=0.01):
        self.optimal_value = optimal_value
        self.tolerance = tolerance

    def should_stop(self, stats) -> bool:
        return abs(stats.best_fitness_overall - self.optimal_value) <= self.tolerance
```

---

## Performance Tips

1. **Population Size**: Typically 100-500 for discrete, 50-200 for continuous
2. **Selection Pressure**: 0.3-0.5 truncation proportion works well
3. **Structure Learning**: More complex structures need more data (larger populations)
4. **Sampling**:
   - PLS/FDA: Fast, good for tree structures
   - Gibbs: Slower, better for dense networks
   - MAP insertion: Boosts convergence on unimodal regions

---

## Citation

If you use these examples in research, please cite:

```bibtex
@software{pateda2025,
  title={PATEDA: Python Adaptation of Tools for Estimation of Distribution Algorithms},
  author={Santana, Roberto and others},
  year={2025},
  url={https://github.com/rsantana-isg/pateda}
}
```

---

## Further Reading

- **MATEDA User Guide**: `Mateda2.0-UserGuide.pdf`
- **PATEDA Design**: `PATEDA_DESIGN.md`
- **Testing Guide**: `TESTING_QUICKSTART.md`
- **MATLAB Mapping**: `MATLAB_PYTHON_MAPPING.md`

---

**Last Updated:** November 19, 2025
**Version:** 1.0
**Maintainer:** Roberto Santana (roberto.santana@ehu.es)
