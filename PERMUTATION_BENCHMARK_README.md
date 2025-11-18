# Permutation-Based EDA Benchmark Suite

## Overview

This benchmark suite provides comprehensive testing for permutation-based Estimation of Distribution Algorithms (EDAs) in pateda. The suite includes three classic NP-hard combinatorial optimization problems and tests three different probabilistic models for permutations.

## Optimization Problems

### 1. Traveling Salesman Problem (TSP)
**File:** `pateda/functions/permutation/tsp.py`

The TSP seeks the shortest Hamiltonian cycle through a set of cities.

**Problem Generators:**
- `create_random_tsp(n_cities, seed)` - Random cities in 2D Euclidean space
- `create_tsp_from_coordinates(coordinates)` - TSP from specific city coordinates

**Objective:** Minimize total tour distance

**Applications:** Logistics, circuit board drilling, genome sequencing

**Reference:**
- Applegate, D.L., et al. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.

### 2. Quadratic Assignment Problem (QAP)
**File:** `pateda/functions/permutation/qap.py`

The QAP assigns facilities to locations to minimize the sum of flow×distance products.

**Problem Generators:**
- `create_random_qap(n, seed, sparse)` - Random QAP instance
- `create_uniform_qap(n, seed)` - Uniform random matrices
- `create_grid_qap(grid_size, seed)` - Grid-based layout problem
- `load_qaplib_instance(flow_matrix, distance_matrix)` - Load QAPLIB benchmark

**Objective:** Minimize total assignment cost

**Applications:** Facility location, keyboard design, hospital layout, turbine balancing

**Reference:**
- Cela, E. (1998). *The Quadratic Assignment Problem: Theory and Algorithms*. Kluwer Academic Publishers.
- QAPLIB: http://coral.ise.lehigh.edu/data-sets/qaplib/

### 3. Linear Ordering Problem (LOP)
**File:** `pateda/functions/permutation/lop.py`

The LOP finds an ordering that maximizes the sum of weights above the diagonal.

**Problem Generators:**
- `create_random_lop(n, seed, symmetric)` - Random LOP instance
- `create_tournament_lop(n, seed)` - Round-robin tournament ranking
- `create_triangular_lop(n, seed)` - Triangular structure
- `create_sparse_lop(n, density, seed)` - Sparse weight matrix
- `load_lolib_instance(weight_matrix)` - Load LOLIB benchmark
- `feedback_arc_set_to_lop(adjacency_matrix)` - Convert from graph problem

**Objective:** Maximize ordering score (or minimize for feedback arc set)

**Applications:** Ranking, input-output economics, archaeology (seriation), graph acyclicity

**Reference:**
- Martí, R., Reinelt, G. (2011). *The Linear Ordering Problem: Exact and Heuristic Methods*. Springer.
- LOLIB: http://grafo.etsii.urjc.es/optsicom/lolib/

## Permutation-Based EDA Models

### 1. Mallows Model with Kendall Distance
**Learning:** `pateda/learning/mallows.py` - `learn_mallows_kendall`
**Sampling:** `pateda/sampling/mallows.py` - `sample_mallows_kendall`

The Mallows model is a probabilistic model for permutations based on a reference permutation (consensus) and a dispersion parameter (theta).

**Properties:**
- Captures global permutation structure
- Based on distance to consensus ranking
- θ parameter controls spread around consensus
- Theoretically well-founded

**Best for:** Problems with strong global structure, TSP with clear optimal regions

**Reference:**
- Mallows, C.L. (1957). "Non-null ranking models". *Biometrika* 44(1-2): 114-130.
- Ceberio, J., et al. (2011). "Introducing the Mallows Model on Estimation of Distribution Algorithms".

### 2. Edge Histogram Model (EHM)
**Learning:** `pateda/learning/histogram.py` - `learn_ehm`
**Sampling:** `pateda/sampling/histogram.py` - `sample_ehm`

The EHM models transitions between consecutive positions in permutations.

**Properties:**
- Captures local edge patterns
- Fast learning and sampling
- Can use symmetric or asymmetric variants
- Includes smoothing parameter (beta_ratio)

**Best for:** Problems with strong adjacency structure, TSP, some QAP instances

**Reference:**
- Ceberio, J., et al. (2015). "A review of distances for the Mallows and Generalized Mallows estimation of distribution algorithms". *Computational Optimization and Applications*.

### 3. Node Histogram Model (NHM)
**Learning:** `pateda/learning/histogram.py` - `learn_nhm`
**Sampling:** `pateda/sampling/histogram.py` - `sample_nhm`

The NHM models the probability of each item appearing at each position.

**Properties:**
- Captures positional preferences
- Very fast learning
- Independence assumption between positions
- Good for problems with position-specific structure

**Best for:** Assignment-type problems, QAP, problems with positional constraints

**Reference:**
- Ceberio, J., et al. (2015). "A review of distances for the Mallows and Generalized Mallows estimation of distribution algorithms".

## Benchmark Test Suite

**File:** `pateda/tests/test_permutation_benchmark.py`

### Test Categories

#### 1. Individual Problem Tests
Tests each problem type with each EDA model:
- `test_tsp_small_mallows()` - TSP (n=10) with Mallows
- `test_tsp_medium_ehm()` - TSP (n=20) with EHM
- `test_tsp_nhm()` - TSP (n=15) with NHM
- `test_qap_small_mallows()` - QAP (n=10) with Mallows
- `test_qap_uniform_ehm()` - Uniform QAP with EHM
- `test_qap_grid_nhm()` - Grid QAP with NHM
- `test_qap_sparse_mallows()` - Sparse QAP with Mallows
- `test_lop_random_mallows()` - Random LOP with Mallows
- `test_lop_tournament_ehm()` - Tournament LOP with EHM
- `test_lop_triangular_nhm()` - Triangular LOP with NHM
- `test_lop_sparse_mallows()` - Sparse LOP with Mallows

#### 2. Comparative Benchmarks
Compare all three EDA models on the same problem:
- `test_comparative_benchmark_tsp()` - Compare all models on TSP-15
- `test_comparative_benchmark_qap()` - Compare all models on QAP-12
- `test_comparative_benchmark_lop()` - Compare all models on LOP-12

#### 3. Scalability Tests
Test performance with increasing problem sizes:
- `test_scalability_tsp()` - TSP with n ∈ {10, 15, 20, 25, 30}
- `test_scalability_qap()` - QAP with n ∈ {8, 10, 12, 15, 20}
- `test_scalability_lop()` - LOP with n ∈ {10, 15, 20, 25, 30}

### Running the Benchmarks

#### Run all quick tests (default):
```bash
pytest pateda/tests/test_permutation_benchmark.py -v
```

#### Run including scalability tests:
```bash
pytest pateda/tests/test_permutation_benchmark.py -v -m slow
```

#### Run as standalone script:
```bash
python pateda/tests/test_permutation_benchmark.py
```

#### Run specific test:
```bash
pytest pateda/tests/test_permutation_benchmark.py::test_comparative_benchmark_tsp -v
```

### Benchmark Metrics

Each benchmark reports:
- **Best Fitness:** Best solution quality found across runs
- **Mean Fitness:** Average solution quality
- **Std Fitness:** Standard deviation of solution quality
- **Time Elapsed:** Average wall-clock time per run
- **Convergence Curve:** Best fitness over generations
- **Best Solution:** The actual best permutation found

### Expected Results

The benchmark suite is designed to:
1. **Verify Correctness:** All tests should pass, indicating algorithms work correctly
2. **Compare Performance:** Comparative tests show which model works best for each problem
3. **Assess Scalability:** Scalability tests reveal how performance degrades with problem size
4. **Validate Implementation:** Results should align with known properties from literature

### Interpreting Results

**TSP:**
- Mallows often performs well due to global distance structure
- EHM excels when good edges are preserved
- Problem difficulty: O(n!) search space, NP-hard

**QAP:**
- All models can be effective depending on problem structure
- Grid-based problems may favor EHM (adjacency patterns)
- Problem difficulty: O(n!) search space, NP-hard

**LOP:**
- NHM can work well for problems with position preferences
- Tournament problems may favor Mallows or EHM
- Problem difficulty: O(n!) search space, NP-hard

## Implementation Notes

### Permutation Representation
- Permutations can be 0-indexed or 1-indexed
- All functions automatically detect and handle both conventions
- Internal processing uses 0-indexed arrays

### Distance Metrics
Available in `pateda/permutation/distances.py`:
- `kendall_distance` - Number of pairwise disagreements
- `cayley_distance` - Minimum number of swaps
- `ulam_distance` - Based on longest increasing subsequence
- `hamming_distance` - Number of positional differences

### Consensus Methods
Available in `pateda/permutation/consensus.py`:
- `find_consensus_borda` - Borda count aggregation
- Other consensus methods for different distance metrics

## Performance Considerations

**Problem Size Guidelines:**
- **Small:** n ≤ 15 (Quick testing, all models)
- **Medium:** 15 < n ≤ 30 (Standard benchmarking)
- **Large:** n > 30 (Scalability testing, use EHM/NHM)

**Population Size Recommendations:**
- Start with pop_size = 5-10 × n
- Increase for harder problems
- Mallows may need larger populations

**Generation Limits:**
- Typical: max_gen = 5-10 × n
- Increase if convergence is slow
- Monitor convergence curves

## References and Further Reading

### Permutation-Based EDAs
1. Ceberio, J., et al. (2012). "A review on estimation of distribution algorithms in permutation-based combinatorial optimization problems". *Progress in Artificial Intelligence* 1(1): 103-117.

2. Ceberio, J., et al. (2015). "A review of distances for the Mallows and Generalized Mallows estimation of distribution algorithms". *Computational Optimization and Applications* 62(2): 545-564.

3. Larrañaga, P., et al. (2012). "A review on probabilistic graphical models in evolutionary computation". *Journal of Heuristics* 18(5): 795-819.

### Problem-Specific References
4. TSP: TSPLIB - http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
5. QAP: QAPLIB - http://coral.ise.lehigh.edu/data-sets/qaplib/
6. LOP: LOLIB - http://grafo.etsii.urjc.es/optsicom/lolib/

### MATEDA Framework
7. Santana, R. (2025). "Mateda 3.0: EDAs and other model-based optimization algorithms with Matlab". Technical Report, University of the Basque Country.

## Contributing

To add new benchmark problems or tests:

1. **New Problem Type:**
   - Create problem class in `pateda/functions/permutation/`
   - Add generator functions
   - Update `__init__.py`

2. **New Tests:**
   - Add test functions to `test_permutation_benchmark.py`
   - Follow naming convention: `test_<problem>_<variant>_<model>`
   - Include assertion and print statement

3. **New EDA Model:**
   - Implement learning in `pateda/learning/`
   - Implement sampling in `pateda/sampling/`
   - Add to benchmark configuration dictionaries

## Troubleshooting

**Common Issues:**

1. **Import Errors:**
   - Ensure pateda is in PYTHONPATH
   - Check all dependencies are installed

2. **Slow Convergence:**
   - Increase population size
   - Increase max generations
   - Try different EDA model

3. **Memory Issues:**
   - Reduce population size
   - Use smaller problem instances
   - Prefer EHM/NHM over Mallows for large n

4. **Unexpected Results:**
   - Check random seed is set correctly
   - Verify problem instance is valid
   - Compare with known optimal solutions if available

## Contact

For questions or issues related to the permutation EDA benchmark suite:
- Repository: https://github.com/rsantana-isg/pateda
- Issues: https://github.com/rsantana-isg/pateda/issues
- Mateda framework: roberto.santana@ehu.eus
