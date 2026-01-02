# MATEDA to PATEDA Porting Roadmap

## Executive Summary

**Current Status**: ~60-70% of MATEDA's core functionality has been ported to pateda
- **Total MATEDA files**: ~220+ MATLAB files across 17 module categories
- **Total PATEDA files**: ~70 Python files across 14 modules
- **Well-covered areas**: Core framework, basic learning algorithms, repairing, selection, permutation distances
- **Major gaps**: Multi-objective optimization, knowledge extraction, local optimization, genetic operators

---

## Current Implementation Status by Module

### ✅ FULLY PORTED (100%)
- **Repairing**: All 4 constraint handling methods
- **Stop Conditions**: Core stopping criteria
- **Core Framework**: EDA infrastructure with enhancements

### ⚠️ PARTIALLY PORTED (50-80%)
- **Learning** (80%): Most probabilistic models, missing Student-t, some advanced features
- **Sampling** (70%): Core samplers available, missing Gibbs, MPE, conditional FDA
- **Selection** (70%): Basic methods done, missing Pareto-based and exponential
- **Permutations** (70%): Mallows models complete, histogram models partial
- **Functions** (60%): Key benchmarks available, some specialized missing
- **Replacement** (50%): Elitism done, missing restricted tournament
- **Seeding** (25%): Only random init, missing biased initialization

### ❌ NOT PORTED (0%)
- **Knowledge Extraction** (0/13 files): Advanced analysis tools
- **Local Optimization** (0/2 files): Hybrid algorithms
- **Crossover** (0/8 files): Recombination operators
- **Mutation** (0/1 file): Genetic mutation
- **Ordering** (0/3 files): Fitness and Pareto ordering
- **Verbose** (0/1 file): Advanced logging

---

## Detailed Gap Analysis

### LEARNING ALGORITHMS

#### ✅ Ported
- `LearnMargProdModel.m` → `umda.py` (Univariate)
- `LearnFDA.m` → `fda.py` (Factorized Distribution)
- `LearnTreeModel.m` → `tree.py` (Tree dependencies)
- `LearnBN.m` → `boa.py` (Bayesian Networks)
- `LearnMOAModel.m` → `markov.py` (Markov networks)
- `LearnGaussian*Model.m` → `gaussian.py` (Gaussian models)
- `Mallows_*_learning.m` → `mallows.py` (All Mallows variants)
- `FactAffinity*.m` → `affinity.py` (Affinity propagation)

#### ❌ Missing
- `LearnTModel.m` - Student-t distribution models
- `FindNeighborhood.m` - Neighborhood structure learning
- `LearnMixture*GaussianModels.m` - Complete mixture implementations
- `LearnEHM.m`, `LearnNHM.m` - Complete histogram models for permutations

### SAMPLING ALGORITHMS

#### ✅ Ported
- `SampleFDA.m`, `SampleBN.m`, `SampleGaussian*.m`
- `MOAGenerate*.m` → `markov.py`
- `Mallows_*_sampling.m` → `mallows.py`

#### ❌ Missing
- `MNGibbsGenerateIndividual.m` - Gibbs sampling for Markov networks
- `FindMPE.m`, `Find_kMPEs.m` - Most Probable Explanation methods
- `SampleMPE_BN.m` - MPE-based sampling
- `SampleCFDA.m` - Conditional FDA sampling
- `SampleMixture*.m` - Complete mixture sampling
- `SampleEHM.m`, `SampleNHM.m` - Complete histogram sampling

### SELECTION METHODS

#### ✅ Ported
- `truncation_selection.m`, `prop_selection.m`, `Boltzmann_selection.m`
- `sus.m` (Stochastic Universal Sampling)
- Plus added: tournament, ranking (not in MATEDA)

#### ❌ Missing
- `exp_selection.m` - Exponential selection
- `FindParetoSet.m` - Pareto set identification
- `NonDominated_selection.m` - Non-dominated selection
- `ParetoFront_selection.m` - Pareto front selection

### SEEDING METHODS

#### ✅ Ported
- `RandomInit.m` → `random_init.py`

#### ❌ Missing
- `Bias_Init.m` - Biased initialization
- `seed_thispop.m` - Custom seeding strategies
- `seeding_unitation_constraint.m` - Constraint-aware seeding

### REPLACEMENT METHODS

#### ✅ Ported
- `elitism.m`, `best_elitism.m` → `elitist.py`

#### ❌ Missing
- `RT_replacement.m` - Restricted Tournament Replacement
- `pop_agregation.m` - Population aggregation strategies

### REPAIRING METHODS ✅ Complete
All ported:
- `SetInBounds_repairing.m`, `SetWithinBounds_repairing.m` → `bounds.py`
- `Trigom_repairing.m` → `trigonometric.py`
- `Unitation_repairing.m` → `unitation.py`

### ORDERING METHODS

#### ❌ All Missing
- `fitness_ordering.m` - Fitness-based ordering
- `Pareto_ordering.m` - Pareto ordering
- `ParetoRank_ordering.m` - Pareto rank ordering

### KNOWLEDGE EXTRACTION

#### ❌ All Missing (13 files)
- `Amount_of_selection.m` - Selection pressure analysis
- `BN_Fitness_Corr.m` - BN-fitness correlation
- `BN_Pop_Prob.m` - Population probability analysis
- `Find_Fitness_Approx.m` - Fitness approximation
- `Generations_entropy.m` - Entropy over generations
- `Individuals_entropy.m` - Individual entropy
- `Mean_Var_Objectives.m` - Objective statistics
- `Most_probable_explanations.m` - MPE analysis
- `ObjectiveDistribution.m` - Objective distribution
- `Probability_monitor.m` - Probability monitoring
- `Realized_heritability.m` - Heritability analysis
- `Response_to_selection.m` - Selection response
- `entropy.m` - General entropy calculations

### LOCAL OPTIMIZATION

#### ❌ All Missing
- `Greedy_search_OffHP.m` - Greedy local search for HP protein
- `local_search_OffHP.m` - Local search for HP protein

### GENETIC OPERATORS

#### ❌ All Missing
**Crossover** (8 files):
- `CXTransposition.m`, `CXTwoPoint.m` - Basic crossover
- `LearnBlockCrossover.m`, `LearnTransposition.m`, `LearnTwoPointCrossover.m` - Learned crossover
- `SampleBlockCrossover.m`, `SampleTransposition.m`, `SampleTwoPointCrossover.m` - Sample crossover

**Mutation** (1 file):
- `BitFlipMutation.m` - Bit flip mutation

### UTILITY FUNCTIONS (OTHERFILES)

#### ⚠️ Partially Covered
Some functionality exists scattered across modules, but not complete:
- `ClusterPointsAffinity.m`, `ClusterPointsKmeans.m` - Clustering
- `IntMutualInf*.m` - Mutual information
- `*Card.m` functions - Cardinality conversions
- `LaplaceEstimator.m` - Laplace smoothing
- `apcluster.m` - Affinity propagation clustering

### PERMUTATION-SPECIFIC

#### ✅ Well Ported
- All distance metrics (Cayley, Kendall, Ulam)
- Mallows models (all 57 files consolidated)
- Consensus methods (Borda, median permutation)
- TSP, QAP, LOP problems

#### ❌ Missing
- `PFSP` (Permutation Flow Shop Problem) - `EvalPFSP.m`, `ReadPFSPInstance.m`
- Complete histogram models implementation
- Some permutation operations utilities

### BENCHMARK FUNCTIONS

#### ✅ Ported
- Deceptive3, Trapn
- IsingModel
- NK Landscapes
- SAT, uBQP
- HP Protein
- OneMax (added)
- Continuous benchmarks (Sphere, Rastrigin, etc.)
- TSP, QAP, LOP

#### ❌ Missing
- Some specialized MATLAB function variants
- PFSP problem evaluation

---

## Incremental Porting Roadmap

### 🔴 PHASE 1: Critical EDA Components (Weeks 1-3)
**Goal**: Complete core EDA functionality for single-objective optimization

#### Week 1: Selection & Replacement
1. **Exponential Selection** (`exp_selection.m` → `selection/exponential.py`)
   - Similar to Boltzmann but different scaling
   - ~50 lines, straightforward port

2. **Restricted Tournament Replacement** (`RT_replacement.m` → `replacement/restricted_tournament.py`)
   - Diversity preservation mechanism
   - ~100 lines, medium complexity

3. **Population Aggregation** (`pop_agregation.m` → `replacement/aggregation.py`)
   - Combine parent and offspring populations
   - ~50 lines, simple

#### Week 2: Seeding Methods
4. **Biased Initialization** (`Bias_Init.m` → `seeding/biased.py`)
   - Initialize with prior knowledge
   - ~80 lines, medium complexity

5. **Custom Seeding** (`seed_thispop.m` → `seeding/custom.py`)
   - Flexible seeding strategies
   - ~60 lines, medium complexity

6. **Constraint-aware Seeding** (`seeding_unitation_constraint.m` → `seeding/constrained.py`)
   - Initialize respecting constraints
   - ~100 lines, medium complexity

#### Week 3: Advanced Sampling
7. **Student-t Model** (`LearnTModel.m` → `learning/student_t.py`)
   - Robust alternative to Gaussian
   - ~150 lines, medium-high complexity
   - Also implement sampling in `sampling/student_t.py`

8. **Gibbs Sampling** (`MNGibbsGenerateIndividual.m` → `sampling/gibbs.py`)
   - MCMC sampling for Markov networks
   - ~200 lines, high complexity

9. **Neighborhood Learning** (`FindNeighborhood.m` → `learning/utils/neighborhood.py`)
   - Learn variable neighborhoods
   - ~100 lines, medium complexity

### 🟠 PHASE 2: Multi-Objective Optimization (Weeks 4-6)
**Goal**: Enable multi-objective EDA capabilities

#### Week 4: Pareto Utilities
10. **Pareto Set Finding** (`FindParetoSet.m` → `selection/utils/pareto.py`)
    - Identify non-dominated solutions
    - ~150 lines, medium complexity

11. **Pareto Ordering** (`Pareto_ordering.m` → `ordering/pareto.py`)
    - Order solutions by dominance
    - ~100 lines, medium complexity

12. **Pareto Rank Ordering** (`ParetoRank_ordering.m` → `ordering/pareto_rank.py`)
    - NSGA-II style ranking
    - ~120 lines, medium complexity

#### Week 5: Multi-Objective Selection
13. **Non-Dominated Selection** (`NonDominated_selection.m` → `selection/non_dominated.py`)
    - Select from Pareto front
    - ~100 lines, medium complexity

14. **Pareto Front Selection** (`ParetoFront_selection.m` → `selection/pareto_front.py`)
    - Maintain diverse Pareto front
    - ~150 lines, medium-high complexity

#### Week 6: Fitness Ordering & Integration
15. **Fitness Ordering** (`fitness_ordering.m` → `ordering/fitness.py`)
    - General fitness-based ordering
    - ~80 lines, simple

16. **Multi-Objective Examples**
    - Create example scripts demonstrating MO-EDA
    - ~200 lines total across 2-3 examples

### 🟡 PHASE 3: Advanced Learning & Sampling (Weeks 7-9)
**Goal**: Complete probabilistic model implementations

#### Week 7: Mixture Models
17. **Complete Mixture of Gaussians** (Complete `learning/gaussian.py` and `sampling/gaussian.py`)
    - Finish mixture implementations
    - ~200 lines additions

18. **Mixture Model Utilities** (Utils for EM algorithm, component selection)
    - ~150 lines

#### Week 8: MPE Methods
19. **Most Probable Explanation** (`FindMPE.m` → `sampling/mpe.py`)
    - Find most likely configurations
    - ~200 lines, high complexity

20. **k-MPE Finding** (`Find_kMPEs.m` → add to `sampling/mpe.py`)
    - Find k most likely configurations
    - ~150 lines, high complexity

21. **MPE Sampling** (`SampleMPE_BN.m` → add to `sampling/mpe.py`)
    - Sample from MPE distributions
    - ~100 lines, medium complexity

#### Week 9: Conditional Models
22. **Conditional FDA** (`SampleCFDA.m` → `sampling/conditional_fda.py`)
    - Conditional factorized sampling
    - ~180 lines, medium-high complexity

23. **Complete Histogram Models** (`LearnEHM.m`, `LearnNHM.m`, `SampleEHM.m`, `SampleNHM.m`)
    - Edge and Node Histogram Models for permutations
    - ~300 lines total, medium complexity

### 🟢 PHASE 4: Hybrid Algorithms & Operators (Weeks 10-12)
**Goal**: Add genetic operators and local optimization

#### Week 10: Local Optimization
24. **Greedy Search** (`Greedy_search_OffHP.m` → `local_optimization/greedy.py`)
    - Greedy local search
    - ~150 lines, medium complexity

25. **Local Search Framework** (`local_search_OffHP.m` → `local_optimization/local_search.py`)
    - General local search
    - ~200 lines, medium complexity

26. **Local Optimization Integration**
    - Integrate with EDA framework
    - ~100 lines in core framework

#### Week 11: Crossover Operators
27. **Basic Crossover** (`CXTransposition.m`, `CXTwoPoint.m` → `crossover/basic.py`)
    - Transposition and two-point crossover
    - ~150 lines, simple-medium

28. **Learned Crossover** (`LearnBlockCrossover.m`, etc. → `crossover/learned.py`)
    - Learn crossover from population structure
    - ~250 lines, medium-high complexity

29. **Sample Crossover** (`SampleBlockCrossover.m`, etc. → `crossover/sample.py`)
    - Sample crossover operators
    - ~200 lines, medium complexity

#### Week 12: Mutation & Integration
30. **Bit Flip Mutation** (`BitFlipMutation.m` → `mutation/bitflip.py`)
    - Standard bit flip mutation
    - ~80 lines, simple

31. **Hybrid Algorithm Examples**
    - Examples combining EDA with genetic operators
    - ~300 lines across 3 examples

### 🔵 PHASE 5: Knowledge Extraction & Analysis (Weeks 13-15)
**Goal**: Advanced analysis and monitoring tools

#### Week 13: Entropy & Information Theory
32. **Entropy Calculations** (`entropy.m` → `knowledge/entropy.py`)
    - General entropy functions
    - ~100 lines, medium complexity

33. **Individuals Entropy** (`Individuals_entropy.m` → `knowledge/individual_entropy.py`)
    - Entropy of individuals
    - ~120 lines, medium complexity

34. **Generations Entropy** (`Generations_entropy.m` → `knowledge/generation_entropy.py`)
    - Entropy over generations
    - ~150 lines, medium complexity

#### Week 14: Model Analysis
35. **BN-Fitness Correlation** (`BN_Fitness_Corr.m` → `knowledge/bn_fitness.py`)
    - Analyze BN structure vs fitness
    - ~180 lines, medium-high complexity

36. **Population Probability** (`BN_Pop_Prob.m` → `knowledge/population_prob.py`)
    - Population probability under model
    - ~150 lines, medium complexity

37. **Fitness Approximation** (`Find_Fitness_Approx.m` → `knowledge/fitness_approx.py`)
    - Approximate fitness from model
    - ~200 lines, high complexity

#### Week 15: Selection & Evolutionary Analysis
38. **Selection Pressure** (`Amount_of_selection.m` → `knowledge/selection_pressure.py`)
    - Measure selection intensity
    - ~120 lines, medium complexity

39. **Heritability** (`Realized_heritability.m` → `knowledge/heritability.py`)
    - Realized heritability analysis
    - ~150 lines, medium complexity

40. **Selection Response** (`Response_to_selection.m` → `knowledge/selection_response.py`)
    - Response to selection
    - ~130 lines, medium complexity

41. **Objective Statistics** (`Mean_Var_Objectives.m`, `ObjectiveDistribution.m` → `knowledge/objectives.py`)
    - Objective value analysis
    - ~150 lines, medium complexity

42. **MPE Analysis** (`Most_probable_explanations.m` → `knowledge/mpe_analysis.py`)
    - Analyze most probable explanations
    - ~180 lines, medium-high complexity

43. **Probability Monitoring** (`Probability_monitor.m` → `knowledge/probability_monitor.py`)
    - Monitor model probabilities
    - ~200 lines, medium-high complexity

### 🟣 PHASE 6: Utilities & Polish (Weeks 16-17)
**Goal**: Complete utility functions and improve integration

#### Week 16: Utility Functions
44. **Clustering Utilities** (`ClusterPointsAffinity.m`, `ClusterPointsKmeans.m` → `utils/clustering.py`)
    - Affinity and k-means clustering
    - ~200 lines, medium complexity

45. **Mutual Information** (`IntMutualInf*.m` → `utils/mutual_information.py`)
    - Mutual information calculations
    - ~180 lines, medium complexity

46. **Cardinality Utilities** (`*Card.m` functions → `utils/cardinality.py`)
    - Cardinality conversions
    - ~150 lines, medium complexity

47. **Permutation Operations** (`Compose.m`, `Invert.m` → `permutation/operations.py`)
    - Permutation composition and inversion
    - ~100 lines, simple

#### Week 17: Final Components
48. **PFSP Problem** (`EvalPFSP.m`, `ReadPFSPInstance.m` → `functions/permutation/pfsp.py`)
    - Permutation Flow Shop Problem
    - ~200 lines, medium complexity

49. **Verbose/Logging** (`simple_verbose.m` → enhance existing logging)
    - Improve logging and progress reporting
    - ~150 lines, simple-medium

50. **Documentation & Examples**
    - Complete documentation
    - Add comprehensive examples for all new features
    - ~500 lines across multiple files

### 🎯 PHASE 7: Testing & Validation (Weeks 18-20)

#### Week 18: Unit Tests
- Write comprehensive unit tests for all new modules
- Ensure 80%+ code coverage
- ~1000+ lines of test code

#### Week 19: Integration Tests
- Test complete EDA workflows
- Benchmark against MATEDA results
- Validate numerical accuracy
- ~500 lines of test code

#### Week 20: Performance & Documentation
- Performance optimization
- Complete API documentation
- Tutorial notebooks
- Migration guide from MATEDA

---

## Priority Recommendations

### Immediate Priorities (Next 2-4 weeks)
1. **Multi-Objective Support** - Major gap, widely needed
   - Pareto utilities and selection methods
   - ~600 lines, high impact

2. **Seeding Methods** - Quick wins, useful functionality
   - Biased and constrained initialization
   - ~240 lines, medium impact

3. **Student-t Models** - Robust alternative to Gaussian
   - ~300 lines, medium-high impact

4. **Restricted Tournament Replacement** - Diversity preservation
   - ~100 lines, medium impact

### Medium-Term Priorities (1-2 months)
1. **Mixture Models** - Complete existing partial implementations
2. **MPE Methods** - Advanced sampling capabilities
3. **Knowledge Extraction** - Analysis tools (at least entropy and selection pressure)
4. **Conditional Models** - Advanced modeling

### Long-Term Enhancements (2-4 months)
1. **Hybrid Algorithms** - Genetic operators and local optimization
2. **Complete Knowledge Extraction** - All 13 analysis tools
3. **Advanced Utilities** - Clustering, MI, etc.
4. **PFSP and remaining benchmarks**

### Optional/Low Priority
1. **Verbose/Logging** - Can use Python's logging framework
2. **Some specialized MATLAB utilities** - May not be needed in Python

---

## Estimated Effort Summary

| Phase | Duration | Lines of Code | Complexity | Impact |
|-------|----------|---------------|------------|--------|
| Phase 1: Critical Components | 3 weeks | ~900 | Medium | High |
| Phase 2: Multi-Objective | 3 weeks | ~700 | Medium-High | Very High |
| Phase 3: Advanced Models | 3 weeks | ~1100 | High | High |
| Phase 4: Hybrid Algorithms | 3 weeks | ~1000 | Medium | Medium-High |
| Phase 5: Knowledge Extraction | 3 weeks | ~1600 | Medium-High | Medium |
| Phase 6: Utilities & Polish | 2 weeks | ~900 | Low-Medium | Medium |
| Phase 7: Testing & Validation | 3 weeks | ~1500+ | Medium | Critical |
| **TOTAL** | **20 weeks** | **~7700+** | **Mixed** | **Complete** |

---

## Success Metrics

### Completion Criteria
- ✅ All core MATEDA algorithms ported
- ✅ Feature parity with MATEDA (excluding MATLAB-specific features)
- ✅ 80%+ test coverage
- ✅ Validation against MATEDA results
- ✅ Complete documentation and examples

### Quality Standards
- Type hints for all public APIs
- Comprehensive docstrings (NumPy style)
- Integration with existing pateda architecture
- Performance within 20% of MATEDA (accounting for MATLAB vs Python)
- Modern Python practices (dataclasses, context managers, etc.)

---

## Notes & Recommendations

### Architecture Considerations
1. **Consistency**: Maintain current pateda architecture and naming conventions
2. **Modularity**: Keep components loosely coupled
3. **Extensibility**: Design for easy addition of new algorithms
4. **Performance**: Consider NumPy/SciPy optimizations
5. **Testing**: Write tests alongside implementation

### Python Enhancements to Consider
1. **Type Safety**: Use mypy for type checking
2. **Async Support**: For parallel evaluation
3. **GPU Acceleration**: JAX/CuPy for large-scale problems
4. **Integration**: scikit-learn compatibility where appropriate
5. **Visualization**: Interactive plots with Plotly
6. **Serialization**: Pickle/JSON for model persistence

### Documentation Priorities
1. Migration guide from MATEDA
2. Algorithm comparison guide
3. Comprehensive API reference
4. Tutorial notebooks for each major feature
5. Performance benchmarking results

### Potential Improvements Over MATEDA
1. Better visualization tools (already started)
2. Modern statistical tools integration
3. Parallel/distributed evaluation
4. Real-time monitoring dashboards
5. Hyperparameter tuning tools
6. AutoML integration capabilities

---

## Conclusion

The current pateda implementation has captured the essential core of MATEDA (~60-70% functionality), with excellent coverage of:
- Core EDA algorithms
- Basic probabilistic models
- Permutation-based EDAs
- Essential operators

The roadmap focuses on incrementally adding:
1. **Critical gaps** first (multi-objective, seeding, advanced sampling)
2. **High-impact features** next (hybrid algorithms, knowledge extraction)
3. **Polish and utilities** last (complete benchmarks, utilities)

Following this 20-week roadmap will achieve **~95% feature parity** with MATEDA while adding Python-specific improvements and modern software engineering practices.

**Recommended Starting Point**: Begin with Phase 1 (Critical Components) and Phase 2 (Multi-Objective), as these provide the highest immediate value and fill the most significant gaps in current functionality.
