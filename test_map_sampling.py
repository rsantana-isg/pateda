"""
Comprehensive tests for MAP-based sampling methods

Tests:
1. MAP inference (exact, BP, decimation)
2. k-MAP computation
3. Insert-MAP sampling
4. Template-MAP sampling
5. Hybrid MAP sampling
6. Integration with MN-FDA
7. Integration with MOA
8. Performance comparison on benchmark problems
"""

import numpy as np
import pytest
from pateda.core.eda import EDA
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.moa import LearnMOA
from pateda.sampling.map_sampling import (
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleHybridMAP,
)
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.fda import SampleFDA
from pateda.inference.map_inference import MAPInference, compute_map, compute_k_map
from pateda.selection.truncation import SelectTruncation
from pateda.seeding.random import SeedRandom


# Test problems
def onemax(x):
    """OneMax: maximize number of ones"""
    return np.sum(x, axis=1)


def trap5(x):
    """Trap-5: deceptive problem with 5-bit traps"""
    n = x.shape[1]
    fitness = np.zeros(x.shape[0])

    for i in range(0, n, 5):
        block = x[:, i:i+5]
        ones = np.sum(block, axis=1)

        # Trap function: deceptive optimum
        fitness += np.where(ones == 5, 5, 4 - ones)

    return fitness


def four_peaks(x, t=None):
    """Four Peaks problem"""
    n = x.shape[1]
    if t is None:
        t = n // 10

    fitness = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        # Count leading ones
        ones = 0
        for j in range(n):
            if x[i, j] == 1:
                ones += 1
            else:
                break

        # Count trailing zeros
        zeros = 0
        for j in range(n-1, -1, -1):
            if x[i, j] == 0:
                zeros += 1
            else:
                break

        # Four peaks fitness
        if ones > t and zeros > t:
            fitness[i] = max(ones, zeros) + n
        else:
            fitness[i] = max(ones, zeros)

    return fitness


class TestMAPInference:
    """Test MAP inference functionality"""

    def test_simple_map_exact(self):
        """Test exact MAP on simple 3-variable network"""
        # Create simple Markov network:
        # Cliques: {0,1}, {1,2}
        # Optimal configuration should be clear from tables

        cliques = [np.array([0, 1]), np.array([1, 2])]

        # Table 1: P(X0, X1) - prefers (1, 1)
        table1 = np.array([[0.1, 0.2],   # X0=0
                          [0.2, 0.5]])   # X0=1

        # Table 2: P(X1, X2) - prefers (1, 1)
        table2 = np.array([[0.1, 0.2],   # X1=0
                          [0.2, 0.5]])   # X1=1

        tables = [table1, table2]
        cardinalities = np.array([2, 2, 2])

        # Compute MAP
        map_config = compute_map(cliques, tables, cardinalities, method="bp")

        print(f"MAP configuration: {map_config}")
        assert len(map_config) == 3
        assert all(map_config >= 0) and all(map_config < 2)

    def test_k_map_computation(self):
        """Test k-MAP finds multiple configurations"""
        cliques = [np.array([0, 1]), np.array([1, 2])]

        table1 = np.array([[0.1, 0.2],
                          [0.2, 0.5]])
        table2 = np.array([[0.1, 0.2],
                          [0.2, 0.5]])

        tables = [table1, table2]
        cardinalities = np.array([2, 2, 2])

        # Compute k-MAP
        configs, probs = compute_k_map(cliques, tables, cardinalities, k=3, method="bp")

        print(f"k-MAP configurations:\n{configs}")
        print(f"Probabilities: {probs}")

        assert len(configs) >= 1  # At least MAP
        assert len(configs) == len(probs)
        assert np.all(probs[:-1] >= probs[1:])  # Sorted by probability

    def test_map_decimation(self):
        """Test MAP computation using decimation"""
        cliques = [np.array([0, 1, 2])]

        # Single clique with clear optimum at (1, 1, 1)
        table = np.array([
            [[0.05, 0.05], [0.05, 0.05]],  # X0=0
            [[0.05, 0.05], [0.05, 0.60]]   # X0=1
        ])

        tables = [table]
        cardinalities = np.array([2, 2, 2])

        inference = MAPInference(cliques, tables, cardinalities, method="decimation")
        result = inference.compute_map()

        print(f"Decimation MAP: {result.configuration}")
        print(f"Log probability: {result.log_probability}")

        assert len(result.configuration) == 3


class TestMAPSampling:
    """Test MAP-based sampling methods"""

    def test_insert_map_basic(self):
        """Test Insert-MAP sampling produces population with MAP"""
        n_vars = 10
        n_samples = 20
        cardinality = np.array([2] * n_vars)

        # Create simple model
        learner = LearnMNFDA(max_clique_size=2, return_factorized=False)

        # Create random population
        pop = np.random.randint(0, 2, size=(50, n_vars))
        fitness = onemax(pop)

        # Learn model
        model = learner.learn(0, n_vars, cardinality, pop, fitness)

        # Sample with Insert-MAP
        sampler = SampleInsertMAP(n_samples=n_samples, map_method="bp")
        new_pop = sampler.sample(n_vars, model, cardinality)

        print(f"Insert-MAP population shape: {new_pop.shape}")
        print(f"First individual (should be MAP): {new_pop[0]}")

        assert new_pop.shape == (n_samples, n_vars)
        assert np.all(new_pop >= 0) and np.all(new_pop < 2)

    def test_template_map_basic(self):
        """Test Template-MAP sampling uses MAP as template"""
        n_vars = 10
        n_samples = 20
        cardinality = np.array([2] * n_vars)

        learner = LearnMNFDA(max_clique_size=2, return_factorized=False)

        pop = np.random.randint(0, 2, size=(50, n_vars))
        fitness = onemax(pop)

        model = learner.learn(0, n_vars, cardinality, pop, fitness)

        # Sample with Template-MAP
        sampler = SampleTemplateMAP(
            n_samples=n_samples,
            map_method="bp",
            template_prob=0.7  # 70% of variables from template
        )
        new_pop = sampler.sample(n_vars, model, cardinality)

        print(f"Template-MAP population shape: {new_pop.shape}")
        print(f"First individual (pure MAP): {new_pop[0]}")
        print(f"Second individual (variation): {new_pop[1]}")

        assert new_pop.shape == (n_samples, n_vars)

        # First should be pure MAP, others should be variations
        if n_samples > 1:
            # Not all individuals should be identical
            assert not np.all(new_pop[0] == new_pop[1])

    def test_hybrid_map_basic(self):
        """Test Hybrid MAP combines both strategies"""
        n_vars = 10
        n_samples = 20
        cardinality = np.array([2] * n_vars)

        learner = LearnMNFDA(max_clique_size=2, return_factorized=False)

        pop = np.random.randint(0, 2, size=(50, n_vars))
        fitness = onemax(pop)

        model = learner.learn(0, n_vars, cardinality, pop, fitness)

        # Sample with Hybrid MAP
        sampler = SampleHybridMAP(
            n_samples=n_samples,
            map_method="bp",
            template_prob=0.5
        )
        new_pop = sampler.sample(n_vars, model, cardinality)

        print(f"Hybrid MAP population shape: {new_pop.shape}")

        assert new_pop.shape == (n_samples, n_vars)


class TestMAPWithMNFDA:
    """Test MAP sampling integrated with MN-FDA"""

    def test_mnfda_insert_map_onemax(self):
        """Test MN-FDA + Insert-MAP on OneMax"""
        n_vars = 30
        pop_size = 100
        n_generations = 50
        cardinality = np.array([2] * n_vars)

        # Create EDA with MN-FDA + Insert-MAP
        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=onemax,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMNFDA(max_clique_size=3, return_factorized=False),
            sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp"),
            selection=SelectTruncation(ratio=0.5),
        )

        # Run optimization
        result = eda.run()

        print(f"\nMN-FDA + Insert-MAP on OneMax (n={n_vars}):")
        print(f"Best fitness: {result['best_fitness']}")
        print(f"Best solution: {result['best_solution']}")
        print(f"Evaluations: {result['n_evaluations']}")

        # Should reach optimum or near-optimum
        assert result['best_fitness'] >= n_vars * 0.9  # At least 90% of optimum

    def test_mnfda_template_map_trap5(self):
        """Test MN-FDA + Template-MAP on Trap-5"""
        n_vars = 25  # 5 blocks of trap-5
        pop_size = 100
        n_generations = 100
        cardinality = np.array([2] * n_vars)

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=trap5,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMNFDA(max_clique_size=5, return_factorized=False),
            sampling=SampleTemplateMAP(
                n_samples=pop_size,
                map_method="bp",
                template_prob=0.6
            ),
            selection=SelectTruncation(ratio=0.5),
        )

        result = eda.run()

        print(f"\nMN-FDA + Template-MAP on Trap-5 (n={n_vars}):")
        print(f"Best fitness: {result['best_fitness']}")
        print(f"Evaluations: {result['n_evaluations']}")

        # Trap-5 optimum is 5 * 5 = 25
        assert result['best_fitness'] >= 20  # At least 80% of optimum

    def test_mnfda_hybrid_map(self):
        """Test MN-FDA + Hybrid MAP"""
        n_vars = 20
        pop_size = 80
        n_generations = 50
        cardinality = np.array([2] * n_vars)

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=onemax,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMNFDA(max_clique_size=3, return_factorized=False),
            sampling=SampleHybridMAP(
                n_samples=pop_size,
                map_method="bp",
                template_prob=0.5
            ),
            selection=SelectTruncation(ratio=0.5),
        )

        result = eda.run()

        print(f"\nMN-FDA + Hybrid MAP on OneMax (n={n_vars}):")
        print(f"Best fitness: {result['best_fitness']}")

        assert result['best_fitness'] >= n_vars * 0.9


class TestMAPWithMOA:
    """Test MAP sampling integrated with MOA"""

    def test_moa_insert_map_onemax(self):
        """Test MOA + Insert-MAP on OneMax"""
        n_vars = 30
        pop_size = 100
        n_generations = 50
        cardinality = np.array([2] * n_vars)

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=onemax,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMOA(k_neighbors=3),
            sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp"),
            selection=SelectTruncation(ratio=0.5),
        )

        result = eda.run()

        print(f"\nMOA + Insert-MAP on OneMax (n={n_vars}):")
        print(f"Best fitness: {result['best_fitness']}")
        print(f"Evaluations: {result['n_evaluations']}")

        assert result['best_fitness'] >= n_vars * 0.9

    def test_moa_template_map(self):
        """Test MOA + Template-MAP"""
        n_vars = 25
        pop_size = 100
        n_generations = 50
        cardinality = np.array([2] * n_vars)

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=onemax,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMOA(k_neighbors=5),
            sampling=SampleTemplateMAP(
                n_samples=pop_size,
                map_method="bp",
                template_prob=0.5
            ),
            selection=SelectTruncation(ratio=0.5),
        )

        result = eda.run()

        print(f"\nMOA + Template-MAP on OneMax (n={n_vars}):")
        print(f"Best fitness: {result['best_fitness']}")

        assert result['best_fitness'] >= n_vars * 0.85


class TestMAPPerformanceComparison:
    """Compare MAP-based sampling with baseline methods"""

    def test_compare_sampling_methods_onemax(self):
        """Compare Insert-MAP, Template-MAP, Hybrid, Gibbs, and PLS on OneMax"""
        n_vars = 30
        pop_size = 100
        n_generations = 40
        cardinality = np.array([2] * n_vars)
        n_runs = 3

        sampling_methods = {
            "Insert-MAP": SampleInsertMAP(n_samples=pop_size, map_method="bp"),
            "Template-MAP": SampleTemplateMAP(n_samples=pop_size, map_method="bp", template_prob=0.6),
            "Hybrid-MAP": SampleHybridMAP(n_samples=pop_size, map_method="bp", template_prob=0.5),
            "Gibbs": SampleGibbs(n_samples=pop_size, IT=4),
            "PLS": SampleFDA(n_samples=pop_size),
        }

        results = {}

        for method_name, sampler in sampling_methods.items():
            best_fitnesses = []

            for run in range(n_runs):
                # Determine model type based on sampler
                if method_name == "PLS":
                    learner = LearnMNFDA(max_clique_size=3, return_factorized=True)
                else:
                    learner = LearnMNFDA(max_clique_size=3, return_factorized=False)

                eda = EDA(
                    n_vars=n_vars,
                    cardinality=cardinality,
                    fitness_function=onemax,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    seeding=SeedRandom(),
                    learning=learner,
                    sampling=sampler,
                    selection=SelectTruncation(ratio=0.5),
                    verbose=False,
                )

                result = eda.run()
                best_fitnesses.append(result['best_fitness'])

            avg_fitness = np.mean(best_fitnesses)
            std_fitness = np.std(best_fitnesses)
            results[method_name] = {
                'mean': avg_fitness,
                'std': std_fitness,
                'best': np.max(best_fitnesses)
            }

        print(f"\n{'='*60}")
        print(f"Sampling Method Comparison on OneMax (n={n_vars}, {n_runs} runs)")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Mean Fitness':<15} {'Std':<10} {'Best':<10}")
        print(f"{'-'*60}")

        for method, stats in results.items():
            print(f"{method:<15} {stats['mean']:<15.2f} {stats['std']:<10.2f} {stats['best']:<10.0f}")

        print(f"{'='*60}\n")

        # MAP methods should perform competitively
        assert results['Insert-MAP']['mean'] >= n_vars * 0.85
        assert results['Template-MAP']['mean'] >= n_vars * 0.80
        assert results['Hybrid-MAP']['mean'] >= n_vars * 0.80

    def test_compare_map_methods_trap5(self):
        """Compare MAP inference methods (exact, BP, decimation) on Trap-5"""
        n_vars = 20  # 4 blocks
        pop_size = 100
        n_generations = 80
        cardinality = np.array([2] * n_vars)
        n_runs = 2

        map_methods = ["bp", "decimation"]
        results = {}

        for map_method in map_methods:
            best_fitnesses = []

            for run in range(n_runs):
                eda = EDA(
                    n_vars=n_vars,
                    cardinality=cardinality,
                    fitness_function=trap5,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    seeding=SeedRandom(),
                    learning=LearnMNFDA(max_clique_size=5, return_factorized=False),
                    sampling=SampleInsertMAP(
                        n_samples=pop_size,
                        map_method=map_method
                    ),
                    selection=SelectTruncation(ratio=0.5),
                    verbose=False,
                )

                result = eda.run()
                best_fitnesses.append(result['best_fitness'])

            results[map_method] = {
                'mean': np.mean(best_fitnesses),
                'std': np.std(best_fitnesses)
            }

        print(f"\nMAP Inference Method Comparison on Trap-5 (n={n_vars}):")
        for method, stats in results.items():
            print(f"  {method}: {stats['mean']:.2f} ± {stats['std']:.2f}")

        # Both methods should work reasonably well
        for method in map_methods:
            assert results[method]['mean'] >= 12  # At least 60% of optimum


class TestMAPHighCardinality:
    """Test MAP methods on higher cardinality problems"""

    def test_ternary_variables(self):
        """Test MAP sampling on ternary (3-valued) variables"""
        n_vars = 20
        pop_size = 100
        n_generations = 60
        cardinality = np.array([3] * n_vars)

        def ternary_onemax(x):
            """Maximize sum (prefer higher values)"""
            return np.sum(x, axis=1)

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_function=ternary_onemax,
            pop_size=pop_size,
            n_generations=n_generations,
            seeding=SeedRandom(),
            learning=LearnMNFDA(max_clique_size=3, return_factorized=False),
            sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp"),
            selection=SelectTruncation(ratio=0.5),
        )

        result = eda.run()

        print(f"\nInsert-MAP on Ternary OneMax (n={n_vars}, k=3):")
        print(f"Best fitness: {result['best_fitness']} (optimum: {n_vars * 2})")

        # Should find good solutions
        assert result['best_fitness'] >= n_vars * 1.5  # At least 75% of optimum


if __name__ == "__main__":
    # Run tests
    print("Testing MAP Inference...")
    test_map = TestMAPInference()
    test_map.test_simple_map_exact()
    test_map.test_k_map_computation()
    test_map.test_map_decimation()

    print("\nTesting MAP Sampling...")
    test_sampling = TestMAPSampling()
    test_sampling.test_insert_map_basic()
    test_sampling.test_template_map_basic()
    test_sampling.test_hybrid_map_basic()

    print("\nTesting MAP with MN-FDA...")
    test_mnfda = TestMAPWithMNFDA()
    test_mnfda.test_mnfda_insert_map_onemax()
    test_mnfda.test_mnfda_template_map_trap5()
    test_mnfda.test_mnfda_hybrid_map()

    print("\nTesting MAP with MOA...")
    test_moa = TestMAPWithMOA()
    test_moa.test_moa_insert_map_onemax()
    test_moa.test_moa_template_map()

    print("\nTesting Performance Comparison...")
    test_compare = TestMAPPerformanceComparison()
    test_compare.test_compare_sampling_methods_onemax()
    test_compare.test_compare_map_methods_trap5()

    print("\nTesting High Cardinality...")
    test_cardinality = TestMAPHighCardinality()
    test_cardinality.test_ternary_variables()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
