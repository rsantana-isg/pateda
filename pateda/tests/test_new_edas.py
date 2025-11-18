"""
Comprehensive tests for newly implemented EDAs from C++.

This test suite covers:
- PBIL (Population-Based Incremental Learning)
- BSC (Bisection)
- MIMIC (Mutual Information Maximization for Input Clustering)
"""

import pytest
import numpy as np
from pateda.learning.pbil import LearnPBIL
from pateda.learning.bsc import LearnBSC
from pateda.learning.mimic import LearnMIMIC
from pateda.sampling.fda import SampleFDA


class TestPBIL:
    """Test Population-Based Incremental Learning (PBIL)"""

    def test_learn_pbil_basic(self):
        """Test basic PBIL learning"""
        np.random.seed(42)
        # Binary population
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)
        fitness = np.sum(population, axis=1)  # Simple OneMax

        learner = LearnPBIL(alpha=0.1)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        assert model is not None
        assert model.structure.shape[0] == n_vars  # One clique per variable
        assert len(model.parameters) == n_vars

    def test_pbil_probabilities_range(self):
        """Test that PBIL probabilities are in valid range"""
        np.random.seed(42)
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)
        fitness = np.sum(population, axis=1)

        learner = LearnPBIL(alpha=0.1)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Check all probabilities are valid
        for prob_table in model.parameters:
            assert np.all(prob_table >= 0)
            assert np.all(prob_table <= 1)
            assert np.isclose(np.sum(prob_table), 1.0)

    def test_pbil_incremental_learning(self):
        """Test that PBIL updates probabilities incrementally"""
        np.random.seed(42)
        n_vars = 10
        cardinality = np.array([2] * n_vars)

        learner = LearnPBIL(alpha=0.5)

        # First generation - all zeros
        population1 = np.zeros((50, n_vars), dtype=int)
        fitness1 = np.sum(population1, axis=1)

        model1 = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population1,
            fitness=fitness1,
        )

        # Get initial probabilities for first variable
        initial_prob = model1.parameters[0][1]  # Probability of value 1

        # Second generation - all ones
        population2 = np.ones((50, n_vars), dtype=int)
        fitness2 = np.sum(population2, axis=1)

        model2 = learner.learn(
            generation=1,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population2,
            fitness=fitness2,
        )

        # Probability should have moved toward 1.0, but not all the way
        updated_prob = model2.parameters[0][1]
        assert updated_prob > initial_prob
        assert updated_prob < 1.0  # Not all the way to 1.0 due to incremental learning

    def test_pbil_on_onemax(self):
        """Test PBIL on OneMax problem"""
        np.random.seed(42)

        def onemax(x):
            return np.sum(x, axis=1)

        n_vars = 20
        pop_size = 100
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        learner = LearnPBIL(alpha=0.1)

        # Run PBIL
        for gen in range(30):
            fitness = onemax(population)

            # Select best 30%
            idx = np.argsort(-fitness)[:30]  # Higher is better
            selected = population[idx]

            # Learn model
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population
            sampler = SampleFDA(n_samples=pop_size)
            population = sampler.sample(n_vars=n_vars, model=model, cardinality=cardinality)

        final_fitness = onemax(population)
        # Should converge reasonably well
        assert np.mean(final_fitness) > n_vars * 0.80

    def test_pbil_with_mutation(self):
        """Test PBIL with mutation"""
        np.random.seed(42)
        n_vars = 10
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (100, n_vars))
        fitness = np.sum(population, axis=1)

        learner = LearnPBIL(alpha=0.1, mutation_prob=0.02, mutation_shift=0.05)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        assert model is not None
        assert len(model.parameters) == n_vars


class TestBSC:
    """Test Bisection (BSC)"""

    def test_learn_bsc_basic(self):
        """Test basic BSC learning"""
        np.random.seed(42)
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnBSC()
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        assert model is not None
        assert model.structure.shape[0] == n_vars
        assert len(model.parameters) == n_vars

    def test_bsc_requires_fitness(self):
        """Test that BSC requires fitness values"""
        np.random.seed(42)
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)

        learner = LearnBSC()

        with pytest.raises(ValueError, match="BSC requires fitness values"):
            model = learner.learn(
                generation=0,
                n_vars=n_vars,
                cardinality=cardinality,
                population=population,
                fitness=None,
            )

    def test_bsc_fitness_weighting(self):
        """Test that BSC properly weights probabilities by fitness"""
        np.random.seed(42)
        n_vars = 5
        cardinality = np.array([2] * n_vars)

        # Create a population where high fitness individuals all have value 1
        # and low fitness individuals all have value 0
        population = np.array(
            [
                [1, 1, 1, 1, 1],  # High fitness
                [1, 1, 1, 1, 1],  # High fitness
                [1, 1, 1, 1, 1],  # High fitness
                [0, 0, 0, 0, 0],  # Low fitness
                [0, 0, 0, 0, 0],  # Low fitness
            ]
        )
        fitness = np.array([10.0, 10.0, 10.0, 1.0, 1.0])

        learner = LearnBSC(normalize_fitness=False)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Probability of value 1 should be weighted toward high fitness
        # Total fitness for value 1: 30, Total fitness for value 0: 2
        # P(1) should be approximately 30/32 = 0.9375
        for i in range(n_vars):
            prob_1 = model.parameters[i][1]
            assert prob_1 > 0.85  # Should be heavily biased toward 1

    def test_bsc_on_onemax(self):
        """Test BSC on OneMax problem"""
        np.random.seed(42)

        def onemax(x):
            return np.sum(x, axis=1).astype(float)

        n_vars = 20
        pop_size = 100
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        learner = LearnBSC()

        # Run BSC
        for gen in range(30):
            fitness = onemax(population)

            # Select best 30%
            idx = np.argsort(-fitness)[:30]
            selected = population[idx]

            # Learn model
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population
            sampler = SampleFDA(n_samples=pop_size)
            population = sampler.sample(n_vars=n_vars, model=model, cardinality=cardinality)

        final_fitness = onemax(population)
        # Should converge reasonably well
        assert np.mean(final_fitness) > n_vars * 0.80


class TestMIMIC:
    """Test Mutual Information Maximization for Input Clustering (MIMIC)"""

    def test_learn_mimic_basic(self):
        """Test basic MIMIC learning"""
        np.random.seed(42)
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnMIMIC()
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        assert model is not None
        assert len(model.parameters) == n_vars  # One table per variable
        assert model.metadata["model_type"] == "MIMIC"
        assert "chain_order" in model.metadata

    def test_mimic_chain_structure(self):
        """Test that MIMIC creates a valid chain structure"""
        np.random.seed(42)
        n_vars = 10
        population = np.random.randint(0, 2, (100, n_vars))
        cardinality = np.array([2] * n_vars)
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnMIMIC()
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Check chain order includes all variables
        chain_order = model.metadata["chain_order"]
        assert len(chain_order) == n_vars
        assert len(set(chain_order)) == n_vars  # All unique

        # First clique should be univariate (root)
        assert model.structure[0, 0] == 0  # No overlapping variables
        assert model.structure[0, 1] == 1  # One new variable

    def test_mimic_on_onemax(self):
        """Test MIMIC on OneMax problem"""
        np.random.seed(42)

        def onemax(x):
            return np.sum(x, axis=1).astype(float)

        n_vars = 20
        pop_size = 150
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        learner = LearnMIMIC()

        # Run MIMIC
        for gen in range(25):
            fitness = onemax(population)

            # Select best 40%
            idx = np.argsort(-fitness)[: int(0.4 * pop_size)]
            selected = population[idx]

            # Learn model
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population
            sampler = SampleFDA(n_samples=pop_size)
            population = sampler.sample(n_vars=n_vars, model=model, cardinality=cardinality)

        final_fitness = onemax(population)
        # Should converge reasonably well
        assert np.mean(final_fitness) > n_vars * 0.75

    def test_mimic_on_correlated_problem(self):
        """Test MIMIC on a problem with dependencies"""
        np.random.seed(42)

        def two_peaks(x):
            """Fitness is high if variables are all 0 or all 1"""
            n_ones = np.sum(x, axis=1)
            n_vars = x.shape[1]
            # Reward patterns close to all 0s or all 1s
            fitness = np.minimum(n_ones, n_vars - n_ones).astype(float)
            return -fitness  # Negate so we maximize (want all same value)

        n_vars = 15
        pop_size = 200
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        learner = LearnMIMIC()

        # Run MIMIC
        best_fitness_history = []
        for gen in range(30):
            fitness = two_peaks(population)
            best_fitness_history.append(np.max(fitness))

            # Select
            idx = np.argsort(-fitness)[: int(0.3 * pop_size)]
            selected = population[idx]

            # Learn model
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population
            sampler = SampleFDA(n_samples=pop_size)
            population = sampler.sample(n_vars=n_vars, model=model, cardinality=cardinality)

        # Should find one of the peaks (all 0s or all 1s)
        final_best_fitness = np.max(two_peaks(population))
        assert final_best_fitness == 0.0  # Perfect solution


class TestComparison:
    """Compare the three new EDAs on standard problems"""

    def test_all_edas_on_onemax(self):
        """Compare PBIL, BSC, and MIMIC on OneMax"""
        np.random.seed(42)

        def onemax(x):
            return np.sum(x, axis=1).astype(float)

        n_vars = 15
        pop_size = 100
        cardinality = np.array([2] * n_vars)
        n_gens = 25

        results = {}

        for name, learner in [
            ("PBIL", LearnPBIL(alpha=0.1)),
            ("BSC", LearnBSC()),
            ("MIMIC", LearnMIMIC()),
        ]:
            np.random.seed(42)  # Same initial population for fair comparison
            population = np.random.randint(0, 2, (pop_size, n_vars))

            for gen in range(n_gens):
                fitness = onemax(population)
                idx = np.argsort(-fitness)[:30]

                model = learner.learn(
                    generation=gen,
                    n_vars=n_vars,
                    cardinality=cardinality,
                    population=population[idx],
                    fitness=fitness[idx],
                )

                sampler = SampleFDA(n_samples=pop_size)
                population = sampler.sample(n_vars=n_vars, model=model, cardinality=cardinality)

            final_fitness = onemax(population)
            results[name] = np.mean(final_fitness)

        # All should converge reasonably well
        for name, mean_fitness in results.items():
            assert mean_fitness > n_vars * 0.7, f"{name} failed to converge on OneMax"
