"""
Comprehensive tests for crossover operators in EDAs.

This test suite covers:
- Two-Point Crossover
- Transposition Operator
- Block Crossover (Symmetric Blind Block Crossover)

Each operator is tested with:
1. Basic functionality tests
2. Integration with EDA framework
3. Application to OneMax optimization problem
"""

import pytest
import numpy as np

from pateda.crossover.two_point import LearnTwoPointCrossover, SampleTwoPointCrossover
from pateda.crossover.transposition import LearnTransposition, SampleTransposition
from pateda.crossover.block import LearnBlockCrossover, SampleBlockCrossover
from pateda.mutation.bitflip import bit_flip_mutation
from pateda.functions.discrete import onemax


class TestTwoPointCrossover:
    """Test Two-Point Crossover operator"""

    def test_learn_basic(self):
        """Test basic learning functionality"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (50, 20))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(20) * 2

        learner = LearnTwoPointCrossover(n_offspring=100)
        model = learner.learn(
            generation=0,
            n_vars=20,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Check model structure
        assert model.structure.shape == (50, 2)  # 100/2 pairs, 2 parents each
        assert model.parameters.shape == (50, 2)  # 50 pairs, 2 points each
        assert model.metadata["model_type"] == "TwoPointCrossover"

        # Check that points are in valid range
        points = model.parameters
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 0] < 20)
        assert np.all(points[:, 1] >= points[:, 0])  # point2 >= point1
        assert np.all(points[:, 1] <= 20)

    def test_sample_basic(self):
        """Test basic sampling functionality"""
        np.random.seed(42)
        n_vars = 20
        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        # Learn model
        learner = LearnTwoPointCrossover(n_offspring=100)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Sample new population
        sampler = SampleTwoPointCrossover(n_samples=100)
        new_pop = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=population,
        )

        # Check output
        assert new_pop.shape == (100, n_vars)
        assert np.all((new_pop == 0) | (new_pop == 1))

    def test_sample_with_mutation(self):
        """Test sampling with mutation"""
        np.random.seed(42)
        n_vars = 20
        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        # Learn model
        learner = LearnTwoPointCrossover(n_offspring=100)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Sample with mutation
        sampler = SampleTwoPointCrossover(
            n_samples=100,
            mutation_fn=bit_flip_mutation,
            mutation_params={"mutation_prob": 0.01},
        )
        new_pop = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=population,
        )

        # Check output
        assert new_pop.shape == (100, n_vars)
        assert np.all((new_pop == 0) | (new_pop == 1))

    def test_on_onemax(self):
        """Test Two-Point Crossover on OneMax problem"""
        np.random.seed(42)
        n_vars = 30
        pop_size = 100
        n_generations = 20
        selection_size = 30

        # Initialize population
        population = np.random.randint(0, 2, (pop_size, n_vars))
        cardinality = np.ones(n_vars) * 2

        best_fitness_history = []

        for gen in range(n_generations):
            # Evaluate
            fitness = onemax(population)
            best_fitness_history.append(np.max(fitness))

            # Select best individuals
            idx = np.argsort(-fitness)[:selection_size]
            selected = population[idx]

            # Learn crossover model
            learner = LearnTwoPointCrossover(n_offspring=pop_size)
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population with mutation
            sampler = SampleTwoPointCrossover(
                n_samples=pop_size,
                mutation_fn=bit_flip_mutation,
                mutation_params={"mutation_prob": 1.0 / n_vars},
            )
            population = sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=selected,
            )

        # Check convergence
        final_fitness = onemax(population)
        assert np.mean(final_fitness) > n_vars * 0.7  # Should improve
        assert best_fitness_history[-1] > best_fitness_history[0]  # Progress


class TestTransposition:
    """Test Transposition operator"""

    def test_learn_basic(self):
        """Test basic learning functionality"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (50, 20))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(20) * 2

        learner = LearnTransposition(n_offspring=100)
        model = learner.learn(
            generation=0,
            n_vars=20,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Check model structure
        trans_params = model.structure
        assert trans_params["individuals"].shape == (100,)
        assert trans_params["lengths"].shape == (100,)
        assert trans_params["locations"].shape == (100,)
        assert trans_params["offsets"].shape == (100,)
        assert model.metadata["model_type"] == "Transposition"

        # Check that parameters are in valid range
        assert np.all(trans_params["individuals"] >= 0)
        assert np.all(trans_params["individuals"] < 50)
        assert np.all(trans_params["lengths"] >= 1)
        assert np.all(trans_params["lengths"] <= 10)  # n_vars/2 = 10
        assert np.all(trans_params["locations"] >= 0)
        assert np.all(trans_params["locations"] < 20)

    def test_sample_basic(self):
        """Test basic sampling functionality"""
        np.random.seed(42)
        n_vars = 20
        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        # Learn model
        learner = LearnTransposition(n_offspring=100)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Sample new population
        sampler = SampleTransposition(n_samples=100)
        new_pop = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=population,
        )

        # Check output
        assert new_pop.shape == (100, n_vars)
        assert np.all((new_pop == 0) | (new_pop == 1))

    def test_on_onemax(self):
        """Test Transposition on OneMax problem"""
        np.random.seed(42)
        n_vars = 30
        pop_size = 100
        n_generations = 20
        selection_size = 30

        # Initialize population
        population = np.random.randint(0, 2, (pop_size, n_vars))
        cardinality = np.ones(n_vars) * 2

        best_fitness_history = []

        for gen in range(n_generations):
            # Evaluate
            fitness = onemax(population)
            best_fitness_history.append(np.max(fitness))

            # Select best individuals
            idx = np.argsort(-fitness)[:selection_size]
            selected = population[idx]

            # Learn transposition model
            learner = LearnTransposition(n_offspring=pop_size)
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population with mutation
            sampler = SampleTransposition(
                n_samples=pop_size,
                mutation_fn=bit_flip_mutation,
                mutation_params={"mutation_prob": 1.0 / n_vars},
            )
            population = sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=selected,
            )

        # Check convergence
        final_fitness = onemax(population)
        assert np.mean(final_fitness) > n_vars * 0.7  # Should improve
        assert best_fitness_history[-1] > best_fitness_history[0]  # Progress


class TestBlockCrossover:
    """Test Block Crossover operator"""

    def test_learn_basic(self):
        """Test basic learning functionality"""
        np.random.seed(42)
        # Create a problem with 4 blocks of 5 variables each (20 total)
        n_vars = 20
        n_classes = 4
        class_size = 5
        symmetry_index = np.array([
            [0, 1, 2, 3, 4],       # Block 1
            [5, 6, 7, 8, 9],       # Block 2
            [10, 11, 12, 13, 14],  # Block 3
            [15, 16, 17, 18, 19],  # Block 4
        ])

        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        learner = LearnBlockCrossover(n_offspring=100, symmetry_index=symmetry_index)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Check model structure
        assert model.structure.shape == (50, 2)  # 100/2 pairs, 2 parents each
        assert model.parameters["masks"].shape == (50, n_classes)
        assert model.metadata["model_type"] == "BlockCrossover"
        assert model.metadata["n_classes"] == n_classes

    def test_sample_basic(self):
        """Test basic sampling functionality"""
        np.random.seed(42)
        # Create a problem with 4 blocks of 5 variables each
        n_vars = 20
        symmetry_index = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ])

        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        # Learn model
        learner = LearnBlockCrossover(n_offspring=100, symmetry_index=symmetry_index)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Sample new population
        sampler = SampleBlockCrossover(
            n_samples=100,
            symmetry_index=symmetry_index,
            mutation_prob=0.0,
        )
        new_pop = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=population,
        )

        # Check output
        assert new_pop.shape == (100, n_vars)
        assert np.all((new_pop == 0) | (new_pop == 1))

    def test_sample_with_mutation(self):
        """Test sampling with integrated mutation"""
        np.random.seed(42)
        n_vars = 20
        symmetry_index = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ])

        population = np.random.randint(0, 2, (50, n_vars))
        fitness = np.sum(population, axis=1)
        cardinality = np.ones(n_vars) * 2

        # Learn model
        learner = LearnBlockCrossover(n_offspring=100, symmetry_index=symmetry_index)
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
        )

        # Sample with mutation
        sampler = SampleBlockCrossover(
            n_samples=100,
            symmetry_index=symmetry_index,
            mutation_prob=0.01,
        )
        new_pop = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=population,
        )

        # Check output
        assert new_pop.shape == (100, n_vars)
        assert np.all((new_pop == 0) | (new_pop == 1))

    def test_on_onemax(self):
        """Test Block Crossover on OneMax problem"""
        np.random.seed(42)
        n_vars = 20
        n_classes = 4
        class_size = 5
        symmetry_index = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ])

        pop_size = 100
        n_generations = 20
        selection_size = 30

        # Initialize population
        population = np.random.randint(0, 2, (pop_size, n_vars))
        cardinality = np.ones(n_vars) * 2

        best_fitness_history = []

        for gen in range(n_generations):
            # Evaluate
            fitness = onemax(population)
            best_fitness_history.append(np.max(fitness))

            # Select best individuals
            idx = np.argsort(-fitness)[:selection_size]
            selected = population[idx]

            # Learn block crossover model
            learner = LearnBlockCrossover(
                n_offspring=pop_size,
                symmetry_index=symmetry_index,
            )
            model = learner.learn(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                population=selected,
                fitness=fitness[idx],
            )

            # Sample new population with mutation
            sampler = SampleBlockCrossover(
                n_samples=pop_size,
                symmetry_index=symmetry_index,
                mutation_prob=1.0 / n_vars,
            )
            population = sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=selected,
            )

        # Check convergence
        final_fitness = onemax(population)
        assert np.mean(final_fitness) > n_vars * 0.7  # Should improve
        assert best_fitness_history[-1] > best_fitness_history[0]  # Progress


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
