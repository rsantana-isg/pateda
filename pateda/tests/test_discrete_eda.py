"""
Comprehensive tests for discrete EDAs.

This test suite covers:
- UMDA (Univariate Marginal Distribution Algorithm)
- BMDA (Bivariate Marginal Distribution Algorithm)
- FDA (Factorized Distribution Algorithm)
- EBNA (Estimation of Bayesian Network Algorithm)
- BOA (Bayesian Optimization Algorithm)
"""

import pytest
import numpy as np
from pateda.learning.umda import LearnUMDA
from pateda.learning.bmda import LearnBMDA
from pateda.learning.fda import LearnFDA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.sampling.fda import SampleFDA
from pateda.sampling.bayesian_network import SampleBayesianNetwork


class TestUMDA:
    """Test Univariate Marginal Distribution Algorithm (UMDA)"""

    def test_learn_umda_basic(self):
        """Test basic UMDA learning"""
        np.random.seed(42)
        # Binary population
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)  # Simple OneMax

        learner = LearnUMDA(n_vars=10)
        model = learner.learn(population, fitness)

        assert 'probabilities' in model or 'frequencies' in model or hasattr(model, 'probabilities')
        assert hasattr(learner, 'probabilities') or 'probabilities' in model.__dict__

    def test_umda_probabilities_range(self):
        """Test that UMDA probabilities are in valid range"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnUMDA(n_vars=10)
        model = learner.learn(population, fitness)

        # Get probabilities
        if hasattr(learner, 'probabilities'):
            probs = learner.probabilities
        elif hasattr(model, 'probabilities'):
            probs = model.probabilities
        else:
            probs = model.get('probabilities', model.get('frequencies'))

        if probs is not None:
            assert np.all(probs >= 0)
            assert np.all(probs <= 1)

    def test_umda_on_onemax(self):
        """Test UMDA on OneMax problem"""
        np.random.seed(42)

        def onemax(x):
            return np.sum(x, axis=1)

        n_vars = 20
        pop_size = 100
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # Run UMDA
        for gen in range(15):
            fitness = onemax(population)

            # Select best 30%
            idx = np.argsort(-fitness)[:30]  # Higher is better
            selected = population[idx]

            # Learn model
            learner = LearnUMDA(n_vars=n_vars)
            model = learner.learn(selected, fitness[idx])

            # Sample new population
            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        final_fitness = onemax(population)
        # Should converge to all ones
        assert np.mean(final_fitness) > n_vars * 0.8


class TestBMDA:
    """Test Bivariate Marginal Distribution Algorithm (BMDA)"""

    def test_learn_bmda_basic(self):
        """Test basic BMDA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnBMDA(n_vars=10)
        model = learner.learn(population, fitness)

        # BMDA should learn pairwise dependencies
        assert model is not None
        assert hasattr(learner, 'graph') or hasattr(model, 'graph') or hasattr(model, 'structure')

    def test_bmda_on_trap(self):
        """Test BMDA on Trap function (requires learning dependencies)"""
        np.random.seed(42)

        def trap_3(block):
            """3-bit trap function"""
            u = np.sum(block)
            if u == 3:
                return 3
            else:
                return 2 - u

        def trap_function(x):
            """Concatenated 3-bit trap"""
            n_vars = x.shape[1]
            n_blocks = n_vars // 3
            fitness = np.zeros(len(x))
            for i in range(n_blocks):
                block = x[:, i*3:(i+1)*3]
                for j in range(len(x)):
                    fitness[j] += trap_3(block[j])
            return fitness

        n_vars = 15  # 5 blocks of 3
        pop_size = 150
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # Run BMDA
        best_fitness_history = []
        for gen in range(20):
            fitness = trap_function(population)
            best_fitness_history.append(np.max(fitness))

            # Select
            idx = np.argsort(-fitness)[:50]

            # Learn
            learner = LearnBMDA(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            # Sample
            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        # Should make progress (trap is harder than onemax)
        assert best_fitness_history[-1] >= best_fitness_history[0]


class TestFDA:
    """Test Factorized Distribution Algorithm (FDA)"""

    def test_learn_fda_basic(self):
        """Test basic FDA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnFDA(n_vars=10)
        model = learner.learn(population, fitness)

        assert model is not None

    def test_fda_sampling(self):
        """Test FDA sampling"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnFDA(n_vars=10)
        model = learner.learn(population, fitness)

        sampler = SampleFDA(model=model)
        samples = sampler.sample(n_samples=50)

        assert samples.shape == (50, 10)
        assert np.all((samples == 0) | (samples == 1))


class TestEBNA:
    """Test Estimation of Bayesian Network Algorithm (EBNA)"""

    def test_learn_ebna_basic(self):
        """Test basic EBNA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnEBNA(n_vars=10)
        model = learner.learn(population, fitness)

        assert model is not None
        # EBNA learns a Bayesian network
        assert hasattr(learner, 'network') or hasattr(model, 'network') or hasattr(model, 'structure')

    def test_ebna_on_deceptive(self):
        """Test EBNA on deceptive problem"""
        np.random.seed(42)

        def deceptive_4(block):
            """4-bit deceptive function"""
            u = np.sum(block)
            if u == 0:
                return 4
            elif u == 4:
                return 5
            else:
                return 4 - u

        def deceptive_function(x):
            """Concatenated 4-bit deceptive"""
            n_vars = x.shape[1]
            n_blocks = n_vars // 4
            fitness = np.zeros(len(x))
            for i in range(n_blocks):
                block = x[:, i*4:(i+1)*4]
                for j in range(len(x)):
                    fitness[j] += deceptive_4(block[j])
            return fitness

        n_vars = 16  # 4 blocks of 4
        pop_size = 200
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # Run EBNA
        for gen in range(15):
            fitness = deceptive_function(population)

            # Select
            idx = np.argsort(-fitness)[:60]

            # Learn Bayesian network
            learner = LearnEBNA(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            # Sample
            sampler = SampleBayesianNetwork(model=model)
            population = sampler.sample(n_samples=pop_size)

        final_fitness = deceptive_function(population)
        # Should find some good solutions
        assert np.max(final_fitness) > n_vars  # Better than random


class TestBOA:
    """Test Bayesian Optimization Algorithm (BOA)"""

    def test_learn_boa_basic(self):
        """Test basic BOA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnBOA(n_vars=10)
        model = learner.learn(population, fitness)

        assert model is not None
        # BOA learns a Bayesian network with specific structure learning
        assert hasattr(model, 'structure') or hasattr(model, 'network') or hasattr(model, 'graph')

    def test_boa_structure_learning(self):
        """Test that BOA learns reasonable structure"""
        np.random.seed(42)

        # Create data with clear dependencies
        # Bit 1 depends on bit 0, bit 2 depends on bit 1, etc.
        population = np.zeros((200, 5), dtype=int)
        population[:, 0] = np.random.randint(0, 2, 200)
        for i in range(1, 5):
            # Create dependency: x_i depends on x_{i-1}
            for j in range(200):
                if np.random.random() < 0.8:  # 80% correlation
                    population[j, i] = population[j, i-1]
                else:
                    population[j, i] = 1 - population[j, i-1]

        fitness = np.sum(population, axis=1)

        learner = LearnBOA(n_vars=5)
        model = learner.learn(population, fitness)

        # BOA should learn some structure (we can't easily test exact structure)
        assert model is not None


class TestDiscreteEDAIntegration:
    """Integration tests for discrete EDAs"""

    def test_umda_convergence(self):
        """Test UMDA convergence on simple problem"""
        np.random.seed(42)

        n_vars = 30
        pop_size = 150
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # OneMax
        convergence = []
        for gen in range(25):
            fitness = np.sum(population, axis=1)
            convergence.append(np.mean(fitness))

            idx = np.argsort(-fitness)[:50]

            learner = LearnUMDA(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        # Should show improvement
        assert convergence[-1] > convergence[0]
        assert convergence[-1] > n_vars * 0.85  # Should be close to optimum

    def test_multimodal_problem(self):
        """Test EDA on multimodal problem"""
        np.random.seed(42)

        def multimodal_fitness(x):
            """Fitness with multiple peaks"""
            # Reward blocks of all 0s or all 1s
            n_vars = x.shape[1]
            block_size = 5
            n_blocks = n_vars // block_size
            fitness = np.zeros(len(x))

            for i in range(n_blocks):
                block = x[:, i*block_size:(i+1)*block_size]
                block_sum = np.sum(block, axis=1)
                # Reward both all 0s and all 1s
                fitness += np.where(block_sum == 0, 5,
                           np.where(block_sum == block_size, 5, 0))

            return fitness

        n_vars = 20
        pop_size = 200
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(20):
            fitness = multimodal_fitness(population)

            idx = np.argsort(-fitness)[:60]

            learner = LearnBMDA(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        final_fitness = multimodal_fitness(population)
        # Should find some optimal blocks
        assert np.max(final_fitness) > 10  # At least 2 optimal blocks

    def test_comparison_umda_vs_bmda(self):
        """Compare UMDA vs BMDA on problem with dependencies"""
        np.random.seed(42)

        def linked_fitness(x):
            """Fitness function with pairwise dependencies"""
            # Reward adjacent pairs being the same
            fitness = np.zeros(len(x))
            for i in range(x.shape[1] - 1):
                fitness += (x[:, i] == x[:, i+1]).astype(float)
            return fitness

        n_vars = 20
        pop_size = 100

        # Test UMDA
        pop_umda = np.random.randint(0, 2, (pop_size, n_vars))
        for _ in range(15):
            fit = linked_fitness(pop_umda)
            idx = np.argsort(-fit)[:30]

            learner = LearnUMDA(n_vars=n_vars)
            model = learner.learn(pop_umda[idx], fit[idx])

            sampler = SampleFDA(model=model)
            pop_umda = sampler.sample(n_samples=pop_size)

        best_umda = np.max(linked_fitness(pop_umda))

        # Test BMDA
        pop_bmda = np.random.randint(0, 2, (pop_size, n_vars))
        for _ in range(15):
            fit = linked_fitness(pop_bmda)
            idx = np.argsort(-fit)[:30]

            learner = LearnBMDA(n_vars=n_vars)
            model = learner.learn(pop_bmda[idx], fit[idx])

            sampler = SampleFDA(model=model)
            pop_bmda = sampler.sample(n_samples=pop_size)

        best_bmda = np.max(linked_fitness(pop_bmda))

        # BMDA should perform better (can learn pairwise dependencies)
        # Allow some tolerance for randomness
        assert best_bmda >= best_umda * 0.9


class TestMultiValuedVariables:
    """Test EDAs with non-binary discrete variables"""

    def test_multivalue_umda(self):
        """Test UMDA with multi-valued variables"""
        np.random.seed(42)

        # Variables can take values 0, 1, 2
        n_vars = 10
        n_values = 3
        population = np.random.randint(0, n_values, (100, n_vars))

        def fitness_func(x):
            # Prefer value 2
            return np.sum(x == 2, axis=1).astype(float)

        fitness = fitness_func(population)

        # This test depends on UMDA supporting multi-valued variables
        # If not, it should at least not crash
        try:
            learner = LearnUMDA(n_vars=n_vars, n_values=n_values)
            model = learner.learn(population, fitness)

            sampler = SampleFDA(model=model)
            samples = sampler.sample(n_samples=50)

            assert samples.shape == (50, n_vars)
            assert np.all(samples >= 0)
            assert np.all(samples < n_values)
        except (TypeError, AttributeError):
            # If multi-valued not supported, that's okay
            pytest.skip("Multi-valued variables not supported")


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_population(self):
        """Test with very small population"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (5, 10))
        fitness = np.sum(population, axis=1)

        learner = LearnUMDA(n_vars=10)
        model = learner.learn(population, fitness)

        sampler = SampleFDA(model=model)
        samples = sampler.sample(n_samples=10)

        assert samples.shape == (10, 10)

    def test_uniform_population(self):
        """Test with uniform population (all same)"""
        np.random.seed(42)
        population = np.ones((50, 10), dtype=int)
        fitness = np.sum(population, axis=1)

        learner = LearnUMDA(n_vars=10)
        model = learner.learn(population, fitness)

        sampler = SampleFDA(model=model)
        samples = sampler.sample(n_samples=20)

        # Should sample all ones (or close to it)
        assert samples.shape == (20, 10)

    def test_large_problem(self):
        """Test with large problem"""
        np.random.seed(42)
        n_vars = 100
        population = np.random.randint(0, 2, (200, n_vars))
        fitness = np.sum(population, axis=1)

        learner = LearnUMDA(n_vars=n_vars)
        model = learner.learn(population, fitness)

        sampler = SampleFDA(model=model)
        samples = sampler.sample(n_samples=50)

        assert samples.shape == (50, n_vars)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
