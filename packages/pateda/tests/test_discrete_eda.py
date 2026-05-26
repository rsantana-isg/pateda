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
from pateda.core.models import FactorizedModel, BayesianNetworkModel


def _learn(learner, pop, fit, n_vars=None):
    """Helper: call learn() with current 5-arg API."""
    nv = n_vars if n_vars is not None else pop.shape[1]
    card = np.full(nv, 2)
    return learner.learn(0, nv, card, pop, fit)


def _sample_fda(model, n_vars, n_samples):
    """Helper: call SampleFDA with current API."""
    card = np.full(n_vars, 2)
    return SampleFDA(n_samples=n_samples).sample(n_vars, model, card)


class TestUMDA:
    """Test Univariate Marginal Distribution Algorithm (UMDA)"""

    def test_learn_umda_basic(self):
        """Test basic UMDA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnUMDA()
        model = _learn(learner, population, fitness)

        assert isinstance(model, FactorizedModel)
        assert model.parameters is not None

    def test_umda_probabilities_range(self):
        """Test that UMDA probabilities are in valid range"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnUMDA()
        model = _learn(learner, population, fitness)

        # parameters is a list of 1D arrays (one per variable)
        for p in model.parameters:
            p_arr = np.asarray(p)
            assert np.all(p_arr >= 0)
            assert np.all(p_arr <= 1 + 1e-9)

    def test_umda_on_onemax(self):
        """Test UMDA on OneMax problem"""
        np.random.seed(42)
        n_vars = 20
        pop_size = 100
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(15):
            fitness = np.sum(population, axis=1).astype(float)
            idx = np.argsort(-fitness)[:30]
            selected = population[idx]

            learner = LearnUMDA()
            model = learner.learn(gen, n_vars, card, selected, fitness[idx])
            population = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)

        final_fitness = np.sum(population, axis=1)
        assert np.mean(final_fitness) > n_vars * 0.8


class TestBMDA:
    """Test Bivariate Marginal Distribution Algorithm (BMDA)"""

    def test_learn_bmda_basic(self):
        """Test basic BMDA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnBMDA()
        model = _learn(learner, population, fitness)

        assert isinstance(model, FactorizedModel)
        assert model.structure is not None

    def test_bmda_on_trap(self):
        """Test BMDA on Trap function (requires learning dependencies)"""
        np.random.seed(42)

        def trap_3(block):
            u = np.sum(block)
            return 3 if u == 3 else 2 - u

        def trap_function(x):
            n_vars = x.shape[1]
            n_blocks = n_vars // 3
            fitness = np.zeros(len(x))
            for i in range(n_blocks):
                block = x[:, i*3:(i+1)*3]
                for j in range(len(x)):
                    fitness[j] += trap_3(block[j])
            return fitness

        n_vars = 15
        pop_size = 150
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        best_fitness_history = []
        for gen in range(20):
            fitness = trap_function(population)
            best_fitness_history.append(np.max(fitness))
            idx = np.argsort(-fitness)[:50]

            learner = LearnBMDA()
            model = learner.learn(gen, n_vars, card, population[idx], fitness[idx])
            population = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)

        assert best_fitness_history[-1] >= best_fitness_history[0]


class TestFDA:
    """Test Factorized Distribution Algorithm (FDA)"""

    def test_learn_fda_basic(self):
        """Test basic FDA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnFDA()
        model = _learn(learner, population, fitness)

        assert isinstance(model, FactorizedModel)

    def test_fda_sampling(self):
        """Test FDA sampling"""
        np.random.seed(42)
        n_vars = 10
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (100, n_vars))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnFDA()
        model = learner.learn(0, n_vars, card, population, fitness)
        samples = SampleFDA(n_samples=50).sample(n_vars, model, card)

        assert samples.shape == (50, n_vars)
        assert np.all((samples == 0) | (samples == 1))


class TestEBNA:
    """Test Estimation of Bayesian Network Algorithm (EBNA)"""

    def test_learn_ebna_basic(self):
        """Test basic EBNA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnEBNA()
        model = _learn(learner, population, fitness)

        assert model is not None
        assert isinstance(model, (FactorizedModel, BayesianNetworkModel))

    def test_ebna_on_deceptive(self):
        """Test EBNA on deceptive problem"""
        np.random.seed(42)

        def deceptive_4(block):
            u = np.sum(block)
            if u == 0:
                return 4
            elif u == 4:
                return 5
            else:
                return 4 - u

        def deceptive_function(x):
            n_vars = x.shape[1]
            n_blocks = n_vars // 4
            fitness = np.zeros(len(x))
            for i in range(n_blocks):
                block = x[:, i*4:(i+1)*4]
                for j in range(len(x)):
                    fitness[j] += deceptive_4(block[j])
            return fitness

        n_vars = 16
        pop_size = 200
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(15):
            fitness = deceptive_function(population)
            idx = np.argsort(-fitness)[:60]

            learner = LearnEBNA()
            model = learner.learn(gen, n_vars, card, population[idx], fitness[idx])
            population = SampleBayesianNetwork(n_samples=pop_size).sample(n_vars, model, card)

        final_fitness = deceptive_function(population)
        assert np.max(final_fitness) > n_vars


class TestBOA:
    """Test Bayesian Optimization Algorithm (BOA)"""

    def test_learn_boa_basic(self):
        """Test basic BOA learning"""
        np.random.seed(42)
        population = np.random.randint(0, 2, (100, 10))
        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnBOA()
        model = _learn(learner, population, fitness)

        assert isinstance(model, (FactorizedModel, BayesianNetworkModel))

    def test_boa_structure_learning(self):
        """Test that BOA learns reasonable structure"""
        np.random.seed(42)
        population = np.zeros((200, 5), dtype=int)
        population[:, 0] = np.random.randint(0, 2, 200)
        for i in range(1, 5):
            for j in range(200):
                population[j, i] = (population[j, i-1]
                                    if np.random.random() < 0.8
                                    else 1 - population[j, i-1])

        fitness = np.sum(population, axis=1).astype(float)

        learner = LearnBOA()
        model = _learn(learner, population, fitness)

        assert isinstance(model, (FactorizedModel, BayesianNetworkModel))


class TestDiscreteEDAIntegration:
    """Integration tests for discrete EDAs"""

    def test_umda_convergence(self):
        """Test UMDA convergence on simple problem"""
        np.random.seed(42)
        n_vars = 30
        pop_size = 150
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        convergence = []
        for gen in range(25):
            fitness = np.sum(population, axis=1).astype(float)
            convergence.append(np.mean(fitness))
            idx = np.argsort(-fitness)[:50]

            learner = LearnUMDA()
            model = learner.learn(gen, n_vars, card, population[idx], fitness[idx])
            population = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)

        assert convergence[-1] > convergence[0]
        assert convergence[-1] > n_vars * 0.85

    def test_multimodal_problem(self):
        """Test EDA on multimodal problem"""
        np.random.seed(42)

        def multimodal_fitness(x):
            block_size = 5
            n_blocks = x.shape[1] // block_size
            fitness = np.zeros(len(x))
            for i in range(n_blocks):
                block = x[:, i*block_size:(i+1)*block_size]
                block_sum = np.sum(block, axis=1)
                fitness += np.where(block_sum == 0, 5,
                           np.where(block_sum == block_size, 5, 0))
            return fitness

        n_vars = 20
        pop_size = 200
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(20):
            fitness = multimodal_fitness(population)
            idx = np.argsort(-fitness)[:60]

            learner = LearnBMDA()
            model = learner.learn(gen, n_vars, card, population[idx], fitness[idx])
            population = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)

        final_fitness = multimodal_fitness(population)
        assert np.max(final_fitness) > 10

    def test_comparison_umda_vs_bmda(self):
        """Compare UMDA vs BMDA on problem with dependencies"""
        np.random.seed(42)

        def linked_fitness(x):
            fitness = np.zeros(len(x))
            for i in range(x.shape[1] - 1):
                fitness += (x[:, i] == x[:, i+1]).astype(float)
            return fitness

        n_vars = 20
        pop_size = 100
        card = np.full(n_vars, 2)

        pop_umda = np.random.randint(0, 2, (pop_size, n_vars))
        for gen in range(15):
            fit = linked_fitness(pop_umda)
            idx = np.argsort(-fit)[:30]
            model = LearnUMDA().learn(gen, n_vars, card, pop_umda[idx], fit[idx])
            pop_umda = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)
        best_umda = np.max(linked_fitness(pop_umda))

        pop_bmda = np.random.randint(0, 2, (pop_size, n_vars))
        for gen in range(15):
            fit = linked_fitness(pop_bmda)
            idx = np.argsort(-fit)[:30]
            model = LearnBMDA().learn(gen, n_vars, card, pop_bmda[idx], fit[idx])
            pop_bmda = SampleFDA(n_samples=pop_size).sample(n_vars, model, card)
        best_bmda = np.max(linked_fitness(pop_bmda))

        assert best_bmda >= best_umda * 0.9


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_population(self):
        """Test with very small population"""
        np.random.seed(42)
        n_vars = 10
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (5, n_vars))
        fitness = np.sum(population, axis=1).astype(float)

        model = LearnUMDA().learn(0, n_vars, card, population, fitness)
        samples = SampleFDA(n_samples=10).sample(n_vars, model, card)

        assert samples.shape == (10, n_vars)

    def test_uniform_population(self):
        """Test with uniform population (all same)"""
        np.random.seed(42)
        n_vars = 10
        card = np.full(n_vars, 2)
        population = np.ones((50, n_vars), dtype=int)
        fitness = np.sum(population, axis=1).astype(float)

        model = LearnUMDA().learn(0, n_vars, card, population, fitness)
        samples = SampleFDA(n_samples=20).sample(n_vars, model, card)

        assert samples.shape == (20, n_vars)

    def test_large_problem(self):
        """Test with large problem"""
        np.random.seed(42)
        n_vars = 100
        card = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (200, n_vars))
        fitness = np.sum(population, axis=1).astype(float)

        model = LearnUMDA().learn(0, n_vars, card, population, fitness)
        samples = SampleFDA(n_samples=50).sample(n_vars, model, card)

        assert samples.shape == (50, n_vars)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
