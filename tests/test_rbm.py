"""
Tests for RBM-based EDAs
"""

import pytest
import numpy as np
import torch

from pateda.learning.rbm import (
    learn_softmax_rbm,
    SoftmaxRBM,
)
from pateda.sampling.rbm import (
    sample_softmax_rbm,
    sample_softmax_rbm_with_surrogate,
)


class TestRBMBasic:
    """Test basic RBM functionality"""

    def test_learn_softmax_rbm_binary(self):
        """Test Softmax RBM learning on binary problems"""
        # Create synthetic binary population
        np.random.seed(42)
        n_samples = 100
        n_vars = 10
        population = np.random.randint(0, 2, (n_samples, n_vars))
        fitness = np.sum(population, axis=1)  # OneMax problem
        cardinality = np.array([2] * n_vars)  # Binary variables

        # Learn RBM model
        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 5, 'epochs': 10, 'batch_size': 16, 'k_cd': 1}
        )

        # Check model structure
        assert 'rbm_state' in model
        assert 'n_vars' in model
        assert 'cardinality' in model
        assert 'n_hidden' in model
        assert model['type'] == 'softmax_rbm'
        assert model['n_vars'] == n_vars
        assert model['n_hidden'] == 5
        assert np.array_equal(model['cardinality'], cardinality)

    def test_learn_softmax_rbm_discrete(self):
        """Test Softmax RBM learning on discrete problems with K>2"""
        np.random.seed(42)
        n_samples = 80
        n_vars = 5
        # Variables with different cardinalities
        cardinality = np.array([3, 4, 2, 5, 3])

        # Generate random discrete population
        population = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            population[:, i] = np.random.randint(0, cardinality[i], n_samples)

        fitness = np.sum(population, axis=1)

        # Learn RBM model
        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 8, 'epochs': 15, 'batch_size': 20}
        )

        # Check model structure
        assert model['type'] == 'softmax_rbm'
        assert model['n_vars'] == n_vars
        assert np.array_equal(model['cardinality'], cardinality)

    def test_sample_softmax_rbm_basic(self):
        """Test basic Softmax RBM sampling"""
        # Create and learn model
        np.random.seed(42)
        n_samples = 100
        n_vars = 8
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (n_samples, n_vars))
        fitness = np.sum(population, axis=1)

        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 4, 'epochs': 10, 'batch_size': 16}
        )

        # Sample from model
        new_population = sample_softmax_rbm(
            model, n_samples=50, cardinality=cardinality,
            params={'n_gibbs_steps': 5, 'burn_in': 50}
        )

        # Check shape and values
        assert new_population.shape == (50, n_vars)
        assert np.all(new_population >= 0)
        assert np.all(new_population < 2)
        assert new_population.dtype == int

    def test_sample_softmax_rbm_discrete(self):
        """Test Softmax RBM sampling with discrete variables"""
        np.random.seed(42)
        n_samples = 80
        n_vars = 5
        cardinality = np.array([3, 4, 2, 5, 3])

        population = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            population[:, i] = np.random.randint(0, cardinality[i], n_samples)

        fitness = np.sum(population, axis=1)

        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 8, 'epochs': 10}
        )

        # Sample from model
        new_population = sample_softmax_rbm(
            model, n_samples=50, cardinality=cardinality,
            params={'n_gibbs_steps': 10, 'burn_in': 80}
        )

        # Check that values respect cardinality
        assert new_population.shape == (50, n_vars)
        for i in range(n_vars):
            assert np.all(new_population[:, i] >= 0)
            assert np.all(new_population[:, i] < cardinality[i])

    def test_rbm_encode_decode(self):
        """Test RBM encoding and decoding"""
        np.random.seed(42)
        n_vars = 5
        cardinality = np.array([2, 3, 4, 2, 3])
        n_hidden = 10

        # Create RBM
        rbm = SoftmaxRBM(n_vars, cardinality, n_hidden)

        # Test population
        test_pop = np.array([
            [0, 1, 2, 1, 0],
            [1, 2, 3, 0, 2],
            [0, 0, 1, 1, 1]
        ])

        # Encode then decode
        encoded = rbm._encode_population(test_pop)
        decoded = rbm._decode_visible(encoded)

        # Should recover original
        assert np.array_equal(decoded, test_pop)


class TestRBMSurrogate:
    """Test RBM energy-based surrogate functionality"""

    def test_sample_with_surrogate(self):
        """Test RBM sampling with energy-based filtering"""
        np.random.seed(42)
        n_samples = 100
        n_vars = 10
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (n_samples, n_vars))
        fitness = np.sum(population, axis=1)

        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 6, 'epochs': 15}
        )

        # Sample with surrogate filtering
        new_population = sample_softmax_rbm_with_surrogate(
            model, n_samples=30, cardinality=cardinality,
            params={
                'oversample_factor': 3,
                'energy_percentile': 40,
                'n_gibbs_steps': 10
            }
        )

        # Check shape and values
        assert new_population.shape == (30, n_vars)
        assert np.all(new_population >= 0)
        assert np.all(new_population < 2)

    def test_free_energy_computation(self):
        """Test free energy computation"""
        np.random.seed(42)
        n_vars = 5
        cardinality = np.array([2] * n_vars)
        n_hidden = 8

        rbm = SoftmaxRBM(n_vars, cardinality, n_hidden)

        # Create test samples
        test_pop = np.random.randint(0, 2, (10, n_vars))
        encoded = rbm._encode_population(test_pop)

        # Compute free energy
        energy = rbm.free_energy(encoded)

        # Should return one energy value per sample
        assert energy.shape == (10,)
        assert torch.is_tensor(energy)


class TestRBMIntegration:
    """Integration tests for RBM with optimization"""

    def test_rbm_optimization_onemax(self):
        """Test RBM on OneMax problem"""
        np.random.seed(42)
        torch.manual_seed(42)

        # OneMax: maximize number of ones
        def onemax(x):
            return -np.sum(x, axis=1)  # Negative for minimization

        # Initialize population
        n_vars = 20
        pop_size = 100
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # Simple EDA loop
        n_generations = 15
        selection_size = int(pop_size * 0.3)

        for gen in range(n_generations):
            # Evaluate
            fitness = onemax(population)

            # Select best
            idx = np.argsort(fitness)[:selection_size]
            selected_pop = population[idx]
            selected_fit = fitness[idx]

            # Learn model
            model = learn_softmax_rbm(
                selected_pop, selected_fit, cardinality,
                params={'n_hidden': n_vars, 'epochs': 10, 'batch_size': 20}
            )

            # Sample new population
            population = sample_softmax_rbm(
                model, n_samples=pop_size, cardinality=cardinality,
                params={'n_gibbs_steps': 10, 'burn_in': 100}
            )

        # Final evaluation
        final_fitness = onemax(population)
        best_fitness = np.min(final_fitness)
        best_solution = population[np.argmin(final_fitness)]

        # Should converge to all ones (fitness = -n_vars)
        assert best_fitness < -15  # Should find mostly ones
        assert np.sum(best_solution) > 15  # Best solution should have many ones

    def test_rbm_optimization_leadingones(self):
        """Test RBM on LeadingOnes problem"""
        np.random.seed(42)
        torch.manual_seed(42)

        # LeadingOnes: count consecutive ones from the start
        def leadingones(x):
            fitness = np.zeros(len(x))
            for i in range(len(x)):
                count = 0
                for j in range(x.shape[1]):
                    if x[i, j] == 1:
                        count += 1
                    else:
                        break
                fitness[i] = -count  # Negative for minimization
            return fitness

        # Initialize
        n_vars = 15
        pop_size = 80
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        # Run optimization
        n_generations = 20
        selection_size = int(pop_size * 0.25)

        for gen in range(n_generations):
            fitness = leadingones(population)
            idx = np.argsort(fitness)[:selection_size]

            model = learn_softmax_rbm(
                population[idx], fitness[idx], cardinality,
                params={'n_hidden': n_vars, 'epochs': 12}
            )

            population = sample_softmax_rbm(
                model, n_samples=pop_size, cardinality=cardinality
            )

        final_fitness = leadingones(population)
        best_fitness = np.min(final_fitness)

        # Should make significant progress
        assert best_fitness < -8  # Should find solutions with many leading ones

    def test_rbm_with_surrogate_optimization(self):
        """Test RBM with energy-based surrogate on optimization"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Simple additive function
        def additive(x):
            # Each bit contributes differently
            weights = np.arange(1, x.shape[1] + 1)
            return -np.dot(x, weights)  # Negative for minimization

        n_vars = 12
        pop_size = 60
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        n_generations = 10
        selection_size = int(pop_size * 0.4)

        for gen in range(n_generations):
            fitness = additive(population)
            idx = np.argsort(fitness)[:selection_size]

            model = learn_softmax_rbm(
                population[idx], fitness[idx], cardinality,
                params={'n_hidden': n_vars, 'epochs': 10}
            )

            # Use surrogate filtering
            population = sample_softmax_rbm_with_surrogate(
                model, n_samples=pop_size, cardinality=cardinality,
                params={'oversample_factor': 2, 'energy_percentile': 50}
            )

        final_fitness = additive(population)
        best_fitness = np.min(final_fitness)

        # Should find good solutions (optimum is all ones)
        assert best_fitness < -50  # Should be close to optimum


class TestRBMEdgeCases:
    """Test edge cases and robustness"""

    def test_rbm_small_population(self):
        """Test RBM with small population"""
        np.random.seed(42)
        n_samples = 20  # Small population
        n_vars = 8
        cardinality = np.array([2] * n_vars)
        population = np.random.randint(0, 2, (n_samples, n_vars))
        fitness = np.sum(population, axis=1)

        # Should handle small population
        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 4, 'epochs': 5}
        )

        new_population = sample_softmax_rbm(
            model, n_samples=10, cardinality=cardinality
        )

        assert new_population.shape == (10, n_vars)

    def test_rbm_large_cardinality(self):
        """Test RBM with large cardinality variables"""
        np.random.seed(42)
        n_samples = 60
        n_vars = 4
        cardinality = np.array([10, 8, 12, 6])  # Large cardinalities

        population = np.zeros((n_samples, n_vars), dtype=int)
        for i in range(n_vars):
            population[:, i] = np.random.randint(0, cardinality[i], n_samples)

        fitness = np.sum(population, axis=1)

        model = learn_softmax_rbm(
            population, fitness, cardinality,
            params={'n_hidden': 15, 'epochs': 10}
        )

        new_population = sample_softmax_rbm(
            model, n_samples=30, cardinality=cardinality
        )

        # Check cardinality constraints
        assert new_population.shape == (30, n_vars)
        for i in range(n_vars):
            assert np.all(new_population[:, i] >= 0)
            assert np.all(new_population[:, i] < cardinality[i])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
