"""
Tests for VAE-based EDAs
"""

import pytest
import numpy as np
import torch

from pateda.learning.vae import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
)
from pateda.sampling.vae import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
)


class TestVAEBasic:
    """Test basic VAE functionality"""

    def test_learn_vae_basic(self):
        """Test basic VAE learning"""
        # Create synthetic population
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        # Learn VAE model
        model = learn_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10, 'batch_size': 16}
        )

        # Check model structure
        assert 'encoder_state' in model
        assert 'decoder_state' in model
        assert 'latent_dim' in model
        assert 'input_dim' in model
        assert 'ranges' in model
        assert model['type'] == 'vae'
        assert model['latent_dim'] == 3
        assert model['input_dim'] == n_vars

    def test_sample_vae_basic(self):
        """Test basic VAE sampling"""
        # Create and learn model
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10, 'batch_size': 16}
        )

        # Sample from model
        new_population = sample_vae(model, n_samples=50)

        # Check shape
        assert new_population.shape == (50, n_vars)

    def test_vae_with_bounds(self):
        """Test VAE sampling with bounds"""
        # Create model
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10}
        )

        # Sample with bounds
        bounds = np.array([[-5, -5, -5, -5, -5], [5, 5, 5, 5, 5]])
        new_population = sample_vae(model, n_samples=50, bounds=bounds)

        # Check bounds are respected
        assert np.all(new_population >= bounds[0])
        assert np.all(new_population <= bounds[1])


class TestExtendedVAE:
    """Test Extended VAE (E-VAE) functionality"""

    def test_learn_extended_vae(self):
        """Test E-VAE learning"""
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        # Learn E-VAE model
        model = learn_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10, 'batch_size': 16}
        )

        # Check model structure
        assert 'encoder_state' in model
        assert 'decoder_state' in model
        assert 'predictor_state' in model
        assert 'fitness_min' in model
        assert 'fitness_max' in model
        assert model['type'] == 'extended_vae'

    def test_sample_extended_vae(self):
        """Test E-VAE sampling"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10}
        )

        # Sample without predictor
        new_pop1 = sample_extended_vae(model, n_samples=50)
        assert new_pop1.shape == (50, 5)

        # Sample with predictor filtering
        new_pop2 = sample_extended_vae(
            model, n_samples=50,
            params={'use_predictor': True, 'predictor_percentile': 30}
        )
        assert new_pop2.shape == (50, 5)

    def test_extended_vae_multiobjective(self):
        """Test E-VAE with multiple objectives"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        # Two objectives
        fitness = np.column_stack([
            np.sum(population ** 2, axis=1),
            np.sum(np.abs(population), axis=1)
        ])

        model = learn_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10}
        )

        assert model['n_objectives'] == 2

        new_population = sample_extended_vae(model, n_samples=50)
        assert new_population.shape == (50, 5)


class TestConditionalExtendedVAE:
    """Test Conditional Extended VAE (CE-VAE) functionality"""

    def test_learn_conditional_extended_vae(self):
        """Test CE-VAE learning"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        # Learn CE-VAE model
        model = learn_conditional_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10, 'batch_size': 16}
        )

        # Check model structure
        assert 'encoder_state' in model
        assert 'conditional_decoder_state' in model
        assert 'predictor_state' in model
        assert model['type'] == 'conditional_extended_vae'

    def test_sample_conditional_extended_vae(self):
        """Test CE-VAE sampling with fitness conditioning"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_conditional_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10}
        )

        # Sample with default target fitness (best from training)
        new_pop1 = sample_conditional_extended_vae(model, n_samples=50)
        assert new_pop1.shape == (50, 5)

        # Sample with specified target fitness
        target_fitness = np.min(fitness)  # Target the best fitness
        new_pop2 = sample_conditional_extended_vae(
            model, n_samples=50,
            params={'target_fitness': target_fitness, 'fitness_noise': 0.05}
        )
        assert new_pop2.shape == (50, 5)

    def test_conditional_vae_multiobjective(self):
        """Test CE-VAE with multiple objectives"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.column_stack([
            np.sum(population ** 2, axis=1),
            np.sum(np.abs(population), axis=1)
        ])

        model = learn_conditional_extended_vae(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 10}
        )

        # Sample with multiobjective target
        target_fitness = np.min(fitness, axis=0)
        new_population = sample_conditional_extended_vae(
            model, n_samples=50,
            params={'target_fitness': target_fitness}
        )
        assert new_population.shape == (50, 5)


class TestVAEIntegration:
    """Integration tests for VAE with optimization"""

    def test_vae_optimization_sphere(self):
        """Test VAE on sphere function"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Sphere function: minimize sum of squares
        def sphere(x):
            return np.sum(x ** 2, axis=1)

        # Initialize population
        n_vars = 10
        pop_size = 50
        population = np.random.uniform(-5, 5, (pop_size, n_vars))

        # Simple EDA loop
        n_generations = 10
        selection_size = int(pop_size * 0.5)

        for gen in range(n_generations):
            # Evaluate
            fitness = sphere(population)

            # Select best
            idx = np.argsort(fitness)[:selection_size]
            selected_pop = population[idx]
            selected_fit = fitness[idx]

            # Learn model
            model = learn_vae(
                selected_pop, selected_fit,
                params={'latent_dim': 5, 'epochs': 20, 'batch_size': 16}
            )

            # Sample new population
            bounds = np.array([[-5] * n_vars, [5] * n_vars])
            population = sample_vae(model, n_samples=pop_size, bounds=bounds)

        # Final evaluation
        final_fitness = sphere(population)
        best_fitness = np.min(final_fitness)

        # Should improve from random initialization
        initial_best = np.min(sphere(np.random.uniform(-5, 5, (pop_size, n_vars))))
        assert best_fitness < initial_best

    def test_extended_vae_optimization(self):
        """Test Extended VAE on optimization problem"""
        np.random.seed(42)
        torch.manual_seed(42)

        def rosenbrock(x):
            return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

        # Initialize
        n_vars = 5
        pop_size = 50
        population = np.random.uniform(-2, 2, (pop_size, n_vars))

        # Run optimization with E-VAE
        n_generations = 5
        selection_size = int(pop_size * 0.4)

        for gen in range(n_generations):
            fitness = rosenbrock(population)
            idx = np.argsort(fitness)[:selection_size]

            model = learn_extended_vae(
                population[idx], fitness[idx],
                params={'latent_dim': 3, 'epochs': 15}
            )

            bounds = np.array([[-2] * n_vars, [2] * n_vars])
            population = sample_extended_vae(model, n_samples=pop_size, bounds=bounds)

        final_fitness = rosenbrock(population)
        # Check that optimization makes progress
        assert np.min(final_fitness) < 1000  # Should find reasonable solutions

    def test_conditional_vae_directed_search(self):
        """Test CE-VAE for fitness-directed search"""
        np.random.seed(42)
        torch.manual_seed(42)

        def ackley(x):
            a, b, c = 20, 0.2, 2 * np.pi
            d = x.shape[1]
            sum1 = np.sum(x**2, axis=1)
            sum2 = np.sum(np.cos(c * x), axis=1)
            return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

        n_vars = 5
        pop_size = 60
        population = np.random.uniform(-5, 5, (pop_size, n_vars))

        # Run with CE-VAE
        for gen in range(8):
            fitness = ackley(population)
            idx = np.argsort(fitness)[:int(pop_size * 0.3)]

            model = learn_conditional_extended_vae(
                population[idx], fitness[idx],
                params={'latent_dim': 4, 'epochs': 15}
            )

            # Target best fitness found so far
            target_fit = np.min(fitness)
            bounds = np.array([[-5] * n_vars, [5] * n_vars])
            population = sample_conditional_extended_vae(
                model, n_samples=pop_size, bounds=bounds,
                params={'target_fitness': target_fit, 'fitness_noise': 0.1}
            )

        final_fitness = ackley(population)
        # Should find good solutions (Ackley minimum is 0)
        assert np.min(final_fitness) < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
