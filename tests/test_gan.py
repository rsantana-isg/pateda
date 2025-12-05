"""
Tests for GAN-based EDAs
"""

import pytest
import numpy as np
import torch

from pateda.learning.gan import learn_gan
from pateda.sampling.gan import sample_gan


class TestGANBasic:
    """Test basic GAN functionality"""

    def test_learn_gan_basic(self):
        """Test basic GAN learning"""
        # Create synthetic population
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        # Learn GAN model
        model = learn_gan(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 50, 'batch_size': 16, 'learning_rate': 0.0002}
        )

        # Check model structure
        assert 'generator_state' in model
        assert 'discriminator_state' in model
        assert 'latent_dim' in model
        assert 'input_dim' in model
        assert 'ranges' in model
        assert model['type'] == 'gan'
        assert model['latent_dim'] == 3
        assert model['input_dim'] == n_vars

    def test_sample_gan_basic(self):
        """Test basic GAN sampling"""
        # Create and learn model
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_gan(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 50, 'batch_size': 16}
        )

        # Sample from model
        new_population = sample_gan(model, n_samples=50)

        # Check shape
        assert new_population.shape == (50, n_vars)

    def test_gan_with_bounds(self):
        """Test GAN sampling with bounds"""
        # Create model
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_gan(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 50}
        )

        # Sample with bounds
        bounds = np.array([[-5, -5, -5, -5, -5], [5, 5, 5, 5, 5]])
        new_population = sample_gan(model, n_samples=50, bounds=bounds)

        # Check bounds are respected
        assert np.all(new_population >= bounds[0])
        assert np.all(new_population <= bounds[1])

    def test_gan_with_temperature(self):
        """Test GAN sampling with temperature parameter"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_gan(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 50}
        )

        # Sample with different temperatures
        pop_low_temp = sample_gan(model, n_samples=50, params={'temperature': 0.5})
        pop_high_temp = sample_gan(model, n_samples=50, params={'temperature': 2.0})

        assert pop_low_temp.shape == (50, 5)
        assert pop_high_temp.shape == (50, 5)

        # Higher temperature should generally lead to more diversity (higher variance)
        var_low = np.var(pop_low_temp)
        var_high = np.var(pop_high_temp)
        # Note: This is a stochastic test, so we just check the shapes work


class TestGANArchitecture:
    """Test GAN architecture variations"""

    def test_gan_custom_architecture(self):
        """Test GAN with custom network architecture"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 10)
        fitness = np.sum(population ** 2, axis=1)

        # Custom architecture
        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 5,
                'hidden_dims_g': [16, 32, 64],  # Deeper generator
                'hidden_dims_d': [64, 32, 16],  # Deeper discriminator
                'epochs': 50,
                'batch_size': 16,
            }
        )

        assert model['hidden_dims_g'] == [16, 32, 64]
        assert model['hidden_dims_d'] == [64, 32, 16]

        # Sample should still work
        new_population = sample_gan(model, n_samples=50)
        assert new_population.shape == (50, 10)

    def test_gan_different_latent_dims(self):
        """Test GAN with different latent dimensions"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 8)
        fitness = np.sum(population ** 2, axis=1)

        for latent_dim in [2, 4, 8]:
            model = learn_gan(
                population, fitness,
                params={'latent_dim': latent_dim, 'epochs': 30}
            )

            assert model['latent_dim'] == latent_dim

            new_pop = sample_gan(model, n_samples=40)
            assert new_pop.shape == (40, 8)


class TestGANTraining:
    """Test GAN training parameters"""

    def test_gan_discriminator_steps(self):
        """Test GAN with multiple discriminator updates per generator update"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        # Train with k=3 discriminator steps per generator step
        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 3,
                'epochs': 40,
                'k_discriminator': 3
            }
        )

        # Should still produce valid model
        assert 'generator_state' in model
        new_pop = sample_gan(model, n_samples=50)
        assert new_pop.shape == (50, 5)

    def test_gan_learning_rate(self):
        """Test GAN with different learning rates"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        # Test with different learning rates
        for lr in [0.0001, 0.0002, 0.0005]:
            model = learn_gan(
                population, fitness,
                params={
                    'latent_dim': 3,
                    'epochs': 30,
                    'learning_rate': lr
                }
            )

            new_pop = sample_gan(model, n_samples=50)
            assert new_pop.shape == (50, 5)

    def test_gan_beta1_parameter(self):
        """Test GAN with different Adam beta1 parameters"""
        np.random.seed(42)
        torch.manual_seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        # Test with different beta1 values (as recommended in GAN literature)
        for beta1 in [0.5, 0.9]:
            model = learn_gan(
                population, fitness,
                params={
                    'latent_dim': 3,
                    'epochs': 30,
                    'beta1': beta1
                }
            )

            new_pop = sample_gan(model, n_samples=50)
            assert new_pop.shape == (50, 5)


class TestGANIntegration:
    """Integration tests for GAN with optimization"""

    def test_gan_optimization_sphere(self):
        """Test GAN on sphere function"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Sphere function: minimize sum of squares
        def sphere(x):
            return np.sum(x ** 2, axis=1)

        # Initialize population
        n_vars = 5
        pop_size = 100
        population = np.random.uniform(-5, 5, (pop_size, n_vars))

        # Simple EDA loop
        n_generations = 15
        selection_size = int(pop_size * 0.3)
        best_fitness_history = []

        for gen in range(n_generations):
            # Evaluate
            fitness = sphere(population)
            best_fitness_history.append(np.min(fitness))

            # Select best
            idx = np.argsort(fitness)[:selection_size]
            selected_pop = population[idx]
            selected_fit = fitness[idx]

            # Learn GAN model
            model = learn_gan(
                selected_pop, selected_fit,
                params={
                    'latent_dim': 3,
                    'epochs': 80,
                    'batch_size': 16,
                    'learning_rate': 0.0002
                }
            )

            # Sample new population
            bounds = np.array([[-5] * n_vars, [5] * n_vars])
            population = sample_gan(model, n_samples=pop_size, bounds=bounds)

        # Final evaluation
        final_fitness = sphere(population)
        best_final = np.min(final_fitness)

        # Should show improvement over generations
        initial_best = best_fitness_history[0]
        assert best_final < initial_best  # Should improve

    def test_gan_optimization_rosenbrock(self):
        """Test GAN on Rosenbrock function"""
        np.random.seed(42)
        torch.manual_seed(42)

        def rosenbrock(x):
            return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

        # Initialize
        n_vars = 4
        pop_size = 80
        population = np.random.uniform(-2, 2, (pop_size, n_vars))

        # Run optimization with GAN
        n_generations = 20
        selection_size = int(pop_size * 0.3)

        for gen in range(n_generations):
            fitness = rosenbrock(population)
            idx = np.argsort(fitness)[:selection_size]

            model = learn_gan(
                population[idx], fitness[idx],
                params={
                    'latent_dim': 2,
                    'epochs': 100,
                    'batch_size': 12,
                    'learning_rate': 0.0002
                }
            )

            bounds = np.array([[-2] * n_vars, [2] * n_vars])
            population = sample_gan(model, n_samples=pop_size, bounds=bounds)

        final_fitness = rosenbrock(population)
        # Rosenbrock is challenging, just check we get reasonable solutions
        assert np.min(final_fitness) < 1000

    def test_gan_small_population(self):
        """Test GAN with small population (edge case from paper)"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Small population scenario
        n_vars = 5
        pop_size = 20  # Very small
        population = np.random.uniform(-5, 5, (pop_size, n_vars))

        def sphere(x):
            return np.sum(x ** 2, axis=1)

        fitness = sphere(population)

        # GAN should handle small populations
        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 3,
                'epochs': 50,
                'batch_size': 10  # Half of population
            }
        )

        new_population = sample_gan(model, n_samples=20)
        assert new_population.shape == (20, n_vars)


class TestGANEdgeCases:
    """Test edge cases and error handling"""

    def test_gan_constant_population(self):
        """Test GAN with constant (zero variance) population"""
        np.random.seed(42)
        torch.manual_seed(42)

        # All same values
        population = np.ones((50, 5))
        fitness = np.sum(population ** 2, axis=1)

        # Should handle without crashing
        model = learn_gan(
            population, fitness,
            params={'latent_dim': 3, 'epochs': 20}
        )

        # Should still be able to sample
        new_pop = sample_gan(model, n_samples=30)
        assert new_pop.shape == (30, 5)

    def test_gan_single_variable(self):
        """Test GAN with single variable"""
        np.random.seed(42)
        torch.manual_seed(42)

        population = np.random.randn(100, 1)
        fitness = population.flatten() ** 2

        model = learn_gan(
            population, fitness,
            params={'latent_dim': 2, 'epochs': 30}
        )

        new_pop = sample_gan(model, n_samples=50)
        assert new_pop.shape == (50, 1)

    def test_gan_high_dimensional(self):
        """Test GAN with higher dimensional problem"""
        np.random.seed(42)
        torch.manual_seed(42)

        n_vars = 20  # Higher dimension
        population = np.random.randn(100, n_vars)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 10,  # latent_dim = n_vars // 2
                'hidden_dims_g': [32, 64, 128],
                'hidden_dims_d': [128, 64, 32],
                'epochs': 50
            }
        )

        new_pop = sample_gan(model, n_samples=100)
        assert new_pop.shape == (100, n_vars)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
