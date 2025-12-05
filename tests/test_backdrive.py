"""
Tests for Backdrive-EDA
"""

import pytest
import numpy as np
import torch

from pateda.learning.backdrive import learn_backdrive
from pateda.sampling.backdrive import (
    sample_backdrive,
    sample_backdrive_adaptive,
)


class TestBackdriveBasic:
    """Test basic Backdrive functionality"""

    def test_learn_backdrive_basic(self):
        """Test basic Backdrive learning"""
        # Create synthetic population
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)

        # Define bounds
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Learn Backdrive model
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 10, 'batch_size': 16}
        )

        # Check model structure
        assert 'hidden_layers' in model.structure
        assert 'activation' in model.structure
        assert 'correlation' in model.metadata
        assert model.metadata['generation'] == 0
        assert model.metadata['n_samples'] == n_samples

        # Check that correlation is reasonable
        assert -1 <= model.metadata['correlation'] <= 1

    def test_learn_backdrive_different_architectures(self):
        """Test Backdrive with different network architectures"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 80
        n_vars = 10
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Test with different hidden layer configurations
        architectures = [
            [50],
            [100, 50],
            [150, 100, 50],
        ]

        for hidden_layers in architectures:
            model = learn_backdrive(
                generation=0,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=population,
                selected_fitness=fitness,
                params={'hidden_layers': hidden_layers, 'epochs': 10}
            )

            assert model.structure['hidden_layers'] == hidden_layers
            assert 'correlation' in model.metadata

    def test_learn_backdrive_activations(self):
        """Test Backdrive with different activation functions"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 80
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        activations = ['tanh', 'relu', 'leaky_relu', 'sigmoid']

        for activation in activations:
            model = learn_backdrive(
                generation=0,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=population,
                selected_fitness=fitness,
                params={'activation': activation, 'epochs': 10}
            )

            assert model.structure['activation'] == activation

    def test_sample_backdrive_basic(self):
        """Test basic Backdrive sampling"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Learn model
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 20, 'batch_size': 16}
        )

        # Sample from model
        new_population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={'n_samples': 50, 'backdrive_iterations': 100}
        )

        # Check shape
        assert new_population.shape == (50, n_vars)

        # Check bounds are respected
        assert np.all(new_population >= cardinality[0])
        assert np.all(new_population <= cardinality[1])

    def test_sample_backdrive_initialization_methods(self):
        """Test Backdrive sampling with different initialization methods"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Learn model
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 20}
        )

        init_methods = ['random', 'perturb_best', 'perturb_selected']

        for init_method in init_methods:
            new_population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness,
                params={
                    'n_samples': 50,
                    'init_method': init_method,
                    'backdrive_iterations': 100,
                }
            )

            assert new_population.shape == (50, n_vars)
            assert np.all(new_population >= cardinality[0])
            assert np.all(new_population <= cardinality[1])

    def test_backdrive_weight_transfer(self):
        """Test weight transfer between generations"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_vars = 5
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Generation 0
        pop_gen0 = np.random.randn(100, n_vars)
        fit_gen0 = np.sum(pop_gen0 ** 2, axis=1, keepdims=True)

        model_gen0 = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=pop_gen0,
            selected_fitness=fit_gen0,
            params={'epochs': 15, 'transfer_weights': False}
        )

        # Generation 1 with weight transfer
        pop_gen1 = np.random.randn(100, n_vars)
        fit_gen1 = np.sum(pop_gen1 ** 2, axis=1, keepdims=True)

        model_gen1 = learn_backdrive(
            generation=1,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=pop_gen1,
            selected_fitness=fit_gen1,
            params={
                'epochs': 15,
                'transfer_weights': True,
                'previous_model': model_gen0
            }
        )

        assert model_gen1.metadata['generation'] == 1
        # Both models should have same architecture
        assert model_gen0.structure['hidden_layers'] == model_gen1.structure['hidden_layers']


class TestBackdriveAdaptive:
    """Test adaptive Backdrive sampling"""

    def test_sample_backdrive_adaptive_basic(self):
        """Test basic adaptive sampling"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Learn model
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 20}
        )

        # Sample with adaptive targets
        new_population = sample_backdrive_adaptive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={
                'n_samples': 60,
                'target_levels': [100, 80, 60],
                'level_fractions': [0.5, 0.3, 0.2],
                'backdrive_iterations': 100,
            }
        )

        # Check shape
        assert new_population.shape == (60, n_vars)

        # Check bounds
        assert np.all(new_population >= cardinality[0])
        assert np.all(new_population <= cardinality[1])

    def test_adaptive_with_different_targets(self):
        """Test adaptive sampling with various target configurations"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 20}
        )

        # Test different target level configurations
        target_configs = [
            ([100, 90], [0.7, 0.3]),
            ([100, 85, 70, 50], [0.4, 0.3, 0.2, 0.1]),
            ([100], [1.0]),
        ]

        for target_levels, level_fractions in target_configs:
            new_population = sample_backdrive_adaptive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness,
                params={
                    'n_samples': 50,
                    'target_levels': target_levels,
                    'level_fractions': level_fractions,
                    'backdrive_iterations': 100,
                }
            )

            assert new_population.shape == (50, n_vars)


class TestBackdriveSurrogate:
    """Test surrogate filtering functionality"""

    def test_backdrive_with_surrogate_filtering(self):
        """Test backdrive with surrogate-based filtering"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 100
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Learn model
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 25}
        )

        # Sample with surrogate filtering
        new_population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={
                'n_samples': 50,
                'use_surrogate': True,
                'oversample_factor': 5,
                'backdrive_iterations': 100,
            }
        )

        # Check shape (should be exactly n_samples after filtering)
        assert new_population.shape == (50, n_vars)

        # Check bounds
        assert np.all(new_population >= cardinality[0])
        assert np.all(new_population <= cardinality[1])

    def test_surrogate_with_different_oversample_factors(self):
        """Test different oversample factors"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_samples = 80
        n_vars = 5
        population = np.random.randn(n_samples, n_vars)
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 20}
        )

        oversample_factors = [2, 3, 5, 10]

        for factor in oversample_factors:
            new_population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness,
                params={
                    'n_samples': 40,
                    'use_surrogate': True,
                    'oversample_factor': factor,
                    'backdrive_iterations': 100,
                }
            )

            # Should always return exactly n_samples after filtering
            assert new_population.shape == (40, n_vars)


class TestBackdriveIntegration:
    """Integration tests for Backdrive with optimization"""

    def test_backdrive_optimization_loop(self):
        """Test complete Backdrive EDA loop runs without errors"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Simple quadratic function (for maximization, negate)
        def quadratic(x):
            if x.ndim == 1:
                return -np.sum((x - 1.0) ** 2)
            return -np.sum((x - 1.0) ** 2, axis=1)

        # Initialize population
        n_vars = 5
        pop_size = 60
        cardinality = np.array([[-3.0] * n_vars, [3.0] * n_vars])
        population = np.random.uniform(-3, 3, (pop_size, n_vars))

        # Simple EDA loop - just verify it runs
        n_generations = 10
        selection_size = int(pop_size * 0.4)

        model = None

        for gen in range(n_generations):
            # Evaluate
            fitness = np.array([quadratic(ind) for ind in population])

            # Select best
            idx = np.argsort(fitness)[-selection_size:]  # Take best (highest)
            selected_pop = population[idx]
            selected_fit = fitness[idx].reshape(-1, 1)

            # Learn model
            model = learn_backdrive(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=selected_pop,
                selected_fitness=selected_fit,
                params={
                    'epochs': 25,
                    'batch_size': 16,
                    'previous_model': model,
                    'transfer_weights': True,
                }
            )

            # Check model was created
            assert model is not None
            assert 'correlation' in model.metadata

            # Sample new population
            population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params={
                    'n_samples': pop_size,
                    'backdrive_iterations': 200,
                    'init_method': 'perturb_best',
                }
            )

            # Check population has correct shape and respects bounds
            assert population.shape == (pop_size, n_vars)
            assert np.all(population >= cardinality[0])
            assert np.all(population <= cardinality[1])

        # Verify final population can be evaluated
        final_fitness = np.array([quadratic(ind) for ind in population])
        assert len(final_fitness) == pop_size
        assert np.all(np.isfinite(final_fitness))

    def test_backdrive_with_weight_transfer(self):
        """Test Backdrive with weight transfer across generations"""
        np.random.seed(42)
        torch.manual_seed(42)

        def simple_func(x):
            # Simple function for testing
            if x.ndim == 1:
                return -np.sum(x ** 2)
            return -np.sum(x ** 2, axis=1)

        # Initialize
        n_vars = 5
        pop_size = 60
        cardinality = np.array([[-3.0] * n_vars, [3.0] * n_vars])
        population = np.random.uniform(-3, 3, (pop_size, n_vars))

        # Run for a few generations with weight transfer
        n_generations = 5
        selection_size = int(pop_size * 0.4)

        model = None

        for gen in range(n_generations):
            fitness = np.array([simple_func(ind) for ind in population])
            idx = np.argsort(fitness)[-selection_size:]

            # Learn with weight transfer
            model = learn_backdrive(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=population[idx],
                selected_fitness=fitness[idx].reshape(-1, 1),
                params={
                    'hidden_layers': [80, 60],
                    'epochs': 20,
                    'previous_model': model,
                    'transfer_weights': True if gen > 0 else False,
                }
            )

            # Verify model was created and has correct generation
            assert model.metadata['generation'] == gen

            population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params={'n_samples': pop_size}
            )

        # Verify final state
        assert population.shape == (pop_size, n_vars)
        assert model.metadata['generation'] == n_generations - 1

    def test_backdrive_adaptive_sampling_integration(self):
        """Test adaptive Backdrive sampling in EDA loop"""
        np.random.seed(42)
        torch.manual_seed(42)

        def test_func(x):
            if x.ndim == 1:
                return -np.sum(x ** 2)
            return -np.sum(x ** 2, axis=1)

        n_vars = 5
        pop_size = 80
        cardinality = np.array([[-3.0] * n_vars, [3.0] * n_vars])
        population = np.random.uniform(-3, 3, (pop_size, n_vars))

        # Run with adaptive sampling
        n_generations = 8
        selection_size = int(pop_size * 0.4)
        model = None

        for gen in range(n_generations):
            fitness = np.array([test_func(ind) for ind in population])
            idx = np.argsort(fitness)[-selection_size:]

            model = learn_backdrive(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=population[idx],
                selected_fitness=fitness[idx].reshape(-1, 1),
                params={'epochs': 20, 'previous_model': model}
            )

            # Use adaptive sampling with multiple target levels
            population = sample_backdrive_adaptive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params={
                    'n_samples': pop_size,
                    'target_levels': [100, 85, 70],
                    'level_fractions': [0.5, 0.3, 0.2],
                    'backdrive_iterations': 200,
                }
            )

            # Verify population properties
            assert population.shape == (pop_size, n_vars)
            assert np.all(population >= cardinality[0])
            assert np.all(population <= cardinality[1])

        # Check final state
        final_fitness = np.array([test_func(ind) for ind in population])
        assert len(final_fitness) == pop_size
        assert np.all(np.isfinite(final_fitness))

    def test_backdrive_with_surrogate_filtering_integration(self):
        """Test Backdrive with surrogate filtering in EDA loop"""
        np.random.seed(42)
        torch.manual_seed(42)

        def test_func(x):
            if x.ndim == 1:
                return -np.sum(x ** 2)
            return -np.sum(x ** 2, axis=1)

        n_vars = 5
        pop_size = 60
        cardinality = np.array([[-3.0] * n_vars, [3.0] * n_vars])
        population = np.random.uniform(-3, 3, (pop_size, n_vars))

        n_generations = 8
        selection_size = int(pop_size * 0.4)
        model = None

        for gen in range(n_generations):
            fitness = np.array([test_func(ind) for ind in population])
            idx = np.argsort(fitness)[-selection_size:]

            model = learn_backdrive(
                generation=gen,
                n_vars=n_vars,
                cardinality=cardinality,
                selected_population=population[idx],
                selected_fitness=fitness[idx].reshape(-1, 1),
                params={'epochs': 25, 'previous_model': model}
            )

            # Sample with surrogate filtering (oversample and select best)
            population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params={
                    'n_samples': pop_size,
                    'use_surrogate': True,
                    'oversample_factor': 3,
                    'backdrive_iterations': 200,
                }
            )

            # Verify surrogate filtering returns correct number of samples
            assert population.shape == (pop_size, n_vars)
            assert np.all(population >= cardinality[0])
            assert np.all(population <= cardinality[1])

        # Check final state
        final_fitness = np.array([test_func(ind) for ind in population])
        assert len(final_fitness) == pop_size
        assert np.all(np.isfinite(final_fitness))


class TestBackdriveEdgeCases:
    """Test edge cases and error handling"""

    def test_small_population(self):
        """Test with very small population"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_vars = 5
        population = np.random.randn(10, n_vars)  # Small population
        fitness = np.sum(population ** 2, axis=1, keepdims=True)
        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 10, 'validation_split': 0.0}  # No validation
        )

        new_population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={'n_samples': 10}
        )

        assert new_population.shape == (10, n_vars)

    def test_identical_fitness(self):
        """Test when all fitness values are identical"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_vars = 5
        population = np.random.randn(50, n_vars)
        fitness = np.ones((50, 1)) * 10.0  # All identical

        cardinality = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        # Should handle this gracefully with a warning
        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 10}
        )

        # Should still be able to sample
        new_population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={'n_samples': 30}
        )

        assert new_population.shape == (30, n_vars)

    def test_single_variable(self):
        """Test with single variable problem"""
        np.random.seed(42)
        torch.manual_seed(42)
        n_vars = 1
        population = np.random.randn(100, n_vars)
        fitness = population ** 2

        cardinality = np.array([[-10.0], [10.0]])

        model = learn_backdrive(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_population=population,
            selected_fitness=fitness,
            params={'epochs': 15, 'hidden_layers': [20]}
        )

        new_population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            current_population=population,
            current_fitness=fitness,
            params={'n_samples': 50}
        )

        assert new_population.shape == (50, n_vars)
        assert np.all(new_population >= -10.0)
        assert np.all(new_population <= 10.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
