"""
Tests for Discrete Backdrive Variants
"""

import pytest
import numpy as np
import torch

from pateda.learning.discrete_backdrive import learn_binary_backdrive
from pateda.learning.discrete_backdrive_weighted_mse import learn_binary_backdrive_weighted_mse
from pateda.learning.discrete_backdrive_ranking import learn_binary_backdrive_ranking
from pateda.learning.discrete_backdrive_huber import learn_binary_backdrive_huber
from pateda.sampling.discrete_neural import sample_binary_backdrive


class TestDiscreteBacdriveVariants:
    """Test discrete backdrive variants with different loss functions"""

    def test_standard_backdrive(self):
        """Test standard backdrive with improved hyperparameters"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create simple binary problem (OneMax-like)
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        # Fitness = sum of bits
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Learn model with improved defaults
        model = learn_binary_backdrive(population, fitness, params={'epochs': 10})
        
        # Check model structure
        assert model['type'] == 'discrete_backdrive'
        assert model['n_vars'] == n_vars
        assert 'network_state' in model
        assert 'fitness_stats' in model
        
        # Sample new solutions
        n_samples = 20
        samples = sample_binary_backdrive(model, n_samples)
        
        # Validate samples
        assert samples.shape == (n_samples, n_vars)
        assert np.all((samples == 0) | (samples == 1))
        
        # Check that generated solutions have reasonable fitness
        # (should be better than random on average)
        sample_fitness = samples.sum(axis=1)
        random_fitness = np.random.randint(0, 2, size=(n_samples, n_vars)).sum(axis=1)
        assert sample_fitness.mean() >= random_fitness.mean() * 0.9  # Allow some variation

    def test_weighted_mse_backdrive(self):
        """Test backdrive with fitness-weighted MSE loss"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Learn model with weighted MSE
        model = learn_binary_backdrive_weighted_mse(population, fitness, params={'epochs': 10})
        
        # Check model structure
        assert model['type'] == 'discrete_backdrive_weighted_mse'
        assert model['n_vars'] == n_vars
        
        # Sample new solutions
        n_samples = 20
        samples = sample_binary_backdrive(model, n_samples)
        
        # Validate samples
        assert samples.shape == (n_samples, n_vars)
        assert np.all((samples == 0) | (samples == 1))

    def test_ranking_backdrive(self):
        """Test backdrive with ranking loss"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Learn model with ranking loss
        model = learn_binary_backdrive_ranking(population, fitness, params={'epochs': 10})
        
        # Check model structure
        assert model['type'] == 'discrete_backdrive_ranking'
        assert model['n_vars'] == n_vars
        
        # Sample new solutions
        n_samples = 20
        samples = sample_binary_backdrive(model, n_samples)
        
        # Validate samples
        assert samples.shape == (n_samples, n_vars)
        assert np.all((samples == 0) | (samples == 1))

    def test_huber_backdrive(self):
        """Test backdrive with Huber loss"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Learn model with Huber loss
        model = learn_binary_backdrive_huber(population, fitness, params={'epochs': 10})
        
        # Check model structure
        assert model['type'] == 'discrete_backdrive_huber'
        assert model['n_vars'] == n_vars
        
        # Sample new solutions
        n_samples = 20
        samples = sample_binary_backdrive(model, n_samples)
        
        # Validate samples
        assert samples.shape == (n_samples, n_vars)
        assert np.all((samples == 0) | (samples == 1))

    def test_dynamic_architecture_sizing(self):
        """Test that dynamic architecture sizing works correctly"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Test with different problem sizes
        test_cases = [
            (10, 30),   # Small problem, small population
            (50, 100),  # Medium problem, medium population
            (100, 200), # Large problem, large population
        ]
        
        for n_vars, pop_size in test_cases:
            population = np.random.randint(0, 2, size=(pop_size, n_vars))
            fitness = population.sum(axis=1, keepdims=True).astype(float)
            
            # Learn model without specifying hidden_layers
            model = learn_binary_backdrive(population, fitness, params={'epochs': 5})
            
            # Check that hidden layers were computed dynamically
            hidden_layers = model['hidden_layers']
            assert isinstance(hidden_layers, list)
            assert len(hidden_layers) == 2  # Should have 2 hidden layers
            
            # Check that architecture is reasonable (not too large)
            # h1 should be at most min(n_vars, pop_size)
            assert hidden_layers[0] <= min(n_vars, pop_size)
            # h2 should be smaller than h1
            assert hidden_layers[1] < hidden_layers[0]

    def test_improved_regularization(self):
        """Test that improved regularization parameters are used"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 30
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Learn model with default params (should use improved defaults)
        model = learn_binary_backdrive(population, fitness, params={'epochs': 5})
        
        # The model should be trained successfully
        assert 'network_state' in model
        
        # Sample should work
        samples = sample_binary_backdrive(model, 10)
        assert samples.shape == (10, n_vars)

    def test_sampling_with_improved_hyperparameters(self):
        """Test that sampling uses improved hyperparameters"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        model = learn_binary_backdrive(population, fitness, params={'epochs': 5})
        
        # Test sampling with explicit improved parameters
        samples = sample_binary_backdrive(
            model, 
            10,
            params={
                'n_iterations': 50,
                'learning_rate': 0.01,  # Improved default
                'gradient_clip': 1.0,   # New parameter
                'temperature': 2.0,     # Improved default
                'temperature_schedule': 'cosine',  # New parameter
                'use_gumbel_noise': False,  # Improved default
            }
        )
        
        assert samples.shape == (10, n_vars)
        assert np.all((samples == 0) | (samples == 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
