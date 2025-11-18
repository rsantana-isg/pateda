"""
Comprehensive tests for Gaussian-based EDAs for continuous optimization.

This test suite covers:
- Univariate Gaussian EDA (Gaussian UMDA)
- Full Multivariate Gaussian EDA
- Mixture of Gaussians EDAs
"""

import pytest
import numpy as np
from pateda.learning.gaussian import (
    learn_gaussian_univariate,
    learn_gaussian_full,
    learn_mixture_gaussian_univariate,
    learn_mixture_gaussian_full,
)
from pateda.sampling.gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_mixture_gaussian_univariate,
    sample_mixture_gaussian_full,
)


class TestGaussianUnivariate:
    """Test Univariate Gaussian EDA (Gaussian UMDA)"""

    def test_learn_gaussian_univariate_basic(self):
        """Test basic learning of univariate Gaussian model"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_univariate(population, fitness)

        assert 'means' in model
        assert 'stds' in model
        assert model['type'] == 'gaussian_univariate'
        assert len(model['means']) == 5
        assert len(model['stds']) == 5
        assert np.all(model['stds'] > 0)  # Ensure positive std

    def test_sample_gaussian_univariate(self):
        """Test sampling from univariate Gaussian model"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_univariate(population, fitness)
        samples = sample_gaussian_univariate(model, n_samples=50)

        assert samples.shape == (50, 5)

    def test_gaussian_univariate_with_bounds(self):
        """Test sampling with variable bounds"""
        np.random.seed(42)
        population = np.random.uniform(-2, 2, (100, 5))
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_univariate(population, fitness)
        bounds = np.array([[-5, -5, -5, -5, -5], [5, 5, 5, 5, 5]])
        samples = sample_gaussian_univariate(model, n_samples=50, bounds=bounds)

        assert np.all(samples >= bounds[0])
        assert np.all(samples <= bounds[1])

    def test_gaussian_univariate_zero_std_prevention(self):
        """Test that zero standard deviation is prevented"""
        # Create population with zero variance in one dimension
        population = np.random.randn(100, 5)
        population[:, 2] = 1.0  # Constant value
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_univariate(population, fitness)

        # Should have small but non-zero std
        assert np.all(model['stds'] > 0)
        assert model['stds'][2] >= 1e-10


class TestGaussianFull:
    """Test Full Multivariate Gaussian EDA"""

    def test_learn_gaussian_full_basic(self):
        """Test learning of full covariance Gaussian model"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_full(population, fitness)

        assert 'mean' in model
        assert 'cov' in model
        assert model['type'] == 'gaussian_full'
        assert len(model['mean']) == 5
        assert model['cov'].shape == (5, 5)

        # Check covariance is symmetric
        assert np.allclose(model['cov'], model['cov'].T)

        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(model['cov'])
        assert np.all(eigenvalues > 0)

    def test_sample_gaussian_full(self):
        """Test sampling from full Gaussian model"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_full(population, fitness)
        samples = sample_gaussian_full(model, n_samples=50)

        assert samples.shape == (50, 5)

    def test_gaussian_full_with_bounds(self):
        """Test sampling with bounds"""
        np.random.seed(42)
        population = np.random.uniform(-2, 2, (100, 5))
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_full(population, fitness)
        bounds = np.array([[-3, -3, -3, -3, -3], [3, 3, 3, 3, 3]])
        samples = sample_gaussian_full(model, n_samples=50, bounds=bounds)

        assert np.all(samples >= bounds[0])
        assert np.all(samples <= bounds[1])

    def test_gaussian_full_variance_scaling(self):
        """Test variance scaling parameter"""
        np.random.seed(42)
        population = np.random.randn(100, 5)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_full(population, fitness)

        # Sample with different variance scaling
        samples1 = sample_gaussian_full(
            model, n_samples=1000,
            params={'var_scaling': 0.5}
        )
        samples2 = sample_gaussian_full(
            model, n_samples=1000,
            params={'var_scaling': 2.0}
        )

        # Higher variance scaling should produce more spread
        var1 = np.var(samples1, axis=0)
        var2 = np.var(samples2, axis=0)
        assert np.all(var2 > var1)


class TestMixtureGaussianUnivariate:
    """Test Mixture of Univariate Gaussians EDA"""

    def test_learn_mixture_univariate_basic(self):
        """Test learning mixture of univariate Gaussians"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {
            'n_clusters': 3,
            'what_to_cluster': 'vars',
            'normalize': True
        }
        model = learn_mixture_gaussian_univariate(population, fitness, params)

        assert 'components' in model
        assert 'n_clusters' in model
        assert model['type'] == 'mixture_gaussian_univariate'
        assert model['n_clusters'] == 3
        assert len(model['components']) == 3

        # Check each component
        for comp in model['components']:
            assert 'means' in comp
            assert 'stds' in comp
            assert 'weight' in comp
            assert len(comp['means']) == 5
            assert len(comp['stds']) == 5
            assert comp['weight'] > 0

        # Weights should sum to approximately 1
        total_weight = sum(comp['weight'] for comp in model['components'])
        assert np.isclose(total_weight, 1.0)

    def test_sample_mixture_univariate(self):
        """Test sampling from mixture model"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {'n_clusters': 3, 'what_to_cluster': 'vars'}
        model = learn_mixture_gaussian_univariate(population, fitness, params)
        samples = sample_mixture_gaussian_univariate(model, n_samples=100)

        assert samples.shape == (100, 5)

    def test_mixture_clustering_on_objectives(self):
        """Test clustering based on objectives"""
        np.random.seed(42)
        # Create population with clear fitness-based clusters
        pop1 = np.random.randn(50, 5) - 2  # Low fitness cluster
        pop2 = np.random.randn(50, 5) + 2  # High fitness cluster
        population = np.vstack([pop1, pop2])
        fitness = np.sum(population**2, axis=1)

        params = {
            'n_clusters': 2,
            'what_to_cluster': 'objs',
            'normalize': True
        }
        model = learn_mixture_gaussian_univariate(population, fitness, params)

        assert len(model['components']) == 2

    def test_mixture_clustering_vars_and_objs(self):
        """Test clustering on both variables and objectives"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {
            'n_clusters': 3,
            'what_to_cluster': 'vars_and_objs',
            'normalize': True
        }
        model = learn_mixture_gaussian_univariate(population, fitness, params)

        assert len(model['components']) == 3


class TestMixtureGaussianFull:
    """Test Mixture of Full Multivariate Gaussians EDA"""

    def test_learn_mixture_full_basic(self):
        """Test learning mixture of full Gaussians"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {'n_clusters': 3, 'what_to_cluster': 'vars'}
        model = learn_mixture_gaussian_full(population, fitness, params)

        assert 'components' in model
        assert model['type'] == 'mixture_gaussian_full'
        assert len(model['components']) == 3

        # Check each component
        for comp in model['components']:
            assert 'mean' in comp
            assert 'cov' in comp
            assert 'weight' in comp
            assert len(comp['mean']) == 5
            assert comp['cov'].shape == (5, 5)

            # Check covariance is symmetric and positive definite
            assert np.allclose(comp['cov'], comp['cov'].T)
            eigenvalues = np.linalg.eigvalsh(comp['cov'])
            assert np.all(eigenvalues > 0)

    def test_sample_mixture_full(self):
        """Test sampling from mixture of full Gaussians"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {'n_clusters': 3}
        model = learn_mixture_gaussian_full(population, fitness, params)
        samples = sample_mixture_gaussian_full(model, n_samples=100)

        assert samples.shape == (100, 5)

    def test_mixture_full_with_variance_scaling(self):
        """Test variance scaling in mixture model"""
        np.random.seed(42)
        population = np.random.randn(200, 5)
        fitness = np.sum(population**2, axis=1)

        params = {'n_clusters': 2}
        model = learn_mixture_gaussian_full(population, fitness, params)

        samples = sample_mixture_gaussian_full(
            model, n_samples=100,
            params={'var_scaling': 0.5}
        )

        assert samples.shape == (100, 5)


class TestGaussianEDAIntegration:
    """Integration tests for Gaussian EDAs in optimization"""

    def test_gaussian_umda_on_sphere(self):
        """Test Gaussian UMDA on sphere function"""
        np.random.seed(42)

        def sphere(x):
            return np.sum(x**2, axis=1)

        # Initialize
        n_vars = 10
        pop_size = 100
        population = np.random.uniform(-5, 5, (pop_size, n_vars))

        # Run EDA
        n_generations = 20
        selection_ratio = 0.3

        best_fitness_history = []

        for gen in range(n_generations):
            fitness = sphere(population)
            best_fitness_history.append(np.min(fitness))

            # Select best
            idx = np.argsort(fitness)[:int(pop_size * selection_ratio)]
            selected_pop = population[idx]

            # Learn and sample
            model = learn_gaussian_univariate(selected_pop, fitness[idx])
            bounds = np.array([[-5]*n_vars, [5]*n_vars])
            population = sample_gaussian_univariate(model, n_samples=pop_size, bounds=bounds)

        # Should improve
        assert best_fitness_history[-1] < best_fitness_history[0]
        assert best_fitness_history[-1] < 10.0  # Should find good solutions

    def test_gaussian_full_on_rosenbrock(self):
        """Test full Gaussian EDA on Rosenbrock function"""
        np.random.seed(42)

        def rosenbrock(x):
            return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

        # Initialize
        n_vars = 5
        pop_size = 100
        population = np.random.uniform(-2, 2, (pop_size, n_vars))

        # Run EDA
        for gen in range(15):
            fitness = rosenbrock(population)

            # Select
            idx = np.argsort(fitness)[:30]

            # Learn full covariance model
            model = learn_gaussian_full(population[idx], fitness[idx])
            bounds = np.array([[-2]*n_vars, [2]*n_vars])
            population = sample_gaussian_full(model, n_samples=pop_size, bounds=bounds)

        final_fitness = rosenbrock(population)
        assert np.min(final_fitness) < 500  # Should make reasonable progress

    def test_mixture_gaussian_on_rastrigin(self):
        """Test mixture Gaussian EDA on Rastrigin (multimodal function)"""
        np.random.seed(42)

        def rastrigin(x):
            n = x.shape[1]
            return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

        n_vars = 5
        pop_size = 150
        population = np.random.uniform(-5.12, 5.12, (pop_size, n_vars))

        # Mixture model should handle multimodality better
        for gen in range(15):
            fitness = rastrigin(population)

            idx = np.argsort(fitness)[:50]

            params = {'n_clusters': 3, 'what_to_cluster': 'vars'}
            model = learn_mixture_gaussian_univariate(
                population[idx], fitness[idx], params
            )

            bounds = np.array([[-5.12]*n_vars, [5.12]*n_vars])
            population = sample_mixture_gaussian_univariate(
                model, n_samples=pop_size, bounds=bounds
            )

        final_fitness = rastrigin(population)
        # Rastrigin minimum is 0, should make progress
        assert np.min(final_fitness) < 50

    def test_comparison_univariate_vs_full(self):
        """Compare univariate vs full covariance on correlated problem"""
        np.random.seed(42)

        # Create correlated optimization problem
        # f(x) = x1^2 + x2^2 + 2*x1*x2 (has correlation)
        def correlated_sphere(x):
            return x[:, 0]**2 + x[:, 1]**2 + 2*x[:, 0]*x[:, 1]

        n_vars = 2
        pop_size = 50

        # Test univariate
        pop_uni = np.random.uniform(-3, 3, (pop_size, n_vars))
        for _ in range(10):
            fit = correlated_sphere(pop_uni)
            idx = np.argsort(fit)[:20]
            model = learn_gaussian_univariate(pop_uni[idx], fit[idx])
            bounds = np.array([[-3]*n_vars, [3]*n_vars])
            pop_uni = sample_gaussian_univariate(model, n_samples=pop_size, bounds=bounds)

        best_uni = np.min(correlated_sphere(pop_uni))

        # Test full covariance
        pop_full = np.random.uniform(-3, 3, (pop_size, n_vars))
        for _ in range(10):
            fit = correlated_sphere(pop_full)
            idx = np.argsort(fit)[:20]
            model = learn_gaussian_full(pop_full[idx], fit[idx])
            bounds = np.array([[-3]*n_vars, [3]*n_vars])
            pop_full = sample_gaussian_full(model, n_samples=pop_size, bounds=bounds)

        best_full = np.min(correlated_sphere(pop_full))

        # Full covariance should perform better on correlated problem
        # (though we allow some variance in random results)
        assert best_full <= best_uni * 1.5  # At least competitive


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_population(self):
        """Test with very small population"""
        np.random.seed(42)
        population = np.random.randn(5, 3)
        fitness = np.sum(population**2, axis=1)

        # Should handle small population
        model = learn_gaussian_univariate(population, fitness)
        samples = sample_gaussian_univariate(model, n_samples=10)
        assert samples.shape == (10, 3)

    def test_single_dimension(self):
        """Test with single dimension"""
        np.random.seed(42)
        population = np.random.randn(100, 1)
        fitness = population.flatten()**2

        model = learn_gaussian_univariate(population, fitness)
        samples = sample_gaussian_univariate(model, n_samples=50)
        assert samples.shape == (50, 1)

    def test_high_dimensions(self):
        """Test with high-dimensional problem"""
        np.random.seed(42)
        n_vars = 50
        population = np.random.randn(100, n_vars)
        fitness = np.sum(population**2, axis=1)

        model = learn_gaussian_full(population, fitness)
        samples = sample_gaussian_full(model, n_samples=50)
        assert samples.shape == (50, n_vars)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
