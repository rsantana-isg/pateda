"""
Tests for Vine Copula Learning and Sampling

This module contains comprehensive tests for the vine copula-based
EDAs implemented in pateda.
"""

import pytest
import numpy as np

# Try to import vine copula modules
try:
    from pateda.learning.vine_copula import (
        learn_vine_copula_cvine,
        learn_vine_copula_dvine,
        learn_vine_copula_auto,
    )
    from pateda.sampling.vine_copula import (
        sample_vine_copula,
        sample_vine_copula_biased,
        sample_vine_copula_conditional,
    )
    import pyvinecopulib as pv
    PYVINECOPULIB_AVAILABLE = True
except ImportError:
    PYVINECOPULIB_AVAILABLE = False


# Skip all tests if pyvinecopulib is not available
pytestmark = pytest.mark.skipif(
    not PYVINECOPULIB_AVAILABLE,
    reason="pyvinecopulib not installed"
)


class TestVineCopulaLearning:
    """Tests for vine copula learning functions"""

    @pytest.fixture
    def correlated_data(self):
        """Generate correlated multivariate normal data for testing"""
        np.random.seed(42)
        n_samples = 200
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.array([
            [1.0, 0.7, 0.4],
            [0.7, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ])
        population = np.random.multivariate_normal(mean, cov, size=n_samples)
        fitness = np.sum(population**2, axis=1)
        return population, fitness

    def test_learn_cvine_basic(self, correlated_data):
        """Test basic C-vine learning"""
        population, fitness = correlated_data
        model = learn_vine_copula_cvine(population, fitness)

        assert model['type'] == 'vine_copula_cvine'
        assert model['structure_type'] == 'cvine'
        assert 'vine_model' in model
        assert 'bounds' in model
        assert model['bounds'].shape == (2, 3)
        assert model['n_vars'] == 3

    def test_learn_cvine_with_params(self, correlated_data):
        """Test C-vine learning with various parameters"""
        population, fitness = correlated_data

        # Test with truncation
        model = learn_vine_copula_cvine(
            population, fitness,
            params={'truncation_level': 1}
        )
        assert model['type'] == 'vine_copula_cvine'

        # Test with different copula family
        model = learn_vine_copula_cvine(
            population, fitness,
            params={'copula_family': 5}  # Clayton
        )
        assert model['copula_family'] == 5

        # Test with family selection
        model = learn_vine_copula_cvine(
            population, fitness,
            params={'select_families': True}
        )
        assert model['type'] == 'vine_copula_cvine'

    def test_learn_dvine_basic(self, correlated_data):
        """Test basic D-vine/R-vine learning"""
        population, fitness = correlated_data
        model = learn_vine_copula_dvine(population, fitness)

        assert model['type'] == 'vine_copula_dvine'
        assert model['structure_type'] == 'rvine'
        assert 'vine_model' in model
        assert 'bounds' in model
        assert model['bounds'].shape == (2, 3)

    def test_learn_dvine_with_params(self, correlated_data):
        """Test D-vine learning with parameters"""
        population, fitness = correlated_data

        # Test with truncation
        model = learn_vine_copula_dvine(
            population, fitness,
            params={'truncation_level': 2}
        )
        assert model['type'] == 'vine_copula_dvine'

        # Test without family selection
        model = learn_vine_copula_dvine(
            population, fitness,
            params={'select_families': False, 'copula_family': 0}
        )
        assert model['copula_family'] == 0

    def test_learn_auto(self, correlated_data):
        """Test automatic vine learning"""
        population, fitness = correlated_data
        model = learn_vine_copula_auto(population, fitness)

        assert model['type'] == 'vine_copula_auto'
        assert model['structure_type'] == 'auto'
        assert 'vine_model' in model
        assert 'bounds' in model

    def test_learn_auto_with_params(self, correlated_data):
        """Test automatic vine learning with custom parameters"""
        population, fitness = correlated_data

        params = {
            'truncation_level': 2,
            'tree_criterion': 'tau',
            'selection_criterion': 'aic'
        }
        model = learn_vine_copula_auto(population, fitness, params=params)
        assert model['type'] == 'vine_copula_auto'

    def test_bounds_extraction(self, correlated_data):
        """Test that bounds are correctly extracted"""
        population, fitness = correlated_data
        model = learn_vine_copula_cvine(population, fitness)

        expected_mins = np.min(population, axis=0)
        expected_maxs = np.max(population, axis=0)

        np.testing.assert_array_almost_equal(model['bounds'][0], expected_mins)
        np.testing.assert_array_almost_equal(model['bounds'][1], expected_maxs)


class TestVineCopulaSampling:
    """Tests for vine copula sampling functions"""

    @pytest.fixture
    def learned_model(self):
        """Create a learned vine copula model for testing"""
        np.random.seed(42)
        n_samples = 200
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        population = np.random.multivariate_normal(mean, cov, size=n_samples)
        fitness = np.sum(population**2, axis=1)
        model = learn_vine_copula_auto(population, fitness)
        return model

    def test_sample_basic(self, learned_model):
        """Test basic sampling from vine copula"""
        n_samples = 100
        samples = sample_vine_copula(learned_model, n_samples)

        assert samples.shape == (n_samples, 3)
        assert np.all(np.isfinite(samples))

        # Check that samples respect bounds
        bounds = learned_model['bounds']
        assert np.all(samples >= bounds[0])
        assert np.all(samples <= bounds[1])

    def test_sample_with_custom_bounds(self, learned_model):
        """Test sampling with custom bounds"""
        n_samples = 50
        custom_bounds = np.array([[-5, -5, -5], [5, 5, 5]])

        samples = sample_vine_copula(
            learned_model, n_samples,
            bounds=custom_bounds
        )

        assert samples.shape == (n_samples, 3)
        assert np.all(samples >= custom_bounds[0])
        assert np.all(samples <= custom_bounds[1])

    def test_sample_with_seeds(self, learned_model):
        """Test reproducibility with random seeds"""
        n_samples = 50
        seeds = [1, 2, 3]

        samples1 = sample_vine_copula(
            learned_model, n_samples,
            params={'seeds': seeds}
        )
        samples2 = sample_vine_copula(
            learned_model, n_samples,
            params={'seeds': seeds}
        )

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_sample_inverse_rosenblatt(self, learned_model):
        """Test sampling using inverse Rosenblatt transform"""
        n_samples = 50
        np.random.seed(42)

        samples = sample_vine_copula(
            learned_model, n_samples,
            params={'use_inverse_rosenblatt': True}
        )

        assert samples.shape == (n_samples, 3)
        assert np.all(np.isfinite(samples))

    def test_sample_biased(self, learned_model):
        """Test biased sampling"""
        n_samples = 100
        samples = sample_vine_copula_biased(learned_model, n_samples)

        assert samples.shape == (n_samples, 3)
        assert np.all(np.isfinite(samples))

    def test_sample_biased_with_exploit_factor(self, learned_model):
        """Test biased sampling with different exploit factors"""
        n_samples = 100

        samples1 = sample_vine_copula_biased(
            learned_model, n_samples,
            params={'exploit_factor': 0.1}
        )
        samples2 = sample_vine_copula_biased(
            learned_model, n_samples,
            params={'exploit_factor': 0.5}
        )

        assert samples1.shape == (n_samples, 3)
        assert samples2.shape == (n_samples, 3)
        # Samples should be different due to different exploit factors
        assert not np.allclose(samples1, samples2)

    def test_sample_conditional(self, learned_model):
        """Test conditional sampling with fixed variables"""
        n_samples = 50
        fixed_vars = {0: 1.5, 2: -0.5}

        samples = sample_vine_copula_conditional(
            learned_model, n_samples,
            fixed_vars=fixed_vars
        )

        assert samples.shape == (n_samples, 3)
        # Check that fixed variables have the correct values
        np.testing.assert_array_almost_equal(samples[:, 0], 1.5)
        np.testing.assert_array_almost_equal(samples[:, 2], -0.5)
        # Second variable should vary
        assert np.std(samples[:, 1]) > 0

    def test_sample_without_clip(self, learned_model):
        """Test sampling without bound clipping"""
        n_samples = 50
        samples = sample_vine_copula(
            learned_model, n_samples,
            params={'clip_bounds': False}
        )

        assert samples.shape == (n_samples, 3)
        assert np.all(np.isfinite(samples))


class TestVineCopulaIntegration:
    """Integration tests for vine copula learning and sampling"""

    def test_learn_sample_roundtrip(self):
        """Test that learning and sampling work together"""
        np.random.seed(42)

        # Create original data
        n_samples = 200
        mean = np.array([2.0, 3.0])
        cov = np.array([[2.0, 0.8], [0.8, 1.5]])
        original_pop = np.random.multivariate_normal(mean, cov, size=n_samples)
        fitness = np.sum(original_pop**2, axis=1)

        # Learn model
        model = learn_vine_copula_auto(original_pop, fitness)

        # Sample from model
        new_pop = sample_vine_copula(model, n_samples=500)

        # Check that sampled data has similar statistics (relaxed tolerance)
        # Note: vine copulas may not perfectly preserve marginal moments
        assert np.abs(np.mean(new_pop, axis=0)[0] - mean[0]) < 1.0
        assert np.abs(np.mean(new_pop, axis=0)[1] - mean[1]) < 1.0

    def test_different_vine_types(self):
        """Test that different vine types can be used interchangeably"""
        np.random.seed(42)

        # Create data
        n_samples = 150
        population = np.random.multivariate_normal(
            [0, 0, 0],
            [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]],
            size=n_samples
        )
        fitness = np.sum(population**2, axis=1)

        # Learn with different methods
        model_cvine = learn_vine_copula_cvine(population, fitness)
        model_dvine = learn_vine_copula_dvine(population, fitness)
        model_auto = learn_vine_copula_auto(population, fitness)

        # All should be able to sample
        samples_cvine = sample_vine_copula(model_cvine, 50)
        samples_dvine = sample_vine_copula(model_dvine, 50)
        samples_auto = sample_vine_copula(model_auto, 50)

        assert samples_cvine.shape == (50, 3)
        assert samples_dvine.shape == (50, 3)
        assert samples_auto.shape == (50, 3)

    def test_high_dimensional(self):
        """Test with higher dimensional data"""
        np.random.seed(42)

        # Create 5-dimensional data
        n_vars = 5
        n_samples = 200
        mean = np.zeros(n_vars)
        # Create a random positive definite covariance matrix
        A = np.random.randn(n_vars, n_vars)
        cov = np.dot(A, A.T) / n_vars + np.eye(n_vars) * 0.5

        population = np.random.multivariate_normal(mean, cov, size=n_samples)
        fitness = np.sum(population**2, axis=1)

        # Learn and sample
        model = learn_vine_copula_auto(population, fitness, params={'truncation_level': 2})
        samples = sample_vine_copula(model, 100)

        assert samples.shape == (100, n_vars)
        assert np.all(np.isfinite(samples))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
