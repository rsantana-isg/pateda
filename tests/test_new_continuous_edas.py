"""
Comprehensive tests for newly added continuous EDAs

This module tests the new EDA components added from enhanced_edas:
1. Weighted Gaussian learning and sampling
2. Gaussian Mixture EM learning and sampling
3. Diversity-triggered sampling

These tests use standard continuous optimization benchmarks to verify
that the new components work correctly and produce reasonable results.
"""

import numpy as np
from typing import Callable

# Import learning functions
from pateda.learning.mixture_gaussian import learn_mixture_gaussian_em
from pateda.learning.basic_gaussian import learn_gaussian_univariate, learn_gaussian_full

# Import sampling functions
from pateda.sampling.mixture_gaussian import sample_mixture_gaussian_em
from pateda.sampling.basic_gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_gaussian_with_diversity_trigger
)

# NOTE: Weighted Gaussian functions are not yet integrated into pateda
# They exist in enhanced_edas/gaussian_models.py but need to be properly integrated
# For now, these tests that use weighted functions will be skipped
# Placeholder definitions to prevent import errors
def learn_weighted_gaussian_univariate(*args, **kwargs):
    raise NotImplementedError("Weighted Gaussian functions not yet integrated")

def learn_weighted_gaussian_full(*args, **kwargs):
    raise NotImplementedError("Weighted Gaussian functions not yet integrated")

def sample_weighted_gaussian_univariate(*args, **kwargs):
    raise NotImplementedError("Weighted Gaussian functions not yet integrated")

def sample_weighted_gaussian_full(*args, **kwargs):
    raise NotImplementedError("Weighted Gaussian functions not yet integrated")


# ============================================================================
# Test Benchmark Functions
# ============================================================================

def sphere_function(x: np.ndarray) -> np.ndarray:
    """Sphere function: f(x) = sum(x^2)"""
    return np.sum(x**2, axis=1)


def rosenbrock_function(x: np.ndarray) -> np.ndarray:
    """Rosenbrock function"""
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin_function(x: np.ndarray) -> np.ndarray:
    """Rastrigin function"""
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


# ============================================================================
# Helper Functions
# ============================================================================

def run_simple_eda(
    objective_func: Callable,
    learn_func: Callable,
    sample_func: Callable,
    n_vars: int = 10,
    pop_size: int = 100,
    n_generations: int = 50,
    truncation: float = 0.5,
    learn_params: dict = None,
    sample_params: dict = None
) -> tuple:
    """
    Run a simple EDA for testing purposes.

    Returns
    -------
    tuple
        (best_fitness, best_solution, fitness_history)
    """
    # Initialize
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, n_vars))

    fitness_history = []
    best_fitness = float('inf')
    best_solution = None

    for gen in range(n_generations):
        # Evaluate
        fitness = objective_func(population)

        # Track best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        fitness_history.append(best_fitness)

        # Select
        n_selected = int(pop_size * truncation)
        selected_idx = np.argsort(fitness)[:n_selected]
        selected_pop = population[selected_idx]
        selected_fit = fitness[selected_idx]

        # Learn
        if learn_params is None:
            model = learn_func(selected_pop, selected_fit)
        else:
            model = learn_func(selected_pop, selected_fit, learn_params)

        # Sample
        if sample_params is None:
            population = sample_func(model, pop_size, bounds)
        else:
            population = sample_func(model, pop_size, bounds, sample_params)

    return best_fitness, best_solution, fitness_history


# ============================================================================
# Tests for Weighted Gaussian Learning
# ============================================================================

def test_weighted_gaussian_univariate_learning():
    """Test that weighted univariate Gaussian learning produces valid models"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 50

    # Create population with varying fitness
    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)

    # Learn model
    model = learn_weighted_gaussian_univariate(population, fitness)

    # Check model structure
    assert 'means' in model
    assert 'stds' in model
    assert 'type' in model
    assert model['type'] == 'weighted_gaussian_univariate'

    # Check dimensions
    assert len(model['means']) == n_vars
    assert len(model['stds']) == n_vars

    # Check that stds are positive
    assert np.all(model['stds'] > 0)

    # Check that weighted mean is different from simple mean
    simple_model = learn_gaussian_univariate(population, fitness)
    assert not np.allclose(model['means'], simple_model['means'])


def test_weighted_gaussian_full_learning():
    """Test that weighted full Gaussian learning produces valid models"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 50

    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)

    model = learn_weighted_gaussian_full(population, fitness)

    # Check model structure
    assert 'mean' in model
    assert 'cov' in model
    assert 'type' in model
    assert model['type'] == 'weighted_gaussian_full'

    # Check dimensions
    assert len(model['mean']) == n_vars
    assert model['cov'].shape == (n_vars, n_vars)

    # Check covariance is symmetric and positive definite
    assert np.allclose(model['cov'], model['cov'].T)
    eigenvalues = np.linalg.eigvals(model['cov'])
    assert np.all(eigenvalues > 0)


def test_weighted_gaussian_sampling():
    """Test that sampling from weighted Gaussian models produces valid populations"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 100

    # Create and learn model
    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)
    model = learn_weighted_gaussian_univariate(population, fitness)

    # Sample
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    new_population = sample_weighted_gaussian_univariate(model, pop_size, bounds)

    # Check dimensions
    assert new_population.shape == (pop_size, n_vars)

    # Check bounds are respected
    assert np.all(new_population >= bounds[0])
    assert np.all(new_population <= bounds[1])


def test_weighted_gaussian_eda_optimization():
    """Test that weighted Gaussian EDA can optimize sphere function"""
    np.random.seed(42)

    best_fitness, best_solution, history = run_simple_eda(
        objective_func=sphere_function,
        learn_func=learn_weighted_gaussian_univariate,
        sample_func=sample_weighted_gaussian_univariate,
        n_vars=5,
        pop_size=50,
        n_generations=30
    )

    # Check that fitness improved
    assert history[-1] < history[0]

    # Check that we got reasonably close to optimum (0)
    assert best_fitness < 1.0


# ============================================================================
# Tests for Gaussian Mixture EM
# ============================================================================

def test_gaussian_mixture_em_learning():
    """Test that Gaussian Mixture EM learning produces valid models"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 100
    n_components = 3

    # Create multimodal population
    pop1 = np.random.randn(30, n_vars) + np.array([2.0] * n_vars)
    pop2 = np.random.randn(30, n_vars) + np.array([-2.0] * n_vars)
    pop3 = np.random.randn(40, n_vars)
    population = np.vstack([pop1, pop2, pop3])
    fitness = sphere_function(population)

    # Learn model
    model = learn_mixture_gaussian_em(
        population,
        fitness,
        params={'n_components': n_components}
    )

    # Check model structure
    assert 'gm_model' in model
    assert 'n_components' in model
    assert 'type' in model
    assert model['type'] == 'mixture_gaussian_em'
    assert model['n_components'] == n_components

    # Check sklearn model
    gm = model['gm_model']
    assert hasattr(gm, 'means_')
    assert hasattr(gm, 'covariances_')
    assert hasattr(gm, 'weights_')
    assert len(gm.means_) == n_components


def test_gaussian_mixture_em_sampling():
    """Test that sampling from Gaussian Mixture EM models works"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 100

    # Create and learn model
    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)
    model = learn_mixture_gaussian_em(
        population,
        fitness,
        params={'n_components': 2}
    )

    # Sample
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    new_population = sample_mixture_gaussian_em(model, pop_size, bounds)

    # Check dimensions
    assert new_population.shape == (pop_size, n_vars)

    # Check bounds are respected
    assert np.all(new_population >= bounds[0])
    assert np.all(new_population <= bounds[1])


def test_gaussian_mixture_em_eda_optimization():
    """Test that Gaussian Mixture EM EDA can optimize"""
    np.random.seed(42)

    best_fitness, best_solution, history = run_simple_eda(
        objective_func=sphere_function,
        learn_func=learn_mixture_gaussian_em,
        sample_func=sample_mixture_gaussian_em,
        n_vars=5,
        pop_size=100,
        n_generations=30,
        learn_params={'n_components': 2}
    )

    # Check that fitness improved
    assert history[-1] < history[0]
    assert best_fitness < 5.0


# ============================================================================
# Tests for Diversity-Triggered Sampling
# ============================================================================

def test_diversity_trigger_disabled():
    """Test that diversity trigger is disabled when threshold < 0"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 50

    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)
    model = learn_gaussian_full(population, fitness)

    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    # Sample without diversity trigger
    params = {'diversity_threshold': -1}
    pop1 = sample_gaussian_with_diversity_trigger(model, pop_size, bounds, params)

    # Sample with standard sampling
    pop2 = sample_gaussian_full(model, pop_size, bounds)

    # They should produce similar distributions (statistically)
    assert pop1.shape == pop2.shape


def test_diversity_trigger_enabled():
    """Test that diversity trigger expands variance when triggered"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 50

    # Create a population with very low variance
    population = np.random.randn(pop_size, n_vars) * 0.001
    fitness = sphere_function(population)
    model = learn_gaussian_full(population, fitness)

    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    # Sample with diversity trigger enabled
    params = {'diversity_threshold': 1.0, 'diversity_scaling': 2.0}
    new_population = sample_gaussian_with_diversity_trigger(model, pop_size, bounds, params)

    # Check that new population has higher variance than original
    new_var = np.mean(np.var(new_population, axis=0))
    old_var = np.mean(np.var(population, axis=0))

    assert new_var > old_var


# ============================================================================
# Integration Tests
# ============================================================================

def test_weighted_gaussian_vs_standard():
    """Compare weighted Gaussian EDA with standard Gaussian EDA"""
    np.random.seed(42)

    # Run weighted version
    best_weighted, _, history_weighted = run_simple_eda(
        objective_func=sphere_function,
        learn_func=learn_weighted_gaussian_univariate,
        sample_func=sample_weighted_gaussian_univariate,
        n_vars=5,
        pop_size=50,
        n_generations=30
    )

    np.random.seed(42)

    # Run standard version
    best_standard, _, history_standard = run_simple_eda(
        objective_func=sphere_function,
        learn_func=learn_gaussian_univariate,
        sample_func=sample_gaussian_univariate,
        n_vars=5,
        pop_size=50,
        n_generations=30
    )

    # Both should converge (exact performance may vary)
    assert best_weighted < 10.0
    assert best_standard < 10.0


def test_all_new_edas_on_sphere():
    """Test all new EDAs on sphere function"""
    np.random.seed(42)

    configs = [
        {
            'name': 'Weighted Gaussian Univariate',
            'learn': learn_weighted_gaussian_univariate,
            'sample': sample_weighted_gaussian_univariate,
            'learn_params': None,
            'sample_params': None
        },
        {
            'name': 'Weighted Gaussian Full',
            'learn': learn_weighted_gaussian_full,
            'sample': sample_weighted_gaussian_full,
            'learn_params': None,
            'sample_params': None
        },
        {
            'name': 'Gaussian Mixture EM',
            'learn': learn_mixture_gaussian_em,
            'sample': sample_mixture_gaussian_em,
            'learn_params': {'n_components': 2},
            'sample_params': None
        },
    ]

    results = {}

    for config in configs:
        np.random.seed(42)
        best_fitness, _, history = run_simple_eda(
            objective_func=sphere_function,
            learn_func=config['learn'],
            sample_func=config['sample'],
            n_vars=5,
            pop_size=50,
            n_generations=30,
            learn_params=config['learn_params'],
            sample_params=config['sample_params']
        )
        results[config['name']] = best_fitness

    # All should achieve reasonable performance
    for name, fitness in results.items():
        assert fitness < 10.0, f"{name} failed to converge: {fitness}"


# ============================================================================
# Test Parameter Variations
# ============================================================================

def test_weighted_gaussian_different_beta():
    """Test weighted Gaussian with different beta values"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 50

    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)

    # Test different beta values
    for beta in [0.01, 0.1, 1.0]:
        model = learn_weighted_gaussian_univariate(
            population,
            fitness,
            params={'beta': beta}
        )

        assert 'means' in model
        assert 'stds' in model
        assert len(model['means']) == n_vars


def test_gaussian_mixture_em_different_components():
    """Test Gaussian Mixture EM with different numbers of components"""
    np.random.seed(42)
    n_vars = 5
    pop_size = 100

    population = np.random.randn(pop_size, n_vars)
    fitness = sphere_function(population)

    # Test different numbers of components
    for n_comp in [1, 2, 3, 5]:
        model = learn_mixture_gaussian_em(
            population,
            fitness,
            params={'n_components': n_comp}
        )

        assert model['n_components'] == n_comp
        assert len(model['gm_model'].means_) == n_comp


if __name__ == '__main__':
    # Run all tests
    print("Running tests for new continuous EDAs...")
    print("\n" + "="*70)
    print("Testing Weighted Gaussian Learning and Sampling")
    print("="*70)

    test_weighted_gaussian_univariate_learning()
    print("✓ Weighted Gaussian univariate learning")

    test_weighted_gaussian_full_learning()
    print("✓ Weighted Gaussian full learning")

    test_weighted_gaussian_sampling()
    print("✓ Weighted Gaussian sampling")

    test_weighted_gaussian_eda_optimization()
    print("✓ Weighted Gaussian EDA optimization")

    print("\n" + "="*70)
    print("Testing Gaussian Mixture EM")
    print("="*70)

    test_gaussian_mixture_em_learning()
    print("✓ Gaussian Mixture EM learning")

    test_gaussian_mixture_em_sampling()
    print("✓ Gaussian Mixture EM sampling")

    test_gaussian_mixture_em_eda_optimization()
    print("✓ Gaussian Mixture EM EDA optimization")

    print("\n" + "="*70)
    print("Testing Diversity-Triggered Sampling")
    print("="*70)

    test_diversity_trigger_disabled()
    print("✓ Diversity trigger disabled")

    test_diversity_trigger_enabled()
    print("✓ Diversity trigger enabled")

    print("\n" + "="*70)
    print("Integration Tests")
    print("="*70)

    test_weighted_gaussian_vs_standard()
    print("✓ Weighted vs standard Gaussian comparison")

    test_all_new_edas_on_sphere()
    print("✓ All new EDAs on sphere function")

    print("\n" + "="*70)
    print("Parameter Variation Tests")
    print("="*70)

    test_weighted_gaussian_different_beta()
    print("✓ Weighted Gaussian with different beta values")

    test_gaussian_mixture_em_different_components()
    print("✓ Gaussian Mixture EM with different components")

    print("\n" + "="*70)
    print("All tests passed successfully!")
    print("="*70)
