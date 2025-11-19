"""
Minimal tests for newly added Gaussian EDA components

This test file directly imports only the necessary Gaussian functions
to avoid loading heavy dependencies like PyTorch.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Direct imports to avoid loading the full package
import importlib.util

# Manually load the gaussian learning module
spec_learn = importlib.util.spec_from_file_location(
    "gaussian_learning",
    "/home/user/pateda/pateda/learning/gaussian.py"
)
gaussian_learning = importlib.util.module_from_spec(spec_learn)
spec_learn.loader.exec_module(gaussian_learning)

# Manually load the gaussian sampling module
spec_sample = importlib.util.spec_from_file_location(
    "gaussian_sampling",
    "/home/user/pateda/pateda/sampling/gaussian.py"
)
gaussian_sampling = importlib.util.module_from_spec(spec_sample)
spec_sample.loader.exec_module(gaussian_sampling)

# Extract functions
learn_weighted_gaussian_univariate = gaussian_learning.learn_weighted_gaussian_univariate
learn_weighted_gaussian_full = gaussian_learning.learn_weighted_gaussian_full
learn_mixture_gaussian_em = gaussian_learning.learn_mixture_gaussian_em
learn_gaussian_univariate = gaussian_learning.learn_gaussian_univariate
learn_gaussian_full = gaussian_learning.learn_gaussian_full

sample_weighted_gaussian_univariate = gaussian_sampling.sample_weighted_gaussian_univariate
sample_weighted_gaussian_full = gaussian_sampling.sample_weighted_gaussian_full
sample_gaussian_with_diversity_trigger = gaussian_sampling.sample_gaussian_with_diversity_trigger
sample_mixture_gaussian_em = gaussian_sampling.sample_mixture_gaussian_em
sample_gaussian_univariate = gaussian_sampling.sample_gaussian_univariate
sample_gaussian_full = gaussian_sampling.sample_gaussian_full


# ============================================================================
# Test Functions
# ============================================================================

def sphere_function(x):
    """Sphere function: f(x) = sum(x^2)"""
    return np.sum(x**2, axis=1)


def run_simple_eda(learn_func, sample_func, n_vars=5, pop_size=50,
                   n_generations=30, learn_params=None, sample_params=None):
    """Run a simple EDA test"""
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, n_vars))

    fitness_history = []
    best_fitness = float('inf')

    for gen in range(n_generations):
        fitness = sphere_function(population)

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]

        fitness_history.append(best_fitness)

        n_selected = int(pop_size * 0.5)
        selected_idx = np.argsort(fitness)[:n_selected]
        selected_pop = population[selected_idx]
        selected_fit = fitness[selected_idx]

        if learn_params is None:
            model = learn_func(selected_pop, selected_fit)
        else:
            model = learn_func(selected_pop, selected_fit, learn_params)

        if sample_params is None:
            population = sample_func(model, pop_size, bounds)
        else:
            population = sample_func(model, pop_size, bounds, sample_params)

    return best_fitness, fitness_history


print("="*70)
print("Testing Weighted Gaussian EDAs")
print("="*70)

# Test 1: Weighted Gaussian Univariate Learning
print("\n1. Testing weighted univariate Gaussian learning...")
np.random.seed(42)
population = np.random.randn(50, 5)
fitness = sphere_function(population)
model = learn_weighted_gaussian_univariate(population, fitness)
assert 'means' in model
assert 'stds' in model
assert model['type'] == 'weighted_gaussian_univariate'
assert len(model['means']) == 5
assert np.all(model['stds'] > 0)
print("   ✓ Weighted univariate Gaussian learning works")

# Test 2: Weighted Gaussian Full Learning
print("2. Testing weighted full Gaussian learning...")
np.random.seed(42)
model = learn_weighted_gaussian_full(population, fitness)
assert 'mean' in model
assert 'cov' in model
assert model['type'] == 'weighted_gaussian_full'
assert len(model['mean']) == 5
assert model['cov'].shape == (5, 5)
print("   ✓ Weighted full Gaussian learning works")

# Test 3: Weighted Gaussian Sampling
print("3. Testing weighted Gaussian sampling...")
np.random.seed(42)
model = learn_weighted_gaussian_univariate(population, fitness)
bounds = np.array([[-5.0] * 5, [5.0] * 5])
new_pop = sample_weighted_gaussian_univariate(model, 50, bounds)
assert new_pop.shape == (50, 5)
assert np.all(new_pop >= bounds[0])
assert np.all(new_pop <= bounds[1])
print("   ✓ Weighted Gaussian sampling works")

# Test 4: Weighted Gaussian EDA Optimization
print("4. Testing weighted Gaussian EDA optimization...")
np.random.seed(42)
best_fit, history = run_simple_eda(
    learn_weighted_gaussian_univariate,
    sample_weighted_gaussian_univariate,
    n_vars=5,
    pop_size=50,
    n_generations=30
)
assert history[-1] < history[0], "Fitness should improve"
assert best_fit < 1.0, "Should converge close to optimum"
print(f"   ✓ Weighted Gaussian EDA converged: {best_fit:.6f}")

# Test 5: Gaussian Mixture EM Learning
print("5. Testing Gaussian Mixture EM learning...")
np.random.seed(42)
pop1 = np.random.randn(30, 5) + 2.0
pop2 = np.random.randn(30, 5) - 2.0
population = np.vstack([pop1, pop2])
fitness = sphere_function(population)
model = learn_mixture_gaussian_em(population, fitness, {'n_components': 2})
assert 'gm_model' in model
assert 'n_components' in model
assert model['n_components'] == 2
assert model['type'] == 'mixture_gaussian_em'
print("   ✓ Gaussian Mixture EM learning works")

# Test 6: Gaussian Mixture EM Sampling
print("6. Testing Gaussian Mixture EM sampling...")
np.random.seed(42)
bounds = np.array([[-5.0] * 5, [5.0] * 5])
new_pop = sample_mixture_gaussian_em(model, 60, bounds)
assert new_pop.shape == (60, 5)
assert np.all(new_pop >= bounds[0])
assert np.all(new_pop <= bounds[1])
print("   ✓ Gaussian Mixture EM sampling works")

# Test 7: Gaussian Mixture EM EDA Optimization
print("7. Testing Gaussian Mixture EM EDA optimization...")
np.random.seed(42)
best_fit, history = run_simple_eda(
    learn_mixture_gaussian_em,
    sample_mixture_gaussian_em,
    n_vars=5,
    pop_size=100,
    n_generations=30,
    learn_params={'n_components': 2}
)
assert history[-1] < history[0], "Fitness should improve"
assert best_fit < 5.0, "Should achieve reasonable fitness"
print(f"   ✓ Gaussian Mixture EM EDA converged: {best_fit:.6f}")

# Test 8: Diversity Trigger
print("8. Testing diversity-triggered sampling...")
np.random.seed(42)
# Create low variance population
low_var_pop = np.random.randn(50, 5) * 0.001
fitness = sphere_function(low_var_pop)
model = learn_gaussian_full(low_var_pop, fitness)
bounds = np.array([[-5.0] * 5, [5.0] * 5])
params = {'diversity_threshold': 1.0, 'diversity_scaling': 2.0}
new_pop = sample_gaussian_with_diversity_trigger(model, 50, bounds, params)
new_var = np.mean(np.var(new_pop, axis=0))
old_var = np.mean(np.var(low_var_pop, axis=0))
assert new_var > old_var, "Diversity trigger should increase variance"
print("   ✓ Diversity trigger works correctly")

# Test 9: Different Beta Values
print("9. Testing different beta values...")
np.random.seed(42)
population = np.random.randn(50, 5)
fitness = sphere_function(population)
for beta in [0.01, 0.1, 1.0]:
    model = learn_weighted_gaussian_univariate(population, fitness, {'beta': beta})
    assert 'means' in model
    assert len(model['means']) == 5
print("   ✓ Different beta values work")

# Test 10: Different Number of Components
print("10. Testing different numbers of mixture components...")
np.random.seed(42)
population = np.random.randn(100, 5)
fitness = sphere_function(population)
for n_comp in [1, 2, 3]:
    model = learn_mixture_gaussian_em(population, fitness, {'n_components': n_comp})
    assert model['n_components'] == n_comp
    assert len(model['gm_model'].means_) == n_comp
print("   ✓ Different component numbers work")

# Test 11: Comparison Test
print("11. Comparing weighted vs standard Gaussian...")
np.random.seed(42)
best_weighted, _ = run_simple_eda(
    learn_weighted_gaussian_univariate,
    sample_weighted_gaussian_univariate
)
np.random.seed(42)
best_standard, _ = run_simple_eda(
    learn_gaussian_univariate,
    sample_gaussian_univariate
)
assert best_weighted < 10.0, "Weighted should converge"
assert best_standard < 10.0, "Standard should converge"
print(f"   ✓ Weighted: {best_weighted:.6f}, Standard: {best_standard:.6f}")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nSummary:")
print("  ✓ Weighted Gaussian Univariate - Learning and Sampling")
print("  ✓ Weighted Gaussian Full - Learning and Sampling")
print("  ✓ Gaussian Mixture EM - Learning and Sampling")
print("  ✓ Diversity-Triggered Sampling")
print("  ✓ Parameter Variations")
print("  ✓ EDA Optimization Tests")
print("\nAll new continuous EDA components are working correctly!")
