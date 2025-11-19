"""
Test suite for GMRF-EDA implementation

This script tests the GMRF-EDA (Gaussian Markov Random Field EDA) implementation
on various continuous optimization benchmark functions.
"""

import numpy as np
from typing import Callable, Tuple
import time

# Make matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualization")

# Import directly from module files to avoid __init__ dependency issues
import sys
import os
import importlib.util

# Load gaussian learning module directly
spec_learning = importlib.util.spec_from_file_location(
    "gaussian_learning",
    os.path.join(os.path.dirname(__file__), "pateda/learning/gaussian.py")
)
gaussian_learning = importlib.util.module_from_spec(spec_learning)
spec_learning.loader.exec_module(gaussian_learning)

# Load gaussian sampling module directly
spec_sampling = importlib.util.spec_from_file_location(
    "gaussian_sampling",
    os.path.join(os.path.dirname(__file__), "pateda/sampling/gaussian.py")
)
gaussian_sampling = importlib.util.module_from_spec(spec_sampling)
spec_sampling.loader.exec_module(gaussian_sampling)

# Load benchmarks module directly
spec_benchmarks = importlib.util.spec_from_file_location(
    "benchmarks",
    os.path.join(os.path.dirname(__file__), "pateda/functions/continuous/benchmarks.py")
)
benchmarks = importlib.util.module_from_spec(spec_benchmarks)
spec_benchmarks.loader.exec_module(benchmarks)

# Extract functions
learn_gmrf_eda = gaussian_learning.learn_gmrf_eda
learn_gmrf_eda_lasso = gaussian_learning.learn_gmrf_eda_lasso
learn_gmrf_eda_elasticnet = gaussian_learning.learn_gmrf_eda_elasticnet
learn_gmrf_eda_lars = gaussian_learning.learn_gmrf_eda_lars
learn_gaussian_univariate = gaussian_learning.learn_gaussian_univariate
learn_gaussian_full = gaussian_learning.learn_gaussian_full

sample_gmrf_eda = gaussian_sampling.sample_gmrf_eda
sample_gaussian_univariate = gaussian_sampling.sample_gaussian_univariate
sample_gaussian_full = gaussian_sampling.sample_gaussian_full

sphere = benchmarks.sphere
rastrigin = benchmarks.rastrigin
rosenbrock = benchmarks.rosenbrock
ackley = benchmarks.ackley


def create_additive_function(n_vars: int, block_size: int = 2) -> Callable:
    """
    Create an additive continuous function with known structure.

    The function is sum of block_size-dimensional subfunctions,
    testing GMRF-EDA's ability to discover the decomposition.
    """
    def additive_func(x: np.ndarray) -> float:
        if x.ndim == 1:
            total = 0.0
            for i in range(0, n_vars, block_size):
                block = x[i:min(i+block_size, n_vars)]
                # Rosenbrock-like function on each block
                total += np.sum(100.0 * (block[1:] - block[:-1]**2)**2 + (1 - block[:-1])**2)
            return total
        else:
            values = np.zeros(len(x))
            for j in range(len(x)):
                values[j] = additive_func(x[j])
            return values

    return additive_func


def run_gmrf_eda_test(
    func: Callable,
    n_vars: int,
    bounds: Tuple[float, float],
    learning_func: Callable,
    learning_name: str,
    n_generations: int = 100,
    pop_size: int = 100,
    selection_ratio: float = 0.5,
    params: dict = None,
    verbose: bool = True
) -> dict:
    """
    Run GMRF-EDA optimization test on a given function.

    Parameters
    ----------
    func : callable
        Objective function to minimize
    n_vars : int
        Number of variables
    bounds : tuple
        (lower, upper) bounds for all variables
    learning_func : callable
        Learning function to use
    learning_name : str
        Name of the learning method
    n_generations : int
        Number of generations
    pop_size : int
        Population size
    selection_ratio : float
        Ratio of population to select
    params : dict
        Parameters for learning function
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary with results
    """
    if params is None:
        params = {}

    # Setup bounds
    lower_bound = np.full(n_vars, bounds[0])
    upper_bound = np.full(n_vars, bounds[1])
    bounds_array = np.array([lower_bound, upper_bound])

    # Initialize population randomly
    population = np.random.uniform(
        lower_bound, upper_bound,
        size=(pop_size, n_vars)
    )

    # Track best fitness over generations
    best_fitness_history = []
    mean_fitness_history = []
    time_history = []

    n_select = int(pop_size * selection_ratio)

    start_time = time.time()

    for gen in range(n_generations):
        gen_start = time.time()

        # Evaluate population
        fitness = func(population)

        # Track statistics
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        mean_fitness = np.mean(fitness)

        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:3d}: Best = {best_fitness:.6e}, Mean = {mean_fitness:.6e}")

        # Selection (truncation selection)
        selected_indices = np.argsort(fitness)[:n_select]
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        # Learning
        model = learning_func(selected_pop, selected_fitness, params)

        # Sampling
        if 'gmrf' in learning_name.lower():
            new_population = sample_gmrf_eda(
                model, pop_size, bounds_array
            )
        elif 'univariate' in learning_name.lower():
            new_population = sample_gaussian_univariate(
                model, pop_size, bounds_array
            )
        else:  # full
            new_population = sample_gaussian_full(
                model, pop_size, bounds_array
            )

        population = new_population

        time_history.append(time.time() - gen_start)

    total_time = time.time() - start_time

    # Final evaluation
    final_fitness = func(population)
    final_best = np.min(final_fitness)
    final_best_solution = population[np.argmin(final_fitness)]

    if verbose:
        print(f"\nFinal best fitness: {final_best:.6e}")
        print(f"Total time: {total_time:.2f}s")
        if 'gmrf' in learning_name.lower():
            print(f"Number of cliques: {len(model['cliques'])}")
            print(f"Cliques: {model['cliques']}")

    return {
        'best_fitness_history': best_fitness_history,
        'mean_fitness_history': mean_fitness_history,
        'time_history': time_history,
        'total_time': total_time,
        'final_best': final_best,
        'final_best_solution': final_best_solution,
        'final_model': model,
    }


def test_basic_functionality():
    """Test basic GMRF-EDA functionality"""
    print("="*80)
    print("TEST 1: Basic Functionality Test")
    print("="*80)

    # Create small test population
    n_vars = 6
    pop_size = 50
    population = np.random.randn(pop_size, n_vars)
    fitness = np.random.rand(pop_size)

    print("\nTesting GMRF-EDA learning methods...")

    # Test LASSO
    print("\n1. Testing LASSO regularization...")
    model_lasso = learn_gmrf_eda_lasso(population, fitness, {'alpha': 0.1})
    print(f"   - Model type: {model_lasso['type']}")
    print(f"   - Number of cliques: {len(model_lasso['cliques'])}")
    print(f"   - Cliques: {model_lasso['cliques']}")

    # Test ElasticNet
    print("\n2. Testing ElasticNet regularization...")
    model_enet = learn_gmrf_eda_elasticnet(population, fitness, {'alpha': 0.1})
    print(f"   - Model type: {model_enet['type']}")
    print(f"   - Number of cliques: {len(model_enet['cliques'])}")
    print(f"   - Cliques: {model_enet['cliques']}")

    # Test LARS
    print("\n3. Testing LARS regularization...")
    model_lars = learn_gmrf_eda_lars(population, fitness)
    print(f"   - Model type: {model_lars['type']}")
    print(f"   - Number of cliques: {len(model_lars['cliques'])}")
    print(f"   - Cliques: {model_lars['cliques']}")

    # Test sampling
    print("\n4. Testing sampling...")
    bounds = np.array([[-5]*n_vars, [5]*n_vars])
    samples = sample_gmrf_eda(model_lasso, 20, bounds)
    print(f"   - Sample shape: {samples.shape}")
    print(f"   - Sample mean: {np.mean(samples, axis=0)}")
    print(f"   - Sample std: {np.std(samples, axis=0)}")

    print("\n✓ Basic functionality test passed!\n")


def test_sphere_function():
    """Test GMRF-EDA on sphere function"""
    print("="*80)
    print("TEST 2: Sphere Function (Separable)")
    print("="*80)

    n_vars = 10

    print(f"\nOptimizing {n_vars}D Sphere function...")
    print("Expected structure: All variables independent")

    results = {}

    # Test GMRF-EDA variants
    for name, learn_func, params in [
        ("GMRF-LASSO", learn_gmrf_eda_lasso, {'alpha': 0.01}),
        ("GMRF-ElasticNet", learn_gmrf_eda_elasticnet, {'alpha': 0.01, 'l1_ratio': 0.5}),
        ("GMRF-LARS", learn_gmrf_eda_lars, {}),
    ]:
        print(f"\n--- {name} ---")
        result = run_gmrf_eda_test(
            sphere, n_vars, (-5.12, 5.12),
            learn_func, name,
            n_generations=50,
            pop_size=100,
            params=params,
            verbose=False
        )
        results[name] = result
        print(f"Final best: {result['final_best']:.6e}")
        print(f"Cliques: {result['final_model']['cliques']}")

    # Also test baselines
    print(f"\n--- Gaussian Univariate (baseline) ---")
    result = run_gmrf_eda_test(
        sphere, n_vars, (-5.12, 5.12),
        learn_gaussian_univariate, "Univariate",
        n_generations=50,
        pop_size=100,
        verbose=False
    )
    results["Univariate"] = result
    print(f"Final best: {result['final_best']:.6e}")

    print("\n✓ Sphere function test completed!\n")
    return results


def test_additive_function():
    """Test GMRF-EDA on additive function with structure"""
    print("="*80)
    print("TEST 3: Additive Function with Block Structure")
    print("="*80)

    n_vars = 10
    block_size = 2
    func = create_additive_function(n_vars, block_size)

    print(f"\nOptimizing {n_vars}D Additive function (block size={block_size})...")
    print(f"Expected structure: {n_vars//block_size} blocks of size {block_size}")

    results = {}

    # Test GMRF-EDA variants
    for name, learn_func, params in [
        ("GMRF-LASSO", learn_gmrf_eda_lasso, {'alpha': 0.05}),
        ("GMRF-ElasticNet", learn_gmrf_eda_elasticnet, {'alpha': 0.05, 'l1_ratio': 0.7}),
    ]:
        print(f"\n--- {name} ---")
        result = run_gmrf_eda_test(
            func, n_vars, (-2.0, 2.0),
            learn_func, name,
            n_generations=100,
            pop_size=150,
            params=params,
            verbose=False
        )
        results[name] = result
        print(f"Final best: {result['final_best']:.6e}")
        print(f"Cliques found: {result['final_model']['cliques']}")

    # Baseline
    print(f"\n--- Gaussian Full (baseline) ---")
    result = run_gmrf_eda_test(
        func, n_vars, (-2.0, 2.0),
        learn_gaussian_full, "Full",
        n_generations=100,
        pop_size=150,
        verbose=False
    )
    results["Full"] = result
    print(f"Final best: {result['final_best']:.6e}")

    print("\n✓ Additive function test completed!\n")
    return results


def test_rosenbrock_function():
    """Test GMRF-EDA on Rosenbrock function"""
    print("="*80)
    print("TEST 4: Rosenbrock Function (Chain-structured)")
    print("="*80)

    n_vars = 10

    print(f"\nOptimizing {n_vars}D Rosenbrock function...")
    print("Expected structure: Chain dependencies between adjacent variables")

    results = {}

    for name, learn_func, params in [
        ("GMRF-LASSO", learn_gmrf_eda_lasso, {'alpha': 0.02}),
        ("GMRF-ElasticNet", learn_gmrf_eda_elasticnet, {'alpha': 0.02, 'l1_ratio': 0.5}),
    ]:
        print(f"\n--- {name} ---")
        result = run_gmrf_eda_test(
            rosenbrock, n_vars, (-5.0, 10.0),
            learn_func, name,
            n_generations=150,
            pop_size=200,
            params=params,
            verbose=False
        )
        results[name] = result
        print(f"Final best: {result['final_best']:.6e}")
        print(f"Cliques found: {result['final_model']['cliques']}")

    print("\n✓ Rosenbrock function test completed!\n")
    return results


def visualize_results(results_dict: dict, title: str, filename: str = None):
    """Visualize optimization results"""
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for {title} (matplotlib not available)")
        return

    plt.figure(figsize=(12, 5))

    # Plot best fitness
    plt.subplot(1, 2, 1)
    for name, results in results_dict.items():
        plt.semilogy(results['best_fitness_history'], label=name, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title(f'{title} - Best Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot mean fitness
    plt.subplot(1, 2, 2)
    for name, results in results_dict.items():
        plt.semilogy(results['mean_fitness_history'], label=name, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Fitness (log scale)')
    plt.title(f'{title} - Mean Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" GMRF-EDA Test Suite")
    print("="*80 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Test 1: Basic functionality
    test_basic_functionality()

    # Test 2: Sphere function
    results_sphere = test_sphere_function()

    # Test 3: Additive function
    results_additive = test_additive_function()

    # Test 4: Rosenbrock function
    results_rosenbrock = test_rosenbrock_function()

    # Visualize results
    print("="*80)
    print("Generating visualization plots...")
    print("="*80)

    visualize_results(results_sphere, "Sphere Function", "gmrf_eda_sphere.png")
    visualize_results(results_additive, "Additive Function", "gmrf_eda_additive.png")
    visualize_results(results_rosenbrock, "Rosenbrock Function", "gmrf_eda_rosenbrock.png")

    print("\n" + "="*80)
    print(" All tests completed successfully!")
    print("="*80 + "\n")

    # Summary
    print("Summary:")
    print("-" * 80)
    print(f"{'Function':<25} {'Method':<20} {'Final Best':<15}")
    print("-" * 80)

    for func_name, results in [
        ("Sphere", results_sphere),
        ("Additive", results_additive),
        ("Rosenbrock", results_rosenbrock),
    ]:
        for method, result in results.items():
            print(f"{func_name:<25} {method:<20} {result['final_best']:<15.6e}")

    print("-" * 80)


if __name__ == "__main__":
    main()
