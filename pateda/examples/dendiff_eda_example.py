"""
Example demonstrating the use of Denoising Diffusion-based EDAs for continuous optimization.

This example shows how to use DenDiff-EDA:
- A novel EDA based on denoising diffusion probabilistic models
- Uses MLP-based denoising network to learn complex distributions
- Demonstrates performance on benchmark functions with large populations
- Measures execution time for learning and sampling
"""

import numpy as np
import time
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff, sample_dendiff_fast


def sphere_function(x):
    """Simple sphere function for testing: f(x) = sum(x_i^2)"""
    return np.sum(x**2, axis=1)


def rosenbrock_function(x):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin_function(x):
    """Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley_function(x):
    """Ackley function"""
    n = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def run_dendiff_eda(
    fitness_function,
    n_vars,
    bounds,
    pop_size=200,
    n_generations=30,
    selection_ratio=0.3,
    use_fast_sampling=False
):
    """
    Run DenDiff-EDA on a given fitness function.

    Parameters
    ----------
    fitness_function : callable
        Function to optimize
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Array of shape (2, n_vars) with [min, max] bounds
    pop_size : int
        Population size (default: 200)
    n_generations : int
        Number of generations (default: 30)
    selection_ratio : float
        Ratio of population to select (default: 0.3)
    use_fast_sampling : bool
        Whether to use fast sampling (default: False)
    """
    print(f"\n{'='*80}")
    print(f"Running DenDiff-EDA on {fitness_function.__name__}")
    print(f"Population size: {pop_size}, Generations: {n_generations}, Variables: {n_vars}")
    print(f"Fast sampling: {use_fast_sampling}")
    print(f"{'='*80}\n")

    selection_size = int(pop_size * selection_ratio)

    # DenDiff parameters
    dendiff_params = {
        'n_timesteps': 1000,  # Standard diffusion steps
        'beta_schedule': 'linear',
        'hidden_dims': [128, 64],
        'time_emb_dim': 32,
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-3
    }

    # Fast sampling parameters
    fast_sampling_params = {
        'ddim_steps': 50,  # Much fewer steps
        'ddim_eta': 0.0    # Deterministic sampling
    }

    # Initialize population
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    # Track metrics
    best_fitness_history = []
    learning_times = []
    sampling_times = []

    # EDA loop
    for gen in range(n_generations):
        # Evaluate
        fitness = fitness_function(population)

        # Track best
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx]
        best_fitness_history.append(best_fitness)

        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6e}")

        # Select best individuals
        idx = np.argsort(fitness)[:selection_size]
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn model
        start_time = time.time()
        model = learn_dendiff(selected_pop, selected_fit, params=dendiff_params)
        learning_time = time.time() - start_time
        learning_times.append(learning_time)

        # Sample new population
        start_time = time.time()
        if use_fast_sampling:
            new_population = sample_dendiff_fast(
                model, n_samples=pop_size, bounds=bounds,
                params=fast_sampling_params
            )
        else:
            new_population = sample_dendiff(model, n_samples=pop_size, bounds=bounds)
        sampling_time = time.time() - start_time
        sampling_times.append(sampling_time)

        # Replace population
        population = new_population

    # Final evaluation
    final_fitness = fitness_function(population)
    final_best_idx = np.argmin(final_fitness)
    final_best_fitness = final_fitness[final_best_idx]
    final_best_solution = population[final_best_idx]

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Initial best fitness: {best_fitness_history[0]:.6e}")
    print(f"Final best fitness:   {final_best_fitness:.6e}")
    print(f"Improvement:          {best_fitness_history[0] / max(final_best_fitness, 1e-10):.2f}x")
    print(f"\nTiming Statistics:")
    print(f"  Average learning time:  {np.mean(learning_times):.3f} ± {np.std(learning_times):.3f} seconds")
    print(f"  Average sampling time:  {np.mean(sampling_times):.3f} ± {np.std(sampling_times):.3f} seconds")
    print(f"  Total learning time:    {np.sum(learning_times):.3f} seconds")
    print(f"  Total sampling time:    {np.sum(sampling_times):.3f} seconds")
    print(f"  Total time per gen:     {np.mean(learning_times) + np.mean(sampling_times):.3f} seconds")
    print(f"{'='*80}\n")

    return {
        'best_fitness_history': best_fitness_history,
        'final_best_fitness': final_best_fitness,
        'final_best_solution': final_best_solution,
        'learning_times': learning_times,
        'sampling_times': sampling_times
    }


def test_large_population_learning():
    """
    Test DenDiff-EDA with large population sizes to demonstrate distribution learning capability.
    """
    print("\n" + "#"*80)
    print("# TEST: LARGE POPULATION DISTRIBUTION LEARNING")
    print("#"*80)

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    print("\nTesting with progressively larger population sizes...")

    population_sizes = [100, 200, 500]

    for pop_size in population_sizes:
        print(f"\n{'='*80}")
        print(f"Population size: {pop_size}")
        print(f"{'='*80}")

        result = run_dendiff_eda(
            sphere_function,
            n_vars,
            bounds,
            pop_size=pop_size,
            n_generations=20,
            selection_ratio=0.3,
            use_fast_sampling=False
        )

        print(f"✓ Successfully completed with pop_size={pop_size}")
        print(f"  Final fitness: {result['final_best_fitness']:.6e}")
        print(f"  Avg time/gen: {np.mean(result['learning_times']) + np.mean(result['sampling_times']):.3f}s")


def compare_sampling_methods():
    """
    Compare standard sampling vs fast sampling.
    """
    print("\n" + "#"*80)
    print("# TEST: STANDARD VS FAST SAMPLING")
    print("#"*80)

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    pop_size = 200

    print("\n1. Standard sampling (1000 diffusion steps):")
    result_standard = run_dendiff_eda(
        sphere_function,
        n_vars,
        bounds,
        pop_size=pop_size,
        n_generations=15,
        use_fast_sampling=False
    )

    print("\n2. Fast sampling (50 DDIM steps):")
    result_fast = run_dendiff_eda(
        sphere_function,
        n_vars,
        bounds,
        pop_size=pop_size,
        n_generations=15,
        use_fast_sampling=True
    )

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Standard sampling:")
    print(f"  Final fitness: {result_standard['final_best_fitness']:.6e}")
    print(f"  Avg sampling time: {np.mean(result_standard['sampling_times']):.3f}s")
    print(f"\nFast sampling:")
    print(f"  Final fitness: {result_fast['final_best_fitness']:.6e}")
    print(f"  Avg sampling time: {np.mean(result_fast['sampling_times']):.3f}s")
    print(f"\nSpeedup: {np.mean(result_standard['sampling_times']) / np.mean(result_fast['sampling_times']):.2f}x")
    print("="*80)


def test_benchmark_functions():
    """
    Test DenDiff-EDA on multiple benchmark functions.
    """
    print("\n" + "#"*80)
    print("# TEST: BENCHMARK FUNCTIONS")
    print("#"*80)

    n_vars = 10
    pop_size = 200
    n_generations = 25

    benchmarks = [
        (sphere_function, np.array([[-5.0] * n_vars, [5.0] * n_vars])),
        (rosenbrock_function, np.array([[-5.0] * n_vars, [5.0] * n_vars])),
        (rastrigin_function, np.array([[-5.12] * n_vars, [5.12] * n_vars])),
        (ackley_function, np.array([[-5.0] * n_vars, [5.0] * n_vars]))
    ]

    results = {}

    for func, bounds in benchmarks:
        result = run_dendiff_eda(
            func,
            n_vars,
            bounds,
            pop_size=pop_size,
            n_generations=n_generations,
            use_fast_sampling=True  # Use fast sampling for efficiency
        )
        results[func.__name__] = result

    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    for func_name, result in results.items():
        print(f"\n{func_name}:")
        print(f"  Initial: {result['best_fitness_history'][0]:.6e}")
        print(f"  Final:   {result['final_best_fitness']:.6e}")
        print(f"  Improvement: {result['best_fitness_history'][0] / max(result['final_best_fitness'], 1e-10):.2f}x")
    print("="*80)


def test_scalability():
    """
    Test scalability to higher dimensions.
    """
    print("\n" + "#"*80)
    print("# TEST: SCALABILITY TO HIGHER DIMENSIONS")
    print("#"*80)

    dimensions = [5, 10, 20, 30]
    pop_size = 300
    n_generations = 15

    results = {}

    for n_vars in dimensions:
        print(f"\n{'='*80}")
        print(f"Testing with {n_vars} dimensions")
        print(f"{'='*80}")

        bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

        result = run_dendiff_eda(
            sphere_function,
            n_vars,
            bounds,
            pop_size=pop_size,
            n_generations=n_generations,
            use_fast_sampling=True
        )

        results[n_vars] = result

    # Print scalability summary
    print("\n" + "="*80)
    print("SCALABILITY SUMMARY")
    print("="*80)
    for n_vars, result in results.items():
        avg_time = np.mean(result['learning_times']) + np.mean(result['sampling_times'])
        print(f"{n_vars:3d} dims: Final={result['final_best_fitness']:.6e}, "
              f"Avg time/gen={avg_time:.3f}s")
    print("="*80)


def main():
    """
    Run all DenDiff-EDA tests and demonstrations.
    """
    print("\n" + "#"*80)
    print("# DENOISING DIFFUSION EDA (DenDiff-EDA) - COMPREHENSIVE TESTS")
    print("#"*80)

    # Test 1: Large population learning
    test_large_population_learning()

    # Test 2: Compare sampling methods
    compare_sampling_methods()

    # Test 3: Benchmark functions
    test_benchmark_functions()

    # Test 4: Scalability
    test_scalability()

    print("\n" + "#"*80)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
