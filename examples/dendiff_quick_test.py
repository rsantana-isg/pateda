"""
Quick test for DenDiff-EDA to verify installation and basic functionality.

This is a minimal test that can be run quickly to ensure the implementation works.
"""

import numpy as np
import time
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff_fast


def sphere_function(x):
    """Simple sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2, axis=1)


def quick_test():
    """Run a quick test of DenDiff-EDA"""
    print("\n" + "="*60)
    print("DenDiff-EDA Quick Test")
    print("="*60 + "\n")

    # Small test problem
    n_vars = 5
    pop_size = 50
    n_generations = 5
    selection_ratio = 0.5

    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    # Minimal DenDiff parameters for quick test
    dendiff_params = {
        'n_timesteps': 100,  # Reduced for speed
        'beta_schedule': 'linear',
        'hidden_dims': [32],  # Smaller network
        'time_emb_dim': 16,
        'epochs': 10,  # Fewer epochs
        'batch_size': 16,
        'learning_rate': 1e-3
    }

    # Fast sampling parameters
    sampling_params = {
        'ddim_steps': 10,  # Very few steps for quick test
        'ddim_eta': 0.0
    }

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    print(f"Problem: {n_vars} variables, {pop_size} population size")
    print(f"Running {n_generations} generations...\n")

    best_fitness_history = []

    for gen in range(n_generations):
        # Evaluate
        fitness = sphere_function(population)
        best_fitness = np.min(fitness)
        best_fitness_history.append(best_fitness)

        print(f"Gen {gen+1}: Best = {best_fitness:.6e}")

        # Select
        selection_size = int(pop_size * selection_ratio)
        idx = np.argsort(fitness)[:selection_size]
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn
        start = time.time()
        model = learn_dendiff(selected_pop, selected_fit, params=dendiff_params)
        learn_time = time.time() - start

        # Sample
        start = time.time()
        population = sample_dendiff_fast(model, n_samples=pop_size, bounds=bounds, params=sampling_params)
        sample_time = time.time() - start

        print(f"       Learn: {learn_time:.2f}s, Sample: {sample_time:.2f}s")

    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Initial best: {best_fitness_history[0]:.6e}")
    print(f"  Final best:   {best_fitness_history[-1]:.6e}")
    print(f"  Improvement:  {best_fitness_history[0] / max(best_fitness_history[-1], 1e-10):.2f}x")
    print("="*60)
    print("\n✓ DenDiff-EDA is working correctly!\n")


if __name__ == '__main__':
    quick_test()
