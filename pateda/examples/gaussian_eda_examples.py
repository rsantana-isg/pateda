"""
Examples demonstrating Gaussian-based EDAs for continuous optimization.

This script shows:
1. Gaussian UMDA (univariate)
2. Full Gaussian EDA (multivariate with covariance)
3. Mixture of Gaussians EDA
"""

import numpy as np
import matplotlib.pyplot as plt
from pateda.learning.gaussian import (
    learn_gaussian_univariate,
    learn_gaussian_full,
    learn_mixture_gaussian_univariate,
)
from pateda.sampling.gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_mixture_gaussian_univariate,
)


# Benchmark functions
def sphere(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2, axis=1)


def rosenbrock(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin(x):
    """Rastrigin function (multimodal)"""
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley(x):
    """Ackley function (multimodal)"""
    a, b, c = 20, 0.2, 2 * np.pi
    d = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(np.cos(c * x), axis=1)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e


def run_gaussian_umda(fitness_func, n_vars, bounds, pop_size=100, n_generations=30):
    """
    Run Gaussian UMDA on a given fitness function.

    Parameters
    ----------
    fitness_func : callable
        Function to minimize
    n_vars : int
        Number of variables
    bounds : tuple
        (lower, upper) bounds for variables
    pop_size : int
        Population size
    n_generations : int
        Number of generations
    """
    print(f"\n{'='*60}")
    print(f"Gaussian UMDA on {fitness_func.__name__}")
    print(f"Variables: {n_vars}, Population: {pop_size}")
    print(f"{'='*60}\n")

    # Initialize
    np.random.seed(42)
    lower, upper = bounds
    population = np.random.uniform(lower, upper, (pop_size, n_vars))

    best_fitness_history = []
    mean_fitness_history = []

    # EDA loop
    for gen in range(n_generations):
        # Evaluate
        fitness = fitness_func(population)

        # Track progress
        best_fitness = np.min(fitness)
        mean_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

        print(f"Gen {gen+1:3d}: Best = {best_fitness:12.6f}, Mean = {mean_fitness:12.6f}")

        # Select best 30%
        selection_size = int(pop_size * 0.3)
        idx = np.argsort(fitness)[:selection_size]
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn Gaussian UMDA model
        model = learn_gaussian_univariate(selected_pop, selected_fit)

        # Sample new population
        bounds_array = np.array([[lower] * n_vars, [upper] * n_vars])
        population = sample_gaussian_univariate(model, n_samples=pop_size, bounds=bounds_array)

    print(f"\nFinal Best: {best_fitness_history[-1]:.6f}")
    print(f"Improvement: {best_fitness_history[0]/max(best_fitness_history[-1], 1e-10):.2f}x\n")

    return best_fitness_history, mean_fitness_history


def run_gaussian_full_eda(fitness_func, n_vars, bounds, pop_size=100, n_generations=30):
    """
    Run Full Gaussian EDA (with covariance) on a given fitness function.

    This version models variable dependencies through a full covariance matrix.
    """
    print(f"\n{'='*60}")
    print(f"Full Gaussian EDA on {fitness_func.__name__}")
    print(f"Variables: {n_vars}, Population: {pop_size}")
    print(f"{'='*60}\n")

    # Initialize
    np.random.seed(42)
    lower, upper = bounds
    population = np.random.uniform(lower, upper, (pop_size, n_vars))

    best_fitness_history = []

    # EDA loop
    for gen in range(n_generations):
        fitness = fitness_func(population)
        best_fitness = np.min(fitness)
        best_fitness_history.append(best_fitness)

        print(f"Gen {gen+1:3d}: Best = {best_fitness:12.6f}")

        # Select best 30%
        selection_size = int(pop_size * 0.3)
        idx = np.argsort(fitness)[:selection_size]

        # Learn full covariance Gaussian model
        model = learn_gaussian_full(population[idx], fitness[idx])

        # Sample with variance scaling (for exploration)
        bounds_array = np.array([[lower] * n_vars, [upper] * n_vars])
        population = sample_gaussian_full(
            model, n_samples=pop_size, bounds=bounds_array,
            params={'var_scaling': 1.0}
        )

    print(f"\nFinal Best: {best_fitness_history[-1]:.6f}\n")
    return best_fitness_history


def run_mixture_gaussian_eda(fitness_func, n_vars, bounds, pop_size=150, n_generations=30):
    """
    Run Mixture of Gaussians EDA for multimodal optimization.

    The mixture model can maintain multiple search regions simultaneously,
    which is beneficial for multimodal functions.
    """
    print(f"\n{'='*60}")
    print(f"Mixture Gaussian EDA on {fitness_func.__name__}")
    print(f"Variables: {n_vars}, Population: {pop_size}")
    print(f"{'='*60}\n")

    # Initialize
    np.random.seed(42)
    lower, upper = bounds
    population = np.random.uniform(lower, upper, (pop_size, n_vars))

    best_fitness_history = []

    # EDA loop
    for gen in range(n_generations):
        fitness = fitness_func(population)
        best_fitness = np.min(fitness)
        best_fitness_history.append(best_fitness)

        print(f"Gen {gen+1:3d}: Best = {best_fitness:12.6f}")

        # Select best 40%
        selection_size = int(pop_size * 0.4)
        idx = np.argsort(fitness)[:selection_size]

        # Learn mixture model
        params = {
            'n_clusters': 3,  # Use 3 mixture components
            'what_to_cluster': 'vars',
            'normalize': True
        }
        model = learn_mixture_gaussian_univariate(
            population[idx], fitness[idx], params
        )

        # Sample from mixture
        bounds_array = np.array([[lower] * n_vars, [upper] * n_vars])
        population = sample_mixture_gaussian_univariate(
            model, n_samples=pop_size, bounds=bounds_array
        )

    print(f"\nFinal Best: {best_fitness_history[-1]:.6f}\n")
    return best_fitness_history


def compare_gaussian_edas():
    """Compare different Gaussian EDA variants"""
    print("\n" + "="*60)
    print("COMPARING GAUSSIAN EDA VARIANTS")
    print("="*60)

    # Test on Rosenbrock (has correlation between variables)
    n_vars = 10
    bounds = (-2.0, 2.0)

    print("\n--- Test 1: Rosenbrock Function (Correlated) ---")
    history_umda = run_gaussian_umda(rosenbrock, n_vars, bounds, n_generations=25)
    history_full = run_gaussian_full_eda(rosenbrock, n_vars, bounds, n_generations=25)

    print("\nResults on Rosenbrock:")
    print(f"  UMDA:         {history_umda[0][-1]:.6f}")
    print(f"  Full Gaussian: {history_full[-1]:.6f}")
    print("  -> Full Gaussian should be better (captures correlations)")

    # Test on Rastrigin (multimodal)
    print("\n--- Test 2: Rastrigin Function (Multimodal) ---")
    bounds_rastrigin = (-5.12, 5.12)
    history_mixture = run_mixture_gaussian_eda(
        rastrigin, n_vars, bounds_rastrigin, pop_size=150, n_generations=25
    )

    print(f"\nMixture Gaussian on Rastrigin: {history_mixture[-1]:.6f}")
    print("  -> Mixture model helps with multimodality")


def example_sphere_optimization():
    """Detailed example: Optimizing Sphere function"""
    print("\n" + "="*60)
    print("DETAILED EXAMPLE: Sphere Function Optimization")
    print("="*60)

    n_vars = 20
    pop_size = 100
    n_generations = 30
    bounds = (-5.0, 5.0)

    print(f"\nProblem: Minimize f(x) = sum(x_i^2)")
    print(f"Global optimum: f(0, 0, ..., 0) = 0")
    print(f"Search space: [{bounds[0]}, {bounds[1]}]^{n_vars}")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_generations}\n")

    # Run UMDA
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    for gen in range(n_generations):
        fitness = sphere(population)

        if gen % 5 == 0 or gen == n_generations - 1:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
            print(f"Generation {gen:3d}:")
            print(f"  Best fitness: {best_fitness:.6f}")
            print(f"  Best solution (first 5): {best_solution[:5]}")

        # Select and learn
        idx = np.argsort(fitness)[:30]
        model = learn_gaussian_univariate(population[idx], fitness[idx])

        # Sample
        bounds_array = np.array([[bounds[0]] * n_vars, [bounds[1]] * n_vars])
        population = sample_gaussian_univariate(model, n_samples=pop_size, bounds=bounds_array)

    print("\n" + "="*60)


def visualize_convergence():
    """Visualize EDA convergence (requires matplotlib)"""
    print("\n" + "="*60)
    print("CONVERGENCE VISUALIZATION")
    print("="*60 + "\n")

    n_vars = 10
    bounds = (-5.0, 5.0)

    # Run on Sphere
    history_sphere = run_gaussian_umda(sphere, n_vars, bounds, n_generations=30)

    # Run on Rosenbrock
    bounds_rosen = (-2.0, 2.0)
    history_rosen = run_gaussian_umda(rosenbrock, n_vars, bounds_rosen, n_generations=30)

    try:
        plt.figure(figsize=(12, 5))

        # Plot Sphere convergence
        plt.subplot(1, 2, 1)
        plt.semilogy(history_sphere[0], 'b-', label='Best')
        plt.semilogy(history_sphere[1], 'r--', label='Mean')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (log scale)')
        plt.title('Convergence on Sphere Function')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Rosenbrock convergence
        plt.subplot(1, 2, 2)
        plt.semilogy(history_rosen[0], 'b-', label='Best')
        plt.semilogy(history_rosen[1], 'r--', label='Mean')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (log scale)')
        plt.title('Convergence on Rosenbrock Function')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gaussian_eda_convergence.png', dpi=150)
        print("Convergence plot saved to 'gaussian_eda_convergence.png'")
        plt.close()

    except ImportError:
        print("Matplotlib not available, skipping visualization")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# GAUSSIAN EDA EXAMPLES")
    print("#"*60)

    # Run examples
    example_sphere_optimization()
    compare_gaussian_edas()

    # Optional visualization
    try:
        visualize_convergence()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*60 + "\n")
