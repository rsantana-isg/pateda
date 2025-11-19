"""
Example demonstrating the use of GAN-based EDAs for continuous optimization.

This example shows how to use GANs (Generative Adversarial Networks) as probabilistic
models in Estimation of Distribution Algorithms for continuous optimization problems.

Implementation based on:
"Generative Adversarial Networks in Estimation of Distribution Algorithms for
Combinatorial Optimization" (Probst, 2016)
"""

import numpy as np
from pateda.learning.gan import learn_gan
from pateda.sampling.gan import sample_gan


def sphere_function(x):
    """Simple sphere function for testing (minimization)"""
    return np.sum(x**2, axis=1)


def rosenbrock_function(x):
    """Rosenbrock function (minimization)"""
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin_function(x):
    """Rastrigin function (minimization)"""
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def run_gan_eda(fitness_function, n_vars, bounds, max_generations=50):
    """
    Run a GAN-based EDA on a given fitness function.

    Parameters
    ----------
    fitness_function : callable
        Function to optimize (minimization)
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Array of shape (2, n_vars) with [min, max] bounds
    max_generations : int
        Maximum number of generations

    Returns
    -------
    best_fitness_history : list
        History of best fitness values
    """
    print(f"\n{'='*60}")
    print(f"Running GAN-EDA on {fitness_function.__name__}")
    print(f"Problem dimension: {n_vars}")
    print(f"{'='*60}\n")

    # Algorithm parameters
    pop_size = 100
    selection_ratio = 0.3
    selection_size = int(pop_size * selection_ratio)

    # GAN training parameters
    gan_params = {
        'latent_dim': max(2, n_vars // 2),
        'hidden_dims_g': [32, 64],  # Generator hidden layers
        'hidden_dims_d': [64, 32],  # Discriminator hidden layers
        'epochs': 100,
        'batch_size': min(16, selection_size // 2),
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'k_discriminator': 1  # Number of discriminator updates per generator update
    }

    # Initialize population randomly
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    # Track best fitness
    best_fitness_history = []
    best_solution = None
    best_fitness = float('inf')

    # EDA loop
    for gen in range(max_generations):
        # Evaluate fitness
        fitness = fitness_function(population)

        # Track overall best
        gen_best_idx = np.argmin(fitness)
        gen_best_fitness = fitness[gen_best_idx]

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx].copy()

        best_fitness_history.append(best_fitness)

        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Check convergence
        if best_fitness < 1e-6:
            print(f"Converged at generation {gen+1}")
            break

        # Select best individuals
        idx = np.argsort(fitness)[:selection_size]
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn GAN model from selected population
        model = learn_gan(selected_pop, selected_fit, params=gan_params)

        # Sample new population from GAN
        new_population = sample_gan(model, n_samples=pop_size, bounds=bounds)

        # Replace population
        population = new_population

    print(f"\nFinal best fitness: {best_fitness:.6f}")
    print(f"Best solution: {best_solution}")
    print(f"Improvement: {best_fitness_history[0]:.6f} -> {best_fitness:.6f}")
    if best_fitness_history[0] > 0:
        print(f"Reduction factor: {best_fitness_history[0] / max(best_fitness, 1e-10):.2f}x\n")

    return best_fitness_history


def compare_functions():
    """Compare GAN-EDA on different benchmark functions"""

    print("\n" + "="*60)
    print("GAN-EDA FOR CONTINUOUS OPTIMIZATION")
    print("="*60)

    results = {}

    # Test on sphere function (easy)
    print("\n--- Testing on Sphere Function ---")
    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    history = run_gan_eda(sphere_function, n_vars, bounds, max_generations=30)
    results['sphere'] = history

    # Test on Rosenbrock function (harder)
    print("\n--- Testing on Rosenbrock Function ---")
    n_vars = 5  # Use smaller dimension for Rosenbrock
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    history = run_gan_eda(rosenbrock_function, n_vars, bounds, max_generations=50)
    results['rosenbrock'] = history

    # Test on Rastrigin function (multimodal)
    print("\n--- Testing on Rastrigin Function ---")
    n_vars = 5
    bounds = np.array([[-5.12] * n_vars, [5.12] * n_vars])
    history = run_gan_eda(rastrigin_function, n_vars, bounds, max_generations=50)
    results['rastrigin'] = history

    # Print comparison
    print("\n" + "="*60)
    print("SUMMARY: Final Best Fitness Values")
    print("="*60)
    for func_name, history in results.items():
        print(f"{func_name:20s}: {history[-1]:.6f}")

    print("\n" + "="*60)
    print("GAN-EDA demonstration completed!")
    print("="*60)

    return results


def demonstrate_hyperparameter_sensitivity():
    """
    Demonstrate the sensitivity of GAN-EDA to hyperparameters.

    This is important because the paper notes that GANs are difficult to tune.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*60 + "\n")

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    pop_size = 100
    selection_size = 30
    max_generations = 20

    # Test different learning rates
    learning_rates = [0.0001, 0.0002, 0.0005, 0.001]

    print("Testing different learning rates on Sphere function:")
    print("-" * 60)

    for lr in learning_rates:
        np.random.seed(42)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

        gan_params = {
            'latent_dim': 5,
            'epochs': 80,
            'batch_size': 16,
            'learning_rate': lr,
        }

        best_fitness = float('inf')

        for gen in range(max_generations):
            fitness = sphere_function(population)
            gen_best = np.min(fitness)
            if gen_best < best_fitness:
                best_fitness = gen_best

            idx = np.argsort(fitness)[:selection_size]
            selected_pop = population[idx]
            selected_fit = fitness[idx]

            model = learn_gan(selected_pop, selected_fit, params=gan_params)
            population = sample_gan(model, n_samples=pop_size, bounds=bounds)

        print(f"Learning rate {lr:.4f}: Final best = {best_fitness:.6f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# GAN-EDA EXAMPLES FOR CONTINUOUS OPTIMIZATION")
    print("#"*60)

    # Run main comparison
    results = compare_functions()

    # Demonstrate hyperparameter sensitivity
    demonstrate_hyperparameter_sensitivity()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("#"*60 + "\n")

    print("\nNote: As mentioned in Probst (2016), GAN-EDAs may not achieve")
    print("competitive performance compared to other EDAs due to:")
    print("  - Noisy training data in early generations")
    print("  - Difficulty in tuning hyperparameters")
    print("  - Complex adversarial training dynamics")
    print("\nThis implementation serves as a research baseline for exploring")
    print("improvements to GAN-based EDAs.\n")
