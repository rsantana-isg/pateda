"""
Example demonstrating the use of VAE-based EDAs for continuous optimization.

This example shows how to use the three VAE variants:
1. VAE: Basic variational autoencoder
2. E-VAE: Extended VAE with fitness predictor
3. CE-VAE: Conditional Extended VAE for fitness-directed search
"""

import numpy as np
from pateda.learning.vae import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
)
from pateda.sampling.vae import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
)


def sphere_function(x):
    """Simple sphere function for testing"""
    return np.sum(x**2, axis=1)


def rosenbrock_function(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def run_vae_eda(fitness_function, n_vars, bounds, variant='vae'):
    """
    Run a VAE-based EDA on a given fitness function.

    Parameters
    ----------
    fitness_function : callable
        Function to optimize
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Array of shape (2, n_vars) with [min, max] bounds
    variant : str
        VAE variant to use: 'vae', 'extended_vae', or 'conditional_extended_vae'
    """
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()} on {fitness_function.__name__}")
    print(f"{'='*60}\n")

    # Algorithm parameters
    pop_size = 100
    n_generations = 30
    selection_ratio = 0.3
    selection_size = int(pop_size * selection_ratio)

    # VAE training parameters
    vae_params = {
        'latent_dim': max(2, n_vars // 2),
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001
    }

    # Initialize population
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    # Track best fitness
    best_fitness_history = []

    # EDA loop
    for gen in range(n_generations):
        # Evaluate
        fitness = fitness_function(population)

        # Track best
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_fitness_history.append(best_fitness)

        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select best individuals
        idx = np.argsort(fitness)[:selection_size]
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn model based on variant
        if variant == 'vae':
            model = learn_vae(selected_pop, selected_fit, params=vae_params)
            new_population = sample_vae(model, n_samples=pop_size, bounds=bounds)

        elif variant == 'extended_vae':
            model = learn_extended_vae(selected_pop, selected_fit, params=vae_params)
            # Can optionally use predictor for filtering
            new_population = sample_extended_vae(
                model, n_samples=pop_size, bounds=bounds,
                params={'use_predictor': False}  # Set to True for surrogate filtering
            )

        elif variant == 'conditional_extended_vae':
            model = learn_conditional_extended_vae(selected_pop, selected_fit, params=vae_params)
            # Target the best fitness found so far
            target_fitness = np.min(selected_fit)
            new_population = sample_conditional_extended_vae(
                model, n_samples=pop_size, bounds=bounds,
                params={'target_fitness': target_fitness, 'fitness_noise': 0.1}
            )

        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Replace population
        population = new_population

    print(f"\nFinal best fitness: {best_fitness_history[-1]:.6f}")
    print(f"Improvement: {best_fitness_history[0]:.6f} -> {best_fitness_history[-1]:.6f}")
    print(f"Reduction factor: {best_fitness_history[0] / max(best_fitness_history[-1], 1e-10):.2f}x\n")

    return best_fitness_history


def compare_vae_variants():
    """Compare the three VAE variants on benchmark functions"""

    print("\n" + "="*60)
    print("COMPARING VAE VARIANTS FOR CONTINUOUS OPTIMIZATION")
    print("="*60)

    # Test on sphere function
    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    results = {}

    for variant in ['vae', 'extended_vae', 'conditional_extended_vae']:
        history = run_vae_eda(sphere_function, n_vars, bounds, variant=variant)
        results[variant] = history

    # Print comparison
    print("\n" + "="*60)
    print("SUMMARY: Final Best Fitness Values")
    print("="*60)
    for variant, history in results.items():
        print(f"{variant:30s}: {history[-1]:.6f}")

    print("\n" + "="*60)
    print("All variants successfully demonstrated!")
    print("="*60)


def example_with_fitness_predictor():
    """Demonstrate E-VAE with fitness predictor for surrogate filtering"""

    print("\n" + "="*60)
    print("EXTENDED VAE WITH FITNESS PREDICTOR (SURROGATE FILTERING)")
    print("="*60 + "\n")

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    pop_size = 100
    selection_size = 30

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    for gen in range(20):
        fitness = sphere_function(population)
        best_fitness = np.min(fitness)
        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select
        idx = np.argsort(fitness)[:selection_size]

        # Learn E-VAE
        model = learn_extended_vae(
            population[idx], fitness[idx],
            params={'latent_dim': 5, 'epochs': 25, 'batch_size': 16}
        )

        # Sample with predictor filtering (oversample and select best predicted)
        population = sample_extended_vae(
            model, n_samples=pop_size, bounds=bounds,
            params={
                'use_predictor': True,
                'predictor_percentile': 40  # Keep best 40% of predicted
            }
        )

    print(f"\nFinal best fitness: {np.min(sphere_function(population)):.6f}")
    print("Predictor successfully used for surrogate-based filtering!\n")


def example_fitness_conditioned_sampling():
    """Demonstrate CE-VAE for fitness-conditioned sampling"""

    print("\n" + "="*60)
    print("CONDITIONAL VAE FOR FITNESS-DIRECTED SAMPLING")
    print("="*60 + "\n")

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    pop_size = 100
    selection_size = 30

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    for gen in range(20):
        fitness = sphere_function(population)
        best_fitness = np.min(fitness)
        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select
        idx = np.argsort(fitness)[:selection_size]

        # Learn CE-VAE
        model = learn_conditional_extended_vae(
            population[idx], fitness[idx],
            params={'latent_dim': 5, 'epochs': 25, 'batch_size': 16}
        )

        # Sample conditioned on target fitness
        # We can target different fitness levels to explore different regions
        target_fitness = best_fitness * 0.8  # Target even better fitness
        population = sample_conditional_extended_vae(
            model, n_samples=pop_size, bounds=bounds,
            params={
                'target_fitness': target_fitness,
                'fitness_noise': 0.15  # Add noise for exploration
            }
        )

    print(f"\nFinal best fitness: {np.min(sphere_function(population)):.6f}")
    print("Successfully used fitness conditioning to direct search!\n")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# VAE-EDA EXAMPLES FOR CONTINUOUS OPTIMIZATION")
    print("#"*60)

    # Run all examples
    compare_vae_variants()
    example_with_fitness_predictor()
    example_fitness_conditioned_sampling()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("#"*60 + "\n")
