"""
Example demonstrating the use of Backdrive-EDA for continuous optimization.

Backdrive-EDA learns an MLP that predicts fitness from solutions, then uses
network inversion (backdrive) to generate high-fitness solutions by propagating
desired fitness values backward through the network.

This implementation follows the approach described in:
- Garciarena et al. (2020). "Envisioning the Benefits of Back-Drive in Evolutionary Algorithms"
- Baluja (2017). "Deep Learning for Explicitly Modeling Optimization Landscapes"
"""

import numpy as np
from pateda.learning.backdrive import learn_backdrive
from pateda.sampling.backdrive import sample_backdrive, sample_backdrive_adaptive


def sphere_function(x):
    """Simple sphere function for testing"""
    return -np.sum(x**2)  # Negative for maximization


def rosenbrock_function(x):
    """Rosenbrock function (minimization)"""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)  # Negative for maximization


def rastrigin_function(x):
    """Rastrigin function (minimization)"""
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))  # Negative for maximization


def run_backdrive_eda(fitness_function, n_vars, bounds, variant='standard'):
    """
    Run a Backdrive-EDA on a given fitness function.

    Parameters
    ----------
    fitness_function : callable
        Function to optimize (should return scalar for maximization)
    n_vars : int
        Number of variables
    bounds : np.ndarray
        Array of shape (2, n_vars) with [min, max] bounds
    variant : str
        Backdrive variant to use: 'standard' or 'adaptive'
    """
    print(f"\n{'='*60}")
    print(f"Running Backdrive-EDA ({variant.upper()}) on {fitness_function.__name__}")
    print(f"{'='*60}\n")

    # Algorithm parameters
    pop_size = 100
    n_generations = 30
    selection_ratio = 0.3
    selection_size = int(pop_size * selection_ratio)

    # Backdrive learning parameters
    learning_params = {
        'hidden_layers': [100, 100],
        'activation': 'tanh',
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 10,
    }

    # Backdrive sampling parameters
    if variant == 'standard':
        sampling_params = {
            'n_samples': pop_size,
            'target_fitness': 100,  # Target best fitness
            'backdrive_iterations': 500,
            'backdrive_lr': 0.01,
            'init_method': 'perturb_best',
            'init_noise': 0.1,
            'use_surrogate': False,
        }
    elif variant == 'adaptive':
        sampling_params = {
            'n_samples': pop_size,
            'target_levels': [100, 90, 80],
            'level_fractions': [0.5, 0.3, 0.2],
            'backdrive_iterations': 500,
            'backdrive_lr': 0.01,
            'init_method': 'perturb_best',
            'init_noise': 0.1,
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Initialize population
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    # Track best fitness
    best_fitness_history = []
    model = None

    # EDA loop
    for gen in range(n_generations):
        # Evaluate
        fitness = np.array([fitness_function(ind) for ind in population])

        # Track best
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_fitness_history.append(best_fitness)

        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select best individuals
        idx = np.argsort(fitness)[-selection_size:]
        selected_pop = population[idx]
        selected_fit = fitness[idx].reshape(-1, 1)

        # Learn model
        learning_params['previous_model'] = model  # For weight transfer
        model = learn_backdrive(
            generation=gen,
            n_vars=n_vars,
            cardinality=bounds,
            selected_population=selected_pop,
            selected_fitness=selected_fit,
            params=learning_params
        )

        print(f"  Model correlation: {model.metadata['correlation']:.4f}")

        # Sample new solutions based on variant
        if variant == 'standard':
            new_population = sample_backdrive(
                n_vars=n_vars,
                model=model,
                cardinality=bounds,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params=sampling_params
            )
        elif variant == 'adaptive':
            new_population = sample_backdrive_adaptive(
                n_vars=n_vars,
                model=model,
                cardinality=bounds,
                current_population=population,
                current_fitness=fitness.reshape(-1, 1),
                params=sampling_params
            )

        # Replace population
        population = new_population

    print(f"\nFinal best fitness: {best_fitness_history[-1]:.6f}")
    print(f"Improvement: {best_fitness_history[0]:.6f} -> {best_fitness_history[-1]:.6f}")
    improvement_factor = (best_fitness_history[-1] - best_fitness_history[0]) / max(abs(best_fitness_history[0]), 1e-10)
    print(f"Relative improvement: {improvement_factor:.2f}x\n")

    return best_fitness_history


def compare_backdrive_variants():
    """Compare standard and adaptive backdrive sampling"""

    print("\n" + "="*60)
    print("COMPARING BACKDRIVE-EDA VARIANTS")
    print("="*60)

    # Test on sphere function
    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    results = {}

    for variant in ['standard', 'adaptive']:
        history = run_backdrive_eda(sphere_function, n_vars, bounds, variant=variant)
        results[variant] = history

    # Print comparison
    print("\n" + "="*60)
    print("SUMMARY: Final Best Fitness Values")
    print("="*60)
    for variant, history in results.items():
        print(f"{variant:20s}: {history[-1]:.6f}")

    print("\n" + "="*60)
    print("All variants successfully demonstrated!")
    print("="*60)


def example_with_surrogate_filtering():
    """Demonstrate surrogate filtering during backdrive sampling"""

    print("\n" + "="*60)
    print("BACKDRIVE-EDA WITH SURROGATE FILTERING")
    print("="*60 + "\n")

    n_vars = 10
    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])
    pop_size = 100
    selection_size = 30

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    model = None

    for gen in range(20):
        fitness = np.array([sphere_function(ind) for ind in population])
        best_fitness = np.max(fitness)
        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select
        idx = np.argsort(fitness)[-selection_size:]

        # Learn Backdrive model
        model = learn_backdrive(
            generation=gen,
            n_vars=n_vars,
            cardinality=bounds,
            selected_population=population[idx],
            selected_fitness=fitness[idx].reshape(-1, 1),
            params={'epochs': 40, 'batch_size': 16, 'previous_model': model}
        )

        # Sample with surrogate filtering (oversample and select best predicted)
        population = sample_backdrive(
            n_vars=n_vars,
            model=model,
            cardinality=bounds,
            current_population=population,
            current_fitness=fitness.reshape(-1, 1),
            params={
                'use_surrogate': True,
                'oversample_factor': 5,  # Generate 5x more and filter
                'backdrive_iterations': 500,
            }
        )

    final_fitness = np.array([sphere_function(ind) for ind in population])
    print(f"\nFinal best fitness: {np.max(final_fitness):.6f}")
    print("Surrogate filtering successfully used!\n")


def example_on_difficult_function():
    """Demonstrate Backdrive-EDA on Rastrigin (multimodal) function"""

    print("\n" + "="*60)
    print("BACKDRIVE-EDA ON RASTRIGIN (MULTIMODAL)")
    print("="*60 + "\n")

    n_vars = 10
    bounds = np.array([[-5.12] * n_vars, [5.12] * n_vars])
    pop_size = 150
    selection_size = 45

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_vars))

    model = None

    for gen in range(30):
        fitness = np.array([rastrigin_function(ind) for ind in population])
        best_fitness = np.max(fitness)
        print(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.6f}")

        # Select
        idx = np.argsort(fitness)[-selection_size:]

        # Learn model with deeper network for complex landscape
        model = learn_backdrive(
            generation=gen,
            n_vars=n_vars,
            cardinality=bounds,
            selected_population=population[idx],
            selected_fitness=fitness[idx].reshape(-1, 1),
            params={
                'hidden_layers': [150, 150, 100],  # Deeper network
                'activation': 'tanh',
                'epochs': 60,
                'batch_size': 16,
                'previous_model': model,
                'transfer_weights': True,
            }
        )

        # Adaptive sampling to maintain diversity
        population = sample_backdrive_adaptive(
            n_vars=n_vars,
            model=model,
            cardinality=bounds,
            current_population=population,
            current_fitness=fitness.reshape(-1, 1),
            params={
                'target_levels': [100, 85, 70, 50],
                'level_fractions': [0.4, 0.3, 0.2, 0.1],
                'backdrive_iterations': 600,
                'init_method': 'perturb_selected',
                'init_noise': 0.15,
            }
        )

    final_fitness = np.array([rastrigin_function(ind) for ind in population])
    print(f"\nFinal best fitness: {np.max(final_fitness):.6f}")
    print("Successfully handled multimodal landscape!\n")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# BACKDRIVE-EDA EXAMPLES FOR CONTINUOUS OPTIMIZATION")
    print("#"*60)

    # Run all examples
    compare_backdrive_variants()
    example_with_surrogate_filtering()
    example_on_difficult_function()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("#"*60 + "\n")
