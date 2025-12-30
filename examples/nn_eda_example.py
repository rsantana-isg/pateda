"""
Example: Neural Network-based EDA with Fitness-Weighted Autoencoder

This example demonstrates the unique NN-EDA approach where:
1. The autoencoder learns to transform [x, f(x)] → [x', f'(x')]
2. Solutions are evaluated during training (acts as local optimizer)
3. Best training solutions can be used directly
4. Additional solutions can be sampled from the learned model

Key Difference from Other EDAs:
- Most evaluations happen during learning, not sampling
- Training process itself optimizes the population
- Sampling is optional but recommended for diversity
"""

import numpy as np
import sys
import os

# Add pateda to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pateda.learning.nn_eda import learn_nn_eda
from pateda.sampling.nn_eda import (
    sample_nn_eda,
    get_best_training_solutions,
    sample_nn_eda_hybrid,
    get_training_statistics
)


def sphere_function(x):
    """Simple sphere function: f(x) = sum(x^2)"""
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def rosenbrock_function(x):
    """Rosenbrock function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    result = np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)
    return result if len(result) > 1 else result[0]


def run_nn_eda_example():
    """Run complete NN-EDA example on a simple optimization problem."""

    print("=" * 80)
    print("NN-EDA EXAMPLE: Fitness-Weighted Autoencoder")
    print("=" * 80)
    print()

    # Problem setup
    n_vars = 10
    pop_size = 100
    objective_function = sphere_function

    print(f"Problem: Sphere function ({n_vars} variables)")
    print(f"Population size: {pop_size}")
    print()

    # Initialize random population
    np.random.seed(42)
    initial_population = np.random.uniform(-5, 5, size=(pop_size, n_vars))
    initial_fitness = objective_function(initial_population)

    print(f"Initial Population:")
    print(f"  Best fitness: {initial_fitness.min():.6f}")
    print(f"  Mean fitness: {initial_fitness.mean():.6f}")
    print(f"  Worst fitness: {initial_fitness.max():.6f}")
    print()

    # ========================================================================
    # PHASE 1: Learn NN-EDA Model
    # ========================================================================
    print("=" * 80)
    print("PHASE 1: LEARNING (with in-training evaluation)")
    print("=" * 80)
    print()

    learning_params = {
        'hidden_dims': [64, 32],
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'list_act_functs': ['relu', 'relu'],
        'fitness_weight_clip': 5.0,
        'store_all_solutions': True,
        'verbose': True
    }

    model_data = learn_nn_eda(
        population=initial_population,
        fitness=initial_fitness,
        objective_function=objective_function,
        params=learning_params
    )

    # Get statistics about training
    print()
    print("=" * 80)
    print("TRAINING STATISTICS")
    print("=" * 80)
    stats = get_training_statistics(model_data)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # ========================================================================
    # PHASE 2: Get Best Solutions from Training
    # ========================================================================
    print("=" * 80)
    print("PHASE 2: EXTRACTING BEST TRAINING SOLUTIONS")
    print("=" * 80)
    print()

    # Get top 50% solutions from training
    n_from_training = pop_size // 2
    training_solutions, training_fitness = get_best_training_solutions(
        model_data,
        n_solutions=n_from_training,
        return_fitness=True
    )

    print(f"Best {n_from_training} solutions from training:")
    print(f"  Best fitness: {training_fitness.min():.6f}")
    print(f"  Mean fitness: {training_fitness.mean():.6f}")
    print(f"  Worst fitness: {training_fitness.max():.6f}")
    print()

    # ========================================================================
    # PHASE 3: Sample Additional Solutions
    # ========================================================================
    print("=" * 80)
    print("PHASE 3: SAMPLING NEW SOLUTIONS")
    print("=" * 80)
    print()

    # Sample remaining solutions
    n_to_sample = pop_size - n_from_training

    sampling_params = {
        'use_training_solutions': True,  # Use training solutions as source
        'add_noise': True,  # Add noise for diversity
        'noise_std': 0.05,
        'n_iterations': 2  # Apply transformation twice
    }

    new_solutions = sample_nn_eda(
        model_data,
        n_samples=n_to_sample,
        params=sampling_params
    )

    # Evaluate new solutions
    new_fitness = objective_function(new_solutions)

    print(f"Generated {n_to_sample} new solutions:")
    print(f"  Best fitness: {new_fitness.min():.6f}")
    print(f"  Mean fitness: {new_fitness.mean():.6f}")
    print(f"  Worst fitness: {new_fitness.max():.6f}")
    print()

    # ========================================================================
    # PHASE 4: Combine and Compare
    # ========================================================================
    print("=" * 80)
    print("PHASE 4: COMBINED RESULTS")
    print("=" * 80)
    print()

    # Combine training and sampled solutions
    all_solutions = np.vstack([training_solutions, new_solutions])
    all_fitness = np.hstack([training_fitness, new_fitness])

    print(f"Combined population ({len(all_solutions)} solutions):")
    print(f"  Best fitness: {all_fitness.min():.6f}")
    print(f"  Mean fitness: {all_fitness.mean():.6f}")
    print(f"  Worst fitness: {all_fitness.max():.6f}")
    print()

    # Compare to initial
    improvement = initial_fitness.min() - all_fitness.min()
    print(f"Improvement over initial best: {improvement:.6f}")
    print(f"  Initial best: {initial_fitness.min():.6f}")
    print(f"  Final best: {all_fitness.min():.6f}")
    print(f"  Improvement factor: {initial_fitness.min() / all_fitness.min():.2f}x")
    print()

    # ========================================================================
    # PHASE 5: Alternative - Hybrid Sampling
    # ========================================================================
    print("=" * 80)
    print("PHASE 5: HYBRID SAMPLING (Recommended)")
    print("=" * 80)
    print()

    # This is the recommended approach - combines training and sampling automatically
    # Pass initial population to transform it (better than random or training solutions)
    hybrid_solutions = sample_nn_eda_hybrid(
        model_data,
        n_samples=pop_size,
        fraction_from_training=0.7,  # 70% from training, 30% generated
        source_population=initial_population,
        source_fitness=initial_fitness,
        params=sampling_params
    )

    hybrid_fitness = objective_function(hybrid_solutions)

    print(f"Hybrid sampling ({pop_size} solutions):")
    print(f"  Best fitness: {hybrid_fitness.min():.6f}")
    print(f"  Mean fitness: {hybrid_fitness.mean():.6f}")
    print(f"  70% from training, 30% by transforming initial population")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("NN-EDA Key Points:")
    print("1. Training phase evaluates many solutions (acts as local optimizer)")
    print(f"   - {model_data['n_evaluations']} evaluations during training")
    print(f"   - Found best fitness: {stats['best_fitness']:.6f}")
    print()
    print("2. Best training solutions can be used directly")
    print(f"   - No additional evaluations needed for training solutions")
    print()
    print("3. Sampling generates additional diverse solutions")
    print(f"   - {n_to_sample} additional evaluations for sampling")
    print()
    print("4. Total improvement:")
    print(f"   - Initial: {initial_fitness.min():.6f}")
    print(f"   - Final: {all_fitness.min():.6f}")
    print(f"   - Improvement: {improvement:.6f} ({improvement/initial_fitness.min()*100:.1f}%)")
    print()
    print("=" * 80)


def run_eda_iteration_example():
    """Example of using NN-EDA in a full EDA loop."""

    print("\n" * 2)
    print("=" * 80)
    print("NN-EDA IN COMPLETE EDA LOOP")
    print("=" * 80)
    print()

    # Setup
    n_vars = 10
    pop_size = 100
    max_generations = 10
    objective_function = sphere_function

    print(f"Running EDA for {max_generations} generations")
    print(f"Problem: {n_vars}-D Sphere")
    print(f"Population size: {pop_size}")
    print()

    # Initialize
    np.random.seed(42)
    population = np.random.uniform(-5, 5, size=(pop_size, n_vars))
    fitness = objective_function(population)

    best_fitness_history = [fitness.min()]
    mean_fitness_history = [fitness.mean()]

    print(f"Generation 0: Best = {fitness.min():.6f}, Mean = {fitness.mean():.6f}")

    # EDA loop
    for gen in range(max_generations):
        # Learn model
        model_data = learn_nn_eda(
            population=population,
            fitness=fitness,
            objective_function=objective_function,
            params={
                'hidden_dims': [64, 32],
                'epochs': 20,
                'batch_size': 16,
                'verbose': False
            }
        )

        # Sample new population (hybrid approach)
        # Pass current population so autoencoder transforms it
        new_population = sample_nn_eda_hybrid(
            model_data,
            n_samples=pop_size,
            fraction_from_training=0.6,  # 60% from training, 40% new
            source_population=population,
            source_fitness=fitness,
            params={'add_noise': True, 'noise_std': 0.05}
        )

        # Evaluate (but training solutions are already evaluated)
        # We only need to evaluate the newly generated ones
        # For simplicity, we evaluate all here
        new_fitness = objective_function(new_population)

        # Update population
        population = new_population
        fitness = new_fitness

        best_fitness_history.append(fitness.min())
        mean_fitness_history.append(fitness.mean())

        print(f"Generation {gen + 1}: Best = {fitness.min():.6f}, Mean = {fitness.mean():.6f}, " +
              f"Training evals = {model_data['n_evaluations']}")

    print()
    print("Final Results:")
    print(f"  Initial best: {best_fitness_history[0]:.6f}")
    print(f"  Final best: {best_fitness_history[-1]:.6f}")
    print(f"  Improvement: {best_fitness_history[0] - best_fitness_history[-1]:.6f}")
    print(f"  Factor: {best_fitness_history[0] / best_fitness_history[-1]:.2f}x")
    print()


if __name__ == '__main__':
    # Run single iteration example
    run_nn_eda_example()

    # Run complete EDA loop
    run_eda_iteration_example()

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED")
    print("=" * 80)
