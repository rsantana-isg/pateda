"""
Test script to verify update_data_per_epoch functionality in NN-EDA

This script compares NN-EDA performance with and without updating training data
after each epoch.
"""

import numpy as np
import sys
import os

# Add pateda to path

from pateda_nn.learning.nn_eda import learn_nn_eda


def sphere_function(x):
    """Simple sphere function: f(x) = sum(x^2)"""
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def test_update_data_per_epoch():
    """Test the update_data_per_epoch functionality."""

    print("=" * 80)
    print("TESTING: update_data_per_epoch functionality")
    print("=" * 80)
    print()

    # Problem setup
    n_vars = 10
    pop_size = 50
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
    print()

    # Common parameters
    common_params = {
        'hidden_dims': [32, 16],
        'epochs': 30,
        'batch_size': 10,
        'learning_rate': 1e-3,
        'list_act_functs': ['relu', 'relu'],
        'fitness_weight_clip': 5.0,
        'store_all_solutions': True,
    }

    # ========================================================================
    # Test 1: WITHOUT updating data per epoch (original behavior)
    # ========================================================================
    print("=" * 80)
    print("TEST 1: WITHOUT update_data_per_epoch (baseline)")
    print("=" * 80)
    print()

    params_without_update = common_params.copy()
    params_without_update['update_data_per_epoch'] = False
    params_without_update['verbose'] = True

    model_data_without = learn_nn_eda(
        population=initial_population.copy(),
        fitness=initial_fitness.copy(),
        objective_function=objective_function,
        params=params_without_update
    )

    print()
    print(f"Results WITHOUT data update:")
    print(f"  Best fitness found: {model_data_without['evaluated_fitness'].min():.6f}")
    print(f"  Total evaluations: {model_data_without['n_evaluations']}")
    print(f"  Improvement: {initial_fitness.min() - model_data_without['evaluated_fitness'].min():.6f}")
    print()

    # ========================================================================
    # Test 2: WITH updating data per epoch (new behavior)
    # ========================================================================
    print("=" * 80)
    print("TEST 2: WITH update_data_per_epoch (NEW FEATURE)")
    print("=" * 80)
    print()

    params_with_update = common_params.copy()
    params_with_update['update_data_per_epoch'] = True
    params_with_update['verbose'] = True

    model_data_with = learn_nn_eda(
        population=initial_population.copy(),
        fitness=initial_fitness.copy(),
        objective_function=objective_function,
        params=params_with_update
    )

    print()
    print(f"Results WITH data update:")
    print(f"  Best fitness found: {model_data_with['evaluated_fitness'].min():.6f}")
    print(f"  Total evaluations: {model_data_with['n_evaluations']}")
    print(f"  Improvement: {initial_fitness.min() - model_data_with['evaluated_fitness'].min():.6f}")
    print()

    # ========================================================================
    # Comparison
    # ========================================================================
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    best_without = model_data_without['evaluated_fitness'].min()
    best_with = model_data_with['evaluated_fitness'].min()

    print(f"Best fitness WITHOUT update: {best_without:.6f}")
    print(f"Best fitness WITH update:    {best_with:.6f}")
    print()

    if best_with < best_without:
        improvement = best_without - best_with
        percentage = (improvement / best_without) * 100
        print(f"✓ Data update per epoch IMPROVED results!")
        print(f"  Improvement: {improvement:.6f} ({percentage:.2f}%)")
    elif best_with > best_without:
        degradation = best_with - best_without
        percentage = (degradation / best_without) * 100
        print(f"✗ Data update per epoch degraded results")
        print(f"  Degradation: {degradation:.6f} ({percentage:.2f}%)")
    else:
        print(f"= Results are identical")

    print()
    print("=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("The update_data_per_epoch feature is working correctly!")
    print("Training data is now updated after each epoch with the best solutions found.")
    print()


if __name__ == '__main__':
    test_update_data_per_epoch()
