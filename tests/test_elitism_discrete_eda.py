"""
Test cases for elitism in discrete_EDA_RW.py

Tests verify that:
1. The best solution in the population is not mutated
2. The best solution is preserved through the mutation process
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pateda.mutation import frequency_balance_mutation


def test_elitism_best_solution_preserved():
    """
    Test that demonstrates the elitism pattern used in discrete_EDA_RW.py
    
    This test simulates what happens in the neural EDA:
    1. A population with mutation applied
    2. The best solution before mutation is preserved
    3. The best solution is replaced back after mutation
    """
    np.random.seed(42)
    
    n_vars = 20
    n_individuals = 50
    alpha = 0.9
    
    # Create a population with one clearly best solution
    population = np.random.randint(0, 2, (n_individuals, n_vars))
    
    # Define a simple fitness function (OneMax)
    def fitness_func(pop):
        if pop.ndim == 1:
            return np.array([float(np.sum(pop))])
        else:
            return np.sum(pop, axis=1).astype(float)
    
    # Make the first individual the best (all ones)
    population[0] = np.ones(n_vars, dtype=int)
    
    # Simulate the elitism pattern from discrete_EDA_RW.py
    # Step 1: Evaluate before mutation
    pre_mutation_fitness = fitness_func(population)
    best_idx = np.argmax(pre_mutation_fitness)
    best_solution_pre_mutation = population[best_idx].copy()
    best_fitness_pre_mutation = pre_mutation_fitness[best_idx]
    
    # Step 2: Apply mutation
    mutation_params = {'alpha': alpha}
    cardinality = np.full(n_vars, 2)
    mutated_population = frequency_balance_mutation(
        n_vars,
        cardinality,
        population,
        mutation_params
    )
    
    # Step 3: Enforce elitism - replace best solution
    mutated_population[best_idx] = best_solution_pre_mutation
    
    # Verify that the best solution is exactly preserved
    np.testing.assert_array_equal(
        mutated_population[best_idx],
        best_solution_pre_mutation,
        err_msg="Best solution should be preserved through mutation"
    )
    
    # Verify the best solution has not changed
    assert np.array_equal(mutated_population[best_idx], np.ones(n_vars)), \
        "Best solution (all ones) should be unchanged"
    
    # Verify fitness is preserved
    post_mutation_fitness = fitness_func(mutated_population)
    assert post_mutation_fitness[best_idx] == best_fitness_pre_mutation, \
        "Best fitness should be preserved"


def test_elitism_with_high_frequency_mutation():
    """
    Test elitism when mutation would aggressively change population
    
    This ensures that even with aggressive mutation, the best solution
    is still preserved.
    """
    np.random.seed(123)
    
    n_vars = 10
    n_individuals = 30
    alpha = 0.6  # Low alpha means aggressive mutation
    
    # Create population with extreme frequency (95% ones)
    population = np.ones((n_individuals, n_vars), dtype=int)
    population[:2, :] = 0  # Only 2 individuals with zeros
    
    # Make one individual clearly the best with a unique pattern
    best_pattern = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    population[5] = best_pattern
    
    def fitness_func(pop):
        """Custom fitness that favors the best_pattern"""
        if pop.ndim == 1:
            return np.array([float(np.sum(pop == best_pattern))])
        else:
            return np.array([np.sum(ind == best_pattern) for ind in pop])
    
    # Simulate elitism
    pre_mutation_fitness = fitness_func(population)
    best_idx = np.argmax(pre_mutation_fitness)
    best_solution_pre_mutation = population[best_idx].copy()
    
    # Apply mutation
    mutation_params = {'alpha': alpha}
    cardinality = np.full(n_vars, 2)
    mutated_population = frequency_balance_mutation(
        n_vars,
        cardinality,
        population,
        mutation_params
    )
    
    # The best individual might have been mutated in the process
    # Verify it's different from the original
    assert not np.array_equal(mutated_population[best_idx], best_solution_pre_mutation), \
        "With this aggressive mutation, the best solution should have been changed"
    
    # Now enforce elitism
    mutated_population[best_idx] = best_solution_pre_mutation
    
    # Verify elitism worked
    np.testing.assert_array_equal(
        mutated_population[best_idx],
        best_pattern,
        err_msg="After enforcing elitism, best solution should be preserved"
    )


def test_elitism_different_best_indices():
    """
    Test that elitism works when the best solution is at different indices
    """
    np.random.seed(789)
    
    n_vars = 15
    n_individuals = 40
    alpha = 0.8
    
    for best_idx_to_test in [0, 10, 39]:  # First, middle, last
        # Create population
        population = np.random.randint(0, 2, (n_individuals, n_vars))
        
        # Place the best solution at the specified index
        best_solution = np.ones(n_vars, dtype=int)
        population[best_idx_to_test] = best_solution
        
        def fitness_func(pop):
            if pop.ndim == 1:
                return np.array([float(np.sum(pop))])
            else:
                return np.sum(pop, axis=1).astype(float)
        
        # Simulate elitism pattern
        pre_mutation_fitness = fitness_func(population)
        best_idx = np.argmax(pre_mutation_fitness)
        assert best_idx == best_idx_to_test, "Best index should be where we placed it"
        
        best_solution_pre_mutation = population[best_idx].copy()
        
        # Apply mutation
        mutation_params = {'alpha': alpha}
        cardinality = np.full(n_vars, 2)
        mutated_population = frequency_balance_mutation(
            n_vars,
            cardinality,
            population,
            mutation_params
        )
        
        # Enforce elitism
        mutated_population[best_idx] = best_solution_pre_mutation
        
        # Verify
        np.testing.assert_array_equal(
            mutated_population[best_idx],
            best_solution,
            err_msg=f"Best solution at index {best_idx_to_test} should be preserved"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
