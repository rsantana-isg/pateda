"""
Test cases for frequency balance mutation

Tests verify that:
1. When freq_ones > alpha, only positions with value 1 are flipped
2. When freq_zeros > alpha, only positions with value 0 are flipped
3. The mutation correctly balances the frequency distribution
"""

import numpy as np
import pytest
from pateda.mutation import frequency_balance_mutation


def test_frequency_balance_flips_only_ones_when_excess_ones():
    """Test that when freq_ones > alpha, only positions with 1 are flipped"""
    np.random.seed(42)
    
    n_vars = 10
    n_individuals = 100
    alpha = 0.7
    
    # Create population where variable 0 has 90% ones (exceeds alpha=0.7)
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:90, 0] = 1  # 90% ones in variable 0
    
    # Store positions that have ones
    ones_positions_before = np.where(population[:, 0] == 1)[0]
    zeros_positions_before = np.where(population[:, 0] == 0)[0]
    
    # Apply mutation
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Check which positions changed
    changed_positions = np.where(population[:, 0] != mutated_pop[:, 0])[0]
    
    # All changed positions should have been ones before mutation
    for pos in changed_positions:
        assert pos in ones_positions_before, f"Position {pos} was changed but was not a 1"
        assert mutated_pop[pos, 0] == 0, f"Position {pos} should be flipped to 0"
    
    # No zeros should have been changed
    for pos in zeros_positions_before:
        assert population[pos, 0] == mutated_pop[pos, 0], f"Position {pos} (which was 0) should not change"
    
    # Verify frequency is now closer to alpha
    freq_ones_after = np.sum(mutated_pop[:, 0]) / n_individuals
    assert freq_ones_after <= alpha + 0.05, f"Frequency of ones after mutation should be <= alpha"


def test_frequency_balance_flips_only_zeros_when_excess_zeros():
    """Test that when freq_zeros > alpha, only positions with 0 are flipped"""
    np.random.seed(42)
    
    n_vars = 10
    n_individuals = 100
    alpha = 0.7
    
    # Create population where variable 0 has 90% zeros (exceeds alpha=0.7)
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:10, 0] = 1  # Only 10% ones, so 90% zeros
    
    # Store positions that have zeros
    ones_positions_before = np.where(population[:, 0] == 1)[0]
    zeros_positions_before = np.where(population[:, 0] == 0)[0]
    
    # Apply mutation
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Check which positions changed
    changed_positions = np.where(population[:, 0] != mutated_pop[:, 0])[0]
    
    # All changed positions should have been zeros before mutation
    for pos in changed_positions:
        assert pos in zeros_positions_before, f"Position {pos} was changed but was not a 0"
        assert mutated_pop[pos, 0] == 1, f"Position {pos} should be flipped to 1"
    
    # No ones should have been changed
    for pos in ones_positions_before:
        assert population[pos, 0] == mutated_pop[pos, 0], f"Position {pos} (which was 1) should not change"
    
    # Verify frequency is now closer to alpha
    freq_zeros_after = 1.0 - np.sum(mutated_pop[:, 0]) / n_individuals
    assert freq_zeros_after <= alpha + 0.05, f"Frequency of zeros after mutation should be <= alpha"


def test_frequency_balance_no_mutation_when_within_alpha():
    """Test that no mutation occurs when frequencies are within alpha threshold"""
    np.random.seed(42)
    
    n_vars = 5
    n_individuals = 100
    alpha = 0.8
    
    # Create population where all variables have balanced frequencies (50/50)
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:50, :] = 1  # 50% ones, 50% zeros
    
    # Apply mutation
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Population should remain unchanged
    np.testing.assert_array_equal(population, mutated_pop)


def test_frequency_balance_multiple_variables():
    """Test that mutation works correctly for multiple variables independently"""
    np.random.seed(42)
    
    n_vars = 3
    n_individuals = 100
    alpha = 0.7
    
    # Create population with different frequency patterns
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:90, 0] = 1  # Variable 0: 90% ones (exceeds alpha)
    population[:10, 1] = 1  # Variable 1: 10% ones, 90% zeros (exceeds alpha)
    population[:50, 2] = 1  # Variable 2: 50% ones (within alpha)
    
    # Apply mutation
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Variable 0: should have some ones flipped to zeros
    freq_ones_var0 = np.sum(mutated_pop[:, 0]) / n_individuals
    assert freq_ones_var0 < 0.9, "Variable 0 frequency should be reduced"
    
    # Variable 1: should have some zeros flipped to ones
    freq_ones_var1 = np.sum(mutated_pop[:, 1]) / n_individuals
    assert freq_ones_var1 > 0.1, "Variable 1 frequency should be increased"
    
    # Variable 2: should remain unchanged
    np.testing.assert_array_equal(population[:, 2], mutated_pop[:, 2])


def test_frequency_balance_alpha_zero():
    """Test that alpha=0 means no mutation"""
    np.random.seed(42)
    
    n_vars = 5
    n_individuals = 100
    alpha = 0.0
    
    # Create population with extreme frequencies
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:95, 0] = 1  # 95% ones
    
    # Apply mutation with alpha=0
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Population should remain unchanged
    np.testing.assert_array_equal(population, mutated_pop)


def test_frequency_balance_computes_correct_number_to_flip():
    """Test that the number of positions to flip is correctly calculated"""
    np.random.seed(42)
    
    n_vars = 5
    n_individuals = 100
    alpha = 0.8
    
    # Create population where variable 0 has 95% ones
    population = np.zeros((n_individuals, n_vars), dtype=int)
    population[:95, 0] = 1
    
    # Apply mutation
    params = {'alpha': alpha}
    mutated_pop = frequency_balance_mutation(n_vars, np.array([2]*n_vars), population, params)
    
    # Number of flips should be (1 - alpha) * n_individuals = 20
    expected_flips = int((1 - alpha) * n_individuals)
    actual_flips = np.sum(population[:, 0] != mutated_pop[:, 0])
    
    assert actual_flips == expected_flips, f"Expected {expected_flips} flips, got {actual_flips}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
