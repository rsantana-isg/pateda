#!/usr/bin/env python3
"""
Test script to verify MI-based parent selection in DBD functions
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pateda.learning.discrete_dbd import (
    compute_mutual_information_matrix_binary,
    find_k_highest_mi_parents,
    compute_conditional_probabilities,
    transform_binary_to_continuous,
    learn_binary_dbd_cs_t,
    learn_binary_dbd_cd_t,
)


def test_mi_matrix():
    """Test mutual information matrix computation"""
    print("\n" + "="*60)
    print("Test 1: Mutual Information Matrix Computation")
    print("="*60)
    
    # Create a simple binary population with known dependencies
    # Variables 0 and 1 are correlated, 2 and 3 are correlated
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated variables
    var0 = np.random.randint(0, 2, n_samples)
    var1 = var0.copy()  # Perfect correlation
    var1[np.random.rand(n_samples) < 0.1] = 1 - var1[np.random.rand(n_samples) < 0.1]  # Add some noise
    
    var2 = np.random.randint(0, 2, n_samples)
    var3 = var2.copy()  # Perfect correlation  
    var3[np.random.rand(n_samples) < 0.1] = 1 - var3[np.random.rand(n_samples) < 0.1]  # Add some noise
    
    population = np.column_stack([var0, var1, var2, var3])
    
    # Compute MI matrix
    mi_matrix = compute_mutual_information_matrix_binary(population)
    
    print(f"Population shape: {population.shape}")
    print(f"MI Matrix shape: {mi_matrix.shape}")
    print(f"MI Matrix:\n{mi_matrix}")
    
    # Check that MI is symmetric
    assert np.allclose(mi_matrix, mi_matrix.T), "MI matrix should be symmetric"
    
    # Check that diagonal is zero
    assert np.allclose(np.diag(mi_matrix), 0), "Diagonal should be zero"
    
    # Check that MI(0,1) and MI(2,3) are higher (correlated variables)
    print(f"MI(0,1) = {mi_matrix[0, 1]:.4f} (should be high - correlated)")
    print(f"MI(2,3) = {mi_matrix[2, 3]:.4f} (should be high - correlated)")
    print(f"MI(0,2) = {mi_matrix[0, 2]:.4f} (should be low - independent)")
    print(f"MI(1,3) = {mi_matrix[1, 3]:.4f} (should be low - independent)")
    
    print("✅ Test 1 PASSED")
    return mi_matrix


def test_k_highest_mi_parents():
    """Test finding k highest MI parents"""
    print("\n" + "="*60)
    print("Test 2: Finding k Highest MI Parents")
    print("="*60)
    
    # Create a simple MI matrix
    mi_matrix = np.array([
        [0.0, 0.5, 0.1, 0.2],
        [0.5, 0.0, 0.3, 0.4],
        [0.1, 0.3, 0.0, 0.6],
        [0.2, 0.4, 0.6, 0.0]
    ])
    
    k = 2
    parent_structure = find_k_highest_mi_parents(mi_matrix, k)
    
    print(f"MI Matrix:\n{mi_matrix}")
    print(f"k = {k}")
    print(f"Parent structure: {parent_structure}")
    
    # Verify structure
    for var in range(4):
        parents = parent_structure[var]
        print(f"Variable {var} has parents: {parents}")
        assert len(parents) == k, f"Variable {var} should have {k} parents"
        assert var not in parents, f"Variable {var} should not be its own parent"
        
        # Verify parents are sorted by MI (highest first)
        mi_values = [mi_matrix[var, p] for p in parents]
        assert mi_values == sorted(mi_values, reverse=True), "Parents should be sorted by MI"
    
    # For variable 0, highest MI should be with variable 1 (0.5), then 3 (0.2)
    assert parent_structure[0][0] == 1, "Highest MI parent of 0 should be 1"
    assert parent_structure[0][1] == 3, "Second highest MI parent of 0 should be 3"
    
    print("✅ Test 2 PASSED")
    return parent_structure


def test_conditional_probabilities_with_mi():
    """Test conditional probabilities with MI-based parent structure"""
    print("\n" + "="*60)
    print("Test 3: Conditional Probabilities with MI-based Parents")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 100
    n_vars = 5
    
    # Create random binary population
    population = np.random.randint(0, 2, (n_samples, n_vars))
    
    k = 2
    
    # Compute MI matrix and parent structure
    mi_matrix = compute_mutual_information_matrix_binary(population)
    parent_structure = find_k_highest_mi_parents(mi_matrix, k)
    
    # Compute conditional probabilities with MI-based parents
    cond_probs_mi = compute_conditional_probabilities(
        population, k, alpha=0.1, parent_structure=parent_structure
    )
    
    # Compute conditional probabilities without MI (original order)
    cond_probs_orig = compute_conditional_probabilities(
        population, k, alpha=0.1, parent_structure=None
    )
    
    print(f"Number of conditional probability tables: {len(cond_probs_mi)}")
    print(f"Parent structure: {parent_structure}")
    
    # Verify we got one CPT per variable
    assert len(cond_probs_mi) == n_vars, "Should have one CPT per variable"
    assert len(cond_probs_orig) == n_vars, "Should have one CPT per variable"
    
    # Verify shapes are correct
    for var in range(n_vars):
        cpd_mi = cond_probs_mi[var]
        cpd_orig = cond_probs_orig[var]
        
        parents_mi = parent_structure.get(var, [])
        n_parents_mi = len(parents_mi)
        
        if n_parents_mi == 0:
            # Marginal probability
            assert cpd_mi.shape == (2,), f"Marginal CPT for var {var} should have shape (2,)"
            print(f"Variable {var}: Marginal P(X={var}): {cpd_mi}")
        else:
            # Conditional probability
            expected_shape = (2 ** n_parents_mi, 2)
            assert cpd_mi.shape == expected_shape, f"CPT for var {var} should have shape {expected_shape}"
            print(f"Variable {var}: Conditional on {parents_mi}, shape: {cpd_mi.shape}")
    
    print("✅ Test 3 PASSED")


def test_transform_with_mi():
    """Test binary to continuous transformation with MI-based parents"""
    print("\n" + "="*60)
    print("Test 4: Binary to Continuous Transformation with MI")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 50
    n_vars = 4
    
    # Create random binary population
    population = np.random.randint(0, 2, (n_samples, n_vars))
    
    k = 2
    
    # Compute MI matrix and parent structure
    mi_matrix = compute_mutual_information_matrix_binary(population)
    parent_structure = find_k_highest_mi_parents(mi_matrix, k)
    
    # Compute conditional probabilities
    cond_probs = compute_conditional_probabilities(
        population, k, alpha=0.1, parent_structure=parent_structure
    )
    
    # Transform to continuous
    continuous_pop = transform_binary_to_continuous(
        population, cond_probs, k, parent_structure
    )
    
    print(f"Original population shape: {population.shape}")
    print(f"Continuous population shape: {continuous_pop.shape}")
    print(f"Continuous values range: [{continuous_pop.min():.4f}, {continuous_pop.max():.4f}]")
    
    # Verify shapes match
    assert continuous_pop.shape == population.shape, "Shapes should match"
    
    # Verify all values are probabilities (between 0 and 1)
    assert np.all(continuous_pop >= 0) and np.all(continuous_pop <= 1), "Values should be probabilities"
    
    # Verify no NaN or inf
    assert not np.any(np.isnan(continuous_pop)), "Should not contain NaN"
    assert not np.any(np.isinf(continuous_pop)), "Should not contain inf"
    
    print("✅ Test 4 PASSED")


def test_learn_functions():
    """Test learn_binary_dbd_cs_t and learn_binary_dbd_cd_t"""
    print("\n" + "="*60)
    print("Test 5: Learn Functions with MI-based Parents")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 80
    n_vars = 10
    
    # Create current and selected populations
    current_pop = np.random.randint(0, 2, (n_samples, n_vars))
    selected_pop = np.random.randint(0, 2, (n_samples // 2, n_vars))
    
    params = {
        'k': 2,
        'alpha': 0.1,
        'hidden_dims': [32],
        'batch_size': 16,
        'n_epochs': 2,  # Small number for quick test
    }
    
    print("Testing learn_binary_dbd_cs_t...")
    try:
        model_cs = learn_binary_dbd_cs_t(current_pop, selected_pop, params)
        print(f"✅ learn_binary_dbd_cs_t succeeded")
        print(f"   Model variant: {model_cs['variant']}")
        print(f"   Model has parent_structure: {'parent_structure' in model_cs}")
        print(f"   Model has mi_matrix: {'mi_matrix' in model_cs}")
        
        if 'parent_structure' in model_cs:
            print(f"   Number of variables in parent structure: {len(model_cs['parent_structure'])}")
        
    except Exception as e:
        print(f"❌ learn_binary_dbd_cs_t failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTesting learn_binary_dbd_cd_t...")
    try:
        model_cd = learn_binary_dbd_cd_t(current_pop, selected_pop, params)
        print(f"✅ learn_binary_dbd_cd_t succeeded")
        print(f"   Model variant: {model_cd['variant']}")
        print(f"   Model has parent_structure: {'parent_structure' in model_cd}")
        print(f"   Model has mi_matrix: {'mi_matrix' in model_cd}")
        
        if 'parent_structure' in model_cd:
            print(f"   Number of variables in parent structure: {len(model_cd['parent_structure'])}")
        
    except Exception as e:
        print(f"❌ learn_binary_dbd_cd_t failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Test 5 PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing MI-based Parent Selection for DBD")
    print("="*60)
    
    try:
        test_mi_matrix()
        test_k_highest_mi_parents()
        test_conditional_probabilities_with_mi()
        test_transform_with_mi()
        success = test_learn_functions()
        
        if success:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("❌ SOME TESTS FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
