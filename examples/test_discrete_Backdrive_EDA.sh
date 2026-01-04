#!/bin/bash
# Test script for discrete_Backdrive_EDA.py
# Tests various configurations to ensure the program works correctly

echo "====================================================================="
echo "Testing discrete_Backdrive_EDA.py - Quick Configuration Tests"
echo "====================================================================="
echo

# Test 1: Basic usage
echo "Test 1: Basic usage with default settings"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5
echo
echo "-------------------------------------------------------------------"
echo

# Test 2: With early stopping
echo "Test 2: With early stopping enabled"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --early-stopping"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --early-stopping
echo
echo "-------------------------------------------------------------------"
echo

# Test 3: Different initialization
echo "Test 3: With perturb_best initialization"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --init perturb_best"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --init perturb_best
echo
echo "-------------------------------------------------------------------"
echo

# Test 4: Different loss function
echo "Test 4: With weighted MSE loss"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --loss weighted_mse"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --loss weighted_mse
echo
echo "-------------------------------------------------------------------"
echo

# Test 5: Adaptive variant
echo "Test 5: Backdrive-Adaptive variant"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --variant backdrive_adaptive"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --variant backdrive_adaptive
echo
echo "-------------------------------------------------------------------"
echo

# Test 6: Different truncation
echo "Test 6: Different truncation percent (0.3)"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.3"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.3
echo
echo "-------------------------------------------------------------------"
echo

# Test 7: All options combined
echo "Test 7: Combined options"
echo "Command: python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --weight-transfer --early-stopping --init perturb_selected --loss ranking"
python discrete_Backdrive_EDA.py 0 OneMax 10 40 10 0.5 --weight-transfer --early-stopping --init perturb_selected --loss ranking
echo
echo "-------------------------------------------------------------------"
echo

echo "====================================================================="
echo "All tests completed!"
echo "====================================================================="
