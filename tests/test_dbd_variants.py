#!/usr/bin/env python3
"""
Test script for DbD variants
Tests all loss functions and options on OneMax problem
"""

import subprocess
import sys
import os

def run_test(name, args):
    """Run a single test case"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cmd = ["python3", "examples/discrete_DbD_EDA.py"] + args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=90
        )
        
        # Check if run completed
        if result.returncode != 0:
            print(f"❌ FAILED: Return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
        
        # Extract best fitness from output
        for line in result.stdout.split('\n'):
            if 'Best Fitness:' in line and 'Best fitness found' not in line:
                try:
                    fitness = float(line.split(':')[1].strip())
                    print(f"✅ PASSED: Best fitness = {fitness}")
                    return True
                except:
                    pass
        
        print(f"⚠️ WARNING: Could not extract fitness")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Timeout")
        return False
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def main():
    """Run all tests"""
    
    # Common parameters: seed, problem, n_vars, pop_size, n_gen, trunc
    base_args = ["0", "OneMax", "20", "80", "5", "0.5"]
    
    tests = [
        # Test different loss functions
        ("DbD with MSE loss", 
         base_args + ["dbd", "relu", "mse", "10", "10", "0", "0.1", "0", "0"]),
        
        ("DbD with Weighted MSE loss",
         base_args + ["dbd", "relu", "weighted_mse", "10", "10", "0", "0.1", "0", "0"]),
        
        ("DbD with Ranking loss",
         base_args + ["dbd", "relu", "ranking", "10", "10", "0", "0.1", "0", "0"]),
        
        ("DbD with Huber loss",
         base_args + ["dbd", "relu", "huber", "10", "10", "0", "0.1", "0", "0"]),
        
        # Test fitness guidance
        ("C-DbD (Conditional with fitness guidance)",
         base_args + ["dbd", "relu", "mse", "10", "10", "0", "0.1", "1", "0"]),
        
        ("C-DbD with Weighted MSE",
         base_args + ["dbd", "relu", "weighted_mse", "10", "10", "0", "0.1", "1", "0"]),
        
        # Test Markov initialization
        ("M-DbD (Markov initialization)",
         base_args + ["dbd", "relu", "mse", "10", "10", "0", "0.1", "0", "1"]),
        
        # Test other variants
        ("DbD-CS with Weighted MSE",
         base_args + ["dbd_cs", "relu", "weighted_mse", "10", "10", "0", "0.1", "0", "0"]),
        
        ("DbD-UC with fitness guidance",
         base_args + ["dbd_uc", "relu", "mse", "10", "10", "0", "0.1", "1", "0"]),
    ]
    
    print(f"\n{'='*60}")
    print("DbD Variants Test Suite")
    print(f"{'='*60}")
    print(f"Testing {len(tests)} configurations...")
    
    passed = 0
    failed = 0
    
    for name, args in tests:
        if run_test(name, args):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print(f"\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
