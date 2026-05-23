"""
Quick test script to verify fitness-based distribution functionality.
"""

import numpy as np

# Test imports
print("Testing imports...")
try:
    from benchmark_dendiff_distributions import (
        sphere_function,
        rastrigin_function,
        generate_empirical_fitness_distribution,
        OBJECTIVE_FUNCTIONS
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test objective functions
print("\nTesting objective functions...")
try:
    x = np.random.randn(10, 5)  # 10 solutions, 5 variables

    # Test sphere
    f_sphere = sphere_function(x)
    assert f_sphere.shape == (10,), f"Expected shape (10,), got {f_sphere.shape}"
    assert np.all(f_sphere >= 0), "Sphere function should be >= 0"
    print(f"✓ Sphere function works (sample output: {f_sphere[0]:.4f})")

    # Test rastrigin
    f_rastrigin = rastrigin_function(x)
    assert f_rastrigin.shape == (10,), f"Expected shape (10,), got {f_rastrigin.shape}"
    print(f"✓ Rastrigin function works (sample output: {f_rastrigin[0]:.4f})")

except Exception as e:
    print(f"✗ Objective function error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test OBJECTIVE_FUNCTIONS dictionary
print("\nTesting OBJECTIVE_FUNCTIONS dictionary...")
try:
    expected_functions = ['sphere', 'ellipsoid', 'rastrigin', 'rosenbrock', 'ackley', 'griewank', 'schwefel']
    for func_name in expected_functions:
        assert func_name in OBJECTIVE_FUNCTIONS, f"{func_name} not in OBJECTIVE_FUNCTIONS"
        assert 'function' in OBJECTIVE_FUNCTIONS[func_name], f"Missing 'function' key for {func_name}"
        assert 'bounds' in OBJECTIVE_FUNCTIONS[func_name], f"Missing 'bounds' key for {func_name}"
        assert 'optimum' in OBJECTIVE_FUNCTIONS[func_name], f"Missing 'optimum' key for {func_name}"
    print(f"✓ All {len(expected_functions)} objective functions registered correctly")
except Exception as e:
    print(f"✗ OBJECTIVE_FUNCTIONS error: {e}")
    sys.exit(1)

# Test empirical fitness distribution generation
print("\nTesting empirical fitness distribution generation...")
try:
    MAT, MAT_fitness, metadata = generate_empirical_fitness_distribution(
        'sphere',
        n_initial=100,
        n_selected=50,
        n_vars=5,
        seed=42
    )

    assert MAT.shape == (50, 5), f"Expected MAT shape (50, 5), got {MAT.shape}"
    assert MAT_fitness.shape == (50,), f"Expected fitness shape (50,), got {MAT_fitness.shape}"
    assert np.all(MAT_fitness[:-1] <= MAT_fitness[1:]), "Fitness should be sorted (best first)"

    print(f"✓ Empirical fitness distribution generated successfully")
    print(f"  MAT shape: {MAT.shape}")
    print(f"  Best fitness: {metadata['best_fitness']:.6f}")
    print(f"  Worst fitness: {metadata['worst_fitness']:.6f}")
    print(f"  Mean fitness: {metadata['mean_fitness']:.6f}")

except Exception as e:
    print(f"✗ Empirical distribution error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test that different objectives produce different fitness distributions
print("\nTesting different objective functions...")
try:
    results = {}
    for obj_name in ['sphere', 'rastrigin', 'rosenbrock']:
        MAT, MAT_fitness, metadata = generate_empirical_fitness_distribution(
            obj_name,
            n_initial=100,
            n_selected=50,
            n_vars=5,
            seed=42
        )
        results[obj_name] = metadata['best_fitness']
        print(f"  {obj_name}: best_fitness = {metadata['best_fitness']:.6f}")

    # Different functions should give different results
    assert len(set(results.values())) > 1, "Different objectives should produce different fitness values"
    print("✓ Different objectives produce different fitness distributions")

except Exception as e:
    print(f"✗ Multiple objectives error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe fitness-based distribution functionality is working correctly.")
print("You can now run the full benchmark with:")
print("  python benchmark_dendiff_distributions.py")
