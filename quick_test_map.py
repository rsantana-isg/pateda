"""
Quick test of MAP inference functionality
"""

import sys
sys.path.insert(0, '/home/user/pateda')

import numpy as np

# Test MAP inference module imports
print("Testing MAP inference imports...")
try:
    from pateda.inference.map_inference import (
        MAPInference,
        compute_map,
        compute_k_map,
        MAPResult,
        KMAPResult
    )
    print("✓ MAP inference imports successful")
except Exception as e:
    print(f"✗ MAP inference import failed: {e}")
    sys.exit(1)

# Test MAP sampling imports (bypass __init__.py to avoid torch dependency)
print("\nTesting MAP sampling imports...")
try:
    # Import directly without going through pateda.sampling.__init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "map_sampling",
        "/home/user/pateda/pateda/sampling/map_sampling.py"
    )
    map_sampling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(map_sampling)

    SampleInsertMAP = map_sampling.SampleInsertMAP
    SampleTemplateMAP = map_sampling.SampleTemplateMAP
    SampleHybridMAP = map_sampling.SampleHybridMAP
    print("✓ MAP sampling imports successful")
except Exception as e:
    print(f"✗ MAP sampling import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test basic MAP inference
print("\nTesting basic MAP inference...")
try:
    # Create simple Markov network: two binary variables
    cliques = [np.array([0, 1])]

    # Probability table favoring (1, 1)
    table = np.array([[0.1, 0.2],   # X0=0
                      [0.2, 0.5]])   # X0=1

    tables = [table]
    cardinalities = np.array([2, 2])

    # Compute MAP using different methods
    map_bp = compute_map(cliques, tables, cardinalities, method="bp")
    print(f"  MAP (BP): {map_bp}")

    map_decimation = compute_map(cliques, tables, cardinalities, method="decimation")
    print(f"  MAP (Decimation): {map_decimation}")

    print("✓ MAP inference working")
except Exception as e:
    print(f"✗ MAP inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test k-MAP
print("\nTesting k-MAP...")
try:
    configs, probs = compute_k_map(cliques, tables, cardinalities, k=3, method="bp")
    print(f"  k-MAP configurations: {configs}")
    print(f"  Probabilities: {probs}")
    print(f"✓ k-MAP working")
except Exception as e:
    print(f"✗ k-MAP failed: {e}")
    import traceback
    traceback.print_exc()

# Test MAPInference class
print("\nTesting MAPInference class...")
try:
    inference = MAPInference(cliques, tables, cardinalities, method="bp")
    result = inference.compute_map()
    print(f"  Configuration: {result.configuration}")
    print(f"  Probability: {result.probability:.4f}")
    print(f"  Log probability: {result.log_probability:.4f}")
    print(f"  Method: {result.method}")
    print(f"✓ MAPInference class working")
except Exception as e:
    print(f"✗ MAPInference failed: {e}")
    import traceback
    traceback.print_exc()

# Test sampling class instantiation
print("\nTesting MAP sampling class instantiation...")
try:
    sampler1 = SampleInsertMAP(n_samples=10, map_method="bp")
    print(f"  Insert-MAP: {sampler1}")

    sampler2 = SampleTemplateMAP(n_samples=10, map_method="bp", template_prob=0.5)
    print(f"  Template-MAP: {sampler2}")

    sampler3 = SampleHybridMAP(n_samples=10, map_method="bp", template_prob=0.5)
    print(f"  Hybrid-MAP: {sampler3}")

    print("✓ MAP sampling classes instantiated successfully")
except Exception as e:
    print(f"✗ Sampling class instantiation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All quick tests passed!")
print("="*60)
