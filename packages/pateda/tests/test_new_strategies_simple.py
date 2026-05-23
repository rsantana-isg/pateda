"""
Simplified test for new sampling strategies that doesn't rely on package imports.
"""
import importlib.util

# Directly load modules without package structure
def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Base path
base_path = "/home/runner/work/pateda/pateda"

# Load required modules
print("Loading modules...")

# Load nn_utils first (required dependency)
nn_utils = load_module_from_path(
    "nn_utils",
    os.path.join(base_path, "learning/nn_utils.py")
)

# Load learning modules
ste_module = load_module_from_path(
    "discrete_dendiff_ste",
    os.path.join(base_path, "learning/discrete_dendiff_ste.py")
)

hard_concrete_module = load_module_from_path(
    "discrete_dendiff_hard_concrete",
    os.path.join(base_path, "learning/discrete_dendiff_hard_concrete.py")
)

deterministic_module = load_module_from_path(
    "discrete_dendiff_deterministic",
    os.path.join(base_path, "learning/discrete_dendiff_deterministic.py")
)

print("✓ Modules loaded successfully\n")

# Now run a simple test
import numpy as np
import torch

def simple_test(strategy_name, learn_fn):
    """Run a simple test for a learning function."""
    print(f"Testing: {strategy_name}")
    print("-" * 50)
    
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create simple data
    n_vars = 10
    pop_size = 20
    population = np.random.randint(0, 2, (pop_size, n_vars)).astype(np.float32)
    fitness = np.sum(population, axis=1)
    
    # Simple params
    params = {
        'n_timesteps': 5,
        'epochs': 3,
        'batch_size': 5,
        'hidden_dims': [16, 8],
        'learning_rate': 1e-3
    }
    
    try:
        # Learn model
        model = learn_fn(population, fitness, params)
        print(f"  ✓ Learning successful")
        print(f"    Model type: {model['type']}")
        print(f"    Input dim: {model['input_dim']}")
        
        # Check model has required keys
        required_keys = ['model_state', 'input_dim', 'type', 'hidden_dims']
        missing_keys = [k for k in required_keys if k not in model]
        if missing_keys:
            print(f"  ✗ Missing keys: {missing_keys}")
            return False
        
        print(f"  ✓ Model structure valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

# Run tests
print("="*70)
print("TESTING NEW ALTERNATIVE SAMPLING STRATEGIES (Simplified)")
print("="*70 + "\n")

results = {}

# Test 1: STE
results['STE'] = simple_test(
    "Straight-Through Estimator (STE)",
    ste_module.learn_discrete_dendiff_ste
)

# Test 2: Hard Concrete
results['Hard Concrete'] = simple_test(
    "Hard Concrete Distribution",
    hard_concrete_module.learn_discrete_dendiff_hard_concrete
)

# Test 3: Deterministic
results['Deterministic'] = simple_test(
    "Deterministic Softmax",
    deterministic_module.learn_discrete_dendiff_deterministic
)

# Summary
print("="*70)
print("TEST SUMMARY")
print("="*70)
for strategy, passed in results.items():
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{strategy:20s}: {status}")

all_passed = all(results.values())
print("="*70)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
print("="*70)

sys.exit(0 if all_passed else 1)
