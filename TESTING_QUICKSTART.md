# DbD-EDA Testing Quick Start Guide

**Branch**: `claude/implement-dbd-eda-01DHskrqrFK7FocN7GTvaXoo`

---

## Prerequisites

### 1. Install PyTorch

```bash
pip install torch>=1.9.0 numpy>=1.21.0
```

Or if you prefer CPU-only version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Verify Installation

```bash
python3 -c "import torch; import numpy; print('PyTorch version:', torch.__version__); print('NumPy version:', numpy.__version__)"
```

Expected output:
```
PyTorch version: 1.x.x
NumPy version: 1.x.x
```

---

## Running Tests

### Quick Validation (5 minutes)

**Recommended first test** - verifies basic functionality:

```bash
cd /home/user/pateda
python3 pateda/examples/dbd_quick_test.py
```

**Expected output:**
```
================================================================================
DbD-EDA QUICK VALIDATION TEST
================================================================================

Test 1: Basic learning and sampling
----------------------------------------
Learning DbD model...
✓ Model learned in X.XXXs
  Input dimension: 5
  Hidden dims: [64, 64]

Sampling from DbD model...
✓ Generated 20 samples in X.XXXs
  Sample shape: (20, 5)
  Sample mean: [...]
  Expected mean ~3.0, actual mean: ~3.0

Test 2: Univariate Gaussian sampling
----------------------------------------
✓ Generated 50 univariate samples
  Original mean: [...]
  Sampled mean: [...]

Test 3: Simple optimization with DbD-CS
----------------------------------------
Running DbD-CS for 10 generations on 5D sphere function...
✓ Optimization completed
  Initial fitness: X.XXXe+XX
  Final fitness: X.XXXe+XX
  Improvement: XXx

================================================================================
ALL TESTS PASSED!
================================================================================
```

**If this passes, your implementation is working!**

---

### Single Variant Test (10 minutes)

Test one variant on the Sphere function:

```bash
cd /home/user/pateda
python3 -c "
from pateda.examples.dbd_eda_example import test_single_variant
test_single_variant('CS', n_vars=10, n_generations=30)
"
```

**Expected output:**
```
================================================================================
Testing DbD-CS on Sphere Function
================================================================================

Generation   1: Best fitness = X.XXXe+XX, No improvement = 0
Generation   2: Best fitness = X.XXXe+XX, No improvement = 0
...
Generation  30: Best fitness = X.XXXe-XX, No improvement = 0

================================================================================
RESULTS for DbD-CS
================================================================================
Initial best fitness: X.XXXe+XX
Final best fitness:   X.XXXe-XX
Improvement:          XXXx
Avg learning time:    X.XXXs
Avg sampling time:    X.XXXs
================================================================================
```

**Success criteria:**
- Final fitness should be < 1e-2
- Improvement should be > 100x
- No errors or crashes

---

### Compare All Variants (30 minutes)

Compare all four variants side-by-side:

```bash
cd /home/user/pateda
python3 -c "
from pateda.examples.dbd_eda_example import compare_all_variants
compare_all_variants(n_vars=10, n_generations=25)
"
```

**Expected output:**
```
================================================================================
VARIANT COMPARISON
================================================================================
Variant    Initial         Final           Improvement  Avg Time/Gen
--------------------------------------------------------------------------------
DbD-CS     X.XXXe+XX       X.XXXe-XX       XXX.XXx      X.XXXs
DbD-CD     X.XXXe+XX       X.XXXe-XX       XXX.XXx      X.XXXs
DbD-UC     X.XXXe+XX       X.XXXe-XX       XXX.XXx      X.XXXs
DbD-US     X.XXXe+XX       X.XXXe-XX       XXX.XXx      X.XXXs
================================================================================
```

**Expected results:**
- DbD-CS and DbD-CD: Best final fitness
- DbD-US: Competitive
- DbD-UC: Slightly worse (focuses on dependency learning)

---

### Benchmark Functions (45 minutes)

Test on multiple benchmark functions:

```bash
cd /home/user/pateda
python3 -c "
from pateda.examples.dbd_eda_example import test_benchmark_functions
test_benchmark_functions()
"
```

Tests all four functions:
- Sphere (unimodal, separable)
- Rosenbrock (unimodal, non-separable)
- Rastrigin (multimodal, separable)
- Ackley (multimodal, non-separable)

---

### Full Test Suite (60-90 minutes)

Run everything:

```bash
cd /home/user/pateda
python3 pateda/examples/dbd_eda_example.py
```

**This will run:**
1. Single variant test (DbD-CS)
2. All variants comparison
3. Benchmark functions test

---

## Interpreting Results

### Good Results Indicators

✅ **Convergence**:
- Fitness decreases over generations
- Final fitness much better than initial
- Smooth convergence (occasional restarts OK)

✅ **Performance**:
- Learning time: 1-5 seconds per generation
- Sampling time: < 1 second per generation
- Total time per generation: < 10 seconds

✅ **Variant Behavior**:
- All variants show improvement
- DbD-CS/CD generally best performance
- DbD-UC/US competitive

✅ **Restart Mechanism**:
- Triggers on multimodal functions (Rastrigin, Ackley)
- Doesn't trigger too frequently (< 3 times)
- Improves fitness after restart

### Potential Issues

❌ **No convergence**:
- Fitness not improving or increasing
- **Check**: Learning rate, epochs, hidden dimensions

❌ **NaN or Inf values**:
- Neural network producing invalid outputs
- **Check**: Normalization, gradient clipping needed?

❌ **Very slow**:
- > 30 seconds per generation
- **Check**: Batch size, hidden dimensions, number of epochs

❌ **Memory errors**:
- Out of memory during training
- **Check**: Reduce batch_size or population size

---

## Common Issues and Solutions

### Issue 1: Import Errors

```
ModuleNotFoundError: No module named 'pateda.learning.dbd'
```

**Solution**:
```bash
cd /home/user/pateda
export PYTHONPATH=/home/user/pateda:$PYTHONPATH
python3 pateda/examples/dbd_quick_test.py
```

### Issue 2: PyTorch Not Found

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
pip3 install torch numpy
```

### Issue 3: Slow Training

If training is too slow, reduce computational cost:

```python
# Modify dbd_params in test scripts:
dbd_params={
    'epochs': 20,           # Reduce from 50
    'batch_size': 64,       # Increase from 32
    'hidden_dims': [32],    # Smaller network
    'num_iterations': 5     # Reduce from 10
}
```

### Issue 4: Poor Convergence

If convergence is poor, try:

```python
dbd_params={
    'epochs': 100,          # More training
    'learning_rate': 5e-4,  # Lower learning rate
    'num_alpha_samples': 20 # More alpha samples
}
```

---

## Recording Results

### Create Results Log

```bash
cd /home/user/pateda
python3 pateda/examples/dbd_quick_test.py 2>&1 | tee results_quick_test.log
python3 pateda/examples/dbd_eda_example.py 2>&1 | tee results_full_suite.log
```

### Save Performance Data

After running tests, document:

1. **System Info**:
   - CPU/GPU used
   - Python version
   - PyTorch version
   - RAM available

2. **Performance Metrics**:
   - Final fitness for each variant
   - Avg time per generation
   - Total runtime

3. **Issues Encountered**:
   - Any errors or warnings
   - Convergence problems
   - Parameter adjustments made

---

## Expected Performance Benchmarks

### 10D Sphere Function (30 generations)

| Variant | Expected Final Fitness | Expected Time/Gen |
|---------|----------------------|-------------------|
| DbD-CS  | < 1e-3               | ~2s               |
| DbD-CD  | < 1e-3               | ~2s               |
| DbD-UC  | < 1e-2               | ~2s               |
| DbD-US  | < 1e-3               | ~2s               |

### 10D Rastrigin Function (30 generations)

| Variant | Expected Final Fitness | Restarts Expected |
|---------|----------------------|-------------------|
| DbD-CS  | < 50                 | 1-2               |
| DbD-CD  | < 50                 | 1-2               |
| DbD-UC  | < 80                 | 2-3               |
| DbD-US  | < 50                 | 1-2               |

---

## Next Steps After Testing

### If Tests Pass ✅

1. **Document results** in test execution log
2. **Run GNBG benchmarks** (24 functions from paper)
3. **Compare with published results**
4. **Consider parameter tuning** for better performance
5. **Ready for publication/use**

### If Tests Fail ❌

1. **Check error messages** carefully
2. **Review** VALIDATION_SUMMARY_DbD_EDA.md
3. **Try** reducing problem complexity (smaller dimensions)
4. **Adjust** hyperparameters
5. **Report issues** with:
   - Error messages
   - System configuration
   - Parameter settings used

---

## Getting Help

### Documentation Files

- `pateda/docs/DbD_EDA_README.md` - Usage guide
- `TEST_PLAN_DbD_EDA.md` - Detailed test plan
- `VALIDATION_SUMMARY_DbD_EDA.md` - Validation status

### Key Parameters to Tune

If you need to adjust performance:

**Learning Parameters**:
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Adam learning rate (default: 1e-3)
- `num_alpha_samples`: Alpha samples per pair (default: 10)

**Network Architecture**:
- `hidden_dims`: Hidden layer sizes (default: [64, 64])

**Sampling Parameters**:
- `num_iterations`: Deblending iterations (default: 10)

**EDA Parameters**:
- `pop_size`: Population size (default: 200)
- `selection_ratio`: Selection ratio (default: 0.3)

**Restart Parameters**:
- `trigger_no_improvement`: Generations before restart (default: 5)
- `diversity_threshold`: Min diversity (default: 1e-6)

---

## Quick Command Reference

```bash
# Install dependencies
pip3 install torch numpy

# Quick test (5 min)
python3 pateda/examples/dbd_quick_test.py

# Single variant (10 min)
python3 -c "from pateda.examples.dbd_eda_example import test_single_variant; test_single_variant('CS')"

# Compare variants (30 min)
python3 -c "from pateda.examples.dbd_eda_example import compare_all_variants; compare_all_variants()"

# Full suite (60-90 min)
python3 pateda/examples/dbd_eda_example.py

# With logging
python3 pateda/examples/dbd_eda_example.py 2>&1 | tee test_results.log
```

---

## Success Checklist

- [ ] PyTorch installed and verified
- [ ] Quick test passes
- [ ] At least one variant successfully optimizes Sphere function
- [ ] All four variants run without errors
- [ ] Fitness improves over generations
- [ ] Restart mechanism triggers appropriately
- [ ] Performance metrics are reasonable
- [ ] Results logged for future reference

---

**Good luck with testing! The implementation is ready to go.**
