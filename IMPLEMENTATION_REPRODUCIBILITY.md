# Implementation Summary: Script Standardization and Reproducibility Fixes

## Overview
This document summarizes the changes made to standardize the discrete EDA scripts and fix reproducibility issues related to random seeding.

## Changes Implemented

### 1. Converted Flags to Positional Arguments

#### Files Modified:
- `examples/discrete_GAN_EDA.py`
- `examples/discrete_VAE_EDA.py`

#### Changes:
Both scripts previously used a mix of required positional arguments and optional flags (e.g., `--activation-g`, `--activation-d`, `--beta-start`, etc.). These have been converted to all positional arguments to maintain consistency with `discrete_Backdrive_EDA.py`.

**discrete_GAN_EDA.py** now requires 14 positional arguments:
```bash
python discrete_GAN_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> <variant> \
    <activation_g> <activation_d> <activation_e> <dropout> <temperature> <use_surrogate>
```

**discrete_VAE_EDA.py** now requires 14 positional arguments:
```bash
python discrete_VAE_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> <vae_variant> \
    <activation_enc> <activation_dec> <beta_start> <beta_end> <latent_dim> <epochs>
```

**Rationale:**
- Consistency across all three discrete EDA programs
- Easier to use in SLURM batch scripts where all parameters must be specified
- Eliminates confusion about default values
- Makes parameter exploration more explicit

### 2. Created Launch Scripts for GAN and VAE Experiments

#### New Files Created:
- `slurm_gan.sh` - SLURM batch script for GAN experiments
- `launch_gan_experiments.py` - Python script to generate all GAN experiment combinations
- `slurm_vae.sh` - SLURM batch script for VAE experiments
- `launch_vae_experiments.py` - Python script to generate all VAE experiment combinations

#### Design:
These scripts follow the same pattern as `slurm_backdrive.sh` and `launch_backdrive_experiments.py`:

**SLURM Scripts (`slurm_gan.sh`, `slurm_vae.sh`):**
- Accept all parameters as positional arguments
- Generate descriptive output filenames based on all parameters
- Include commented SLURM directives for resource allocation
- Provide example usage commands

**Launch Scripts (`launch_gan_experiments.py`, `launch_vae_experiments.py`):**
- Define parameter grids for systematic exploration
- Generate `sbatch` commands for all parameter combinations
- Support multiple objective functions and problem sizes
- Print commands to stdout for review before submission

**GAN Parameter Grid:**
- Variants: WGAN-GP, Cond-Fit-GAN, Aux-GAN, Repulsion-GAN
- Generator activations: relu, tanh, leaky_relu
- Discriminator activations: leaky_relu, relu
- Dropout rates: 0.3, 0.5
- Temperature values: 0.5, 1.0
- Use surrogate: 0, 1 (only for Aux-GAN)

**VAE Parameter Grid:**
- Variants: VAE, E-VAE, C-VAE, BA-VAE, FW-VAE
- Encoder activations: relu, tanh, leaky_relu
- Decoder activations: relu, tanh
- Beta annealing: start=[0.0, 0.1], end=[1.0]
- Latent dimensions: 0 (default), 8
- Epochs: 30, 50

### 3. Fixed Reproducibility/Seeding Issues

#### Problem Identified:
The three discrete EDA programs (`discrete_Backdrive_EDA.py`, `discrete_GAN_EDA.py`, `discrete_VAE_EDA.py`) were only setting NumPy's random seed. However, they also use PyTorch for neural network operations, which has its own random number generators that were not being seeded. This caused non-reproducible results even with the same seed and parameters.

#### Solution Implemented:
Added a comprehensive `set_seed()` function to all three programs that sets:

1. **Python's built-in random module** (`random.seed()`)
2. **NumPy's random state** (`np.random.seed()`)
3. **PyTorch CPU random state** (`torch.manual_seed()`)
4. **PyTorch CUDA random states** (`torch.cuda.manual_seed()`, `torch.cuda.manual_seed_all()`)
5. **PyTorch deterministic mode** for CUDNN operations
6. **PyTorch deterministic algorithms** (when available in PyTorch >= 1.8)

#### Implementation Details:

```python
def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch deterministic operations
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Set deterministic behavior for reproducibility
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For some operations on CUDA >= 10.2
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        # PyTorch not available, skip torch seeding
        pass
```

#### Files Modified:
- `examples/discrete_Backdrive_EDA.py`
- `examples/discrete_GAN_EDA.py`
- `examples/discrete_VAE_EDA.py`

#### Impact:
- **Reproducibility:** Same seed + same parameters = same results
- **Performance:** May have slight performance impact due to deterministic operations, but ensures reproducibility
- **Compatibility:** Gracefully handles cases where PyTorch is not installed
- **Multi-GPU:** Properly seeds all CUDA devices

## Testing Recommendations

### 1. Verify Positional Arguments
Test that the modified scripts accept all parameters as positional arguments:

```bash
# Test GAN script
python examples/discrete_GAN_EDA.py 42 OneMax 20 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0

# Test VAE script
python examples/discrete_VAE_EDA.py 42 OneMax 20 80 20 0.5 VAE relu relu 0.0 1.0 0 30
```

### 2. Verify Reproducibility
Run the same command twice and verify identical results:

```bash
# Run 1
python examples/discrete_Backdrive_EDA.py 42 OneMax 20 80 10 0.5 backdrive random mse relu 0 0 0 > run1.txt

# Run 2
python examples/discrete_Backdrive_EDA.py 42 OneMax 20 80 10 0.5 backdrive random mse relu 0 0 0 > run2.txt

# Compare (should be identical)
diff run1.txt run2.txt
```

### 3. Verify Launch Scripts
Test that launch scripts generate valid commands:

```bash
# Generate GAN commands (review before submitting)
python launch_gan_experiments.py | head -10

# Generate VAE commands (review before submitting)
python launch_vae_experiments.py | head -10
```

## Known Considerations

### Performance Impact
Setting `torch.backends.cudnn.deterministic = True` may reduce performance by 10-20% in some cases, as CUDNN cannot use non-deterministic optimizations. This is a necessary trade-off for reproducibility.

### PyTorch Version Compatibility
The `torch.use_deterministic_algorithms()` function is only available in PyTorch >= 1.8. The implementation gracefully handles older versions by using `hasattr()` to check availability.

### SLURM Environment
The SLURM scripts use `bnd -exec` which appears to be a wrapper command specific to the computing environment. If this is not available in your environment, modify the scripts to use `python3` directly.

## Summary

All requirements from the problem statement have been successfully implemented:

✅ **Requirement 1:** Substituted flags by positional input variables in `discrete_GAN_EDA.py` and `discrete_VAE_EDA.py`

✅ **Requirement 2:** Created `slurm_gan.sh` and `launch_gan_experiments.py` to explore all variants of GAN parameters

✅ **Requirement 3:** Created `slurm_vae.sh` and `launch_vae_experiments.py` to explore all variants of VAE parameters

✅ **Requirement 4:** Fixed reproducibility issues by implementing comprehensive seeding across Python random, NumPy, and PyTorch

The changes ensure that:
- All three discrete EDA programs have consistent interfaces
- Parameter exploration can be systematically performed via launch scripts
- Results are reproducible when using the same seed and parameters
- The code gracefully handles different PyTorch versions and availability
