# Dendiff Parameter Analysis for Rank-Based Selection

## Overview

`benchmark_dendiff_parameter_analysis.py` is a focused extension of the benchmark suite that provides comprehensive analysis of how dendiff parameters affect approximation quality specifically for **Rank-based selection with SP=2.0**.

## Key Features

### 1. **Fixed Distribution Method**
   - **Rank-based Selection with SP=2.0**
   - Provides consistent baseline across all parameter variations
   - Robust across different fitness landscapes

### 2. **Comprehensive Objective Function Coverage**
   - **12 benchmark functions** (exceeds the 10+ requirement)
   - Diverse characteristics:
     - **Unimodal**: sphere, ellipsoid, zakharov, sum_powers
     - **Multimodal**: rastrigin, ackley, griewank, schwefel, levy, styblinski_tang
     - **Nonseparable**: rosenbrock, dixon_price
     - **Various landscapes**: smooth, plate-shaped, valley-shaped, deceptive

### 3. **Four Statistical Metrics**
   - **JS Divergence**: Distributional similarity vs independent reference
   - **KL Divergence (vs MAT)**: Quality of training set approximation
   - **Signed Fitness Difference**: Improvement indicator (negative = success)
   - **Std Ratio**: Variance preservation ratio

### 4. **Three Parameter Analyses**

#### Analysis 1: Effect of Timesteps
- **Range**: 100, 200, 300, 400, 500 timesteps
- **Fixed**: epochs=100, architecture=[128,64]
- **Purpose**: Find optimal diffusion steps for quality vs cost

#### Analysis 2: Effect of Training Epochs
- **Range**: 20, 40, 60, 80, 100 epochs
- **Fixed**: timesteps=500, architecture=[128,64]
- **Purpose**: Determine convergence requirements

#### Analysis 3: Effect of Network Architecture
- **Architectures tested**:
  - `[64, 32]` - Small network
  - `[128, 64]` - Medium network (baseline)
  - `[256, 128]` - Large network
  - `[256, 128, 64]` - Deep network
  - `[512, 256]` - Very large network
- **Fixed**: epochs=100, timesteps=500
- **Purpose**: Understand capacity requirements

## Additional Objective Functions

The script adds 5 new benchmark functions to reach 12 total:

| Function | Type | Characteristics |
|----------|------|----------------|
| Levy | Multimodal | Many local minima |
| Zakharov | Unimodal | Plate-shaped |
| Sum of Powers | Unimodal | Nonseparable |
| Dixon-Price | Unimodal | Nonseparable |
| Styblinski-Tang | Multimodal | Many local minima |

## Usage

```bash
python3 benchmark_dendiff_parameter_analysis.py
```

## Output Structure

### For Each Analysis:

1. **Per-Objective Results**
   - Detailed table showing all 4 metrics for each configuration
   - One table per objective function

2. **Summary Statistics**
   - Average metrics across all objectives
   - Identifies general trends and optimal parameters

3. **Final Recommendations Section**
   - Parameter selection guidelines based on problem type
   - Success indicators for good approximations
   - Rank-based selection insights

## Key Findings Template

After running the analysis, you'll get:

### Timesteps Findings:
- Optimal range identification
- Quality vs computational cost trade-offs
- Minimum viable timesteps for acceptable quality

### Epochs Findings:
- Convergence speed analysis
- Risk of underfitting vs overfitting
- Optimal training duration

### Architecture Findings:
- Capacity requirements for different problem types
- Small vs large network performance
- Deep vs wide network comparisons

### Objective Characteristics:
- Which functions are easiest/hardest for dendiff
- Correlation between function properties and approximation quality
- Recommended configurations for different problem classes

## Interpretation Guide

### Success Indicators:
✅ **Good Approximation**:
- JS Divergence < 0.5
- KL Divergence < 0.5
- |Signed Diff| small
- Std Ratio ≥ 1.0

⚠️ **Warning Signs**:
- KL Divergence > 1.0 (underfitting)
- Large positive Signed Diff (missing good solutions)
- Std Ratio < 0.5 (losing diversity)

### Parameter Selection Strategy:

**Start Conservative**:
1. timesteps=300, epochs=60, architecture=[128,64]
2. Run quick test on your objective function
3. Check metrics

**Scale Up if Needed**:
- High divergence → Increase architecture
- Signed diff issues → Increase epochs
- Still poor → Increase timesteps to 500

## Example Output

```
ANALYSIS 1: EFFECT OF TIMESTEPS (Rank-based Selection, SP=2.0)
================================================================================

Objective: SPHERE
--------------------------------------------------------------------------------
Timesteps    JS Div       KL Div (MAT)    Signed Diff     Std Ratio
--------------------------------------------------------------------------------
100          0.4521       0.5234          +0.123456       0.9876
200          0.3845       0.4123          +0.056789       1.0234
300          0.3234       0.3456          -0.012345       1.0567
400          0.3012       0.3123          -0.034567       1.0789
500          0.2876       0.2987          -0.045678       1.0912

...

TIMESTEPS SUMMARY: Average Across All Objectives
================================================================================
Timesteps    Avg JS Div      Avg KL Div      Avg |Signed Diff|   Avg Std Ratio
--------------------------------------------------------------------------------
100          0.5123          0.6234          0.145678            0.9234
200          0.4234          0.5123          0.098765            1.0123
300          0.3567          0.4234          0.067890            1.0456
400          0.3234          0.3789          0.045678            1.0678
500          0.3012          0.3456          0.034567            1.0834
```

## Recommended Configurations

Based on analysis results:

### Simple Problems (sphere, ellipsoid, zakharov):
```python
dendiff_params = {
    'epochs': 60,
    'n_timesteps': 300,
    'hidden_dims': [128, 64]
}
```

### Moderate Problems (rastrigin, ackley, levy):
```python
dendiff_params = {
    'epochs': 80,
    'n_timesteps': 400,
    'hidden_dims': [256, 128]
}
```

### Complex Problems (schwefel, rosenbrock, dixon_price):
```python
dendiff_params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [256, 128, 64]
}
```

## Integration with Main Benchmark

This script complements `benchmark_enhanced_dendiff_distributions.py` by:
- Focusing specifically on Rank-based selection
- Providing deeper parameter analysis
- Testing more objective functions
- Offering clearer parameter selection guidelines

## Computational Requirements

Approximate runtime with default settings:
- 12 objectives × 16 configurations ≈ 192 evaluations
- Each evaluation: ~10-30 seconds (depending on parameters)
- **Total**: ~30-90 minutes

Recommendations for faster analysis:
- Reduce to 6-8 objectives
- Use smaller timestep increments
- Test fewer architectures

## Future Extensions

Potential additions:
- Beta schedule variations (linear, cosine)
- Sample size effects
- Different selection pressures (SP=1.5, 2.5, 3.0)
- Interaction effects between parameters
- Learning rate variations
