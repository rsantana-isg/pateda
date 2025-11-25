# Dendiff Parameter Analysis for Rank-Based Selection - GNBG Benchmark

## Overview

`benchmark_dendiff_parameter_analysis_gnbg.py` is a variant of the parameter analysis script that uses the **24 GNBG (Generalized Numerical Benchmark Generator) functions** instead of the standard 12 benchmark functions.

This provides a comprehensive evaluation of dendiff parameter effects across a well-established, diverse benchmark suite used in continuous optimization research.

## GNBG Benchmark Suite

The GNBG benchmark consists of 24 problem instances with systematically varying characteristics:

### Problem Categories

| Category | Functions | Characteristics |
|----------|-----------|----------------|
| **Unimodal Separable** | f1-f2 | Easiest problems, single optimum, independent variables |
| **Unimodal Non-separable** | f3-f4 | Single optimum, variable interactions |
| **Multimodal Separable** | f5-f6 | Multiple optima, independent variables |
| **Multimodal Non-separable** | f7-f12 | Multiple optima, variable interactions |
| **Highly Multimodal Separable** | f13-f14 | Many optima, independent variables |
| **Highly Multimodal Non-separable** | f15-f20 | Many optima, strong interactions |
| **Hybrid Compositions** | f21-f24 | Combined function types, most complex |

### Key Differences from Standard Benchmark

| Aspect | Standard Benchmark | GNBG Benchmark |
|--------|-------------------|----------------|
| **Number of Functions** | 12 functions | 24 functions |
| **Source** | Classic test functions + additions | Systematic generator |
| **Shift Handling** | External shift parameter | Built-in shift |
| **Categorization** | Ad-hoc grouping | Systematic categorization |
| **Problem Diversity** | Manually selected | Configurable generation |
| **Validation** | Individual function validation | Suite-wide validation |

## Script Structure

The GNBG variant is **identical** to the original `benchmark_dendiff_parameter_analysis.py` except for:

### Changes:
1. **Objective Functions**: Uses 24 GNBG functions instead of 12 standard functions
2. **Function Loading**: Loads functions from `.mat` files using GNBG_class
3. **No Shift Parameter**: GNBG has built-in shifts, so `use_shift=False`
4. **GNBG-Specific Descriptions**: Problem categorization based on GNBG paper

### Unchanged:
- All analysis functions (timesteps, epochs, architecture)
- Statistical metrics (JS Div, KL Div, Signed Diff, Std Ratio)
- Rank-based selection with SP=2.0
- Parameter ranges and test configurations
- Output format and recommendations

## Requirements

### GNBG Instance Files

The script requires GNBG problem instance files (`.mat` format) located at:
```
pateda/functions/GNBG_Instances.Python-main/f1.mat through f24.mat
```

### Dependencies

```python
- numpy
- scipy
- torch (for dendiff)
- All dependencies from benchmark_enhanced_dendiff_distributions.py
```

### GNBG Class

Requires `GNBG_class.py` from `enhanced_edas/` directory.

## Usage

```bash
python3 benchmark_dendiff_parameter_analysis_gnbg.py
```

## Output Structure

Identical to the original parameter analysis:

### For Each Analysis:

1. **Per-Function Results**
   - Detailed table for each of 24 GNBG functions
   - All 4 metrics for each configuration

2. **Summary Statistics**
   - Average metrics across all 24 functions
   - Identifies general trends

3. **Final Recommendations**
   - GNBG category-specific parameter guidelines
   - Success indicators

## Computational Requirements

**Significantly longer than standard benchmark:**

- 24 functions (vs 12) = 2× more evaluations
- Each GNBG function evaluation may be more expensive
- **Estimated runtime**: 60-180 minutes (vs 30-90 minutes for standard)

### Recommendations for Faster Testing:

```python
# Test subset of GNBG functions
objectives_to_analyze = [
    'gnbg_f1',   # Unimodal separable
    'gnbg_f3',   # Unimodal non-separable
    'gnbg_f5',   # Multimodal separable
    'gnbg_f10',  # Multimodal non-separable
    'gnbg_f15',  # Highly multimodal non-separable
    'gnbg_f21'   # Hybrid composition
]
```

## Expected Findings

### By Problem Category:

**Unimodal Separable (f1-f2)**:
- Lowest divergence values
- Small network [128,64] sufficient
- Fewer timesteps (300) adequate

**Multimodal Non-separable (f7-f12)**:
- Moderate divergence
- Medium network [256,128] recommended
- More timesteps (400-500) beneficial

**Highly Multimodal (f13-f20)**:
- Highest divergence values
- Large/deep networks needed
- Maximum timesteps (500) required

**Hybrid Compositions (f21-f24)**:
- Very high divergence
- Largest networks [512,256] or [256,128,64]
- Maximum epochs and timesteps

### Example Output:

```
ANALYSIS 1: EFFECT OF TIMESTEPS (Rank-based Selection, SP=2.0)
================================================================================

Objective: GNBG_F1
--------------------------------------------------------------------------------
Timesteps    JS Div       KL Div (MAT)    Signed Diff     Std Ratio
--------------------------------------------------------------------------------
100          0.3245       0.4123          +0.023456       1.0234
200          0.2876       0.3456          -0.012345       1.0456
300          0.2534       0.2987          -0.034567       1.0678
400          0.2412       0.2756          -0.045678       1.0789
500          0.2345       0.2645          -0.056789       1.0912

...

TIMESTEPS SUMMARY: Average Across All GNBG Functions
================================================================================
Timesteps    Avg JS Div      Avg KL Div      Avg |Signed Diff|   Avg Std Ratio
--------------------------------------------------------------------------------
100          0.4567          0.5678          0.123456            1.0123
200          0.3876          0.4567          0.089012            1.0345
300          0.3234          0.3789          0.056789            1.0567
400          0.2987          0.3456          0.034567            1.0678
500          0.2876          0.3234          0.023456            1.0789
```

## Recommended Configurations

Based on GNBG category:

### Unimodal Problems (f1-f4):
```python
dendiff_params = {
    'epochs': 60,
    'n_timesteps': 300,
    'hidden_dims': [128, 64]
}
```

### Multimodal Problems (f5-f12):
```python
dendiff_params = {
    'epochs': 80,
    'n_timesteps': 400,
    'hidden_dims': [256, 128]
}
```

### Highly Multimodal Problems (f13-f20):
```python
dendiff_params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [256, 128, 64]
}
```

### Hybrid Compositions (f21-f24):
```python
dendiff_params = {
    'epochs': 100,
    'n_timesteps': 500,
    'hidden_dims': [512, 256]  # or [256, 128, 64]
}
```

## Comparison with Standard Benchmark

### Advantages of GNBG:

1. **More Functions**: 24 vs 12 = better statistical significance
2. **Systematic Coverage**: Explicitly covers problem types
3. **Standardized**: Well-known benchmark in optimization community
4. **Diverse**: More problem characteristics tested
5. **Validated**: Extensively used in research

### Advantages of Standard:

1. **Faster**: Half the functions = quicker analysis
2. **Simpler**: Classic functions, easier to understand
3. **Lightweight**: No .mat file dependencies
4. **Flexible**: Easy to add custom functions

## Use Cases

### Use GNBG Benchmark When:
- Need comprehensive, standardized evaluation
- Publishing research requiring established benchmarks
- Want to test across systematically varied problem characteristics
- Need comparison with other papers using GNBG

### Use Standard Benchmark When:
- Quick parameter tuning for specific problem types
- Prototyping and development
- Limited computational budget
- Testing specific function characteristics

## Integration with Research

This GNBG variant enables:

- **Reproducible Research**: Standard benchmark suite
- **Fair Comparison**: With other GNBG-based studies
- **Publication**: Results suitable for academic papers
- **Comprehensive**: Covers wide problem space

## References

When using this GNBG variant, please cite:

```bibtex
@article{yazdani2023gnbg,
  title={GNBG: A Generalized and Configurable Benchmark Generator for Continuous Numerical Optimization},
  author={Yazdani, Danial and Omidvar, Mohammad Nabi and Yazdani, Donya and Deb, Kalyanmoy and Gandomi, Amir H},
  journal={arXiv preprint arXiv:2312.07083},
  year={2023}
}

@article{gandomi2023gnbg,
  title={GNBG-Generated Test Suite for Box-Constrained Numerical Global Optimization},
  author={Gandomi, Amir H and Yazdani, Danial and Omidvar, Mohammad Nabi and Deb, Kalyanmoy},
  journal={arXiv preprint arXiv:2312.07034},
  year={2023}
}
```

## Future Extensions

Potential additions:
- Higher dimensional variants (30D, 50D GNBG instances)
- Cross-dimensional analysis
- Comparison of GNBG vs standard functions
- Correlation analysis between problem features and dendiff performance
- Meta-learning for automatic parameter selection based on GNBG category

## Files

- `benchmark_dendiff_parameter_analysis_gnbg.py` - Main GNBG analysis script
- `PARAMETER_ANALYSIS_GNBG_README.md` - This documentation
- `pateda/functions/GNBG_Instances.Python-main/*.mat` - GNBG problem instances (24 files)
- `enhanced_edas/GNBG_class.py` - GNBG class implementation
