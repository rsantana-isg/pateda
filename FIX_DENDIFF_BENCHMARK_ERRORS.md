# Fix for Dendiff Benchmark Errors

## Problem

When running `benchmark_enhanced_dendiff_distributions.py`, the following error occurred:

```
ValueError: Fewer non-zero entries in p than size
```

This error appeared in the Boltzmann distribution selection (and potentially in rank-based selection) when calling `np.random.choice()` with `replace=False`.

## Root Cause

The issue occurs when:

1. **Boltzmann Selection**: When computing Boltzmann probabilities using `exp(-fitness_normalized / temperature)`, if the fitness range is large relative to the temperature, many solutions get probabilities that are effectively zero due to numerical precision limits.

2. **Rank-based Selection**: Similarly, when computing rank-based probabilities with high selection pressure, solutions with poor ranks may get zero or negative weights (clipped to 0).

3. **NumPy Constraint**: `np.random.choice()` with `replace=False` requires that the number of non-zero probabilities is at least as large as the `size` parameter (number of samples to select).

Example scenario:
- `n_initial=1000` solutions generated
- `n_selected=500` solutions to be selected
- But only 300 solutions have non-zero probabilities after Boltzmann/rank-based weighting
- NumPy raises: "Fewer non-zero entries in p than size"

## Solution

Modified both `generate_boltzmann_distribution()` and `generate_rank_based_distribution()` functions in `benchmark_dendiff_distributions.py` to:

1. **Count non-zero probabilities** before attempting to sample:
   ```python
   non_zero_count = np.sum(probabilities > 0)
   ```

2. **Adjust sample size** if necessary:
   ```python
   if non_zero_count < n_selected:
       print(f"Warning: Only {non_zero_count} non-zero probabilities, "
             f"adjusting n_selected from {n_selected} to {non_zero_count}")
       n_selected_actual = non_zero_count
   else:
       n_selected_actual = n_selected
   ```

3. **Use adjusted size** in sampling:
   ```python
   selected_indices = np.random.choice(
       n_initial,
       size=n_selected_actual,
       replace=False,
       p=probabilities
   )
   ```

4. **Update metadata** to reflect actual number selected:
   ```python
   metadata = {
       ...
       'n_selected': n_selected_actual,
       'selection_ratio': n_selected_actual / n_initial,
       ...
   }
   ```

## Files Modified

- `benchmark_dendiff_distributions.py`:
  - `generate_boltzmann_distribution()` function (lines ~433-476)
  - `generate_rank_based_distribution()` function (lines ~560-603)

## Changes Summary

### generate_boltzmann_distribution()
- Added check for non-zero probability count
- Adjusts `n_selected` to match available non-zero probabilities
- Prints warning when adjustment is needed
- Updates metadata with actual `n_selected`

### generate_rank_based_distribution()
- Same changes as Boltzmann distribution
- Handles cases where selection pressure causes many zero-probability solutions

## Testing

The fix ensures that:
1. No `ValueError` is raised when selecting solutions
2. The actual number of selected solutions is correctly tracked in metadata
3. The benchmark can run successfully with various temperature and selection pressure parameters
4. A warning is printed when the selection size is adjusted, making it transparent to users

## Impact

- **Backward compatible**: Functions still return the same data types
- **Transparent**: Warnings inform users when adjustments occur
- **Robust**: Handles edge cases gracefully without crashing
- **Accurate**: Metadata correctly reflects actual number of selected solutions

## Example Warning Output

```
Warning: Only 347 non-zero probabilities, adjusting n_selected from 500 to 347
```

This allows the benchmark to continue running while informing the user about the adjustment.
