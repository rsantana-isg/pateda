"""
Frequency Balance Mutation for Multi-value EDAs (cardinality > 2)

Implements a mutation operator that ensures no variable has an excessive
frequency of any particular value in the population.

Frequency Balance Mutation (Multi-value):
-----------------------------------------
This mutation operator prevents premature convergence by maintaining
diversity in each variable's distribution. For each variable:

1. Compute frequency of each value (0, 1, ..., c-1) in the population
2. For any value whose frequency exceeds threshold alpha:
   - Randomly select (1-alpha) * pop_size positions that have that value
   - Replace those values with a randomly chosen different value
     from the remaining c-1 values

Purpose:
--------
- Prevents any variable from becoming too biased toward a single value
- Maintains population diversity throughout evolution
- Helps EDAs explore more of the search space
- Guards against premature convergence
- Generalizes the binary frequency balance mutation to cardinality c > 2

Parameters:
-----------
alpha : float
    Maximum allowed frequency for any single value (e.g., 0.95)
    If alpha=0, no mutation is applied
    If frequency of value v > alpha, replace (1-alpha) portion of
    positions carrying value v with random other values

Example:
--------
If alpha=0.95 and a variable has value 2 appearing in 98% of individuals
for a c=4 problem:
- Threshold exceeded: 0.98 > 0.95
- Select (1-0.95) = 5% of positions carrying value 2
- Replace each with a randomly chosen value from {0, 1, 3}
- This reduces the frequency of value 2

Implementation:
---------------
For each variable independently:
1. For each value v in {0, ..., c-1}:
   a. Calculate freq_v = count(column == v) / pop_size
   b. If freq_v > alpha:
      - n_to_replace = min(int((1-alpha) * pop_size), count(column == v))
      - Select n_to_replace random positions carrying value v
      - Replace each with a random value from {0, ..., c-1} without {v}
"""

from typing import Dict, Any
import numpy as np


def frequency_balance_multivalue_mutation(
    n_vars: int,
    cardinality: np.ndarray,
    population: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Apply frequency balance mutation to a multi-value population (cardinality > 2)

    For each variable, if the frequency of any single value exceeds alpha,
    replace (1-alpha) portion of those positions with randomly chosen
    different values to restore balance.

    Args:
        n_vars: Number of variables
        cardinality: Variable cardinalities array (one entry per variable)
        population: Population to mutate (n_individuals, n_vars)
        params: Dictionary with mutation parameters
               - 'alpha': Maximum allowed frequency threshold (required)
                         If alpha=0, no mutation is applied

    Returns:
        Mutated population (n_individuals, n_vars)

    Raises:
        ValueError: If alpha is not provided or is invalid
    """
    if "alpha" not in params:
        raise ValueError("alpha is required in params")

    alpha = params["alpha"]

    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # If alpha is 0, no mutation is applied
    if alpha == 0:
        return population.copy()

    # Create a copy to avoid modifying the input
    new_pop = population.copy()

    n_individuals = population.shape[0]

    # Process each variable independently
    for var_idx in range(n_vars):
        c = int(cardinality[var_idx])
        var_column = new_pop[:, var_idx]

        # For each possible value, check if its frequency exceeds alpha
        for val in range(c):
            positions_with_val = np.where(var_column == val)[0]
            freq_val = len(positions_with_val) / n_individuals

            if freq_val > alpha:
                # Calculate how many positions to replace
                n_to_replace = int((1 - alpha) * n_individuals)
                n_to_replace = min(n_to_replace, len(positions_with_val))

                if n_to_replace == 0:
                    continue

                # Randomly select positions to replace
                replace_indices = np.random.choice(
                    positions_with_val, n_to_replace, replace=False
                )

                # Other available values (all values except val)
                other_values = np.array([v for v in range(c) if v != val])

                # Replace each selected position with a random other value
                new_pop[replace_indices, var_idx] = np.random.choice(
                    other_values, n_to_replace
                )

                # Refresh column view after modifications
                var_column = new_pop[:, var_idx]

    return new_pop
