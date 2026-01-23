"""
Frequency Balance Mutation for Binary EDAs

Implements a mutation operator that ensures no variable has an excessive
frequency of ones or zeros in the population.

Frequency Balance Mutation:
---------------------------
This mutation operator prevents premature convergence by maintaining
diversity in each variable's distribution. For each variable:

1. Compute frequency of ones and zeros in the population
2. If frequency of ones OR zeros exceeds threshold alpha:
   - Randomly select (1-alpha) * pop_size positions
   - Flip those bits to balance the distribution

Purpose:
--------
- Prevents any variable from becoming too biased toward 0 or 1
- Maintains population diversity throughout evolution
- Helps EDAs explore more of the search space
- Guards against premature convergence

Parameters:
-----------
alpha : float
    Maximum allowed frequency for ones or zeros (e.g., 0.95)
    If alpha=0, no mutation is applied
    If frequency > alpha, flip (1-alpha) portion of positions

Example:
--------
If alpha=0.95 and a variable has 98% ones in population:
- Threshold exceeded: 0.98 > 0.95
- Flip (1-0.95) = 5% of population for that variable
- This reduces the frequency of ones

Implementation:
---------------
For each variable independently:
1. Calculate freq_ones = sum(column) / pop_size
2. Calculate freq_zeros = 1 - freq_ones
3. If max(freq_ones, freq_zeros) > alpha:
   - Select (1-alpha) * pop_size random positions
   - Flip bits at those positions
"""

from typing import Dict, Any
import numpy as np


def frequency_balance_mutation(
    n_vars: int,
    cardinality: np.ndarray,
    population: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Apply frequency balance mutation to a binary population
    
    For each variable, if the frequency of ones or zeros exceeds alpha,
    flip (1-alpha) portion of the population to restore balance.

    Args:
        n_vars: Number of variables
        cardinality: Variable cardinalities (not used, kept for interface compatibility)
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
        # Get the column for this variable
        var_column = new_pop[:, var_idx]
        
        # Calculate frequency of ones
        freq_ones = np.sum(var_column) / n_individuals
        freq_zeros = 1.0 - freq_ones
        
        # Check if either frequency exceeds alpha
        if freq_ones > alpha or freq_zeros > alpha:
            # Calculate how many positions to flip
            n_to_flip = int((1 - alpha) * n_individuals)

           
            # Only flip if n_to_flip > 0
            if n_to_flip == 0:
                continue
            
            # Determine which class is the majority and get positions with that class
            if freq_ones > alpha:
                # Too many ones - flip only positions that have value 1
                majority_positions = np.where(var_column == 1)[0]
            else:
                # Too many zeros - flip only positions that have value 0
                majority_positions = np.where(var_column == 0)[0]
            
            # Ensure we don't flip more positions than available in the majority class
            n_to_flip = min(n_to_flip, len(majority_positions))
            
            # Randomly select positions from the majority class to flip
            flip_indices = np.random.choice(majority_positions, n_to_flip, replace=False)
            
            # Flip the selected positions for this variable
            new_pop[flip_indices, var_idx] = 1 - new_pop[flip_indices, var_idx]

    return new_pop
