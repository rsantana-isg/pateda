"""
Bit-Flip Mutation for Binary EDAs

Implements the classic bit-flip mutation operator for binary variables.

Bit-Flip Mutation:
------------------
A mutation operator for binary (0/1) variables that flips each bit with
a specified probability. For each variable in each individual:
- With probability p: flip the bit (0→1 or 1→0)
- With probability (1-p): leave the bit unchanged

This is the standard mutation operator used in genetic algorithms for
binary-encoded problems.

Mutation Probability:
---------------------
The mutation probability p is typically set based on problem size:
- Common choice: p = 1/n (where n = number of variables)
- This gives an expected ~1 mutation per individual
- Can be tuned: higher p increases exploration, lower p refines search

Applications in EDAs:
---------------------
In EDAs, mutation serves several purposes:
1. Maintains population diversity
2. Allows exploration of regions not covered by the model
3. Prevents premature convergence
4. Compensates for model inaccuracies

Mutation can be applied:
- After crossover operations
- After probabilistic sampling
- As part of hybrid EDA-GA approaches

Implementation:
---------------
This function generates a random mask where each element is True with
probability p, then flips the bits at those positions.

Equivalent to MATEDA's BitFlipMutation.m

References:
-----------
- Back, T. (1993). "Optimal mutation rates in genetic search." ICGA 1993.
- Mühlenbein, H. (1992). "How genetic algorithms really work: Mutation and
  hillclimbing." PPSN 1992.
- MATEDA-2.0 User Guide
- Last MATLAB version: 12/21/2020. Roberto Santana (roberto.santana@ehu.es)
"""

from typing import Dict, Any
import numpy as np


def bit_flip_mutation(
    n_vars: int,
    cardinality: np.ndarray,
    population: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Apply bit-flip mutation to a binary population

    Args:
        n_vars: Number of variables
        cardinality: Variable cardinalities (not used, kept for interface compatibility)
        population: Population to mutate (n_individuals, n_vars)
        params: Dictionary with mutation parameters
               - 'mutation_prob': Probability of flipping each bit (required)

    Returns:
        Mutated population (n_individuals, n_vars)

    Raises:
        ValueError: If mutation_prob is not provided or is invalid
    """
    if "mutation_prob" not in params:
        raise ValueError("mutation_prob is required in params")

    mutation_prob = params["mutation_prob"]

    if not 0 <= mutation_prob <= 1:
        raise ValueError(f"mutation_prob must be in [0, 1], got {mutation_prob}")

    # Create a copy to avoid modifying the input
    new_pop = population.copy()

    n_individuals = population.shape[0]

    # Generate mutation mask: True where mutations should occur
    mutation_mask = np.random.rand(n_individuals, n_vars) < mutation_prob

    # Flip bits where mask is True
    # For binary variables: flip means 1 - value
    new_pop[mutation_mask] = 1 - population[mutation_mask]

    return new_pop
