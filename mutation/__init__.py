"""
Mutation operators for EDAs

This module implements mutation operators that can be applied after crossover
or sampling operations in EDAs.

Mutation in EDAs:
-----------------
While traditional EDAs rely primarily on probabilistic sampling, hybrid approaches
can incorporate mutation operators to maintain diversity and improve exploration.
Mutation can be applied:
- After crossover operations
- After probabilistic sampling
- As a standalone variation operator

Available Mutation Operators:
------------------------------
- Bit-Flip Mutation: Flips binary values with a specified probability

References:
-----------
- MATEDA-2.0 User Guide
- Larrañaga, P., & Lozano, J. A. (Eds.). (2002). "Estimation of Distribution
  Algorithms: A New Tool for Evolutionary Computation."
"""

from pateda.mutation.bitflip import bit_flip_mutation

__all__ = [
    "bit_flip_mutation",
]
