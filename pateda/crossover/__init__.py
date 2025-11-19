"""
Crossover operators for EDAs

This module implements non-probabilistic crossover operators that follow
the EDA paradigm of separating learning and sampling phases.

Crossover in EDAs:
------------------
Unlike traditional genetic algorithms where crossover directly combines parent
solutions, EDAs split crossover into two phases:

1. Learning Phase: Determines crossover parameters (mating pools, crossover points, etc.)
   from the selected population. This creates a "model" that specifies how crossover
   will be applied.

2. Sampling Phase: Applies the crossover operator according to the learned model to
   generate new offspring solutions.

This separation allows crossover to be treated uniformly with other EDA components
and enables more sophisticated crossover strategies.

Available Crossover Operators:
------------------------------
- Two-Point Crossover: Classic GA operator with two crossover points
- Block Crossover: Exchanges complete blocks defined by symmetry/equivalence classes
- Transposition: Moves a segment of the genome to a different location (toroidal)

Each operator consists of:
- LearnXXX: Learning method that creates the crossover model
- SampleXXX: Sampling method that applies the crossover

References:
-----------
- MATEDA-2.0 User Guide, Section 4.5: "Non probabilistic models"
- Santana, R., McKay, B., & Lozano, M. (2013). "Symmetry in evolutionary and
  estimation of distribution algorithms." CEC 2013.
"""

from pateda.crossover.two_point import LearnTwoPointCrossover, SampleTwoPointCrossover
from pateda.crossover.transposition import LearnTransposition, SampleTransposition
from pateda.crossover.block import LearnBlockCrossover, SampleBlockCrossover

__all__ = [
    "LearnTwoPointCrossover",
    "SampleTwoPointCrossover",
    "LearnTransposition",
    "SampleTransposition",
    "LearnBlockCrossover",
    "SampleBlockCrossover",
]
