"""Discrete non-binary multi-objective benchmark problems.

Multi-objective (vector-valued) functions over integer / categorical
representations.  Each callable returns an objective vector to be maximised,
matching pateda's ``maximize=True`` convention.

* Braid bi-objective -- jointly minimise the quantum-gate approximation error
  and the number of matrices in the braid (see
  :mod:`pateda.functions.discrete_non_binary.problems.braid`).
"""

from pateda.functions.discrete_non_binary.multiobjective.braid_biobjective import (
    braid_raw_objectives,
    braid_biobjective,
    create_braid_biobjective_function,
    make_fibonacci_braid_biobjective,
    make_icosahedral_braid_biobjective,
)

__all__ = [
    "braid_raw_objectives",
    "braid_biobjective",
    "create_braid_biobjective_function",
    "make_fibonacci_braid_biobjective",
    "make_icosahedral_braid_biobjective",
]
