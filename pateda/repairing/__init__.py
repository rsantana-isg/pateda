"""
Repairing module for constraint handling in PATEDA.

This module provides various constraint repair mechanisms for both discrete
and continuous optimization problems. Repairing methods are used to modify
solutions that violate constraints, making them feasible.

Available repair methods:
- Unitation repairing: For binary problems with cardinality constraints
- Trigonometric repairing: For continuous problems with periodic bounds [0, 2π]
- Bounds repairing: For continuous problems with box constraints

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

from pateda.repairing.unitation import unitation_repairing
from pateda.repairing.unitation_method import UnitationRepairing
from pateda.repairing.trigonometric import trigonometric_repairing
from pateda.repairing.bounds import (
    set_in_bounds_repairing,
    set_within_bounds_repairing,
)

__all__ = [
    "unitation_repairing",
    "UnitationRepairing",
    "trigonometric_repairing",
    "set_in_bounds_repairing",
    "set_within_bounds_repairing",
]
