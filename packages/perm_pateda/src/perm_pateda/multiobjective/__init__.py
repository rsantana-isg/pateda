"""
Multi-objective optimization for permutation problems.

Provides:
- Scalarization functions (weighted sum, Tchebycheff)
- Pareto dominance and archive management
- MEDA/D-MK: Decomposition-based EDA with Kernels of Mallows Models (Cayley)
- MEDA/D-KENDALL: Decomposition-based EDA with Mallows Kendall
- MEDA/D-ULAM: Decomposition-based EDA with Mallows Ulam
- MEDA/D-GMKENDALL: Decomposition-based EDA with Generalized Mallows Kendall
- MEDA/D-GMCAYLEY: Decomposition-based EDA with Generalized Mallows Cayley
"""

from perm_pateda.multiobjective.scalarization import weighted_sum, tchebycheff
from perm_pateda.multiobjective.pareto import dominates, pareto_front, ParetoArchive
from perm_pateda.multiobjective.meda_d_mk import (
    MEDA_D_MK,
    MEDA_D_KENDALL,
    MEDA_D_ULAM,
    MEDA_D_GMKENDALL,
    MEDA_D_GMCAYLEY,
)

__all__ = [
    "weighted_sum", "tchebycheff",
    "dominates", "pareto_front", "ParetoArchive",
    "MEDA_D_MK",
    "MEDA_D_KENDALL",
    "MEDA_D_ULAM",
    "MEDA_D_GMKENDALL",
    "MEDA_D_GMCAYLEY",
]