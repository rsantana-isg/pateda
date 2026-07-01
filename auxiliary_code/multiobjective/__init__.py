"""
Multi-objective optimization for permutation problems.

Provides:
- Scalarization functions (weighted sum, Tchebycheff)
- Pareto dominance and archive management
- MOEA/D framework for permutation-based multi-objective optimization
- Fourier-enhanced multi-objective optimization (FPD-MOEAD)
- MEDA/D-MK: Decomposition-based EDA with Kernels of Mallows Models
- MEDA/D-F: Decomposition-based EDA with Fourier Decomposition Models
"""

from src.multiobjective.scalarization import weighted_sum, tchebycheff
from src.multiobjective.pareto import dominates, pareto_front, ParetoArchive
from src.multiobjective.moead import MOEAD
from src.multiobjective.fourier_moead import FourierMOEAD
from src.multiobjective.meda_d_mk import MEDA_D_MK
from src.multiobjective.meda_d_f import MEDA_D_F

__all__ = [
    "weighted_sum", "tchebycheff",
    "dominates", "pareto_front", "ParetoArchive",
    "MOEAD",
    "FourierMOEAD",
    "MEDA_D_MK",
    "MEDA_D_F",
]
