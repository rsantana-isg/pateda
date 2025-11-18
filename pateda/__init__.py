"""
PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A Python port of MATEDA-3.0
"""

__version__ = "0.1.0"
__author__ = "Roberto Santana (original MATEDA), Claude (Python port)"

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.core.models import (
    Model,
    FactorizedModel,
    TreeModel,
    BayesianNetworkModel,
    GaussianModel,
)

__all__ = [
    "EDA",
    "EDAComponents",
    "Model",
    "FactorizedModel",
    "TreeModel",
    "BayesianNetworkModel",
    "GaussianModel",
]
