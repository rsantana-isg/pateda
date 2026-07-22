"""
PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A Python port of MATEDA-3.0
"""

__version__ = "0.2.0"
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

# Plug-and-play algorithm wrappers
from pateda.algorithms.discrete import (
    UMDA,
    BMDA,
    TreeEDA,
    TreeEDAR,
    MIMIC,
    PBIL,
    EBNA,
    BOA,
    AffEDA,
    MKEDA,
    MTED,
    MNFDA,
    MNFDAR,
    MNFDAG,
    MNFDAGR,
    MOA,
    CUMDA,
    CFDA,
    FDA,
)

from pateda.algorithms.continuous import (
    GaussianUMDA,
    GaussianEDA,
    MixtureGaussianEDA,
    GMRFEDA,
    VineEDA,
    CVineEDA,
    RVineEDA,
)

__all__ = [
    # Core
    "EDA",
    "EDAComponents",
    "Model",
    "FactorizedModel",
    "TreeModel",
    "BayesianNetworkModel",
    "GaussianModel",
    # Discrete algorithms
    "UMDA",
    "BMDA",
    "TreeEDA",
    "TreeEDAR",
    "MIMIC",
    "PBIL",
    "EBNA",
    "BOA",
    "AffEDA",
    "MKEDA",
    "MTED",
    "MNFDA",
    "MNFDAR",
    "MNFDAG",
    "MNFDAGR",
    "MOA",
    "CUMDA",
    "CFDA",
    "FDA",
    # Continuous algorithms
    "GaussianUMDA",
    "GaussianEDA",
    "MixtureGaussianEDA",
    "GMRFEDA",
    "VineEDA",
    "CVineEDA",
    "RVineEDA",
]
