"""
PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A Python port of MATEDA-3.0
"""

__version__ = "0.2.5"
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
    EBNA_BIC,
    EBNA_K2,
    EBNA_PC,
    BOA,
    HBOA,
    HBOA_Light_A1,
    HBOA_Light_A2,
    HBOA_Light_A3,
    HBOA_Light_A4,
    HBOA_Light_A5,
    LFDA,
    SARTRE_EDA,
    BINOTEARS_EDA,
    PCBN_EDA,
    HSARTRE_EDA,
    HBINOTEARS_EDA,
    AffEDA,
    AffEDASparse,
    MKEDA,
    MTED,
    MNFDA,
    MNFDAR,
    MNFDASparse,
    MNFDAS,
    MNFDAP,
    MNFDAF,
    MNFDASSparse,
    MNFDAG,
    MNFDAGR,
    MNEDAG,
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
    "EBNA_BIC",
    "EBNA_K2",
    "EBNA_PC",
    "BOA",
    "HBOA",
    "HBOA_Light_A1",
    "HBOA_Light_A2",
    "HBOA_Light_A3",
    "HBOA_Light_A4",
    "HBOA_Light_A5",
    "LFDA",
    "SARTRE_EDA",
    "BINOTEARS_EDA",
    "PCBN_EDA",
    "HSARTRE_EDA",
    "HBINOTEARS_EDA",
    "AffEDA",
    "AffEDASparse",
    "MKEDA",
    "MTED",
    "MNFDA",
    "MNFDAR",
    "MNFDASparse",
    "MNFDAS",
    "MNFDAP",
    "MNFDAF",
    "MNFDASSparse",
    "MNFDAG",
    "MNFDAGR",
    "MNEDAG",
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
