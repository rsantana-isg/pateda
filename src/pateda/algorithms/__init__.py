"""
Plug-and-play EDA algorithm wrappers.

Discrete algorithms::

    from pateda.algorithms import UMDA, BMDA, TreeEDA, ...

Continuous algorithms::

    from pateda.algorithms import GaussianUMDA, GaussianEDA, ...

Permutation algorithms live in the separate ``perm_pateda`` package.
"""

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
    # Discrete
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
    # Continuous
    "GaussianUMDA",
    "GaussianEDA",
    "MixtureGaussianEDA",
    "GMRFEDA",
    "VineEDA",
    "CVineEDA",
    "RVineEDA",
]
