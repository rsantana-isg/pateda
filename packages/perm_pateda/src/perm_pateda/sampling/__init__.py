"""Permutation sampling methods.

Histogram samplers (edge / node) and distance-based samplers for the Mallows
and Generalized Mallows models under Kendall's-tau and Cayley distances.
"""

from perm_pateda.sampling.histogram import SampleEHM, SampleNHM, sample_ehm, sample_nhm
from perm_pateda.sampling.mallows import (
    SampleMallowsKendall,
    SampleMallowsCayley,
    SampleGeneralizedMallowsKendall,
    SampleGeneralizedMallowsCayley,
    sample_mallows_kendall,
    sample_mallows_cayley,
)

__all__ = [
    "SampleEHM",
    "SampleNHM",
    "sample_ehm",
    "sample_nhm",
    "SampleMallowsKendall",
    "SampleMallowsCayley",
    "SampleGeneralizedMallowsKendall",
    "SampleGeneralizedMallowsCayley",
    "sample_mallows_kendall",
    "sample_mallows_cayley",
]
