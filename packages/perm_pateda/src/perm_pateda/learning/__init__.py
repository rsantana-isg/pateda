"""Permutation model learning methods.

Histogram models (edge / node) and distance-based exponential models
(Mallows and Generalized Mallows under Kendall's-tau and Cayley distances).
"""

from perm_pateda.learning.histogram import LearnEHM, LearnNHM, learn_ehm, learn_nhm
from perm_pateda.learning.mallows import (
    LearnMallowsKendall,
    LearnMallowsCayley,
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
    learn_mallows_kendall,
)

__all__ = [
    "LearnEHM",
    "LearnNHM",
    "learn_ehm",
    "learn_nhm",
    "LearnMallowsKendall",
    "LearnMallowsCayley",
    "LearnGeneralizedMallowsKendall",
    "LearnGeneralizedMallowsCayley",
    "learn_mallows_kendall",
]
