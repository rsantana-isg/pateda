"""Permutation model learning methods.

Histogram models (edge / node) and distance-based exponential models
(Mallows and Generalized Mallows under Kendall's-tau and Cayley distances).
"""

from perm_pateda.learning.histogram import LearnEHM, LearnNHM, learn_ehm, learn_nhm
from perm_pateda.learning.mallows import (
    LearnMallowsKendall,
    LearnMallowsCayley,
    LearnMallowsUlam,
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
    learn_mallows_kendall,
    learn_mallows_cayley,
    learn_mallows_ulam,
)

from perm_pateda.learning.plackett_luce import LearnPlackettLuce
from perm_pateda.learning.mixture_plackett_luce import LearnPlackettLuceMixture
from perm_pateda.learning.hamming_kmm import LearnHammingKMM

__all__ = [
    "LearnEHM",
    "LearnNHM",
    "learn_ehm",
    "learn_nhm",
    "LearnMallowsKendall",
    "LearnMallowsCayley",
    "LearnMallowsUlam",
    "LearnGeneralizedMallowsKendall",
    "LearnGeneralizedMallowsCayley",
    "learn_mallows_kendall",
    "learn_mallos_cayley",
    "learn_mallows_ulam",
    "LearnPlackettLuce"
    "LearnPlackettLuceMixture",
    "LearnHammingKMM",
]
