"""Probabilistic model learning methods"""

from pateda.learning.fda import LearnFDA
from pateda.learning.umda import LearnUMDA
from pateda.learning.bmda import LearnBMDA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.tree import LearnTreeModel
from pateda.learning.affinity import (
    LearnAffinityFactorization,
    LearnAffinityFactorizationElim,
)

# Gaussian learning functions for continuous optimization
from pateda.learning.gaussian import (
    learn_gaussian_univariate,
    learn_gaussian_full,
    learn_mixture_gaussian_univariate,
    learn_mixture_gaussian_full,
)

# Permutation-based learning functions
from pateda.learning.mallows import LearnMallowsKendall, learn_mallows_kendall
from pateda.learning.histogram import LearnEHM, LearnNHM, learn_ehm, learn_nhm

__all__ = [
    "LearnFDA",
    "LearnUMDA",
    "LearnBMDA",
    "LearnEBNA",
    "LearnBOA",
    "LearnMarkovChain",
    "LearnMixtureTrees",
    "LearnTreeModel",
    "LearnAffinityFactorization",
    "LearnAffinityFactorizationElim",
    "learn_gaussian_univariate",
    "learn_gaussian_full",
    "learn_mixture_gaussian_univariate",
    "learn_mixture_gaussian_full",
    "LearnMallowsKendall",
    "learn_mallows_kendall",
    "LearnEHM",
    "LearnNHM",
    "learn_ehm",
    "learn_nhm",
]
