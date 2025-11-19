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
from pateda.learning.pbil import LearnPBIL
from pateda.learning.bsc import LearnBSC
from pateda.learning.mimic import LearnMIMIC

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

# Vine copula learning functions for continuous optimization
from pateda.learning.vine_copula import (
    learn_vine_copula_cvine,
    learn_vine_copula_dvine,
    learn_vine_copula_auto,
)

# VAE learning functions for continuous optimization
from pateda.learning.vae import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
)

# Backdrive learning functions for continuous optimization
from pateda.learning.backdrive import learn_backdrive

# GAN learning functions for continuous optimization
from pateda.learning.gan import learn_gan

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
    "LearnPBIL",
    "LearnBSC",
    "LearnMIMIC",
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
    "learn_vine_copula_cvine",
    "learn_vine_copula_dvine",
    "learn_vine_copula_auto",
    "learn_vae",
    "learn_extended_vae",
    "learn_conditional_extended_vae",
    "learn_backdrive",
    "learn_gan",
]
