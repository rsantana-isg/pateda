"""Sampling methods"""

from pateda.sampling.fda import SampleFDA
from pateda.sampling.cumda import SampleCUMDA, SampleCUMDARange
from pateda.sampling.cfda import (
    SampleCFDA,
    SampleCFDARange,
    SampleCFDAWeighted,
)
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.markov import SampleMarkovChain, SampleMarkovChainForward
from pateda.sampling.mixture_trees import SampleMixtureTrees, SampleMixtureTreesDirect
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.map_sampling import (
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleHybridMAP,
)

# Gaussian sampling functions for continuous optimization
from pateda.sampling.basic_gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_gaussian_with_diversity_trigger,
)
from pateda.sampling.mixture_gaussian import (
    sample_mixture_gaussian_univariate,
    sample_mixture_gaussian_full,
    sample_mixture_gaussian_em,
)
from pateda.sampling.gmrf_eda import (
    sample_gmrf_eda,
)

# Permutation-based sampling functions
from pateda.sampling.mallows import SampleMallowsKendall, sample_mallows_kendall
from pateda.sampling.histogram import SampleEHM, SampleNHM, sample_ehm, sample_nhm

# Vine copula sampling functions for continuous optimization
from pateda.sampling.vine_copula import (
    sample_vine_copula,
    sample_vine_copula_biased,
    sample_vine_copula_conditional,
)

# VAE sampling functions for continuous optimization (requires PyTorch)
try:
    from pateda.sampling.vae import (
        sample_vae,
        sample_extended_vae,
        sample_conditional_extended_vae,
    )
except ImportError:
    sample_vae = None
    sample_extended_vae = None
    sample_conditional_extended_vae = None

# Backdrive sampling functions for continuous optimization (requires PyTorch)
try:
    from pateda.sampling.backdrive import (
        sample_backdrive,
        sample_backdrive_adaptive,
    )
except ImportError:
    sample_backdrive = None
    sample_backdrive_adaptive = None

# GAN sampling functions for continuous optimization (requires PyTorch)
try:
    from pateda.sampling.gan import sample_gan
except ImportError:
    sample_gan = None

__all__ = [
    "SampleFDA",
    "SampleCUMDA",
    "SampleCUMDARange",
    "SampleCFDA",
    "SampleCFDARange",
    "SampleCFDAWeighted",
    "SampleBayesianNetwork",
    "SampleMarkovChain",
    "SampleMarkovChainForward",
    "SampleMixtureTrees",
    "SampleMixtureTreesDirect",
    "SampleGibbs",
    "SampleInsertMAP",
    "SampleTemplateMAP",
    "SampleHybridMAP",
    "sample_gaussian_univariate",
    "sample_gaussian_full",
    "sample_gaussian_with_diversity_trigger",
    "sample_mixture_gaussian_univariate",
    "sample_mixture_gaussian_full",
    "sample_mixture_gaussian_em",
    "sample_gmrf_eda",
    "SampleMallowsKendall",
    "sample_mallows_kendall",
    "SampleEHM",
    "SampleNHM",
    "sample_ehm",
    "sample_nhm",
    "sample_vine_copula",
    "sample_vine_copula_biased",
    "sample_vine_copula_conditional",
    "sample_vae",
    "sample_extended_vae",
    "sample_conditional_extended_vae",
    "sample_backdrive",
    "sample_backdrive_adaptive",
    "sample_gan",
]
