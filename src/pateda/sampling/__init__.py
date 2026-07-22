"""Sampling methods"""

from pateda.sampling.fda import SampleFDA
from pateda.sampling.int_fda import SampleIntFDA
from pateda.sampling.network_crossover import SampleNetworkCrossover
from pateda.sampling.regularized_markov import SampleRegularizedMarkov
from pateda.sampling.partial import SamplePartialFDA
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
from pateda.sampling.kmap_sampling import (
    SampleInsertKMAP,
    SampleTemplateKMAP,
)
from pateda.sampling.basic_gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_gaussian_with_diversity_trigger,
    SampleGaussianUnivariate,
    SampleGaussianFull,
)
from pateda.sampling.mixture_gaussian import (
    sample_mixture_gaussian_univariate,
    sample_mixture_gaussian_full,
    sample_mixture_gaussian_em,
)
from pateda.sampling.gmrf_eda import sample_gmrf_eda

try:
    from pateda.sampling.vine_copula import (
        sample_vine_copula,
        sample_vine_copula_biased,
        sample_vine_copula_conditional,
    )
except ImportError:
    sample_vine_copula = None
    sample_vine_copula_biased = None
    sample_vine_copula_conditional = None

__all__ = [
    "SampleFDA", "SampleIntFDA", "SampleNetworkCrossover",
    "SampleRegularizedMarkov",
    "SamplePartialFDA", "SampleCUMDA", "SampleCUMDARange",
    "SampleCFDA", "SampleCFDARange", "SampleCFDAWeighted",
    "SampleBayesianNetwork", "SampleMarkovChain", "SampleMarkovChainForward",
    "SampleMixtureTrees", "SampleMixtureTreesDirect", "SampleGibbs",
    "SampleInsertMAP", "SampleTemplateMAP", "SampleHybridMAP",
    "SampleInsertKMAP", "SampleTemplateKMAP",
    "sample_gaussian_univariate", "sample_gaussian_full",
    "sample_gaussian_with_diversity_trigger",
    "SampleGaussianUnivariate", "SampleGaussianFull",
    "sample_mixture_gaussian_univariate", "sample_mixture_gaussian_full",
    "sample_mixture_gaussian_em", "sample_gmrf_eda",
    "sample_vine_copula", "sample_vine_copula_biased",
    "sample_vine_copula_conditional",
]
