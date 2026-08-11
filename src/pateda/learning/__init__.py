"""Probabilistic model learning methods"""

from pateda.learning.fda import LearnFDA
from pateda.learning.umda import LearnUMDA
from pateda.learning.cumda import LearnCUMDA
from pateda.learning.cfda import (
    LearnCFDA,
    create_pairwise_chain_cliques,
    create_block_cliques,
    create_overlapping_windows_cliques,
)
from pateda.learning.bmda import LearnBMDA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.learning.hboa import LearnHBOA
from pateda.learning.lfda import LearnLFDA
from pateda.learning.bn_extra import (
    LearnSARTRE,
    LearnBINOTEARS,
    LearnPCBN,
    LearnHSARTRE,
    LearnHBINOTEARS,
)
from pateda.learning.pada import LearnPADA
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.regularized_markov import LearnRegularizedMarkov
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_m import LearnTreeModelM
from pateda.learning.int_fda import LearnIntFDA
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.mnfda_r import LearnMNFDAR
from pateda.learning.mnfdag_r import LearnMNFDAGR
from pateda.learning.mnedag import LearnMNEDAG
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.moa import LearnMOA
from pateda.learning.interaction_learning import (
    find_matrix_interactions_SAT,
    find_matrix_interactions_ising,
    find_matrix_interactions_nk,
    find_matrix_interactions_additive_decomposable,
    find_matrix_interactions_RNA_design,
)
from pateda.learning.affinity import (
    LearnAffinityFactorization,
    LearnAffinityFactorizationElim,
)
from pateda.learning.affinity_sparse import LearnAffinitySparse
from pateda.learning.pbil import LearnPBIL
from pateda.learning.mimic import LearnMIMIC
from pateda.learning.basic_gaussian import (
    learn_gaussian_univariate,
    learn_gaussian_full,
    LearnGaussianUnivariate,
    LearnGaussianFull,
)
from pateda.learning.mixture_gaussian import (
    learn_mixture_gaussian_univariate,
    learn_mixture_gaussian_full,
    learn_mixture_gaussian_em,
)
from pateda.learning.gmrf_eda import (
    learn_gmrf_eda,
    learn_gmrf_eda_lasso,
    learn_gmrf_eda_elasticnet,
    learn_gmrf_eda_lars,
)
try:
    from pateda.learning.vine_copula import (
        learn_vine_copula_cvine,
        learn_vine_copula_dvine,
        learn_vine_copula_auto,
    )
except ImportError:
    learn_vine_copula_cvine = None
    learn_vine_copula_dvine = None
    learn_vine_copula_auto = None

__all__ = [
    "LearnFDA", "LearnUMDA", "LearnCUMDA", "LearnCFDA",
    "create_pairwise_chain_cliques", "create_block_cliques",
    "create_overlapping_windows_cliques",
    "LearnBMDA", "LearnEBNA", "LearnBOA",
    "LearnHBOA", "LearnLFDA", "LearnPADA", "LearnMarkovChain",
    "LearnSARTRE", "LearnBINOTEARS", "LearnPCBN",
    "LearnHSARTRE", "LearnHBINOTEARS",
    "LearnMixtureTrees", "LearnTreeModel",
    "LearnMNFDA",
    "LearnMNFDAG",
    "LearnMNFDAR", "LearnMNFDAGR", "LearnMNEDAG",
    "LearnTreeModelR", "LearnTreeModelM", "LearnMOA", "LearnIntFDA",
    "LearnRegularizedMarkov",
    "find_matrix_interactions_SAT", "find_matrix_interactions_ising",
    "find_matrix_interactions_nk",
    "find_matrix_interactions_additive_decomposable",
    "find_matrix_interactions_RNA_design",
    "LearnAffinityFactorization", "LearnAffinityFactorizationElim",
    "LearnAffinitySparse",
    "LearnPBIL", "LearnMIMIC",
    "learn_gaussian_univariate", "learn_gaussian_full",
    "LearnGaussianUnivariate", "LearnGaussianFull",
    "learn_mixture_gaussian_univariate", "learn_mixture_gaussian_full",
    "learn_mixture_gaussian_em",
    "learn_gmrf_eda", "learn_gmrf_eda_lasso",
    "learn_gmrf_eda_elasticnet", "learn_gmrf_eda_lars",
    "learn_vine_copula_cvine", "learn_vine_copula_dvine",
    "learn_vine_copula_auto",
]
