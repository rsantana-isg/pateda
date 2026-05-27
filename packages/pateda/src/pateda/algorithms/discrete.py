"""
Plug-and-play discrete EDA wrappers.

Each class assembles the appropriate pateda components and exposes a clean
constructor + run() interface so users do not need to know the internal
component architecture.

Usage example::

    from pateda import UMDA
    import numpy as np

    def onemax(x):
        return np.sum(x)

    alg = UMDA(n_vars=20, cardinality=2, fitness_func=onemax,
               pop_size=200, n_gen=50, random_seed=42)
    stats, cache = alg.run()
    print("Best:", stats.best_fitness_overall)
"""

from typing import Callable, Optional, Union
import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding.random_init import RandomInit
from pateda.seeding.seeding_unitation_constraint import SeedingUnitationConstraint
from pateda.selection.truncation import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.learning.umda import LearnUMDA
from pateda.learning.bmda import LearnBMDA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.mimic import LearnMIMIC
from pateda.learning.pbil import LearnPBIL
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.learning.affinity import LearnAffinityFactorization
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfda_r import LearnMNFDAR
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.mnfdag_r import LearnMNFDAGR
from pateda.learning.moa import LearnMOA
from pateda.learning.cumda import LearnCUMDA
from pateda.learning.cfda import LearnCFDA
from pateda.learning.fda import LearnFDA
from pateda.learning.bsc import LearnBSC

from pateda.sampling.fda import SampleFDA
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.markov import SampleMarkovChain
from pateda.sampling.mixture_trees import SampleMixtureTrees
from pateda.sampling.cumda import SampleCUMDA
from pateda.sampling.cfda import SampleCFDA

from pateda.core.components import LearningMethod
from pateda.algorithms.base import _BaseEDA


class _BSCLearnerAdapter(LearningMethod):
    """Flatten (N,1) fitness arrays to 1-D before delegating to LearnBSC."""

    def __init__(self, alpha: float, normalize_fitness: bool):
        self._inner = LearnBSC(alpha=alpha, normalize_fitness=normalize_fitness)

    def learn(self, generation, n_vars, cardinality, population, fitness, **params):
        fit = np.asarray(fitness)
        if fit.ndim > 1:
            fit = fit.reshape(-1)
        return self._inner.learn(generation, n_vars, cardinality, population, fit, **params)


def _to_cardinality(cardinality, n_vars: int) -> np.ndarray:
    """Convert int or array-like cardinality to a 1-D numpy array."""
    if isinstance(cardinality, (int, float)):
        return np.full(n_vars, int(cardinality))
    return np.asarray(cardinality, dtype=int)


def _make_components(
    learner,
    sampler,
    pop_size: int,
    selection_ratio: float,
    n_gen: int,
    elitism: bool,
) -> EDAComponents:
    """Assemble standard EDA components."""
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=selection_ratio),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
        replacement=ElitistReplacement(n_elite=1) if elitism else None,
    )
    return components


# ---------------------------------------------------------------------------
# UMDA
# ---------------------------------------------------------------------------

class UMDA(_BaseEDA):
    """
    Univariate Marginal Distribution Algorithm (UMDA).

    Models variables as independent, learning marginal frequencies from the
    selected population.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    cardinality : int or array-like
        Number of values per variable.  Pass an int for binary/same-cardinality
        problems, or a 1-D array for mixed cardinalities.
    fitness_func : callable
        Function mapping an individual (1-D array) to a scalar fitness value
        (higher is better).
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
    selection_ratio : float
        Fraction of the population used for learning.
    elitism : bool
        Whether to preserve the best individual across generations.
    alpha : float
        Laplace smoothing parameter (0 = no smoothing).
    random_seed : int or None
        Seed for the random number generator.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnUMDA(alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# BMDA
# ---------------------------------------------------------------------------

class BMDA(_BaseEDA):
    """
    Bivariate Marginal Distribution Algorithm (BMDA).

    Learns pairwise dependencies using chi-square tests to build a forest of
    bivariate marginals.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnBMDA(alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# TreeEDA
# ---------------------------------------------------------------------------

class TreeEDA(_BaseEDA):
    """
    Tree-structured EDA (Tree-EDA).

    Learns a maximum-weight spanning tree of pairwise mutual information to
    model variable dependencies.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnTreeModel(alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# TreeEDAR (Tree-EDA with root selection)
# ---------------------------------------------------------------------------

class TreeEDAR(_BaseEDA):
    """
    Tree-EDA with root-based variable ordering (Tree-EDA_r).

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        interaction_matrix: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        if interaction_matrix is None:
            # Default: full interaction matrix (all pairs allowed) —
            # equivalent to unrestricted Tree-EDA but using the restricted
            # learning code path.
            interaction_matrix = np.ones((n_vars, n_vars), dtype=int)
        learner = LearnTreeModelR(interaction_matrix=interaction_matrix, alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MIMIC
# ---------------------------------------------------------------------------

class MIMIC(_BaseEDA):
    """
    Mutual Information Maximization for Input Clustering (MIMIC).

    Learns a chain-structured model that maximizes mutual information between
    consecutive variables.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMIMIC(alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# PBIL
# ---------------------------------------------------------------------------

class PBIL(_BaseEDA):
    """
    Population-Based Incremental Learning (PBIL).

    Maintains a probability vector updated incrementally from selected
    individuals using a learning rate:

        P_new = (1 - alpha) * P_old + alpha * freq(selected)

    With ``alpha = 1`` PBIL reduces exactly to UMDA; with smaller values it
    blends the new frequencies with the previous probability vector to
    smooth updates across generations.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Learning rate for probability update (default 0.5).

        Note: the original Baluja (1994) paper recommends ``alpha = 0.05``-
        ``0.1`` for runs of *thousands* of generations.  For the typical EDA
        regime (``n_gen`` in the tens or low hundreds) such small values
        leave the probability vector close to uniform — i.e. PBIL barely
        explores beyond random search.  We therefore default to ``0.5``,
        which gives a fair speed/memory trade-off; pass a smaller value
        when running for many more generations.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnPBIL(alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# EBNA
# ---------------------------------------------------------------------------

class EBNA(_BaseEDA):
    """
    Estimation of Bayesian Network Algorithm (EBNA).

    Learns a Bayesian network structure using a score-and-search approach.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnEBNA(alpha=alpha)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# BOA
# ---------------------------------------------------------------------------

class BOA(_BaseEDA):
    """
    Bayesian Optimization Algorithm (BOA).

    Learns a Bayesian network using a greedy structure learning approach with
    a BIC/MDL scoring criterion.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnBOA()
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# AffEDA
# ---------------------------------------------------------------------------

class AffEDA(_BaseEDA):
    """
    Affinity-based EDA (Aff-EDA).

    Uses affinity propagation on mutual information to discover variable
    cliques, then builds a factorized model over these cliques.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    max_clique_size : int
        Maximum number of variables per clique.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        max_clique_size: int = 5,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnAffinityFactorization(max_clique_size=max_clique_size, alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MKEDA (k-order Markov chain EDA)
# ---------------------------------------------------------------------------

class MKEDA(_BaseEDA):
    """
    k-order Markov Chain EDA (MK-EDA).

    Models variables as a k-order Markov chain, where each variable depends
    on the k preceding variables in a fixed ordering.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    k : int
        Markov order (default 1).
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        k: int = 1,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMarkovChain(k=k, alpha=alpha)
        sampler = SampleMarkovChain(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MTED (Mixture of Trees EDA)
# ---------------------------------------------------------------------------

class MTED(_BaseEDA):
    """
    Mixture of Trees EDA (MT-EDA).

    Combines multiple tree-structured models in a mixture, enabling modelling
    of multimodal distributions.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    n_trees : int
        Number of tree components in the mixture.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        n_trees: int = 5,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMixtureTrees(n_components=n_trees, alpha=alpha)
        sampler = SampleMixtureTrees(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MNFDA
# ---------------------------------------------------------------------------

class MNFDA(_BaseEDA):
    """
    Markov Network Factorized Distribution Algorithm (MN-FDA).

    Learns a Markov network structure from pairwise chi-square independence
    tests, then samples using FDA (factorized distribution) sampling.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    max_clique_size : int
        Maximum clique size for the Markov network.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        max_clique_size: int = 3,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMNFDA(max_clique_size=max_clique_size, return_factorized=True)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MNFDAR (MN-FDA with random restarts)
# ---------------------------------------------------------------------------

class MNFDAR(_BaseEDA):
    """
    MN-FDA with random ordering (MN-FDA_r).

    Like MN-FDA but uses random variable orderings during sampling to
    improve exploration.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, max_clique_size, random_seed : see MNFDA.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        max_clique_size: int = 3,
        interaction_matrix: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        if interaction_matrix is None:
            # Default: full interaction matrix (all pairs allowed).
            interaction_matrix = np.ones((n_vars, n_vars), dtype=int)
        learner = LearnMNFDAR(
            interaction_matrix=interaction_matrix,
            max_clique_size=max_clique_size,
            return_factorized=True,
        )
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MNFDAG (MN-FDA with GBN structure)
# ---------------------------------------------------------------------------

class MNFDAG(_BaseEDA):
    """
    MN-FDA with augmented graph (MN-FDAg).

    Extends MN-FDA by augmenting the Markov network structure with additional
    edges based on mutual information scores.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, max_clique_size, random_seed : see MNFDA.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        max_clique_size: int = 3,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        # return_factorized=False → returns MarkovNetworkModel required by SampleGibbs
        learner = LearnMNFDAG(max_clique_size=max_clique_size, return_factorized=False)
        sampler = SampleGibbs(n_samples=pop_size, IT=2)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MNFDAGR (MN-FDAg with random restarts)
# ---------------------------------------------------------------------------

class MNFDAGR(_BaseEDA):
    """
    MN-FDAg with random ordering (MN-FDAg_r).

    Like MN-FDAg but uses random variable orderings during Gibbs sampling.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, max_clique_size, random_seed : see MNFDA.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        max_clique_size: int = 3,
        interaction_matrix: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        if interaction_matrix is None:
            # Default: full interaction matrix (all pairs allowed).
            # return_factorized=False → returns MarkovNetworkModel required by
            # SampleGibbs.
            interaction_matrix = np.ones((n_vars, n_vars), dtype=int)
        learner = LearnMNFDAGR(
            interaction_matrix=interaction_matrix,
            max_clique_size=max_clique_size,
            return_factorized=False,
        )
        sampler = SampleGibbs(n_samples=pop_size, IT=2)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MOA (Markovianity-based Optimization Algorithm)
# ---------------------------------------------------------------------------

class MOA(_BaseEDA):
    """
    Markovianity-Based Optimization Algorithm (MOA).

    Learns local Markov neighborhoods for each variable and samples via
    Gibbs sampling.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    k_neighbors : int
        Number of Markov neighbors per variable.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        k_neighbors: int = 3,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMOA(k_neighbors=k_neighbors)
        sampler = SampleGibbs(n_samples=pop_size, IT=2)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# CUMDA (Constrained UMDA)
# ---------------------------------------------------------------------------

class CUMDA(_BaseEDA):
    """
    Constrained UMDA (CUMDA).

    Binary EDA that enforces a fixed number of ones in every solution via
    Stochastic Universal Sampling.

    Parameters
    ----------
    n_vars : int
        Number of binary variables.
    cardinality : int or array-like
        Must be 2 (binary).
    fitness_func : callable
        Fitness function (higher is better).
    n_ones : int
        Exact number of ones required in each solution.
    pop_size, n_gen, selection_ratio, elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        n_ones: int,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnCUMDA(alpha=alpha)
        sampler = SampleCUMDA(n_samples=pop_size, n_ones=n_ones)
        components = EDAComponents(
            seeding=SeedingUnitationConstraint(),
            seeding_params={'num_ones': n_ones},
            selection=TruncationSelection(ratio=selection_ratio),
            learning=learner,
            sampling=sampler,
            stop_condition=MaxGenerations(n_gen),
            replacement=ElitistReplacement(n_elite=1) if elitism else None,
        )
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# CFDA (Constrained FDA)
# ---------------------------------------------------------------------------

class CFDA(_BaseEDA):
    """
    Constrained Factorized Distribution Algorithm (CFDA).

    Binary EDA that enforces a fixed unitation constraint while using a
    tree-structured factorized model.

    Parameters
    ----------
    n_vars : int
        Number of binary variables.
    cardinality : int or array-like
        Must be 2 (binary).
    fitness_func : callable
        Fitness function (higher is better).
    n_ones : int
        Exact number of ones required in each solution.
    pop_size, n_gen, selection_ratio, elitism, random_seed : see UMDA.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        n_ones: int,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnCFDA()
        sampler = SampleCFDA(n_samples=pop_size, n_ones=n_ones)
        components = EDAComponents(
            seeding=SeedingUnitationConstraint(),
            seeding_params={'num_ones': n_ones},
            selection=TruncationSelection(ratio=selection_ratio),
            learning=learner,
            sampling=sampler,
            stop_condition=MaxGenerations(n_gen),
            replacement=ElitistReplacement(n_elite=1) if elitism else None,
        )
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# FDA (Factorized Distribution Algorithm)
# ---------------------------------------------------------------------------

class FDA(_BaseEDA):
    """
    Factorized Distribution Algorithm (FDA).

    Mühlenbein, Mahnig & Rodriguez (1999).  FDA represents the joint
    probability distribution as a product of factors (cliques).  When no
    clique structure is provided, FDA defaults to a univariate (UMDA-like)
    factorization, but it can also accept an explicit clique decomposition
    coming from a known factorization of the problem (e.g. from a junction
    tree or a domain-specific factorization).

    The clique-structure matrix follows the MATEDA convention: each row is
    ``[n_overlap, n_new, overlap_indices..., new_indices...]``.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    cliques : np.ndarray or None, default None
        Optional clique structure matrix.  If ``None``, a univariate
        decomposition is used.
    alpha : float, default 1.0
        Laplace smoothing pseudo-count for the probability tables.  The
        default of 1.0 matches the original MATEDA-2.0 implementation; pass
        0.0 to disable smoothing.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        cliques: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnFDA(cliques=cliques, alpha=alpha)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# BSC (Bisection / fitness-weighted univariate EDA)
# ---------------------------------------------------------------------------

class BSC(_BaseEDA):
    """
    Bisection EDA (BSC).

    BSC estimates univariate marginals using a fitness-weighted scheme:
    ``P(X_i = k) = sum(fitness of individuals with X_i = k) / sum(all fitness)``.
    Variables that appear in high-fitness individuals get higher
    probabilities, biasing the search toward fitter regions but potentially
    reducing diversity.

    Originally introduced in MATEDA-1.0 (Inza et al. 2000) as part of an
    EDA-based approach to feature subset selection.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float, default 0.0
        Laplace smoothing pseudo-count (0 means no smoothing).
    normalize_fitness : bool, default True
        If True, fitness values are min-max normalised before weighting.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[int, np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        alpha: float = 0.0,
        normalize_fitness: bool = True,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = _BSCLearnerAdapter(alpha=alpha, normalize_fitness=normalize_fitness)
        sampler = SampleFDA(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)
