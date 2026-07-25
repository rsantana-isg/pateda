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
from pateda.replacement.niching import RestrictedTournamentReplacement
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.learning.umda import LearnUMDA
from pateda.learning.bmda import LearnBMDA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.mimic import LearnMIMIC
from pateda.learning.pbil import LearnPBIL
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.learning.lfda import LearnLFDA
from pateda.learning.hboa import LearnHBOA, LearnHBOALight
from pateda.learning.bn_extra import (
    LearnSARTRE,
    LearnBINOTEARS,
    LearnPCBN,
    LearnHSARTRE,
    LearnHBINOTEARS,
)
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

from pateda.sampling.fda import SampleFDA
from pateda.sampling.bayesian_network import SampleBayesianNetwork, SampleLocalStructureBN
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.markov import SampleMarkovChain
from pateda.sampling.mixture_trees import SampleMixtureTrees
from pateda.sampling.cumda import SampleCUMDA
from pateda.sampling.cfda import SampleCFDA

from pateda.core.components import LearningMethod
from pateda.algorithms.base import _BaseEDA


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
        alpha: float = 1.0,
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
        alpha: float = 1.0,
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
        alpha: float = 1.0,
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
        alpha: float = 1.0,
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
        alpha: float = 1.0,
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

    This class is the *flexible* EBNA of Larranaga et al. (2000, "Combinatorial
    Optimization by Learning and Simulation of Bayesian Networks", UAI): the
    ``score_metric`` argument selects one of the three structure-learning
    paradigms of the original proposal, exposed individually as the
    :class:`EBNA_BIC`, :class:`EBNA_K2` and :class:`EBNA_PC` convenience classes:

    * ``"bic"`` -- penalized maximum likelihood (Bayesian Information Criterion)
      with greedy add-only arc search (the classic ``EBNA_BIC``; default).
    * ``"k2"``  -- Bayesian marginal likelihood, the K2/Cooper-Herskovits metric
      (``EBNA_K2``; the paper's ``EBNA_K2+pen`` adds an explicit complexity
      penalty on top -- see :class:`EBNA_K2`).
    * ``"pc"`` / ``"stable_pc"`` -- constraint-based structure learning by
      detecting conditional independencies with the PC algorithm (``EBNA_PC``).

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see UMDA.
    alpha : float
        Laplace smoothing pseudo-count (0 = no smoothing).  For the
        constraint-based metrics ``"pc"``/``"stable_pc"`` it is also the source
        of the conditional-independence significance level in ``bayes_nets``
        (``alpha_ci = alpha`` when ``0 < alpha < 1``, else 0.05).
    max_parents : int
        Maximum number of parents per variable.
    score_metric : str
        Structure-learning method passed to ``bayes_nets`` (see above and
        :meth:`LearnEBNA`).  Default ``"bic"``.
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
        alpha: float = 1.0,
        max_parents: Optional[int] = 2,
        score_metric: str = "bic",
        warm_start: bool = False,
        penalty: Optional[Union[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnEBNA(max_parents=max_parents, score_metric=score_metric,
                            alpha=alpha, warm_start=warm_start, penalty=penalty)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# Flexible EBNA family (Larranaga et al., 2000): one class per score metric
# ---------------------------------------------------------------------------

class EBNA_BIC(EBNA):
    """
    EBNA with the **penalized maximum likelihood** (BIC) score -- the original
    ``EBNA_BIC`` of Etxeberria & Larranaga (1999), with the paper's
    **warm-started search** that makes it depart from ``LFDA``.

    Faithful to Larranaga et al. (2000): the first generation learns the network
    from an arc-less graph with **Algorithm B** (greedy add-only BIC search);
    every subsequent generation runs an **add/delete/reverse local search
    warm-started from the previous generation's DAG**.  ``LFDA``, by contrast,
    relearns add-only from scratch each generation with a tunable penalty weight
    -- so the two now differ genuinely in *search strategy*, not just defaults.
    Set ``warm_start=False`` to recover the plain from-scratch add-only search.

    Parameters
    ----------
    See :class:`EBNA`.  ``score_metric`` is fixed to ``"bic"``; ``warm_start``
    defaults to ``True``.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, alpha=1.0, max_parents=2,
                 warm_start=True, random_seed=None):
        super().__init__(n_vars, cardinality, fitness_func, pop_size, n_gen,
                         selection_ratio, elitism, alpha, max_parents,
                         score_metric="bic", warm_start=warm_start,
                         random_seed=random_seed)


class EBNA_K2(EBNA):
    """
    EBNA with the **penalized Bayesian marginal likelihood** -- the
    ``EBNA_K2+pen`` of Larranaga et al. (2000).

    The K2/Cooper-Herskovits marginal likelihood ``log p(D | S)`` (Dirichlet
    prior) already penalizes complexity *implicitly*; ``EBNA_K2+pen`` adds an
    **explicit penalty** ``- f(N) dim(S)`` on top and, via the Etxeberria et al.
    (1997) theorem, an **automatic per-variable parent bound** (used when
    ``max_parents=None``).  This maps to the ``bayes_nets`` ``method="k2_pen"``.

    Parameters
    ----------
    See :class:`EBNA`.  ``score_metric`` is fixed to ``"k2_pen"``.
    penalty : {"bic", "aic"} or float
        Explicit-penalty weight ``f(N)``: ``"bic"`` -> ``0.5 log N`` (default),
        ``"aic"`` -> 1, or a float constant.  ``penalty=0`` recovers plain K2.
    max_parents : int or None
        ``None`` (default) uses the automatic Etxeberria per-variable bound.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, alpha=1.0, max_parents=None,
                 penalty="bic", random_seed=None):
        super().__init__(n_vars, cardinality, fitness_func, pop_size, n_gen,
                         selection_ratio, elitism, alpha, max_parents,
                         score_metric="k2_pen", penalty=penalty,
                         random_seed=random_seed)


class EBNA_PC(EBNA):
    """
    EBNA with **constraint-based** structure learning -- the ``EBNA_PC`` of
    Larranaga et al. (2000), which recovers the network by *detecting
    conditional independencies* with the PC algorithm rather than by
    score-and-search.

    The original paper uses the PC algorithm with chi-square independence tests
    at significance ``alpha_ci = 0.01``.  Here the (order-independent) PC-Stable
    of ``bayes_nets`` is used by default; the CI significance level is taken from
    ``alpha`` when ``0 < alpha < 1`` (otherwise the ``bayes_nets`` default of
    0.05).  Set ``score_metric="pc"`` for the plain (order-dependent) PC.

    Parameters
    ----------
    See :class:`EBNA`.  ``score_metric`` is restricted to ``"pc"`` /
    ``"stable_pc"`` (default ``"stable_pc"``).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, alpha=1.0, max_parents=2,
                 score_metric="stable_pc", random_seed=None):
        if score_metric not in ("pc", "stable_pc"):
            raise ValueError("EBNA_PC requires score_metric in {'pc', 'stable_pc'}, "
                             f"got {score_metric!r}")
        super().__init__(n_vars, cardinality, fitness_func, pop_size, n_gen,
                         selection_ratio, elitism, alpha, max_parents,
                         score_metric=score_metric, random_seed=random_seed)


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
        max_parents: int = 3,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnBOA(max_parents=max_parents)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class LFDA(_BaseEDA):
    """
    Learning Factorized Distribution Algorithm (LFDA).

    A BN-based EDA that scores structures with the Bayesian Information
    Criterion and searches greedily, with an explicit bound on the number of
    parents and a weight on the BIC complexity penalty (``bic_weight``).  Like
    EBNA and BOA it is a score-and-search learner.

    ``bic_weight`` **is** the LFDA penalty weight ``alpha`` of Muehlenbein &
    Mahnig (1999): ``bic_weight=1`` is standard Schwarz BIC (the paper's
    ``alpha=0.5`` in bit form), ``>1`` yields sparser networks, ``<1`` denser
    ones -- the tunable complexity penalization that is LFDA's defining
    ingredient.  See ``docs/LFDA_departure_from_EBNA.md`` for how ``LFDA`` and
    ``EBNA``/``EBNA_BIC`` relate and how to make them depart further.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see :class:`UMDA`.
    max_parents : int
        Maximum parents per variable.
    bic_weight : float
        Multiplier on the BIC complexity penalty (>1 sparser, <1 denser).
    alpha : float
        Laplace smoothing pseudo-count.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, max_parents=4,
                 bic_weight=1.0, alpha=1.0, random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnLFDA(max_parents=max_parents, bic_weight=bic_weight, alpha=alpha)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# BN-based EDAs with alternative (non score-and-search) structure learners
# ---------------------------------------------------------------------------

def _make_niching_components(learner, sampler, pop_size, selection_ratio,
                             n_gen, window_size):
    """Standard components but with restricted-tournament (niching) replacement,
    the diversity-preserving replacement used by hBOA."""
    return EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=selection_ratio),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
        replacement=RestrictedTournamentReplacement(window_size=window_size),
    )


class SARTRE_EDA(_BaseEDA):
    """
    SARTRE-EDA: a BN-based EDA (structured like EBNA) that learns the network
    with SARTRE -- an order-based method that prunes spurious edges with a
    sparse additive / group-lasso model.  Different learning paradigm from
    EBNA/BOA/LFDA, fast, and scales to large ``n``.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see :class:`UMDA`.
    max_parents : int
        Maximum parents per variable.
    alpha : float
        Laplace smoothing pseudo-count.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, max_parents=4, alpha=1.0,
                 random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnSARTRE(max_parents=max_parents, alpha=alpha)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class BINOTEARS_EDA(_BaseEDA):
    """
    BINOTEARS-EDA: a BN-based EDA (structured like EBNA) that learns the network
    with BINOTEARS, a differentiable continuous-optimization structure learner
    with an acyclicity constraint.

    .. note:: BINOTEARS supports **binary variables only**.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see :class:`UMDA`.
    max_parents, alpha : see :class:`SARTRE_EDA`.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, max_parents=4, alpha=1.0,
                 random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnBINOTEARS(max_parents=max_parents, alpha=alpha)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class PCBN_EDA(_BaseEDA):
    """
    PCBN-EDA: a BN-based EDA (structured like EBNA) that learns the network with
    the constraint-based PC-Stable algorithm in a **bounded conditioning-order**
    form (a hard cap on the conditioning-set size), which is what lets the
    constraint-based paradigm scale to ``n >= 100``.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see :class:`UMDA`.
    max_cond_set_size : int
        Order limit of the PC independence tests (2--3 recommended).
    max_parents, alpha : see :class:`SARTRE_EDA`.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, elitism=True, max_cond_set_size=3,
                 max_parents=4, alpha=1.0, random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnPCBN(max_cond_set_size=max_cond_set_size,
                            max_parents=max_parents, alpha=alpha)
        sampler = SampleBayesianNetwork(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class HSARTRE_EDA(_BaseEDA):
    """
    HSARTRE-EDA: the hBOA-style upgrade of :class:`SARTRE_EDA`.  SARTRE discovers
    the skeleton, a decision graph learns the local (context-specific) structure
    restricted to that skeleton, and **restricted-tournament (niching)
    replacement** preserves diversity -- the two ingredients hBOA adds to BOA.

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    random_seed : see :class:`UMDA`.
    max_parents : int
        Maximum parents in the decision-graph refinement (larger than the plain
        variant, since local structure controls the parameter growth).
    window_size : int
        Window size of the restricted-tournament (niching) replacement.
    local_structure : str
        "dg" (decision graph, default) or "dt" (decision tree).
    alpha : float
        Laplace smoothing pseudo-count.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20,
                 local_structure="dg", alpha=1.0, random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnHSARTRE(max_parents=max_parents,
                               local_structure=local_structure, alpha=alpha)
        sampler = SampleLocalStructureBN(n_samples=pop_size)
        components = _make_niching_components(learner, sampler, pop_size,
                                              selection_ratio, n_gen, window_size)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class HBINOTEARS_EDA(_BaseEDA):
    """
    HBINOTEARS-EDA: the hBOA-style upgrade of :class:`BINOTEARS_EDA` (decision
    graph over the BINOTEARS skeleton + restricted-tournament niching).

    .. note:: Binary variables only (inherited from BINOTEARS).

    Parameters
    ----------
    See :class:`HSARTRE_EDA`.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20,
                 local_structure="dg", alpha=1.0, random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnHBINOTEARS(max_parents=max_parents,
                                  local_structure=local_structure, alpha=alpha)
        sampler = SampleLocalStructureBN(n_samples=pop_size)
        components = _make_niching_components(learner, sampler, pop_size,
                                              selection_ratio, n_gen, window_size)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


class HBOA(_BaseEDA):
    """
    Hierarchical Bayesian Optimization Algorithm (hBOA) as a plug-and-play EDA.

    Learns a Bayesian network whose local conditional distributions are
    represented with decision trees/graphs (so many parents can be afforded),
    samples them directly through :class:`SampleLocalStructureBN` (the compact
    structure is exploited at sampling, never materialising the dense table),
    and preserves diversity with restricted-tournament (niching) replacement --
    the complete hBOA of Pelikan (2005).

    Parameters
    ----------
    n_vars, cardinality, fitness_func, pop_size, n_gen, selection_ratio,
    random_seed : see :class:`UMDA`.
    max_parents : int
        Maximum parents per variable (larger than BOA, since the local
        structure controls the parameter growth).
    local_structure : str
        "dg" (decision graph, default) or "dt" (decision tree).
    window_size : int
        Window size of the restricted-tournament (niching) replacement.
    alpha : float
        Laplace/Dirichlet smoothing pseudo-count.
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, local_structure="dg",
                 window_size=20, alpha=1.0, random_seed=None):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnHBOA(max_parents=max_parents, local_structure=local_structure,
                            alpha=alpha)
        sampler = SampleLocalStructureBN(n_samples=pop_size)
        components = _make_niching_components(learner, sampler, pop_size,
                                              selection_ratio, n_gen, window_size)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# HBOA-Light family (A1-A5): faster decision-tree/graph local structure
# ---------------------------------------------------------------------------

def _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                     selection_ratio, window_size, max_parents, alpha,
                     random_seed, **learn_kwargs):
    """Assemble an HBOA-Light EDA (compact local-structure BN + niching)."""
    card = _to_cardinality(cardinality, n_vars)
    learner = LearnHBOALight(max_parents=max_parents, alpha=alpha, **learn_kwargs)
    sampler = SampleLocalStructureBN(n_samples=pop_size)
    components = _make_niching_components(learner, sampler, pop_size,
                                          selection_ratio, n_gen, window_size)
    return EDA(pop_size, n_vars, fitness_func, card, components,
               random_seed=random_seed)


class HBOA_Light_A1(_BaseEDA):
    """
    HBOA-Light **A1** -- decision-*tree* CPDs instead of decision graphs.

    Drops the (expensive, combinatorial) decision-graph leaf-merging step of
    hBOA and keeps context-specific decision-*tree* CPDs (Boutilier et al.,
    1996).  ~4-5x cheaper structure search than the ``dg`` build while retaining
    most of the context-specific-independence benefit.  See
    ``docs/Fast_DG_Learning.md`` (A1).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20, alpha=1.0,
                 random_seed=None):
        eda = _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                               selection_ratio, window_size, max_parents, alpha,
                               random_seed, method="dt", local_structure="dt")
        super().__init__(eda)


class HBOA_Light_A2(_BaseEDA):
    """
    HBOA-Light **A2** -- decision graphs with **top-k mutual-information
    candidate-parent pruning**.

    Restricts each variable's parent search to its ``top_k`` highest-MI
    neighbours (de Campos & Ji, 2011 style constraint), turning the O(n)
    candidate loop into O(k) while keeping the full ``dg`` local structure.  See
    ``docs/Fast_DG_Learning.md`` (A2).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20, alpha=1.0,
                 top_k=10, random_seed=None):
        eda = _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                               selection_ratio, window_size, max_parents, alpha,
                               random_seed, method="dg", local_structure="dg",
                               candidate_parents=f"mi:{int(top_k)}")
        super().__init__(eda)


class HBOA_Light_A3(_BaseEDA):
    """
    HBOA-Light **A3** -- decision graphs with **cached sufficient statistics**
    (naive-Bayes "independent information gain") split scoring.

    Scores candidate splits from precomputed statistics instead of rescanning
    the selected set (Su & Zhang, 2006), giving the same score at lower per-split
    cost.  See ``docs/Fast_DG_Learning.md`` (A3).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20, alpha=1.0,
                 random_seed=None):
        eda = _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                               selection_ratio, window_size, max_parents, alpha,
                               random_seed, method="dg", local_structure="dg",
                               fast_local_scoring=True)
        super().__init__(eda)


class HBOA_Light_A4(_BaseEDA):
    """
    HBOA-Light **A4** -- **bounded** decision graphs grown with the cheaper
    **MDL** split score.

    Caps the local structure at ``max_leaves`` and grows it with the MDL/BIC
    split gain instead of the more expensive (and denser) K2 gain (Muehlenbein &
    Mahnig).  See ``docs/Fast_DG_Learning.md`` (A4).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20, alpha=1.0,
                 max_leaves=32, split_score="mdl", random_seed=None):
        eda = _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                               selection_ratio, window_size, max_parents, alpha,
                               random_seed, method="dg", local_structure="dg",
                               max_leaves=max_leaves, split_score=split_score)
        super().__init__(eda)


class HBOA_Light_A5(_BaseEDA):
    """
    HBOA-Light **A5** -- **non-search** decision-graph construction (Tree-in-Tree
    / Naive Decision Graph).

    Builds the decision graph in a single constructive pass (grow a tree, then
    merge leaves; Zhu & Shoaran, 2021) instead of a combinatorial hill-climb,
    the decisive win that keeps decision-graph EDAs tractable at ``n = 64/100``.
    See ``docs/Fast_DG_Learning.md`` (A5).
    """

    def __init__(self, n_vars, cardinality, fitness_func, pop_size=100, n_gen=50,
                 selection_ratio=0.5, max_parents=6, window_size=20, alpha=1.0,
                 random_seed=None):
        eda = _make_hboa_light(n_vars, cardinality, fitness_func, pop_size, n_gen,
                               selection_ratio, window_size, max_parents, alpha,
                               random_seed, method="dg_ndg", local_structure="dg")
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
        alpha: float = 1.0,
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
        alpha: float = 1.0,
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
    weight_learning : str
        Method for learning mixture weights: "uniform", "em", or
        "fitness_proportional".
    use_priors : bool
        If True, add priors to prevent premature convergence (mutation-like
        effect as described in RepMutMTFDA).
    use_adaptive : bool
        If True, stop learning early when the model probability of the data
        reaches ``adaptive_mu`` (avoids overfitting).
    adaptive_mu : float
        Threshold for adaptive learning (default 0.9).
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
        alpha: float = 1.0,
        weight_learning: str = "uniform",
        use_priors: bool = False,
        use_adaptive: bool = False,
        adaptive_mu: float = 0.9,
        random_seed: Optional[int] = None,
    ):
        card = _to_cardinality(cardinality, n_vars)
        learner = LearnMixtureTrees(
            n_components=n_trees,
            alpha=alpha,
            weight_learning=weight_learning,
            use_priors=use_priors,
            use_adaptive=use_adaptive,
            truncation_ratio=selection_ratio,
            adaptive_mu=adaptive_mu,
        )
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
        alpha: float = 1.0,
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
