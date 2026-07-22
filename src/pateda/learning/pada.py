"""
Polytree Approximation of Distribution Algorithm (PADA) learning

PADA (Soto, Ochoa, Acid & de Campos, 1999) restricts the probabilistic model
to a *polytree*: a singly connected Bayesian network, i.e. a directed acyclic
graph whose underlying undirected skeleton contains no loops.  A polytree is
strictly more expressive than a tree, because a variable may have more than
one parent and head-to-head nodes (colliders) X -> Z <- Y can be represented,
yet it stays far cheaper than a general Bayesian network: with n variables it
has at most n-1 edges, inference and sampling are linear, and the parameters
can be reliably estimated from the modest sample sizes available inside an
EDA.

Structure Learning (the LPA algorithm)
--------------------------------------
Unlike EBNA, BOA and LFDA, which are score-and-search algorithms, PADA learns
its structure with *independence tests*.  The dependency measure Dep(a, b) is
the Kullback-Leibler dependency, i.e. the (conditional) mutual information
computed on the empirical distribution of the selected population.  The five
steps of the LPA algorithm are:

1. Insert the pair <a, b> into a candidate list L when Dep(a, b) > e0.
2. Remove <a, b> from L when Dep(a, b | c) < e1 for some third variable c,
   i.e. when c explains away the dependency between a and b.
3. Rank the surviving pairs by the global dependency degree
   DepG(a, b) = min(Dep(a, b), min_c Dep(a, b | c)).
4. Add edges in that order, up to n-1 edges, skipping any edge that would
   close a cycle.  This keeps the skeleton singly connected.
5. Orient.  For every path a - c - b in the skeleton, conditioning on a
   head-to-head node *raises* the dependency between its parents whereas
   conditioning on any other middle node *lowers* it, so
   Dep(a, b | c) > Dep(a, b) implies the collider a -> c <- b.  Remaining
   edges are oriented without creating new head-to-head patterns.

The two thresholds e0 and e1 are the algorithm's only real parameters.  They
scale as 1/N, which is why PADA's behaviour depends on the population size:
larger populations make the tests stricter in absolute terms and admit more
structure.  Here they are derived from ``alpha_ci`` unless given explicitly.

Variants
--------
- ``dep_mode="global"`` (default) is PADA proper: it performs the conditional
  tests of steps 2-3 and ranks by DepG.  Cost is cubic in the number of
  independence tests.
- ``dep_mode="marginal"`` is the PADA1 variant: it uses only the first-order
  (marginal) tests, skipping step 2 and ranking by Dep(a, b).  Cost is
  quadratic.  The orientation step is unchanged, so the result is still a
  polytree, just built from a cruder ranking.

``n_cond_candidates`` restricts the conditioning variables c of steps 2-3 to
the few variables most strongly dependent on a or b, which keeps the global
variant tractable on large problems.

Comparison with the other BN-based EDAs in pateda
-------------------------------------------------
- Tree-EDA (:class:`~pateda.learning.tree.LearnTreeModel`): every variable has
  at most one parent.  PADA relaxes this to allow multiple parents as long as
  the skeleton stays singly connected, so it can capture colliders that a
  tree cannot.
- EBNA / BOA / LFDA: unrestricted DAGs learned by score-and-search.  They are
  more expressive but need more data per parameter and more computation.
  PADA occupies the middle ground and degrades gracefully when the selected
  population is small.

When to use
-----------
- Problems whose interaction graph is sparse and close to loop-free.
- Small populations, where the parameters of a dense Bayesian network cannot
  be estimated reliably.
- As a cheaper, more expressive alternative to Tree-EDA.

References
----------
- Soto, M.R., Ochoa, A., Acid, S., & de Campos, L.M. (1999). "Introducing the
  polytree approximation of distribution algorithm." Second Symposium on
  Artificial Intelligence (CIMAF-99), pp. 360-367.
- Ochoa, A., Muehlenbein, H., & Soto, M.R. (2000). "A Factorized Distribution
  Algorithm Using Single Connected Bayesian Networks." PPSN VI, pp. 787-796.
- Ochoa, A., Muehlenbein, H., & Soto, M.R. (2000). "Factorized Distribution
  Algorithms Using Bayesian Networks Bounded by Single Connected Graphs."
- Rebane, G., & Pearl, J. (1987). "The recovery of causal poly-trees from
  statistical data." UAI-87, pp. 175-182.
- Acid, S., & de Campos, L.M. (1995). "Approximations of causal networks by
  polytrees: An empirical study."
"""

from typing import Any, Optional
import numpy as np

from bayes_nets import BayesianNetwork
from bayes_nets.polytree_learning import PolytreeLPALearner

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.utils.weights import normalize_probabilities


class LearnPADA(LearningMethod):
    """
    Learn a PADA (Polytree Approximation of Distribution Algorithm) model

    PADA learns a singly connected Bayesian network (a polytree) with the LPA
    algorithm, using conditional-mutual-information independence tests rather
    than a score-and-search procedure.

    Structure learning is delegated to
    :class:`bayes_nets.polytree_learning.PolytreeLPALearner` and parameter
    learning to :class:`bayes_nets.BayesianNetwork`; this class adapts the
    result to the pateda :class:`~pateda.core.models.BayesianNetworkModel`
    contract (``structure`` = adjacency matrix, ``parameters`` = dict mapping
    each variable to ``{"parents": [...], "cpd": array}``).
    """

    def __init__(
        self,
        alpha_ci: float = 0.05,
        e0: Optional[float] = None,
        e1: Optional[float] = None,
        dep_mode: str = "global",
        n_cond_candidates: Optional[int] = 5,
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
    ):
        """
        Initialize PADA learning

        Args:
            alpha_ci: Significance level used to derive the dependency
                thresholds e0 and e1 when they are not given explicitly.
                Both then scale as 1/N.
            e0: Explicit marginal dependency threshold (None = derive from
                alpha_ci and the sample size)
            e1: Explicit conditional dependency threshold (None = derive from
                alpha_ci and the sample size)
            dep_mode: "global" for PADA (conditional tests, cubic) or
                "marginal" for the PADA1 variant (first-order tests only,
                quadratic)
            n_cond_candidates: Restrict the conditioning variables of steps
                2-3 to this many most-dependent variables (None = condition on
                every variable, the literal cubic algorithm)
            max_parents: Optional cap on the number of parents per variable.
                None uses the ``bayes_nets`` rule of thumb.  The polytree
                constraint already bounds the total number of edges by n-1.
            alpha: Laplace/Dirichlet smoothing for the CPD estimation
        """
        if dep_mode not in ("global", "marginal"):
            raise ValueError(
                f"dep_mode must be 'global' or 'marginal', got {dep_mode!r}"
            )
        self.alpha_ci = alpha_ci
        self.e0 = e0
        self.e1 = e1
        self.dep_mode = dep_mode
        self.n_cond_candidates = n_cond_candidates
        self.max_parents = max_parents
        self.alpha = alpha

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> BayesianNetworkModel:
        """
        Learn PADA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for PADA learning)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)

        # Customized selection: bayes_nets expects sample_weights as a
        # probability vector summing to 1 (None -> uniform 1/N).
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        learner = PolytreeLPALearner(
            alpha_ci=self.alpha_ci,
            e0=self.e0,
            e1=self.e1,
            dep_mode=self.dep_mode,
            n_cond_candidates=self.n_cond_candidates,
            max_parents=self.max_parents,
        )
        adjacency = learner.learn(
            data, n_vars, cardinality, sample_weights=sample_weights
        )

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        bn.set_structure(adjacency)
        bn.learn_parameters(data, alpha=self.alpha, sample_weights=sample_weights)

        # Adapt to the pateda BayesianNetworkModel contract.
        model = BayesianNetworkModel(
            structure=bn.to_adjacency_matrix(),
            parameters=bn.cpds,
            metadata={
                "generation": generation,
                "model_type": "PADA",
                "dep_mode": self.dep_mode,
                "alpha_ci": self.alpha_ci,
                "e0": self.e0,
                "e1": self.e1,
                "max_parents": self.max_parents,
            },
        )

        return model
