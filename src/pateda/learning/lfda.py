"""
Learning Factorized Distribution Algorithm (LFDA) learning

LFDA is the learning counterpart of FDA.  FDA
(:class:`~pateda.learning.fda.LearnFDA`) is given a factorization of the
search distribution derived analytically from the structure of the additively
decomposable fitness function, and only estimates the marginals of that fixed
factorization.  LFDA (Muehlenbein & Mahnig, 1999) removes the requirement of
knowing the factorization in advance: it *learns* a Bayesian network from the
selected population at every generation and uses the learned network as the
factorization.  FDA is recovered when the learned structure coincides with
the true one.

Structure Learning
------------------
LFDA scores structures with the Bayesian Information Criterion and searches
greedily, adding one arc at a time:

    BIC(B, D, w) = sum_i log P(D_i | Pa_i, theta_ML) - w * (log N / 2) * |theta|

Two control parameters define the algorithm, and both matter in practice:

- ``max_parents`` (k in the paper).  A hard bound on the number of parents of
  any variable.  Muehlenbein & Mahnig emphasize that BIC alone does not
  prevent the search from spending its whole sample on a few dense nodes, so
  LFDA always runs with an explicit k.
- ``bic_weight`` (the weighting factor alpha in the paper).  A multiplier on
  the BIC complexity penalty.  ``bic_weight = 1`` gives standard BIC;
  ``bic_weight > 1`` penalizes complexity more, yielding sparser networks;
  ``bic_weight < 1`` yields denser networks.  Tuning it is how LFDA trades
  model accuracy against the amount of data (the selected population) that is
  available to estimate the parameters.

Since the population is resampled every generation, the amount of data is
small and fixed, and the penalty weight is the main lever on overfitting.

Comparison with EBNA and BOA
----------------------------
All three learn a general Bayesian network by score-and-search, so they are
close relatives; the differences are in emphasis:

- EBNA (:class:`~pateda.learning.ebna.LearnEBNA`) also uses greedy search with
  BIC/AIC/K2, but is presented as a general BN-learning EDA and does not
  expose a penalty weight.  LFDA is the same search made explicit as an
  approximation to FDA's factorization, with the penalty weight as a
  first-class parameter.
- BOA (:class:`~pateda.learning.boa.LearnBOA`) uses the K2/BDe Bayesian score
  with a variable ordering rather than an information criterion.

Computational Complexity
------------------------
- Greedy search: O(n^2 * k * m * r^k), n=variables, m=samples, r=cardinality,
  k=max_parents.
- Cycle checking on every candidate arc, as no variable ordering is assumed.

When to use
-----------
- Additively decomposable functions whose factorization is unknown, i.e. the
  setting where FDA would be the right algorithm if the structure were given.
- Whenever the population size is small enough that the strength of the
  complexity penalty visibly changes the learned model.

References
----------
- Muehlenbein, H., & Mahnig, T. (1999). "FDA - A scalable evolutionary
  algorithm for the optimization of additively decomposed functions."
  Evolutionary Computation, 7(4):353-376.
- Muehlenbein, H., & Mahnig, T. (1999). "The Factorized Distribution Algorithm
  for additively decomposed functions." CEC 1999, pp. 752-759.
- Schwarz, G. (1978). "Estimating the dimension of a model."
  The Annals of Statistics, 6(2):461-464.
"""

from typing import Any, List, Optional
import numpy as np

from bayes_nets import BayesianNetwork, BICScoringMethod
from bayes_nets.structure_learning import GreedyHillClimbLearner

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.utils.weights import normalize_probabilities


class _WeightedBICScoringMethod(BICScoringMethod):
    """BIC with an adjustable weight on the complexity penalty.

    ``local_score = log-likelihood - weight * (k / 2) * log(N)``

    The parent class implements the ``weight = 1`` case, so the weighted score
    is obtained by adding back ``(1 - weight) * (k / 2) * log(N)``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 1.0,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(alpha=alpha, sample_weights=sample_weights)
        self.weight = weight

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        bic = super().local_score(var, parents, data, cardinality)
        if self.weight == 1.0:
            return bic
        # Number of free parameters of the CPD of var given parents.
        n_parent_configs = int(np.prod([int(cardinality[p]) for p in parents])) if parents else 1
        n_params = n_parent_configs * (int(cardinality[var]) - 1)
        penalty = 0.5 * n_params * np.log(data.shape[0])
        return bic + (1.0 - self.weight) * penalty


class LearnLFDA(LearningMethod):
    """
    Learn an LFDA (Learning Factorized Distribution Algorithm) model

    LFDA learns a Bayesian network by greedy hill-climbing on a BIC score
    whose complexity penalty can be reweighted, subject to a hard bound on
    the number of parents per variable.

    Structure and parameter learning are delegated to ``bayes_nets``; this
    class adapts the result to the pateda
    :class:`~pateda.core.models.BayesianNetworkModel` contract
    (``structure`` = adjacency matrix, ``parameters`` = dict mapping each
    variable to ``{"parents": [...], "cpd": array}``).
    """

    def __init__(
        self,
        max_parents: int = 4,
        bic_weight: float = 1.0,
        alpha: float = 1.0,
        limit_joint_table_size: bool = True,
    ):
        """
        Initialize LFDA learning

        Args:
            max_parents: Maximum number of parents per variable (k in the
                paper).  LFDA always runs with an explicit bound.
            bic_weight: Weight on the BIC complexity penalty (alpha in the
                paper).  1.0 = standard BIC, > 1.0 = sparser networks,
                < 1.0 = denser networks.
            alpha: Laplace/Dirichlet smoothing used when estimating the
                log-likelihood and the final CPDs
            limit_joint_table_size: If True, only allow parent sets whose
                conditional-table size (variable + parents) is <= n_samples
        """
        if bic_weight < 0:
            raise ValueError(f"bic_weight must be non-negative, got {bic_weight}")
        self.max_parents = max_parents
        self.bic_weight = bic_weight
        self.alpha = alpha
        self.limit_joint_table_size = limit_joint_table_size

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
        Learn LFDA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for LFDA learning)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)

        # Customized selection: bayes_nets expects sample_weights as a
        # probability vector summing to 1 (None -> uniform 1/N).
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        # Sample weights are embedded in the scorer, so they are not passed
        # again to learn(); the hill-climber would otherwise re-wrap them.
        scoring = _WeightedBICScoringMethod(
            weight=self.bic_weight,
            alpha=self.alpha,
            sample_weights=sample_weights,
        )
        learner = GreedyHillClimbLearner(
            scoring=scoring,
            max_parents=self.max_parents,
            limit_table_size=self.limit_joint_table_size,
        )
        adjacency = learner.learn(data, n_vars, cardinality)

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        bn.set_structure(adjacency)
        bn.learn_parameters(data, alpha=self.alpha, sample_weights=sample_weights)

        # Adapt to the pateda BayesianNetworkModel contract.
        model = BayesianNetworkModel(
            structure=bn.to_adjacency_matrix(),
            parameters=bn.cpds,
            metadata={
                "generation": generation,
                "model_type": "LFDA",
                "max_parents": self.max_parents,
                "bic_weight": self.bic_weight,
                "alpha": self.alpha,
            },
        )

        return model
