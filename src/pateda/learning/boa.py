"""
Bayesian Optimization Algorithm (BOA) learning

BOA is a sophisticated EDA that learns Bayesian networks to model the probability
distribution of promising solutions. Unlike tree models, BOA allows each variable
to have multiple parents, enabling it to capture complex multivariate dependencies.

Bayesian Networks:
A Bayesian network is a directed acyclic graph (DAG) where:
- Nodes represent random variables
- Directed edges represent probabilistic dependencies
- Each variable Xᵢ has a conditional probability distribution P(Xᵢ | parents(Xᵢ))

The joint probability factorizes as:
    P(X) = ∏ᵢ₌₁ⁿ P(Xᵢ | parents(Xᵢ))

BOA Structure Learning:
BOA uses the K2 algorithm for greedy structure search:
1. Start with a variable ordering (predefined or learned)
2. For each variable in order:
   a. Start with no parents
   b. Greedily add parents that improve the score metric
   c. Stop when score doesn't improve or max_parents reached
3. The ordering constraint ensures the resulting graph is acyclic

Scoring Metrics:
- K2: Bayesian-Dirichlet metric with uniform prior
  Score = log P(D | G) with Dirichlet prior
- BD: Bayesian-Dirichlet with adjustable prior strength (alpha parameter)
- BIC: Bayesian Information Criterion (penalizes complexity)

Decision Trees/Graphs (Advanced):
For efficiency, BOA can use decision trees or decision graphs to compactly
represent conditional probability distributions, especially when:
- Variables have high cardinality
- Parent sets are large
- CPDs have local structure (similar probabilities for many configurations)

Advantages over simpler EDAs:
- Can model complex multivariate dependencies (not just pairwise)
- More expressive than tree models
- Theoretical guarantees for convergence on certain problem classes

Computational Complexity:
- Structure learning: O(n² * m * k^p) where n=variables, m=samples, k=cardinality, p=max parents
- More expensive than UMDA or tree models
- Trade-off between model complexity and learning efficiency

When to use:
- Problems with strong multivariate dependencies
- Building blocks are known to involve multiple variables
- Can afford higher computational cost for better model quality

Equivalent to MATEDA's LearnBOA.m

References:
- Pelikan, M., Goldberg, D.E., & Cantú-Paz, E. (1999). "BOA: The Bayesian
  Optimization Algorithm." GECCO 1999, pp. 525-532.
- Pelikan, M. (2005). "Hierarchical Bayesian Optimization Algorithm: Toward a
  New Generation of Evolutionary Algorithms." Springer.
- Cooper, G.F., & Herskovits, E. (1992). "A Bayesian method for the induction
  of probabilistic networks from data." Machine Learning, 9(4):309-347.
- MATEDA-2.0 User Guide, Section 4.2: "Bayesian network based factorizations"
"""

from typing import Any, Optional
import numpy as np

from bayes_nets import BayesianNetwork

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.utils.weights import normalize_probabilities


class LearnBOA(LearningMethod):
    """
    Learn a BOA (Bayesian Optimization Algorithm) model

    BOA learns a Bayesian network using sophisticated scoring metrics and
    structure learning algorithms.

    Structure and parameter learning are delegated to
    :class:`bayes_nets.BayesianNetwork`; this class adapts the result to the
    pateda :class:`~pateda.core.models.BayesianNetworkModel` contract
    (``structure`` = adjacency matrix, ``parameters`` = dict mapping each
    variable to ``{"parents": [...], "cpd": array}``).
    """

    def __init__(
        self,
        max_parents: int = 3,
        score_metric: str = "k2",
        metric_alpha: float = 1.0,
        use_decision_graphs: bool = False,
        ordering: Optional[np.ndarray] = None,
        limit_joint_table_size: bool = True,
    ):
        """
        Initialize BOA learning

        Args:
            max_parents: Maximum number of parents per variable
            score_metric: Scoring metric ("k2", "bd", "bic")
            metric_alpha: Alpha parameter for BD metric (prior strength)
            use_decision_graphs: Use decision graphs for compact CPD representation
            ordering: Variable ordering for K2 algorithm (if None, use natural order)
            limit_joint_table_size: If True, only allow parent sets whose
                conditional-table size (variable + parents) is <= n_samples
        """
        self.max_parents = max_parents
        self.score_metric = score_metric
        self.metric_alpha = metric_alpha
        self.use_decision_graphs = use_decision_graphs
        self.ordering = ordering
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
        Learn BOA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)

        # Customized selection: bayes_nets expects sample_weights as a
        # probability vector summing to 1 (None -> uniform 1/N).
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        # Map BOA's "bd" (Bayesian-Dirichlet) metric onto the K2 scoring used by
        # bayes_nets; both are Dirichlet-multinomial marginal-likelihood scores.
        method = "k2" if self.score_metric == "bd" else self.score_metric

        # K2 requires a variable ordering; default to the natural order.
        if self.ordering is not None:
            ordering = np.asarray(self.ordering, dtype=int)
        else:
            ordering = np.arange(n_vars)

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        bn.fit(
            data,
            method=method,
            max_parents=self.max_parents,
            alpha=self.metric_alpha,
            ordering=ordering,
            limit_table_size=self.limit_joint_table_size,
            sample_weights=sample_weights,
        )

        # Adapt to the pateda BayesianNetworkModel contract.
        adjacency = bn.to_adjacency_matrix()
        cpds = bn.cpds

        model = BayesianNetworkModel(
            structure=adjacency,
            parameters=cpds,
            metadata={
                "generation": generation,
                "model_type": "BOA",
                "max_parents": self.max_parents,
                "score_metric": self.score_metric,
                "metric_alpha": self.metric_alpha,
                "ordering": ordering,
            },
        )

        return model
