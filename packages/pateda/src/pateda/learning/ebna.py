"""
Estimation of Bayesian Networks Algorithm (EBNA) learning

EBNA is an EDA that learns Bayesian networks to capture dependencies between
variables. Unlike tree models which restrict each variable to have at most one
parent, EBNA allows multiple parents but typically limits the number (local structure).

EBNA was one of the first EDAs to use general Bayesian networks for optimization,
introduced by Etxeberria and Larrañaga (1999). It uses greedy hill-climbing for
structure learning with information-theoretic scoring metrics.

Structure Learning:
EBNA uses greedy search to find a good Bayesian network structure:
1. For each variable:
   a. Start with no parents
   b. Greedily add parents that maximize the score
   c. Ensure acyclicity: only add edge if it doesn't create a cycle
   d. Stop when max_parents reached or no improvement
2. The search is local and greedy (not guaranteed to find global optimum)

Scoring Metrics:
- BIC (Bayesian Information Criterion): Balances fit and complexity
  BIC = log P(D | θ_ML, G) - (k/2) log(n)
  where k is the number of parameters, n is the sample size
- AIC (Akaike Information Criterion): Similar to BIC with different penalty
  AIC = log P(D | θ_ML, G) - k
- K2: Bayesian scoring metric (same as BOA)

The BIC/AIC penalty terms prevent overfitting by penalizing complex structures.

Key Features:
- Local structure: Limited number of parents (typically 2-3)
- Cycle detection: Ensures the learned graph is a DAG
- Flexible scoring: Supports multiple metrics (BIC, AIC, K2)
- Optional fixed structure: Can use predefined DAG if domain knowledge available

Comparison with BOA:
- EBNA: Greedy local search, any variable ordering
- BOA: K2 algorithm requires variable ordering, often uses decision graphs
- EBNA: Typically uses BIC/AIC scoring
- BOA: Typically uses K2/BD (Bayesian) scoring
- Both: Learn general Bayesian networks with multiple parents

Computational Complexity:
- Greedy search: O(n² * max_parents * m * k^p)
  where n=variables, m=samples, k=cardinality, p=max parents
- Cycle checking adds overhead compared to ordered approaches (K2)

When to use:
- Problems with multivariate dependencies
- Don't have good variable ordering (unlike K2/BOA)
- Want BIC/AIC model selection
- Need balance between UMDA/Tree (too simple) and unrestricted BN (too complex)

Equivalent to MATEDA's LearnEBNA.m

References:
- Etxeberria, R., & Larrañaga, P. (1999). "Global optimization using Bayesian
  networks." Second Symposium on Artificial Intelligence (CIMAF-99), pp. 332-339.
- Larrañaga, P., Etxeberria, R., Lozano, J.A., & Peña, J.M. (2000). "Optimization
  in continuous domains by learning and simulation of Gaussian networks."
  Workshop on Optimization by Building and Using Probabilistic Models, GECCO-2000.
- MATEDA-2.0 User Guide, Section 4.2: "Bayesian network based factorizations"
"""

from typing import Any, Optional
import numpy as np

from bayes_nets import BayesianNetwork

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.utils.weights import normalize_probabilities


class LearnEBNA(LearningMethod):
    """
    Learn an EBNA (Estimation of Bayesian Networks Algorithm) model

    EBNA learns a Bayesian network where each variable can have multiple parents,
    but the number of parents is typically limited (local structure).
    Uses scoring metrics like BIC or AIC for structure learning.

    Structure and parameter learning are delegated to
    :class:`bayes_nets.BayesianNetwork`; this class adapts the result to the
    pateda :class:`~pateda.core.models.BayesianNetworkModel` contract
    (``structure`` = adjacency matrix, ``parameters`` = dict mapping each
    variable to ``{"parents": [...], "cpd": array}``).
    """

    def __init__(
        self,
        max_parents: int = 2,
        score_metric: str = "bic",
        alpha: float = 0.0,
        structure: Optional[np.ndarray] = None,
        limit_joint_table_size: bool = True,
    ):
        """
        Initialize EBNA learning

        Args:
            max_parents: Maximum number of parents per variable
            score_metric: Scoring metric for structure learning ("bic", "aic", "k2")
            alpha: Smoothing parameter for probability estimation
            structure: Pre-defined structure (DAG as adjacency matrix)
                      If provided, structure learning is skipped
            limit_joint_table_size: If True, only allow parent sets whose
                conditional-table size (variable + parents) is <= n_samples
        """
        self.max_parents = max_parents
        self.score_metric = score_metric
        self.alpha = alpha
        self.structure = structure
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
        Learn EBNA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for EBNA learning)
            **params: Additional parameters

        Returns:
            Learned BayesianNetworkModel
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)

        # Customized selection: bayes_nets expects sample_weights as a
        # probability vector summing to 1 (None -> uniform 1/N).
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)

        if self.structure is not None:
            # Fixed structure: only estimate the CPDs.
            bn.set_structure(np.asarray(self.structure, dtype=int))
            bn.learn_parameters(data, alpha=self.alpha, sample_weights=sample_weights)
        else:
            # Learn structure and parameters with the requested scoring metric.
            bn.fit(
                data,
                method=self.score_metric,
                max_parents=self.max_parents,
                alpha=self.alpha,
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
                "model_type": "EBNA",
                "max_parents": self.max_parents,
                "score_metric": self.score_metric,
            },
        )

        return model
