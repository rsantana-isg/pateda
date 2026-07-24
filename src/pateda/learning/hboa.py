"""
Hierarchical Bayesian Optimization Algorithm (hBOA) learning

hBOA is BOA extended so that it can solve *hierarchically* decomposable
problems, in which building blocks on one level are composed into building
blocks of the next level and the meaning of a partial solution only becomes
apparent once its context is fixed.  Pelikan (2005) identifies three
ingredients that a hierarchical optimizer needs, and hBOA adds the two that
plain BOA lacks.

Differences between BOA and hBOA
--------------------------------
Both algorithms learn a Bayesian network over the selected population and
sample it to produce the new population, so the *global* structure (the DAG)
plays the same role in both.  They differ in how the local conditional
distributions are represented and in how new solutions enter the population:

1. Local structure of the CPDs.
   BOA stores each conditional distribution p(Xᵢ | Paᵢ) as a full conditional
   probability table (CPT) with one row per parent configuration.  The table
   has ∏_{Xⱼ ∈ Paᵢ} rⱼ rows, so its size grows exponentially with the number
   of parents and BOA must keep max_parents small (typically 2-3).

   hBOA stores each conditional distribution as a *decision tree* or
   *decision graph* over the parents.  Only the parent configurations that
   the data actually distinguishes get their own parameters; configurations
   that behave alike share a leaf.  This is the central difference: it makes
   the number of parameters grow with the *complexity of the dependency*
   rather than with the number of parents, so hBOA can afford many more
   parents per variable than BOA on the same population size.

2. Merging of equivalent contexts (decision graphs).
   A decision *tree* can only split contexts apart.  A decision *graph* can
   additionally merge two leaves into one, letting distinct parent
   configurations share a single set of parameters.  Merging is what allows
   hBOA to express "these several contexts are equivalent", which is exactly
   the situation created by hierarchical problems where a low-level building
   block has already been chunked.  ``local_structure="dg"`` selects this
   representation; ``"dt"`` restricts hBOA to trees.

3. Niching in the replacement step.
   BOA replaces the population generationally.  hBOA uses restricted
   tournament replacement (RTR): each new solution competes against the most
   similar member of a random subset of the current population and replaces
   it only if it is better.  Preserving alternative partial solutions until
   their context is resolved is essential on hierarchical problems.

   RTR is a *replacement* component, not a learning one, so it is orthogonal
   to this class.  Following the task specification, this module is used with
   pateda's default truncation selection and the replacement method chosen in
   the ``EDA`` configuration; combine it with a niching replacement from
   ``pateda.replacement`` to obtain the complete hBOA of Pelikan (2005).

4. Scoring.
   BOA scores a tabular parent set with K2/BDe or BIC.  hBOA scores a *split*
   in the decision tree/graph: ``"dt"`` uses an MDL (BIC-style) criterion that
   accepts a split only when its likelihood gain outweighs the extra leaves,
   ``"dg"`` uses the Bayesian-Dirichlet score of Chickering, Heckerman and
   Meek (1997) over the graph, which also governs leaf merges.

Implementation note
-------------------
The decision tree / decision graph is used by ``bayes_nets`` as the *scoring*
model that drives the structure search: a parent set is accepted on the basis
of how compactly its CPD can be represented locally.  The resulting DAG is
then parameterized with tabular CPDs so that the model can be sampled by the
standard :class:`~pateda.sampling.bayesian_network.SampleBayesianNetwork`.
The local structure therefore shapes *which* dependencies hBOA keeps (its
practical effect on optimization) while sampling stays interchangeable with
the other Bayesian-network EDAs in pateda.

Because the local structure absorbs the cost of large parent sets, the
default ``max_parents`` here is larger than BOA's, and ``limit_joint_table_size``
defaults to ``False``: the tabular-table-size guard that protects BOA from
overfitting a huge CPT is precisely the restriction hBOA is designed to lift.

Computational Complexity
------------------------
- Structure search: hill-climbing with add/delete/reverse moves, each move
  rescored by building a local tree/graph, so more expensive per move than
  BOA's table lookup but tolerant of far larger parent sets.
- Overall the most expensive of the discrete BN-based EDAs in pateda.

When to use
-----------
- Hierarchically decomposable problems (HIFF, hierarchical trap).
- Problems with many overlapping dependencies where BOA's parent limit binds.
- Ising spin glasses and MAXSAT, the benchmarks of Pelikan & Goldberg (2003).

References
----------
- Pelikan, M., & Goldberg, D.E. (2001). "Escaping hierarchical traps with
  competent genetic algorithms." GECCO 2001, pp. 511-518.
- Pelikan, M., & Goldberg, D.E. (2003). "Hierarchical BOA Solves Ising Spin
  Glasses and MAXSAT." GECCO 2003, pp. 1271-1282.
- Pelikan, M. (2005). "Hierarchical Bayesian Optimization Algorithm: Toward a
  New Generation of Evolutionary Algorithms." Springer.
- Chickering, D.M., Heckerman, D., & Meek, C. (1997). "A Bayesian approach to
  learning Bayesian networks with local structure." UAI-97, pp. 80-89.
- Friedman, N., & Goldszmidt, M. (1996). "Learning Bayesian networks with
  local structure." UAI-96, pp. 252-262.
"""

from typing import Any, Optional
import numpy as np

from bayes_nets import BayesianNetwork

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.utils.weights import normalize_probabilities


class LearnHBOA(LearningMethod):
    """
    Learn an hBOA (hierarchical Bayesian Optimization Algorithm) model

    hBOA learns a Bayesian network whose conditional distributions are
    represented with decision trees or decision graphs, which lets each
    variable take many more parents than BOA can afford with tabular CPDs.

    Structure and parameter learning are delegated to
    :class:`bayes_nets.BayesianNetwork`; this class adapts the result to the
    pateda :class:`~pateda.core.models.BayesianNetworkModel` contract
    (``structure`` = adjacency matrix, ``parameters`` = dict mapping each
    variable to ``{"parents": [...], "cpd": array}``).

    See the module docstring for the differences with respect to
    :class:`~pateda.learning.boa.LearnBOA`.
    """

    def __init__(
        self,
        max_parents: int = 6,
        local_structure: str = "dg",
        alpha: float = 1.0,
        max_tree_depth: Optional[int] = None,
        limit_joint_table_size: bool = False,
    ):
        """
        Initialize hBOA learning

        Args:
            max_parents: Maximum number of parents per variable.  Larger than
                BOA's default because the local structure keeps the number of
                parameters under control.
            local_structure: Local CPD representation used during structure
                search: "dg"/"decision_graph" (default, allows leaf merging)
                or "dt"/"decision_tree" (splits only).
            alpha: Dirichlet/Laplace smoothing used by the local scorer and by
                the final tabular parameter estimation
            max_tree_depth: Maximum depth of the local decision tree/graph
                (None = unbounded, limited only by max_parents)
            limit_joint_table_size: If True, only allow parent sets whose
                tabular conditional-table size (variable + parents) is
                <= n_samples.  Defaults to False for hBOA, since bounding the
                tabular table size defeats the purpose of local structure.
        """
        if local_structure not in ("dt", "decision_tree", "dg", "decision_graph"):
            raise ValueError(
                "local_structure must be 'dt'/'decision_tree' or "
                f"'dg'/'decision_graph', got {local_structure!r}"
            )
        self.max_parents = max_parents
        self.local_structure = local_structure
        self.alpha = alpha
        self.max_tree_depth = max_tree_depth
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
        Learn hBOA model from population

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

        # "dt" -> decision-tree MDL scorer, "dg" -> decision-graph Bayesian
        # scorer.  Both are driven by a stable hill-climbing search.
        method = "dt" if self.local_structure in ("dt", "decision_tree") else "dg"

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        # Use the decision tree/graph both to *score* the structure search and
        # to *represent and sample* the local CPDs (``local_structure=method``),
        # so the compact context-specific structure is exploited at sampling
        # time by SampleLocalStructureBN -- the faithful hBOA of Pelikan (2005),
        # rather than flattening the CPDs to dense tables.
        bn.fit(
            data,
            method=method,
            local_structure=method,
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
                "model_type": "HBOA",
                "max_parents": self.max_parents,
                "local_structure": self.local_structure,
                "alpha": self.alpha,
                "max_tree_depth": self.max_tree_depth,
            },
        )

        return model
