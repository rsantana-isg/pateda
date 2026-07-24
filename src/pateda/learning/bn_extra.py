"""
Additional Bayesian-network structure learners for plug-and-play EDAs.

This module adds structure-learning methods whose *paradigm* differs from the
score-and-search greedy search used by EBNA, BOA and LFDA (all three of which
are close relatives; see :mod:`pateda.learning.lfda`).  They are the learners
behind the SARTRE-EDA, BINOTEARS-EDA, PCBN-EDA and their hierarchical (decision
graph + niching) counterparts:

* :class:`LearnSARTRE` -- SARTRE, an order-based method that prunes spurious
  edges through a sparse additive model fitted by group-lasso regression.
* :class:`LearnBINOTEARS` -- BINOTEARS, which casts *binary*-data structure
  learning as a differentiable continuous optimization with an acyclicity
  constraint (binary variables only).
* :class:`LearnPCBN` -- the constraint-based PC algorithm in a **bounded
  conditioning-order** form (order-independent PC-Stable with a hard cap on the
  size of the conditioning set), which is what makes PC affordable at
  ``n >= 100``.
* :class:`LearnHSARTRE`, :class:`LearnHBINOTEARS` -- the hierarchical-BOA style
  upgrade of the two scalable learners: the base method discovers the skeleton
  and a decision graph is then learned *restricted to that skeleton* so the
  local conditional distributions capture context-specific independence and
  more parents can be afforded.  Pair these with a niching replacement (e.g.
  :class:`~pateda.replacement.niching.RestrictedTournamentReplacement`) to
  obtain the complete hBOA-style algorithm; the plug-and-play EDA classes
  ``HSARTRE_EDA`` / ``HBINOTEARS_EDA`` do exactly that.

Design note (decision graphs)
-----------------------------
In the current ``bayes_nets`` the decision graph is used as the *scoring* model
that drives which parents are kept; the final CPDs are tabular so that the
standard :class:`~pateda.sampling.bayesian_network.SampleBayesianNetwork` can
sample them.  ``LearnHSARTRE`` / ``LearnHBINOTEARS`` therefore obtain the
decision-graph benefit by running the ``"dg"`` learner constrained to the base
skeleton via ``interaction_matrix``.  Should ``bayes_nets`` later expose a
first-class "given a DAG, fit and sample decision-graph CPDs" primitive (e.g.
``fit(method="sartre", local_structure="dg")``), only :meth:`_refine` below
needs to change.
"""

from typing import Any, Optional
import numpy as np

from bayes_nets import BayesianNetwork

from pateda.core.components import LearningMethod
from pateda.core.models import BayesianNetworkModel
from pateda.learning.ebna import LearnEBNA, _acyclic_orientation
from pateda.learning.utils.weights import normalize_probabilities


# ---------------------------------------------------------------------------
# Thin single-method learners (reuse the generic LearnEBNA machinery)
# ---------------------------------------------------------------------------
class LearnSARTRE(LearnEBNA):
    """Learn a BN with SARTRE (sparse additive / group-lasso edge pruning).

    A drop-in replacement for :class:`~pateda.learning.ebna.LearnEBNA` that
    fixes the ``bayes_nets`` method to ``"sartre"``.  SARTRE is fast and scales
    to large ``n``, and gives the most accurate skeletons ($F_1$) in the
    benchmark of the accompanying study.
    """

    def __init__(self, max_parents: int = 4, alpha: float = 1.0,
                 limit_joint_table_size: bool = True):
        super().__init__(max_parents=max_parents, score_metric="sartre",
                         alpha=alpha, limit_joint_table_size=limit_joint_table_size,
                         model_type="SARTRE")


class LearnBINOTEARS(LearnEBNA):
    """Learn a BN with BINOTEARS (differentiable structure learning).

    A drop-in replacement for :class:`~pateda.learning.ebna.LearnEBNA` that
    fixes the ``bayes_nets`` method to ``"binotears"``.

    .. note::
       BINOTEARS supports **binary variables only**.  On problems with any
       cardinality > 2 the underlying learner raises; use it only on binary
       problems (Trap, Deceptive, Ising, OneMax, UBQP, ...).
    """

    def __init__(self, max_parents: int = 4, alpha: float = 1.0,
                 limit_joint_table_size: bool = True):
        super().__init__(max_parents=max_parents, score_metric="binotears",
                         alpha=alpha, limit_joint_table_size=limit_joint_table_size,
                         model_type="BINOTEARS")


# ---------------------------------------------------------------------------
# Bounded-conditioning-order PC (constraint-based)
# ---------------------------------------------------------------------------
class LearnPCBN(LearningMethod):
    """Constraint-based BN learning with the order-independent PC-Stable
    algorithm and a **bounded conditioning-set size**.

    Plain PC/PC-Stable conditions on ever larger separating sets, which makes
    the number of independence tests grow steeply with ``n`` and prevents the
    method from finishing on the largest instances.  Capping the conditioning
    set at ``max_cond_set_size`` (the ``k`` of order-limited PC) keeps the cost
    manageable while still recovering the skeleton of sparse graphs, so the
    method scales to ``n >= 100``.

    Args:
        max_cond_set_size: Hard cap on the size of the conditioning set in the
            independence tests (the order limit; 2--3 is a good default).
        max_parents: Maximum parents per variable in the oriented DAG.
        alpha_ci: Significance level of the conditional-independence tests.
        stable: Use the order-independent PC-Stable (default) or plain PC.
        alpha: Laplace smoothing for the final CPD estimation.
    """

    def __init__(self, max_cond_set_size: int = 3, max_parents: int = 4,
                 alpha_ci: float = 0.05, stable: bool = True, alpha: float = 1.0):
        self.max_cond_set_size = int(max_cond_set_size)
        self.max_parents = int(max_parents)
        self.alpha_ci = float(alpha_ci)
        self.stable = bool(stable)
        self.alpha = float(alpha)

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray,
              population: np.ndarray, fitness: np.ndarray,
              **params: Any) -> BayesianNetworkModel:
        from bayes_nets.structure_learning import PCLearner, StablePCLearner

        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        cls_pc = StablePCLearner if self.stable else PCLearner
        learner = cls_pc(alpha_ci=self.alpha_ci,
                         max_cond_set_size=self.max_cond_set_size,
                         max_parents=self.max_parents)
        adjacency = learner.learn(data, n_vars, cardinality,
                                  sample_weights=sample_weights)

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        adjacency = np.asarray(adjacency, dtype=int)
        # CPDAGs from constraint-based search can be cyclic/undirected; repair.
        bn.set_structure(_acyclic_orientation(adjacency))
        bn.learn_parameters(data, alpha=self.alpha, sample_weights=sample_weights)

        return BayesianNetworkModel(
            structure=bn.to_adjacency_matrix(),
            parameters=bn.cpds,
            metadata={
                "generation": generation,
                "model_type": "PCBN",
                "max_cond_set_size": self.max_cond_set_size,
                "max_parents": self.max_parents,
                "stable": self.stable,
            },
        )


# ---------------------------------------------------------------------------
# Hierarchical (decision-graph + skeleton) composite learners
# ---------------------------------------------------------------------------
class _LearnDGRefined(LearningMethod):
    """Base class: learn the structure with *base_method* and represent the
    local conditional distributions with a decision tree/graph.

    This is the learning half of the hBOA-style upgrade: the base method
    (SARTRE, BINOTEARS, ...) decides *which* interactions exist, and the
    decision graph decides *how* the local conditional distribution over those
    interacting parents is represented, capturing context-specific independence
    so that more parents can be afforded than with tabular CPDs.

    The composition is done natively by ``bayes_nets``:
    ``bn.fit(method=base_method, local_structure="dg")`` learns the structure
    with the base learner and then fits decision-graph CPDs, which are stored in
    ``bn.cpds[var]["local"]`` and sampled directly (never materialising the
    dense table) by
    :class:`~pateda.sampling.bayesian_network.SampleLocalStructureBN`.
    """

    base_method: str = "sartre"
    model_type: str = "H-BN"

    def __init__(self, max_parents: int = 6, local_structure: str = "dg",
                 alpha: float = 1.0):
        if local_structure not in ("dt", "decision_tree", "dg", "decision_graph"):
            raise ValueError("local_structure must be 'dt' or 'dg'")
        self.max_parents = int(max_parents)
        self.local_structure = "dt" if local_structure in ("dt", "decision_tree") else "dg"
        self.alpha = float(alpha)

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray,
              population: np.ndarray, fitness: np.ndarray,
              **params: Any) -> BayesianNetworkModel:
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)
        sample_weights = normalize_probabilities(params.get("p"), data.shape[0])

        bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
        # Base structure learner + decision-graph local CPDs, in one call.
        bn.fit(
            data, method=self.base_method, local_structure=self.local_structure,
            max_parents=self.max_parents, alpha=self.alpha,
            limit_table_size=False, sample_weights=sample_weights,
        )

        # Base learners used here (sartre, binotears) return DAGs; guard anyway.
        if not bn.is_dag():
            bn.set_structure(_acyclic_orientation(bn.to_adjacency_matrix()))
            bn.learn_local_structure(data, structure=self.local_structure,
                                     sample_weights=sample_weights)

        return BayesianNetworkModel(
            structure=bn.to_adjacency_matrix(),
            parameters=bn.cpds,
            metadata={
                "generation": generation,
                "model_type": self.model_type,
                "base_method": self.base_method,
                "local_structure": self.local_structure,
                "max_parents": self.max_parents,
            },
        )


class LearnHSARTRE(_LearnDGRefined):
    """SARTRE skeleton refined with a decision graph (hBOA-style local structure)."""
    base_method = "sartre"
    model_type = "HSARTRE"


class LearnHBINOTEARS(_LearnDGRefined):
    """BINOTEARS skeleton refined with a decision graph (hBOA-style local structure).

    Binary variables only (inherited from BINOTEARS).
    """
    base_method = "binotears"
    model_type = "HBINOTEARS"
