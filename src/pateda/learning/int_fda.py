"""
Int_FDA (Tree-based Factorized Distribution Algorithm for Integers) — learning

Int_FDA is a tree-based FDA conceived for problems whose variables have *very
high cardinality*.  Its defining feature is that the learned model is **not a
probability model**: it never materializes conditional probability tables.
Instead, the model is the tree structure together with two auxiliary index
tables and the *selected population itself*.  This is what makes it practical
when the cardinality ``c`` is large: a tree FDA that stored bivariate
marginals would need ``O(c² · n²)`` memory, whereas Int_FDA needs only
``O(N · n + c · n)`` (``N`` = size of the selected set), because the empirical
conditional ``p(X_j | X_i)`` is represented *implicitly* by pointers into the
selected population rather than by a ``c × c`` table.

This module implements the *learning* half.  The companion sampler,
:class:`~pateda.sampling.int_fda.SampleIntFDA`, reconstructs new individuals
by copying genes from donor vectors of the selected population, guided by the
tree and the index tables — see that module for the generation algorithm.

Relationship to the other FDAs in pateda
----------------------------------------
Int_FDA shares its *structure learning* with the Tree-EDA
(:class:`~pateda.learning.tree.LearnTreeModel`): a Chow-Liu maximum-weight
spanning tree over a pairwise mutual-information matrix.  The two differ only
in what fills the tree:

- ``LearnTreeModel`` estimates, for every edge, a conditional probability
  table ``P(child | parent)`` of size ``c_parent × c_child``.  Sampling draws
  from those tables.  Memory and time grow with ``c²``.
- ``LearnIntFDA`` estimates *nothing* numerically.  It stores index tables so
  that "pick a random selected vector whose parent equals value ``k``" is an
  ``O(1)`` operation, and copies the child gene from that donor.  The sampled
  distribution is *exactly* the Chow-Liu tree distribution of the selected set
  (same as Tree-EDA would define), but no ``c²`` table is ever built.

This mirrors the "non-probabilistic model" paradigm already used by the
crossover operators in pateda (see
:class:`~pateda.crossover.block.LearnBlockCrossover`), where the model is a set
of indices plus the population rather than a probability distribution.

Auxiliary tables (following Santana, Ochoa & Soto, 2002)
--------------------------------------------------------
For the selected population ``S`` of ``N`` vectors and ``n`` variables:

- ``PopulValues`` — shape ``(N, n)``.  Column ``i`` holds the *row indices* of
  ``S`` sorted in ascending order of the value that variable ``i`` takes.  Thus
  all vectors sharing a given value of variable ``i`` occupy a contiguous block
  of rows in column ``i``.
- ``ParentIndices`` — one array per variable ``i``, of length
  ``cardinality[i] + 1``, holding the cumulative counts
  ``ParentIndices[i][v] = #{ vectors with X_i < v }``.  Consequently the block
  of ``PopulValues[:, i]`` corresponding to value ``v`` is the half-open range
  ``[ParentIndices[i][v], ParentIndices[i][v+1])``.  (In the paper this table
  is stored as the *last* index of each value with a ``-1`` sentinel; the
  cumulative form used here is the equivalent boundary representation and
  removes the special-casing of the first value.)

Together, ``ParentIndices`` locates the block for a parent value in ``O(1)`` and
``PopulValues`` turns a position in that block into a donor-vector index.

Mutual information for high cardinality
---------------------------------------
The mutual-information matrix that drives the spanning tree is computed
*without* allocating any ``c × c`` array (which would defeat the purpose).  For
each pair ``(i, j)`` only the *observed* joint value-pairs are enumerated — at
most ``N`` of them — so the cost is ``O(n² · N log N)`` time and ``O(N)`` extra
memory per pair, regardless of the cardinality.  The raw mutual information
``I(X_i, X_j) = Σ p(a,b) log( p(a,b) / (p(a) p(b)) )`` is used as the edge
weight, exactly as in the paper's Chow-Liu construction.

References
----------
- Santana, R., Ochoa, A., & Soto, M. R. (2002). "Solving problems with integer
  representation using a tree based Factorized Distribution Algorithm."
  (Int-Tree / Int_FDA.)  Center of Mathematics and Theoretical Physics,
  ICIMAF, Havana, Cuba.
- Chow, C., & Liu, C. (1968). "Approximating discrete probability distributions
  with dependence trees." IEEE Transactions on Information Theory, 14(3).
- Baluja, S., & Davies, S. (1997). "Using optimal dependency-trees for
  combinatorial optimization." ICML 1997.
"""

from typing import Any, List, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import Model
from pateda.learning.tree import LearnTreeModel


class LearnIntFDA(LearningMethod):
    """
    Learn an Int_FDA (tree-based FDA for integers) model.

    The learned :class:`~pateda.core.models.Model` stores, instead of
    probability tables:

    - ``structure``: the tree as a cliques array
      ``[n_parents, n_new, parent_or_root_idx, child_idx]`` (the same format
      produced by :class:`~pateda.learning.tree.LearnTreeModel`, ordered so
      every parent precedes its children).
    - ``parameters``: a dict with

        * ``"selected_population"``: the selected set ``S`` (``N × n`` int array),
        * ``"popul_values"``: the ``PopulValues`` table (``N × n`` int array),
        * ``"parent_indices"``: the ``ParentIndices`` tables (list of
          ``cardinality[i] + 1`` cumulative-count arrays).

    The companion sampler is
    :class:`~pateda.sampling.int_fda.SampleIntFDA`.

    Note on customized selection: Int_FDA is a non-probabilistic model whose
    generation copies genes uniformly from the stored selected set, so it does
    not consume the per-individual weight vector ``p`` that fitness-weighted
    selection produces; the ``p`` argument is accepted and ignored (the
    default "uniform" weighting passes ``p = None`` anyway).
    """

    def __init__(self, mi_threshold: float = 1e-4):
        """
        Initialize Int_FDA learning.

        Args:
            mi_threshold: Minimum mutual information for an edge to be added to
                the tree.  Pairs below the threshold produce isolated roots
                (a forest), exactly as in
                :class:`~pateda.learning.tree.LearnTreeModel`.
        """
        self.mi_threshold = mi_threshold
        # Reused only for its (well tested) maximum-weight-spanning-tree
        # construction; the MI matrix is computed here in a cardinality-safe way.
        self._tree_builder = LearnTreeModel(mi_threshold=mi_threshold)

    # ------------------------------------------------------------------
    # Mutual information (high-cardinality safe)
    # ------------------------------------------------------------------
    def _mutual_information_matrix(
        self, population: np.ndarray, n_vars: int
    ) -> np.ndarray:
        """
        Compute the pairwise raw mutual-information matrix.

        Only observed value-pairs are enumerated, so no ``c × c`` table is ever
        allocated and the memory cost is ``O(N)`` per variable pair, whatever
        the cardinality.

        Args:
            population: Selected population (N, n_vars).
            n_vars: Number of variables.

        Returns:
            Symmetric (n_vars, n_vars) matrix of raw mutual information.
        """
        n_samples = population.shape[0]
        mi = np.zeros((n_vars, n_vars))

        # Dense per-variable relabelling of values to [0, n_distinct):
        # inverse[i] maps every row to the rank of its value, counts[i] holds
        # the marginal counts.  Both are O(N) regardless of cardinality.
        inverse: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        for i in range(n_vars):
            _, inv, cnt = np.unique(
                population[:, i], return_inverse=True, return_counts=True
            )
            inverse.append(inv.astype(np.int64))
            counts.append(cnt.astype(np.float64))

        for i in range(n_vars - 1):
            inv_i = inverse[i]
            n_i = counts[i]
            for j in range(i + 1, n_vars):
                inv_j = inverse[j]
                n_j = counts[j]
                k_j = n_j.shape[0]

                # Joint counts over *observed* pairs only.  Encode each pair as
                # a single integer key so a 1-D np.unique suffices (faster than
                # np.unique(axis=0)); the key range is < N² which fits int64.
                key = inv_i * k_j + inv_j
                uniq_key, joint = np.unique(key, return_counts=True)
                a = uniq_key // k_j
                b = uniq_key % k_j

                p_ab = joint / n_samples
                # I = Σ p(a,b) log( N · n_ab / (n_a · n_b) )
                mi_ij = float(
                    np.sum(p_ab * np.log(n_samples * joint / (n_i[a] * n_j[b])))
                )
                mi_ij = max(0.0, mi_ij)
                mi[i, j] = mi_ij
                mi[j, i] = mi_ij

        return mi

    # ------------------------------------------------------------------
    # Auxiliary tables
    # ------------------------------------------------------------------
    @staticmethod
    def _build_auxiliary_tables(
        population: np.ndarray, cardinality: np.ndarray
    ):
        """
        Build the ``PopulValues`` and ``ParentIndices`` tables.

        Args:
            population: Selected population (N, n_vars), integer valued.
            cardinality: Per-variable cardinalities.

        Returns:
            Tuple ``(popul_values, parent_indices)`` where

            - ``popul_values`` has shape (N, n_vars); column ``i`` are the row
              indices of ``population`` sorted by the value of variable ``i``;
            - ``parent_indices`` is a list of 1-D int arrays; entry ``i`` has
              length ``cardinality[i] + 1`` and holds cumulative counts so that
              value ``v`` of variable ``i`` occupies rows
              ``[parent_indices[i][v], parent_indices[i][v+1])`` of
              ``popul_values[:, i]``.
        """
        n_samples, n_vars = population.shape

        # PopulValues: stable argsort per column keeps equal-valued vectors in a
        # deterministic order (irrelevant to correctness, helps reproducibility).
        popul_values = np.argsort(population, axis=0, kind="stable").astype(np.int64)

        parent_indices: List[np.ndarray] = []
        for i in range(n_vars):
            card_i = int(cardinality[i])
            col = population[:, i].astype(np.int64)
            if col.size and int(col.max()) >= card_i:
                raise ValueError(
                    f"Variable {i} has value {int(col.max())} >= its declared "
                    f"cardinality {card_i}; check the cardinality vector."
                )
            # Cumulative counts: cum[v] = #{ rows with X_i < v }, length card+1.
            # bincount gives per-value counts; a prefix sum turns them into the
            # block boundaries used by the sampler.
            per_value = np.bincount(col, minlength=card_i).astype(np.int64)
            cum = np.empty(card_i + 1, dtype=np.int64)
            cum[0] = 0
            np.cumsum(per_value, out=cum[1:])
            parent_indices.append(cum)

        return popul_values, parent_indices

    # ------------------------------------------------------------------
    # Learning entry point
    # ------------------------------------------------------------------
    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Model:
        """
        Learn an Int_FDA model from the selected population.

        Args:
            generation: Current generation number.
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            population: Selected population to learn from.
            fitness: Fitness values (unused by Int_FDA).
            **params: Additional parameters.  ``p`` (fitness weights) is
                accepted for interface compatibility but ignored — see the
                class docstring.

        Returns:
            A :class:`~pateda.core.models.Model` whose ``structure`` is the tree
            and whose ``parameters`` hold the selected population and the index
            tables (no probability tables).
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)

        # 1. Structure: Chow-Liu maximum-weight spanning tree on the raw MI.
        mi_matrix = self._mutual_information_matrix(data, n_vars)
        cliques = self._tree_builder._create_tree_structure(mi_matrix, n_vars)

        # 2. Auxiliary tables + the selected population itself (the "model").
        popul_values, parent_indices = self._build_auxiliary_tables(data, cardinality)

        model = Model(
            structure=cliques,
            parameters={
                "selected_population": data,
                "popul_values": popul_values,
                "parent_indices": parent_indices,
            },
            metadata={
                "generation": generation,
                "model_type": "IntFDA",
                "mi_threshold": self.mi_threshold,
                "n_selected": data.shape[0],
                "cardinality": cardinality,
                "mi_matrix": mi_matrix,
            },
        )

        return model
