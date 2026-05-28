"""
Bivariate Marginal Distribution Algorithm (BMDA) learning

Equivalent to MATEDA's LearnBMDA.m
BMDA models pairwise dependencies between variables.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnBMDA(LearningMethod):
    """
    Learn a BMDA (Bivariate Marginal Distribution Algorithm) model

    BMDA represents the probability distribution using bivariate marginals,
    capturing pairwise dependencies between variables. It uses a tree structure
    or complete pairwise model.
    """

    def __init__(
        self,
        structure: Optional[np.ndarray] = None,
        structure_learning: str = "tree",
        alpha: float = 0.0,
    ):
        """
        Initialize BMDA learning

        Args:
            structure: Pre-defined dependency structure (adjacency matrix or edge list)
                      If None, structure is learned from data
            structure_learning: Method for learning structure ("tree", "complete", "greedy")
                              - "tree": Learn maximum spanning tree based on mutual information
                              - "complete": Use all pairwise dependencies
                              - "greedy": Greedy structure learning
            alpha: Smoothing parameter for probability estimation
        """
        self.structure = structure
        self.structure_learning = structure_learning
        self.alpha = alpha

    def _mutual_information(
        self, var1_data: np.ndarray, var2_data: np.ndarray, k1: int, k2: int
    ) -> float:
        """
        Calculate mutual information between two variables

        Args:
            var1_data: Data for first variable
            var2_data: Data for second variable
            k1: Cardinality of first variable
            k2: Cardinality of second variable

        Returns:
            Mutual information value
        """
        n = len(var1_data)

        # Calculate joint and marginal counts
        joint_counts = np.zeros((k1, k2))
        for i in range(n):
            joint_counts[int(var1_data[i]), int(var2_data[i])] += 1

        # Add small constant to avoid log(0)
        joint_counts += 1e-10
        joint_probs = joint_counts / np.sum(joint_counts)

        # Marginal probabilities
        marginal1 = np.sum(joint_probs, axis=1)
        marginal2 = np.sum(joint_probs, axis=0)

        # Calculate mutual information
        mi = 0.0
        for i in range(k1):
            for j in range(k2):
                if joint_probs[i, j] > 1e-10:
                    mi += joint_probs[i, j] * np.log(
                        joint_probs[i, j] / (marginal1[i] * marginal2[j])
                    )

        return mi

    def _learn_tree_structure(
        self, population: np.ndarray, n_vars: int, cardinality: np.ndarray
    ) -> np.ndarray:
        """
        Learn tree structure using maximum spanning tree on mutual information

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Adjacency matrix representing tree structure
        """
        # Calculate mutual information between all pairs
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mi = self._mutual_information(
                    population[:, i],
                    population[:, j],
                    int(cardinality[i]),
                    int(cardinality[j]),
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Find maximum spanning tree using Prim's algorithm
        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        visited = np.zeros(n_vars, dtype=bool)
        visited[0] = True

        for _ in range(n_vars - 1):
            max_mi = -1
            best_i, best_j = -1, -1

            for i in range(n_vars):
                if visited[i]:
                    for j in range(n_vars):
                        if not visited[j] and mi_matrix[i, j] > max_mi:
                            max_mi = mi_matrix[i, j]
                            best_i, best_j = i, j

            if best_i >= 0:
                adjacency[best_i, best_j] = 1
                adjacency[best_j, best_i] = 1
                visited[best_j] = True

        return adjacency

    def _build_directed_tree(
        self, adjacency: np.ndarray, mi_matrix: Optional[np.ndarray] = None
    ):
        """
        Convert an undirected (forest of) tree adjacency into a sequence of
        directed edges (parent → child) via BFS plus the set of roots.

        Args:
            adjacency: Symmetric 0/1 matrix describing the (forest) edges.
            mi_matrix: Optional MI matrix used to pick the root of each tree
                component as the variable with the highest total MI; falls
                back to the lowest-index variable when not given.

        Returns:
            roots, edges, order
                - roots: list of int, one per connected component
                - edges: list of (parent, child) tuples in BFS order
                - order: list of int giving the BFS visit order over all
                  variables (used to build the clique array).
        """
        n_vars = adjacency.shape[0]
        visited = np.zeros(n_vars, dtype=bool)
        roots = []
        edges = []
        order = []

        # Iterate over connected components so disconnected variables (the
        # spanning-tree learner always connects everything, but a user-provided
        # adjacency may be a forest) are also handled.
        while not visited.all():
            unvisited = np.where(~visited)[0]
            if mi_matrix is not None:
                # Pick the unvisited variable with the highest total MI as
                # the root of its component.
                mi_sums = mi_matrix[unvisited].sum(axis=1)
                root = int(unvisited[int(np.argmax(mi_sums))])
            else:
                root = int(unvisited[0])
            roots.append(root)

            queue = [root]
            visited[root] = True
            order.append(root)

            while queue:
                node = queue.pop(0)
                for j in range(n_vars):
                    if adjacency[node, j] and not visited[j]:
                        edges.append((node, j))
                        visited[j] = True
                        order.append(j)
                        queue.append(j)

        return roots, edges, order

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn BMDA model from population

        The model is encoded as a tree-structured factorization compatible
        with :class:`pateda.sampling.fda.SampleFDA`:

        - exactly one ``[0, 1, root]`` clique per connected component,
          carrying the marginal P(root);
        - one ``[1, 1, parent, child]`` clique per non-root variable,
          carrying the conditional table P(child | parent).

        Storing each edge as a joint ``[0, 2, i, j]`` factor (as an earlier
        version did) caused ``SampleFDA`` to re-sample every shared variable
        independently for every edge it appeared in, which destroyed the
        learned dependencies and produced essentially marginal samples.

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for BMDA learning)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel with bivariate (tree) structure
        """
        alpha = params.get("alpha", self.alpha)
        pop_size = population.shape[0]
        cardinality_int = np.asarray(cardinality, dtype=int)

        # ------------------------------------------------------------------
        # 1) Learn the undirected dependency graph (max-spanning tree)
        # ------------------------------------------------------------------
        if self.structure is not None:
            adjacency = np.asarray(self.structure, dtype=int)
            mi_matrix = None
        elif self.structure_learning == "tree":
            # Compute MI matrix once for both structure learning and root choice.
            mi_matrix = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    mi = self._mutual_information(
                        population[:, i],
                        population[:, j],
                        int(cardinality_int[i]),
                        int(cardinality_int[j]),
                    )
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

            # Prim's algorithm — same logic as _learn_tree_structure but reuses
            # the MI matrix we just computed.
            adjacency = np.zeros((n_vars, n_vars), dtype=int)
            visited = np.zeros(n_vars, dtype=bool)
            visited[0] = True
            for _ in range(n_vars - 1):
                max_mi = -np.inf
                best_i, best_j = -1, -1
                for i in range(n_vars):
                    if visited[i]:
                        for j in range(n_vars):
                            if not visited[j] and mi_matrix[i, j] > max_mi:
                                max_mi = mi_matrix[i, j]
                                best_i, best_j = i, j
                if best_j >= 0:
                    adjacency[best_i, best_j] = 1
                    adjacency[best_j, best_i] = 1
                    visited[best_j] = True
        elif self.structure_learning == "complete":
            # Complete graph cannot be decomposed as a tree; warn and degrade
            # to a max-spanning tree from a uniform MI matrix (random tree).
            mi_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
            adjacency = np.zeros((n_vars, n_vars), dtype=int)
            for i in range(n_vars - 1):
                adjacency[i, i + 1] = 1
                adjacency[i + 1, i] = 1
        else:
            raise ValueError(f"Unknown structure_learning method: {self.structure_learning}")

        # ------------------------------------------------------------------
        # 2) Convert the undirected tree into a directed factorization
        # ------------------------------------------------------------------
        roots, edges, order = self._build_directed_tree(adjacency, mi_matrix)

        # ------------------------------------------------------------------
        # 3) Marginal P(var) for every variable (used both for roots and as a
        #    denominator when deriving conditional tables).
        # ------------------------------------------------------------------
        # Use at least 1 pseudo-count per cell so that unseen values get
        # non-zero probability and high-cardinality marginals are smoothed.
        smooth = max(alpha, 1.0)
        marginal_probs = []
        for v in range(n_vars):
            k = int(cardinality_int[v])
            counts = np.bincount(population[:, v].astype(int), minlength=k).astype(float)
            counts = counts + smooth
            marginal_probs.append(counts / counts.sum())

        # ------------------------------------------------------------------
        # 4) Build cliques in BFS order so every parent is sampled before its
        #    children (this is what SampleFDA assumes).
        # ------------------------------------------------------------------
        cliques: list = []
        tables: list = []

        # First, emit one root clique per connected component.
        root_set = set(roots)
        for root in roots:
            cliques.append([0, 1, root, 0])
            tables.append(marginal_probs[root])

        # Then, emit one P(child | parent) clique per directed edge.
        for parent, child in edges:
            k_p = int(cardinality_int[parent])
            k_c = int(cardinality_int[child])

            # Joint count P(parent, child)
            joint_counts = np.zeros((k_p, k_c))
            pv = population[:, parent].astype(int)
            cv = population[:, child].astype(int)
            np.add.at(joint_counts, (pv, cv), 1)

            # Conditional P(child | parent) with a Dirichlet prior centred at
            # the child marginal.  Sparse parent rows (high-cardinality parent,
            # few samples per value) fall back to the child marginal rather
            # than to uniform, preserving the effective distribution.
            c_counts = np.bincount(cv, minlength=k_c).astype(float)
            child_marginal = (c_counts + smooth) / (pop_size + smooth * k_c)
            row_sums = joint_counts.sum(axis=1, keepdims=True)
            cond = (joint_counts + smooth * child_marginal[np.newaxis, :]) / (row_sums + smooth)

            cliques.append([1, 1, parent, child])
            tables.append(cond)

        # ------------------------------------------------------------------
        # 5) Pack cliques into the (n_cliques, max_len) numpy structure
        #    SampleFDA expects.
        # ------------------------------------------------------------------
        max_len = max(len(c) for c in cliques)
        cliques_array = np.zeros((len(cliques), max_len), dtype=int)
        for i, clique in enumerate(cliques):
            cliques_array[i, : len(clique)] = clique

        model = FactorizedModel(
            structure=cliques_array,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "BMDA",
                "structure_learning": self.structure_learning,
                "alpha": alpha,
                "adjacency": adjacency,
                "n_roots": len(roots),
                "n_edges": len(edges),
            },
        )

        return model
