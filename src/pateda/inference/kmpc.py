"""
K Most Probable Configurations (KMPC) algorithms for EDA probabilistic models.

Implements efficient algorithms for finding the K configurations with highest
probability from the probabilistic models used in EDAs:

- KMPCUnivariate : Nilsson's partition scheme for independent variable models (UMDA)
- KMPCTree       : K-MAP for tree-structured Bayesian networks via max-product + partition
- KMPCJunctionTree : General K-MAP for factorized models (FDA, MNFDA) via Nilsson's MFP
- KMPCBayesianNetwork : K-MAP for Bayesian networks (EBNA) via CPD factorization

All classes expose a ``compute(k)`` method returning a ``KMPCResult``.

Dispatch helper ``compute_kmpc(model, cardinality, k)`` selects the appropriate
class automatically based on the model type.

References
----------
Nilsson, D. (1998). An efficient algorithm for finding the M most probable
  configurations in probabilistic expert systems. Statistics and Computing, 8.
Echegoyen, C. et al. (2010). Analyzing the k Most Probable Solutions in EDAs
  Based on Bayesian Networks. In: Exploitation of Linkage Learning.
Santana, R. (2013). Message Passing Methods for Estimation of Distribution
  Algorithms Based on Markov Networks.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pateda.core.models import (
    BayesianNetworkModel,
    FactorizedModel,
    MarkovNetworkModel,
    Model,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KMPCResult:
    """Result of a K most probable configurations search.

    Attributes
    ----------
    configurations : ndarray of shape (k, n_vars)
        Configurations in descending order of probability.
    log_probabilities : ndarray of shape (k,)
        Log-probabilities of each configuration.
    probabilities : ndarray of shape (k,)
        Probabilities of each configuration (may underflow for very small values).
    k_found : int
        Number of distinct configurations actually returned (<= k requested).
    method : str
        Algorithm variant used.
    """

    configurations: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    k_found: int
    method: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level helpers shared by all KMPC variants
# ---------------------------------------------------------------------------

def _eval_log_prob_factorized(
    config: np.ndarray,
    clique_vars_list: List[np.ndarray],
    tables: List[np.ndarray],
) -> float:
    """Log probability of *config* under a factorized model.

    Each factor contributes log table[x_{C1}, x_{C2}, ...] where C1, C2, …
    are the variable indices in the clique.
    """
    log_p = 0.0
    for clique_vars, table in zip(clique_vars_list, tables):
        idx = tuple(int(config[v]) for v in clique_vars)
        try:
            val = table[idx]
        except (IndexError, TypeError):
            val = 1e-300
        log_p += np.log(max(float(val), 1e-300))
    return log_p


def _greedy_conditional_map(
    evidence: Dict[int, int],
    clique_vars_list: List[np.ndarray],
    tables: List[np.ndarray],
    cardinality: np.ndarray,
    var_order: List[int],
) -> np.ndarray:
    """Find the MAP configuration conditional on evidence using a greedy pass.

    For each free variable (in *var_order* order) the method computes a
    log-marginal by accumulating contributions from all cliques that contain
    that variable and whose other members are already fixed, then picks the
    argmax.

    This is exact for tree models (given topological *var_order*) and a good
    approximation for general factorized models.

    Parameters
    ----------
    evidence : dict mapping variable index → fixed value
    clique_vars_list : list of variable-index arrays (one per clique)
    tables : list of probability tables (one per clique)
    cardinality : (n_vars,) array of per-variable cardinalities
    var_order : processing order for free variables (must respect dependencies)

    Returns
    -------
    config : ndarray of shape (n_vars,)
    """
    n_vars = len(cardinality)
    config = np.full(n_vars, -1, dtype=int)

    for var, val in evidence.items():
        config[var] = int(val)

    for var in var_order:
        if config[var] >= 0:
            continue  # fixed by evidence

        card_v = int(cardinality[var])
        log_marginal = np.zeros(card_v)

        for clique_vars, table in zip(clique_vars_list, tables):
            cv_list = list(clique_vars)
            if var not in cv_list:
                continue

            var_pos = cv_list.index(var)

            # Check that all other variables in this clique are fixed
            fixed_idx = []
            all_fixed = True
            for i, v in enumerate(cv_list):
                if v == var:
                    continue
                if config[v] < 0:
                    all_fixed = False
                    break
                fixed_idx.append((i, int(config[v])))

            if not all_fixed:
                # Use value 0 for unset variables (approximation)
                fixed_idx = []
                for i, v in enumerate(cv_list):
                    if v == var:
                        continue
                    val_v = int(config[v]) if config[v] >= 0 else 0
                    fixed_idx.append((i, val_v))

            n_dims = len(cv_list)
            for cand_val in range(card_v):
                idx = [0] * n_dims
                idx[var_pos] = cand_val
                for (pos, fv) in fixed_idx:
                    idx[pos] = fv
                try:
                    prob = float(table[tuple(idx)])
                except (IndexError, TypeError):
                    prob = 1e-300
                log_marginal[cand_val] += np.log(max(prob, 1e-300))

        config[var] = int(np.argmax(log_marginal))

    # Fallback: fill any still-unset variables with 0
    config[config < 0] = 0
    return config


def _extract_clique_vars_from_structure(
    structure: np.ndarray,
) -> Tuple[List[np.ndarray], List[int]]:
    """Parse the MATEDA-style clique structure matrix.

    Each row of *structure* has layout:
      [n_overlap, n_new, overlap_vars..., new_vars...]

    Returns
    -------
    clique_vars_list : list of variable-index arrays (overlap + new vars)
    var_order : list of new variables introduced, in clique processing order
    """
    n_cliques = structure.shape[0]
    clique_vars_list: List[np.ndarray] = []
    var_order: List[int] = []

    for c in range(n_cliques):
        n_overlap = int(structure[c, 0])
        n_new = int(structure[c, 1])
        overlap_vars = structure[c, 2:2 + n_overlap].astype(int)
        new_vars = structure[c, 2 + n_overlap:2 + n_overlap + n_new].astype(int)
        all_vars = np.concatenate([overlap_vars, new_vars])
        clique_vars_list.append(all_vars)
        var_order.extend(new_vars.tolist())

    return clique_vars_list, var_order


# ---------------------------------------------------------------------------
# KMPC for univariate (independent) models
# ---------------------------------------------------------------------------

class KMPCUnivariate:
    """Exact K most probable configurations for independent variable models.

    The distribution factorizes as p(x) = ∏ᵢ pᵢ(xᵢ).  The algorithm uses
    Nilsson's partition scheme adapted for independent distributions:

    1.  x¹ = MAP: for each variable independently pick its most probable value.
    2.  Partition H\\{x¹} into n disjoint subsets Sᵢ:
          Sᵢ = {x : x₁=x¹₁, …, x_{i-1}=x¹_{i-1}, xᵢ≠x¹ᵢ}
    3.  The maximum of Sᵢ is obtained by fixing the prefix, taking the
        second-best value for variable i, and MAP values for i+1..n.
    4.  Use a max-heap to select the best partition element as x².
    5.  Refine partitions iteratively for x³, x⁴, …, xᴷ.

    Complexity: O(K · n · log K) heap operations.

    Parameters
    ----------
    univariate_tables : list of 1-D probability arrays, one per variable
    cardinality : (n_vars,) integer array
    """

    def __init__(
        self,
        univariate_tables: List[np.ndarray],
        cardinality: np.ndarray,
    ) -> None:
        self.n_vars = len(univariate_tables)
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.log_probs = [
            np.log(np.maximum(t, 1e-300)) for t in univariate_tables
        ]
        # Pre-compute sorted value orders (highest prob first)
        self.sorted_vals = [
            np.argsort(-lp) for lp in self.log_probs
        ]

    def _eval_log_prob(self, config: np.ndarray) -> float:
        return float(sum(
            self.log_probs[i][int(config[i])] for i in range(self.n_vars)
        ))

    def _best_alt_val(self, var: int, exclude_val: int) -> Optional[int]:
        """Best value for *var* that is different from *exclude_val*."""
        for v in self.sorted_vals[var]:
            if int(v) != exclude_val:
                return int(v)
        return None

    def compute(self, k: int) -> KMPCResult:
        """Return the K most probable configurations.

        Parameters
        ----------
        k : int  (number of configurations to find)

        Returns
        -------
        KMPCResult
        """
        n = self.n_vars
        x_star = np.array([int(sv[0]) for sv in self.sorted_vals])
        lp_star = self._eval_log_prob(x_star)

        configs: List[np.ndarray] = [x_star.copy()]
        logps: List[float] = [lp_star]

        if k == 1:
            return self._make_result(configs, logps)

        # Heap elements: (neg_log_prob, tie_breaker, config, split_idx)
        # split_idx: the variable that defines this partition element
        # The config was obtained by: prefix from parent up to split_idx-1,
        # best alt value at split_idx, MAP for split_idx+1..n-1.
        heap: List = []
        counter = 0

        def push_initial_partitions(ref: np.ndarray):
            nonlocal counter
            for i in range(n):
                alt = self._best_alt_val(i, int(ref[i]))
                if alt is None:
                    continue
                cand = ref.copy()
                cand[i] = alt
                # For j > i keep MAP values (= ref[j] since ref is MAP)
                lp = self._eval_log_prob(cand)
                heapq.heappush(heap, (-lp, counter, cand.copy().tolist(), i, ref.copy().tolist()))
                counter += 1

        push_initial_partitions(x_star)
        seen = {tuple(x_star.tolist())}

        while len(configs) < k and heap:
            neg_lp, _, cand_list, split_idx, ref_list = heapq.heappop(heap)
            cand = np.array(cand_list)
            ref = np.array(ref_list)

            key = tuple(cand.tolist())
            if key in seen:
                continue
            seen.add(key)

            configs.append(cand.copy())
            logps.append(-neg_lp)

            # Generate new partition elements for i = split_idx .. n-1
            for i in range(split_idx, n):
                alt = self._best_alt_val(i, int(cand[i]))
                if alt is None:
                    continue
                new_cand = cand.copy()
                new_cand[i] = alt
                # For j > i reset to MAP values (argmax for each variable)
                for j in range(i + 1, n):
                    new_cand[j] = int(self.sorted_vals[j][0])
                new_lp = self._eval_log_prob(new_cand)
                heapq.heappush(
                    heap, (-new_lp, counter, new_cand.tolist(), i, cand.tolist())
                )
                counter += 1

        return self._make_result(configs, logps)

    def _make_result(
        self, configs: List[np.ndarray], logps: List[float]
    ) -> KMPCResult:
        cfgs = np.array(configs)
        lps = np.array(logps)
        probs = np.exp(np.clip(lps, -500, 0))
        return KMPCResult(
            configurations=cfgs,
            log_probabilities=lps,
            probabilities=probs,
            k_found=len(configs),
            method="kmpc_univariate",
        )


# ---------------------------------------------------------------------------
# KMPC for tree-structured models
# ---------------------------------------------------------------------------

class KMPCTree:
    """K most probable configurations for tree-structured Bayesian networks.

    Wraps the general junction-tree KMPC but exploits the tree structure:
    - The clique ordering in a tree (root → children) ensures that the
      greedy conditional MAP is exact (no approximation).
    - Nilsson's partition scheme is applied with this exact conditional MAP.

    Parameters
    ----------
    structure : ndarray of shape (n_vars, 4)
        Tree clique matrix from ``LearnTreeModel``.
    tables : list of ndarray
        Probability tables (root: 1-D marginal; non-root: 2-D conditional).
    cardinality : (n_vars,) array
    """

    def __init__(
        self,
        structure: np.ndarray,
        tables: List[np.ndarray],
        cardinality: np.ndarray,
    ) -> None:
        self.structure = structure
        self.tables = tables
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.clique_vars_list, self.var_order = _extract_clique_vars_from_structure(
            structure
        )
        self.n_vars = len(cardinality)

    def compute(self, k: int) -> KMPCResult:
        """Return the K most probable configurations."""
        engine = _KMPCPartition(
            clique_vars_list=self.clique_vars_list,
            tables=self.tables,
            cardinality=self.cardinality,
            var_order=self.var_order,
            method_name="kmpc_tree",
        )
        return engine.compute(k)


# ---------------------------------------------------------------------------
# KMPC for general factorized (junction tree) models
# ---------------------------------------------------------------------------

class KMPCJunctionTree:
    """K most probable configurations for general factorized models.

    Applies Nilsson's partition scheme with greedy conditional MAP.
    Suitable for FDA, MNFDA, and any other model stored as a
    ``FactorizedModel`` (MATEDA-style clique structure).

    Parameters
    ----------
    structure : ndarray
        Factorized model clique structure matrix.
    tables : list of ndarray
        Factor probability tables.
    cardinality : (n_vars,) array
    """

    def __init__(
        self,
        structure: np.ndarray,
        tables: List[np.ndarray],
        cardinality: np.ndarray,
    ) -> None:
        self.structure = structure
        self.tables = tables
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.clique_vars_list, self.var_order = _extract_clique_vars_from_structure(
            structure
        )
        self.n_vars = len(cardinality)

    def compute(self, k: int) -> KMPCResult:
        """Return the K most probable configurations."""
        engine = _KMPCPartition(
            clique_vars_list=self.clique_vars_list,
            tables=self.tables,
            cardinality=self.cardinality,
            var_order=self.var_order,
            method_name="kmpc_junction_tree",
        )
        return engine.compute(k)


# ---------------------------------------------------------------------------
# KMPC for Bayesian network models
# ---------------------------------------------------------------------------

class KMPCBayesianNetwork:
    """K most probable configurations for Bayesian network models.

    Converts the BN CPDs to a factorized representation and applies
    Nilsson's partition scheme.

    Parameters
    ----------
    bn_model : BayesianNetworkModel
        Model containing structure (a bayes_nets BayesianNetwork / networkx
        DiGraph or any object exposing the same ``nodes()`` / ``predecessors()``
        API).
    cardinality : (n_vars,) array
    """

    def __init__(
        self,
        bn_model: BayesianNetworkModel,
        cardinality: np.ndarray,
    ) -> None:
        self.bn_model = bn_model
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.n_vars = len(cardinality)
        self._clique_vars_list, self._tables, self._var_order = (
            self._extract_factorized(bn_model, self.cardinality)
        )

    @staticmethod
    def _extract_factorized(
        bn_model: BayesianNetworkModel,
        cardinality: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Convert BN CPDs into factorized cliques and tables."""
        clique_vars_list: List[np.ndarray] = []
        tables: List[np.ndarray] = []
        var_order: List[int] = []

        structure = bn_model.structure
        parameters = bn_model.parameters

        try:
            # Duck-typed BayesianNetwork API (bayes_nets / networkx-style:
            # nodes(), predecessors()).
            nodes = list(structure.nodes())
            n_vars = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}

            # Topological ordering
            try:
                import networkx as nx
                topo = list(nx.topological_sort(structure))
            except Exception:
                topo = nodes

            for node in topo:
                node_idx = node_to_idx[node]
                parents = list(structure.predecessors(node))
                parent_indices = [node_to_idx[p] for p in parents]

                # Get CPD for this node
                if isinstance(parameters, dict):
                    cpd = parameters.get(node, None)
                else:
                    # Assume list ordered by nodes()
                    cpd = parameters[nodes.index(node)] if parameters else None

                if cpd is None:
                    # Uniform distribution fallback
                    card = int(cardinality[node_idx])
                    table = np.ones(card) / card
                    clique_vars_list.append(np.array([node_idx]))
                    tables.append(table)
                    var_order.append(node_idx)
                    continue

                # Extract CPD table
                try:
                    # CPD exposing a get_values() method (bayes_nets-style CPD)
                    vals = np.asarray(cpd.get_values())
                    card_node = int(cardinality[node_idx])
                    if parent_indices:
                        # Shape: (card_node, prod(card_parents))
                        # Reshape to (card_parent1, card_parent2, ..., card_node)
                        parent_cards = [int(cardinality[pi]) for pi in parent_indices]
                        table = vals.T.reshape(parent_cards + [card_node])
                        all_vars = np.array(parent_indices + [node_idx])
                    else:
                        table = vals.ravel()
                        all_vars = np.array([node_idx])

                    clique_vars_list.append(all_vars)
                    tables.append(table)
                    var_order.append(node_idx)
                except Exception:
                    card = int(cardinality[node_idx])
                    table = np.ones(card) / card
                    clique_vars_list.append(np.array([node_idx]))
                    tables.append(table)
                    var_order.append(node_idx)

        except Exception:
            # Fallback: treat as independent
            n_vars = len(cardinality)
            for i in range(n_vars):
                card = int(cardinality[i])
                table = np.ones(card) / card
                clique_vars_list.append(np.array([i]))
                tables.append(table)
                var_order.append(i)

        return clique_vars_list, tables, var_order

    def compute(self, k: int) -> KMPCResult:
        """Return the K most probable configurations."""
        engine = _KMPCPartition(
            clique_vars_list=self._clique_vars_list,
            tables=self._tables,
            cardinality=self.cardinality,
            var_order=self._var_order,
            method_name="kmpc_bayesian_network",
        )
        return engine.compute(k)


# ---------------------------------------------------------------------------
# Core partition engine (shared by Tree, JunctionTree, BayesianNetwork)
# ---------------------------------------------------------------------------

class _KMPCPartition:
    """Nilsson's partition scheme for factorized distributions.

    Partition phase
    ---------------
    After finding xᴸ, the space H\\{x¹,…,xᴸ} is partitioned into disjoint
    subsets Sᵢ = {x : x_{var_order[0..i-1]} = xᴸ_{…}, x_{var_order[i]} ≠ xᴸ_{…}}.

    Candidate phase
    ---------------
    The best configuration in each Sᵢ is found via greedy conditional MAP
    (exact for trees, approximate for general factorized models).

    Identification phase
    --------------------
    The best candidate across all Sᵢ becomes xᴸ⁺¹.

    Parameters
    ----------
    clique_vars_list : list of variable-index arrays
    tables : list of factor tables
    cardinality : (n_vars,) integer array
    var_order : processing order for greedy conditional MAP
    method_name : label for the returned KMPCResult
    """

    def __init__(
        self,
        clique_vars_list: List[np.ndarray],
        tables: List[np.ndarray],
        cardinality: np.ndarray,
        var_order: List[int],
        method_name: str = "kmpc_partition",
    ) -> None:
        self.clique_vars_list = clique_vars_list
        self.tables = tables
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.var_order = var_order
        self.n_vars = len(cardinality)
        self.method_name = method_name

    def _eval_log_prob(self, config: np.ndarray) -> float:
        return _eval_log_prob_factorized(config, self.clique_vars_list, self.tables)

    def _find_map(self, evidence: Dict[int, int]) -> np.ndarray:
        return _greedy_conditional_map(
            evidence,
            self.clique_vars_list,
            self.tables,
            self.cardinality,
            self.var_order,
        )

    def _best_config_for_partition(
        self, prefix_config: np.ndarray, split_pos: int
    ) -> Optional[np.ndarray]:
        """Best configuration in partition element defined by split_pos.

        The partition element is:
          {x : x[var_order[0..split_pos-1]] = prefix[…], x[var_order[split_pos]] ≠ prefix[…]}

        Strategy:
        1. Fix all variables at positions 0..split_pos-1 to prefix values.
        2. For position split_pos, try all values ≠ prefix value.
        3. For each alternative, run conditional MAP for positions split_pos+1..n-1.
        4. Return the overall best.
        """
        fixed_var = self.var_order[split_pos]
        exclude_val = int(prefix_config[fixed_var])
        card_v = int(self.cardinality[fixed_var])

        evidence_prefix: Dict[int, int] = {
            self.var_order[j]: int(prefix_config[self.var_order[j]])
            for j in range(split_pos)
        }

        best_config: Optional[np.ndarray] = None
        best_lp = -np.inf

        for alt_val in range(card_v):
            if alt_val == exclude_val:
                continue
            ev = dict(evidence_prefix)
            ev[fixed_var] = alt_val
            cand = self._find_map(ev)
            lp = self._eval_log_prob(cand)
            if lp > best_lp:
                best_lp = lp
                best_config = cand

        return best_config

    def compute(self, k: int) -> KMPCResult:
        """Run the partition scheme to find K most probable configurations."""
        n_order = len(self.var_order)
        if n_order == 0:
            n_order = self.n_vars

        # Find MAP (x¹)
        x_star = self._find_map({})
        lp_star = self._eval_log_prob(x_star)

        configs: List[np.ndarray] = [x_star.copy()]
        logps: List[float] = [lp_star]

        if k == 1:
            return self._make_result(configs, logps)

        # Heap: (-log_prob, tie_breaker, cand_list, split_pos, ref_list)
        heap: List = []
        counter = 0

        def push_partitions(ref: np.ndarray, start_pos: int):
            nonlocal counter
            for i in range(start_pos, n_order):
                cand = self._best_config_for_partition(ref, i)
                if cand is None:
                    continue
                lp = self._eval_log_prob(cand)
                heapq.heappush(
                    heap,
                    (-lp, counter, cand.tolist(), i, ref.tolist()),
                )
                counter += 1

        push_partitions(x_star, 0)
        seen = {tuple(x_star.tolist())}

        while len(configs) < k and heap:
            neg_lp, _, cand_list, split_pos, ref_list = heapq.heappop(heap)
            cand = np.array(cand_list)
            key = tuple(cand_list)

            if key in seen:
                continue
            seen.add(key)

            configs.append(cand.copy())
            logps.append(-neg_lp)

            # Refine partitions: generate new partition elements from cand
            push_partitions(cand, split_pos)

        return self._make_result(configs, logps)

    def _make_result(
        self, configs: List[np.ndarray], logps: List[float]
    ) -> KMPCResult:
        cfgs = np.array(configs)
        lps = np.array(logps)
        probs = np.exp(np.clip(lps, -500, 0))
        return KMPCResult(
            configurations=cfgs,
            log_probabilities=lps,
            probabilities=probs,
            k_found=len(configs),
            method=self.method_name,
        )


# ---------------------------------------------------------------------------
# Public dispatch helper
# ---------------------------------------------------------------------------

def compute_kmpc(
    model: Model,
    cardinality: np.ndarray,
    k: int,
) -> KMPCResult:
    """Compute the K most probable configurations for any EDA model.

    Automatically dispatches to the appropriate KMPC algorithm based on
    the model type.

    Parameters
    ----------
    model : FactorizedModel | MarkovNetworkModel | BayesianNetworkModel
        Learned probabilistic model from an EDA.
    cardinality : (n_vars,) integer array
        Number of possible values for each variable.
    k : int
        Number of most probable configurations to find.

    Returns
    -------
    KMPCResult

    Examples
    --------
    >>> result = compute_kmpc(model, cardinality, k=10)
    >>> top10_configs = result.configurations
    >>> top10_probs = result.probabilities
    """
    cardinality = np.asarray(cardinality, dtype=int)

    if isinstance(model, FactorizedModel):
        structure = model.structure
        tables = model.parameters
        model_type = model.metadata.get("model_type", "")

        if model_type == "UMDA":
            # Pure univariate: exact Nilsson partition
            clique_vars_list, var_order = _extract_clique_vars_from_structure(structure)
            # tables[i] is the 1-D marginal for variable i
            uni_tables = []
            for cvl, table in zip(clique_vars_list, tables):
                uni_tables.append(np.asarray(table).ravel())
            return KMPCUnivariate(uni_tables, cardinality).compute(k)

        elif model_type == "Tree":
            return KMPCTree(structure, tables, cardinality).compute(k)

        else:
            # General factorized (FDA, MNFDA, …)
            return KMPCJunctionTree(structure, tables, cardinality).compute(k)

    elif isinstance(model, MarkovNetworkModel):
        # Represent as junction tree
        cliques = model.structure
        tables = model.parameters
        clique_vars_list = [np.asarray(c) for c in cliques]
        n_vars = int(cardinality.shape[0])
        var_order = list(range(n_vars))  # Default ordering
        engine = _KMPCPartition(
            clique_vars_list=clique_vars_list,
            tables=tables,
            cardinality=cardinality,
            var_order=var_order,
            method_name="kmpc_markov_network",
        )
        return engine.compute(k)

    elif isinstance(model, BayesianNetworkModel):
        return KMPCBayesianNetwork(model, cardinality).compute(k)

    else:
        raise TypeError(
            f"Unsupported model type for KMPC: {type(model)}. "
            "Expected FactorizedModel, MarkovNetworkModel, or BayesianNetworkModel."
        )
