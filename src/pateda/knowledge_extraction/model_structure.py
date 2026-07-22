"""
Structure extraction for exploiting learned models in variation and search.

Most EDAs in pateda *sample* the learned probabilistic model (probabilistic
logic sampling, Gibbs, most-probable configurations).  The graphical
*structure* of the model can also be exploited in other ways -- network
crossover and substructural neighborhood search -- that do not sample the
model's probabilities at all, only its dependency graph.  This module provides
the two generic structure views those operators need, working uniformly over
the main discrete model classes in pateda (Bayesian networks, trees,
factorized / Markov-network models) and over raw problem-defined interaction
matrices:

- :func:`model_to_linkage_graph` -- the undirected *linkage graph* ``G`` (a
  symmetric 0/1 adjacency matrix) whose edges join variables the model
  considers dependent.  Used by network crossover to grow connected crossover
  masks (Hauschild & Pelikan, 2010).

- :func:`model_to_substructures` -- a list of *substructures* (groups of
  variable indices) induced by the model.  Used by substructural neighborhood
  search to hill-climb over the joint values of each group (Lima, Pelikan,
  Sastry, Butz, Goldberg & Lobo, 2006).  Several substructure definitions are
  supported (see the ``mode`` argument), reproducing the parental / children /
  parental+children neighborhoods of the paper as well as clique-based and
  generic neighborhood-based groupings.

Both accept either a learned :class:`~pateda.core.models.Model` (any of the
pateda classes) or a raw square interaction matrix, so exactly the same
operators run with a model learned online by an EDA *or* with a linkage graph
known a priori from the problem (an additive function's block structure, an
Ising lattice, a UBQP weight matrix, a SAT clause graph, ...).

References
----------
- Hauschild, M., & Pelikan, M. (2010). "Network crossover performance on NK
  landscapes and deceptive problems." GECCO 2010 / MEDAL Report 2010003.
- Lima, C. F., Pelikan, M., Sastry, K., Butz, M., Goldberg, D. E., & Lobo, F.
  G. (2006). "Substructural Neighborhoods for Local Search in the Bayesian
  Optimization Algorithm." PPSN IX / MEDAL Report 2006007.
"""

from typing import Any, List, Optional
import numpy as np

from pateda.knowledge_extraction.network_measures import model_to_adjacency


# ---------------------------------------------------------------------------
# Directed adjacency (parent -> child), when the model provides one
# ---------------------------------------------------------------------------

def _directed_adjacency(source: Any, n_vars: Optional[int]):
    """Return ``(adj, is_directed)`` for a model or a raw matrix.

    ``adj[i, j] == 1`` is read as an arc ``i -> j`` when ``is_directed`` is
    True (Bayesian networks, trees); otherwise the matrix is symmetric.
    """
    adj, directed = model_to_adjacency(source, n_vars=n_vars)
    adj = (np.asarray(adj) != 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj, directed


# ---------------------------------------------------------------------------
# Linkage graph (undirected) for network crossover
# ---------------------------------------------------------------------------

def model_to_linkage_graph(source: Any, n_vars: Optional[int] = None) -> np.ndarray:
    """
    Extract the undirected linkage graph ``G`` from a model or interaction matrix.

    Parameters
    ----------
    source : Model or np.ndarray
        A learned pateda model (``BayesianNetworkModel``, ``TreeModel``,
        ``FactorizedModel``, ``MarkovNetworkModel``, ...) or a raw square 0/1
        (or weighted) interaction matrix.
    n_vars : int, optional
        Number of variables (inferred when omitted).

    Returns
    -------
    np.ndarray
        Symmetric ``(n_vars, n_vars)`` 0/1 adjacency matrix with zero diagonal.
        Directed models are symmetrized (an arc ``i -> j`` becomes an edge).
    """
    adj, _ = _directed_adjacency(source, n_vars)
    g = ((adj + adj.T) > 0).astype(int)
    np.fill_diagonal(g, 0)
    return g


# ---------------------------------------------------------------------------
# Substructures for substructural neighborhood search
# ---------------------------------------------------------------------------

def _cliques_of(source: Any, n_vars: int) -> Optional[List[np.ndarray]]:
    """Return the clique member sets of a factorized / Markov-network model, or
    ``None`` if ``source`` does not expose a clique structure."""
    cls = type(source).__name__
    if cls not in ("FactorizedModel", "MarkovNetworkModel"):
        return None
    structure = np.asarray(source.structure)
    if structure.ndim == 1:
        structure = structure.reshape(1, -1)
    cliques = []
    for row in structure:
        n_nb = int(row[0])
        n_new = int(row[1])
        members = [int(v) for v in row[2:2 + n_nb + n_new] if 0 <= int(v) < n_vars]
        if members:
            cliques.append(np.array(sorted(set(members)), dtype=int))
    return cliques


def model_to_substructures(
    source: Any,
    n_vars: Optional[int] = None,
    mode: str = "neighborhood",
    max_size: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract the list of substructures induced by a model or interaction matrix.

    A *substructure* is a group of variable indices that the model ties
    together; substructural local search explores the joint values of each
    group.  The available definitions follow Lima et al. (2006) and generalize
    them to non-Bayesian models:

    ``"parental"``     -- for each variable ``i``: ``{i} ∪ parents(i)``.
    ``"children"``     -- for each variable ``i``: ``{i} ∪ children(i)``.
    ``"both"``         -- for each variable ``i``: ``{i} ∪ parents(i) ∪ children(i)``.
    ``"neighborhood"`` -- for each variable ``i``: ``{i} ∪ undirected-neighbors(i)``
                          (the generic default; the Markov blanket for undirected
                          models, and equivalent to ``"both"`` for a BN).
    ``"clique"``       -- the cliques / factors of a factorized or Markov-network
                          model (falls back to ``"neighborhood"`` otherwise).

    For directed models (Bayesian networks, trees) parents/children are read
    from the arcs; for undirected models (factorized / Markov) every mode that
    refers to parents or children uses the symmetric neighbors instead.

    Parameters
    ----------
    source : Model or np.ndarray
        Learned model or raw interaction matrix.
    n_vars : int, optional
        Number of variables (inferred when omitted).
    mode : str
        Substructure definition (see above).
    max_size : int, optional
        Drop substructures with more than ``max_size`` variables (keeps the
        ``2^K`` enumeration of substructural search tractable).  ``None`` keeps
        all groups.

    Returns
    -------
    list of np.ndarray
        Deduplicated substructures, each a sorted 1-D array of variable indices.
        Always includes every variable at least once (singletons are added for
        variables that appear in no group), so a search over the substructures
        can still reach every variable.
    """
    if mode not in ("parental", "children", "both", "neighborhood", "clique"):
        raise ValueError(
            "mode must be 'parental', 'children', 'both', 'neighborhood' or "
            f"'clique', got {mode!r}"
        )

    adj, directed = _directed_adjacency(source, n_vars)
    n = adj.shape[0] if n_vars is None else n_vars

    subs: List[np.ndarray] = []

    if mode == "clique":
        cliques = _cliques_of(source, n)
        if cliques is not None:
            subs = cliques
        else:
            mode = "neighborhood"  # fall back below

    if mode != "clique":
        sym = ((adj + adj.T) > 0).astype(int)
        for i in range(n):
            if mode == "parental":
                nb = np.where(adj[:, i] > 0)[0] if directed else np.where(sym[i] > 0)[0]
            elif mode == "children":
                nb = np.where(adj[i, :] > 0)[0] if directed else np.where(sym[i] > 0)[0]
            elif mode == "both":
                if directed:
                    nb = np.union1d(np.where(adj[:, i] > 0)[0],
                                    np.where(adj[i, :] > 0)[0])
                else:
                    nb = np.where(sym[i] > 0)[0]
            else:  # neighborhood
                nb = np.where(sym[i] > 0)[0]
            group = np.array(sorted(set([i]) | set(int(v) for v in nb)), dtype=int)
            subs.append(group)

    # Optional size cap.
    if max_size is not None:
        subs = [s for s in subs if len(s) <= max_size]

    # Deduplicate (order-independent) while preserving order.
    seen = set()
    unique: List[np.ndarray] = []
    covered = set()
    for s in subs:
        key = tuple(s.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(s)
            covered.update(key)

    # Guarantee every variable is reachable: add singletons for any left out
    # (e.g. isolated variables, or all groups dropped by max_size).
    for i in range(n):
        if i not in covered:
            unique.append(np.array([i], dtype=int))

    return unique
