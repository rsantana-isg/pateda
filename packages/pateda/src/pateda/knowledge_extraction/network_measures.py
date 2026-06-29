"""
Network measures for the structural analysis of probabilistic models learned
by EDAs.

This module implements the *network-theoretic* analysis of the graphical
component of the probabilistic graphical models (PGMs) learned by Estimation of
Distribution Algorithms, following:

  * R. Santana, R. Armañanzas, C. Bielza, P. Larrañaga,
    "Network measures for information extraction in evolutionary algorithms",
    International Journal of Computational Intelligence Systems, 6(6):1163-1188,
    2013.
  * R. Santana, C. Bielza, J. A. Lozano, P. Larrañaga,
    "Mining probabilistic models learned by EDAs in the optimization of
    multi-objective problems", GECCO-2009, pp. 445-452.

The analysis has two steps (Algorithm 1 of the first paper):

  1. *Extraction of the structure*: every learned model is mapped to a graph
     (adjacency matrix).  Bayesian networks give directed graphs; tree models
     give directed trees; factorized / Markov-network models give the
     undirected interaction graph induced by their cliques.
  2. *Computation of network measures*: a set of local and global topological
     descriptors is computed for each graph.

The collection of measures in :func:`compute_network_measures` corresponds to
the 13 descriptors of Section 2.3 of the first paper (``dagdif``, ``Ndensity``,
``indegree``, ``outdegree``, edge betweenness, ``pair dist.``, ``reachability``,
clustering coefficient, ``shortcut prob.``, motif numbers for ``Z=3`` and
``Z=4``, maximum modularity and vertex participation coefficient) plus a few
additional classic measures (characteristic path length, radius, diameter,
assortativity, maximum clique size, number of connected components).

Author: Roberto Santana (roberto.santana@ehu.eus)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover - networkx is a declared dependency
    _HAS_NX = False


# Disconnected vertices are assigned a very high, unattainable distance value
# (Section 2.3, "pair dist.").  Used when averaging pairwise distances.
DISCONNECTED_DISTANCE = 1e6


# ---------------------------------------------------------------------------
# Structure extraction: model -> adjacency matrix
# ---------------------------------------------------------------------------

def cliques_to_adjacency(structure: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Build the undirected interaction graph induced by a clique/factor structure.

    The clique matrix follows the MATEDA / pateda convention where each row is::

        [n_neighbors, n_new_vars, neighbor_indices..., new_var_indices...]

    All variables that appear in the same clique are pairwise connected
    (the standard interaction-graph reading used for FDA and Markov-network
    models).

    Parameters
    ----------
    structure : np.ndarray
        Clique matrix of shape (n_cliques, k).
    n_vars : int
        Number of variables (graph nodes).

    Returns
    -------
    np.ndarray
        Symmetric (n_vars, n_vars) 0/1 adjacency matrix with zero diagonal.
    """
    adj = np.zeros((n_vars, n_vars), dtype=int)
    structure = np.asarray(structure)
    if structure.ndim == 1:
        structure = structure.reshape(1, -1)

    for row in structure:
        n_nb = int(row[0])
        n_new = int(row[1])
        members = [int(v) for v in row[2:2 + n_nb + n_new]]
        members = [v for v in members if 0 <= v < n_vars]
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                a, b = members[a_idx], members[b_idx]
                if a != b:
                    adj[a, b] = 1
                    adj[b, a] = 1
    return adj


def tree_to_adjacency(structure: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Build a directed adjacency matrix (parent -> child) from a tree structure.

    Tree rows follow ``[n_parents, n_new_vars, parent_idx, child_idx]``; a row
    with ``n_parents == 1`` contributes the arc ``parent_idx -> child_idx``
    (root rows have ``n_parents == 0`` and contribute no arc).

    Returns a directed (asymmetric) 0/1 adjacency matrix.
    """
    adj = np.zeros((n_vars, n_vars), dtype=int)
    structure = np.asarray(structure)
    if structure.ndim == 1:
        structure = structure.reshape(1, -1)

    for row in structure:
        n_parents = int(row[0])
        if n_parents >= 1 and len(row) >= 4:
            parent = int(row[2])
            child = int(row[3])
            if 0 <= parent < n_vars and 0 <= child < n_vars and parent != child:
                adj[parent, child] = 1
    return adj


def model_to_adjacency(
    model: Any, n_vars: Optional[int] = None
) -> Tuple[np.ndarray, bool]:
    """
    Map a learned EDA model (or a raw structure) to an adjacency matrix.

    Handles the pateda model classes (``BayesianNetworkModel``, ``TreeModel``,
    ``FactorizedModel``, ``MarkovNetworkModel``, ``GaussianModel``) as well as
    raw numpy arrays (square adjacency matrices or clique matrices).

    Parameters
    ----------
    model : Any
        A pateda ``Model`` instance, an object exposing ``structure`` /
        ``adjacency_matrix`` / ``graph``, or a numpy array.
    n_vars : int, optional
        Number of variables.  Inferred when possible.

    Returns
    -------
    (adjacency, is_directed) : (np.ndarray, bool)
        The 0/1 adjacency matrix and whether it should be read as directed.
    """
    # Identify the model type by class name to avoid hard imports / cycles.
    cls = type(model).__name__

    structure = None
    if hasattr(model, "adjacency_matrix"):
        structure = model.adjacency_matrix
    elif hasattr(model, "structure"):
        structure = model.structure
    elif hasattr(model, "graph"):
        structure = model.graph
    elif isinstance(model, np.ndarray):
        structure = model
        cls = "ndarray"

    if structure is None:
        raise ValueError(f"Cannot extract a structure from model of type {cls!r}")

    structure = np.asarray(structure)

    # A square matrix is treated as an adjacency matrix directly.
    is_square = structure.ndim == 2 and structure.shape[0] == structure.shape[1]

    if cls in ("BayesianNetworkModel",) or (is_square and cls in ("ndarray", "GaussianModel")):
        adj = (np.asarray(structure) != 0).astype(int)
        np.fill_diagonal(adj, 0)
        directed = not np.array_equal(adj, adj.T)
        return adj, directed

    if cls == "TreeModel":
        n = n_vars if n_vars is not None else int(structure.shape[0])
        return tree_to_adjacency(structure, n), True

    if cls in ("FactorizedModel", "MarkovNetworkModel"):
        if n_vars is None:
            members = structure[:, 2:] if structure.ndim == 2 else structure[2:]
            n = int(np.max(members)) + 1 if members.size else int(structure.shape[0])
        else:
            n = n_vars
        return cliques_to_adjacency(structure, n), False

    # Fall-backs: square -> adjacency; otherwise assume clique matrix.
    if is_square:
        adj = (structure != 0).astype(int)
        np.fill_diagonal(adj, 0)
        return adj, not np.array_equal(adj, adj.T)

    n = n_vars if n_vars is not None else int(np.max(structure[:, 2:])) + 1
    return cliques_to_adjacency(structure, n), False


def to_networkx(adjacency: np.ndarray, directed: bool):
    """Build a networkx graph from a 0/1 adjacency matrix."""
    if not _HAS_NX:
        raise ImportError("networkx is required for network measures")
    create_using = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(np.asarray(adjacency), create_using=create_using)
    return G


# ---------------------------------------------------------------------------
# Local / global network measures
# ---------------------------------------------------------------------------

def network_density(adjacency: np.ndarray, directed: bool = True) -> float:
    """Connection density ``Ndensity``: edges / (n^2 - n)."""
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    n_edges = int(np.sum(adj != 0))
    if not directed:
        n_edges = int(np.sum(np.triu(adj != 0, k=1)))
        return n_edges / (n * (n - 1) / 2.0)
    return n_edges / float(n * n - n)


def degree_statistics(adjacency: np.ndarray, directed: bool = True) -> Dict[str, Any]:
    """In/out/total degree statistics and the degree distribution."""
    adj = (np.asarray(adjacency) != 0).astype(int)
    indeg = adj.sum(axis=0)          # arcs arriving at j  (column sums)
    outdeg = adj.sum(axis=1)         # arcs leaving i      (row sums)
    if directed:
        total = indeg + outdeg
    else:
        total = adj.sum(axis=1)
        indeg = outdeg = total
    max_deg = int(total.max()) if total.size else 0
    degree_distribution = np.bincount(total.astype(int), minlength=max_deg + 1)
    return {
        "indegree_mean": float(np.mean(indeg)) if indeg.size else 0.0,
        "outdegree_mean": float(np.mean(outdeg)) if outdeg.size else 0.0,
        "degree_mean": float(np.mean(total)) if total.size else 0.0,
        "degree_max": max_deg,
        "indegree": indeg,
        "outdegree": outdeg,
        "degree": total,
        "degree_distribution": degree_distribution,
    }


def clustering_coefficient(G) -> Dict[str, Any]:
    """Average and per-node clustering coefficient (Fagiolo for digraphs)."""
    per_node = nx.clustering(G)
    values = np.array(list(per_node.values()), dtype=float) if per_node else np.array([0.0])
    return {"clustering_mean": float(np.mean(values)), "clustering_per_node": per_node}


def betweenness(G) -> Dict[str, float]:
    """Average vertex and edge betweenness centrality."""
    vbc = nx.betweenness_centrality(G)
    ebc = nx.edge_betweenness_centrality(G) if G.number_of_edges() else {}
    v_vals = np.array(list(vbc.values()), dtype=float) if vbc else np.array([0.0])
    e_vals = np.array(list(ebc.values()), dtype=float) if ebc else np.array([0.0])
    return {
        "vertex_betweenness_mean": float(np.mean(v_vals)),
        "edge_betweenness_mean": float(np.mean(e_vals)),
    }


def _pairwise_distances(G) -> List[float]:
    """All ordered pairwise shortest-path lengths, with disconnected pairs set
    to :data:`DISCONNECTED_DISTANCE`."""
    nodes = list(G.nodes())
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    dists = []
    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            dists.append(lengths.get(s, {}).get(t, DISCONNECTED_DISTANCE))
    return dists


def distance_measures(G) -> Dict[str, float]:
    """``pair dist.``, ``reachability`` and characteristic path length."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return {"pair_distance_mean": 0.0, "reachability_mean": 0.0,
                "characteristic_path_length": 0.0}
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    pair_dists, reach, finite = [], [], []
    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            if t in lengths.get(s, {}):
                d = lengths[s][t]
                pair_dists.append(d)
                finite.append(d)
                reach.append(1.0)
            else:
                pair_dists.append(DISCONNECTED_DISTANCE)
                reach.append(0.0)
    return {
        "pair_distance_mean": float(np.mean(pair_dists)),
        "reachability_mean": float(np.mean(reach)),
        "characteristic_path_length": float(np.mean(finite)) if finite else 0.0,
    }


def shortcut_probability(G) -> float:
    """Fraction of arcs that are *shortcuts* (range ``g_ij > 2``).

    The range of an arc ``e_ij`` is the shortest path from ``j`` to ``i`` after
    removing the arc; if this length is greater than 2 the arc is a shortcut.
    """
    edges = list(G.edges())
    if not edges:
        return 0.0
    n_shortcuts = 0
    for (u, v) in edges:
        H = G.copy()
        H.remove_edge(u, v)
        try:
            g_ij = nx.shortest_path_length(H, source=v, target=u)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            g_ij = np.inf
        if g_ij > 2:
            n_shortcuts += 1
    return n_shortcuts / float(len(edges))


def eccentricity_measures(G) -> Dict[str, float]:
    """Radius and diameter computed over the largest (strongly) connected part."""
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        return {"radius": 0.0, "diameter": 0.0}
    if G.is_directed():
        comps = list(nx.strongly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    comps = [c for c in comps if len(c) > 1]
    if not comps:
        return {"radius": 0.0, "diameter": 0.0}
    largest = max(comps, key=len)
    sub = G.subgraph(largest)
    try:
        ecc = nx.eccentricity(sub)
        vals = np.array(list(ecc.values()), dtype=float)
        return {"radius": float(vals.min()), "diameter": float(vals.max())}
    except nx.NetworkXError:
        return {"radius": 0.0, "diameter": 0.0}


def assortativity(G) -> float:
    """Degree assortativity coefficient (NaN -> 0.0)."""
    if G.number_of_edges() == 0:
        return 0.0
    try:
        a = nx.degree_assortativity_coefficient(G)
    except Exception:
        return 0.0
    return float(a) if np.isfinite(a) else 0.0


def connected_components_stats(G) -> Dict[str, float]:
    """Number and average size of (weakly) connected components."""
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    sizes = [len(c) for c in comps]
    return {
        "n_components": len(sizes),
        "avg_component_size": float(np.mean(sizes)) if sizes else 0.0,
        "largest_component_size": int(max(sizes)) if sizes else 0,
    }


def clique_stats(G) -> Dict[str, int]:
    """Maximum clique size and number of maximal cliques (on the skeleton)."""
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_edges() == 0:
        return {"max_clique_size": 1 if H.number_of_nodes() else 0,
                "n_maximal_cliques": H.number_of_nodes()}
    cliques = list(nx.find_cliques(H))
    sizes = [len(c) for c in cliques]
    return {"max_clique_size": int(max(sizes)) if sizes else 0,
            "n_maximal_cliques": len(cliques)}


# ---------------------------------------------------------------------------
# Motifs
# ---------------------------------------------------------------------------

# Names of the connected triad types (directed 3-node motifs) in the networkx
# triadic-census ordering; the leading disconnected types are excluded.
_CONNECTED_TRIADS = [
    "021D", "021U", "021C", "111D", "111U", "030T", "030C",
    "201", "120D", "120U", "120C", "210", "300",
]


def triad_census(G) -> Dict[str, int]:
    """Directed 3-node triad census (16 classes; networkx ordering)."""
    if not G.is_directed():
        G = G.to_directed()
    return dict(nx.triadic_census(G))


def motif_number(G, size: int = 3) -> int:
    """Total number of *connected* induced sub-graphs of the given size.

    This is the motif number of Section 2.2 (sum over the motif frequency
    spectrum).  For ``size == 3`` it is obtained from the triad census; for
    larger sizes connected node subsets are enumerated (suitable for the small
    graphs learned by EDAs).
    """
    from itertools import combinations

    if size == 3 and G.is_directed():
        census = triad_census(G)
        return int(sum(census[t] for t in _CONNECTED_TRIADS))

    H = G
    nodes = list(H.nodes())
    count = 0
    for combo in combinations(nodes, size):
        sub = H.subgraph(combo)
        ucc = sub.to_undirected() if sub.is_directed() else sub
        if nx.is_connected(ucc):
            count += 1
    return count


def motif_spectrum(G, size: int = 3) -> Dict[str, int]:
    """Frequency of each connected motif (isomorphism) class of the given size.

    Classes are keyed by a Weisfeiler-Lehman graph hash, giving the motif
    frequency spectrum without an explicit enumeration of all motif classes.
    """
    from itertools import combinations
    from collections import defaultdict

    spectrum: Dict[str, int] = defaultdict(int)
    nodes = list(G.nodes())
    for combo in combinations(nodes, size):
        sub = G.subgraph(combo)
        ucc = sub.to_undirected() if sub.is_directed() else sub
        if not nx.is_connected(ucc):
            continue
        h = nx.weisfeiler_lehman_graph_hash(sub, iterations=2)
        spectrum[h] += 1
    return dict(spectrum)


# ---------------------------------------------------------------------------
# Modularity and participation
# ---------------------------------------------------------------------------

def detect_communities(G, seed: int = 0) -> List[set]:
    """Community partition via the Louvain method (on the undirected skeleton)."""
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_edges() == 0:
        return [{node} for node in H.nodes()]
    try:
        return list(nx.community.louvain_communities(H, seed=seed))
    except Exception:
        return list(nx.community.greedy_modularity_communities(H))


def max_modularity(G, seed: int = 0) -> Tuple[float, List[set]]:
    """Maximum modularity value and the corresponding community partition."""
    H = G.to_undirected() if G.is_directed() else G
    communities = detect_communities(H, seed=seed)
    if H.number_of_edges() == 0:
        return 0.0, communities
    try:
        q = nx.community.modularity(H, communities)
    except Exception:
        q = 0.0
    return float(q), communities


def participation_coefficient(G, communities: Optional[List[set]] = None) -> Dict[str, Any]:
    """Vertex participation coefficient (Guimera-Amaral).

    ``P_i = 1 - sum_s (k_is / k_i)^2`` where ``k_is`` is the number of links of
    node ``i`` to community ``s`` and ``k_i`` its degree.  Returns the mean and
    the per-node values.
    """
    H = G.to_undirected() if G.is_directed() else G
    if communities is None:
        communities = detect_communities(H)
    membership = {}
    for c_idx, comm in enumerate(communities):
        for node in comm:
            membership[node] = c_idx

    per_node = {}
    for node in H.nodes():
        neighbors = list(H.neighbors(node))
        k_i = len(neighbors)
        if k_i == 0:
            per_node[node] = 0.0
            continue
        counts: Dict[int, int] = {}
        for nb in neighbors:
            c = membership.get(nb, -1)
            counts[c] = counts.get(c, 0) + 1
        per_node[node] = 1.0 - sum((k_is / k_i) ** 2 for k_is in counts.values())
    vals = np.array(list(per_node.values()), dtype=float) if per_node else np.array([0.0])
    return {"participation_mean": float(np.mean(vals)), "participation_per_node": per_node}


def dagdif(adjacency_t: np.ndarray, adjacency_t1: np.ndarray) -> int:
    """Number of different arcs between two graphs (Hamming distance of arcs)."""
    a = (np.asarray(adjacency_t) != 0).astype(int)
    b = (np.asarray(adjacency_t1) != 0).astype(int)
    return int(np.sum(a != b))


# ---------------------------------------------------------------------------
# Aggregate driver
# ---------------------------------------------------------------------------

def compute_network_measures(
    model: Any,
    n_vars: Optional[int] = None,
    previous_adjacency: Optional[np.ndarray] = None,
    include_motifs_z4: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute the full collection of network measures for one learned model.

    Parameters
    ----------
    model : Any
        Learned model, raw adjacency matrix or clique matrix (see
        :func:`model_to_adjacency`).
    n_vars : int, optional
        Number of variables.
    previous_adjacency : np.ndarray, optional
        Adjacency matrix of the previous generation, used for ``dagdif``.
    include_motifs_z4 : bool
        Whether to compute the (more expensive) ``Z=4`` motif number.
    seed : int
        Seed for the community detection used by modularity.

    Returns
    -------
    dict
        Scalar network measures plus the ``adjacency`` matrix and ``directed``
        flag.  The scalar keys mirror the 13 descriptors of the reference paper.
    """
    adjacency, directed = model_to_adjacency(model, n_vars=n_vars)
    n = adjacency.shape[0]
    G = to_networkx(adjacency, directed)

    measures: Dict[str, Any] = {
        "n_vars": n,
        "n_edges": int(np.sum(adjacency != 0)) if directed
        else int(np.sum(np.triu(adjacency != 0, k=1))),
        "directed": directed,
    }
    measures["density"] = network_density(adjacency, directed)

    deg = degree_statistics(adjacency, directed)
    measures["indegree_mean"] = deg["indegree_mean"]
    measures["outdegree_mean"] = deg["outdegree_mean"]
    measures["degree_mean"] = deg["degree_mean"]
    measures["degree_max"] = deg["degree_max"]
    measures["degree_distribution"] = deg["degree_distribution"]

    measures.update(clustering_coefficient(G))
    measures.update(betweenness(G))
    measures.update(distance_measures(G))
    measures["shortcut_probability"] = shortcut_probability(G)
    measures.update(eccentricity_measures(G))
    measures["assortativity"] = assortativity(G)
    measures.update(connected_components_stats(G))
    measures.update(clique_stats(G))

    measures["motif_number_z3"] = motif_number(G, size=3)
    if include_motifs_z4 and n >= 4:
        measures["motif_number_z4"] = motif_number(G, size=4)
    else:
        measures["motif_number_z4"] = 0

    q, communities = max_modularity(G, seed=seed)
    measures["max_modularity"] = q
    measures["n_communities"] = len(communities)
    measures.update(participation_coefficient(G, communities))

    if previous_adjacency is not None and previous_adjacency.shape == adjacency.shape:
        measures["dagdif"] = dagdif(previous_adjacency, adjacency)
    else:
        measures["dagdif"] = 0

    measures["adjacency"] = adjacency
    return measures


# Scalar keys produced by :func:`compute_network_measures` (excludes arrays).
SCALAR_MEASURE_KEYS = [
    "n_edges", "density", "indegree_mean", "outdegree_mean", "degree_mean",
    "degree_max", "clustering_mean", "vertex_betweenness_mean",
    "edge_betweenness_mean", "pair_distance_mean", "reachability_mean",
    "characteristic_path_length", "shortcut_probability", "radius", "diameter",
    "assortativity", "n_components", "avg_component_size",
    "largest_component_size", "max_clique_size", "n_maximal_cliques",
    "motif_number_z3", "motif_number_z4", "max_modularity", "n_communities",
    "participation_mean", "dagdif",
]


def edge_frequency_matrix(adjacencies: Sequence[np.ndarray]) -> np.ndarray:
    """Fraction of generations in which each arc appears (frequency matrix).

    This is the *coincidence / frequency matrix* of the structure-mining
    literature: ``M[i, j]`` is the relative frequency of arc ``i -> j`` over the
    supplied sequence of adjacency matrices.
    """
    adjacencies = [np.asarray(a) for a in adjacencies if np.asarray(a).size]
    if not adjacencies:
        return np.zeros((0, 0))
    n = adjacencies[0].shape[0]
    acc = np.zeros((n, n), dtype=float)
    for a in adjacencies:
        if a.shape == (n, n):
            acc += (a != 0).astype(float)
    return acc / len(adjacencies)


def aggregate_degree_distribution(
    adjacencies: Sequence[np.ndarray], directed: bool = True
) -> np.ndarray:
    """Average number of vertices having each degree, across the sequence.

    Reproduces the degree-distribution descriptor of the *Mining* paper
    (average number of vertices for each vertex degree).  Returns an array
    ``d`` where ``d[k]`` is the mean count of vertices of degree ``k``.
    """
    dists = []
    max_len = 0
    for a in adjacencies:
        a = np.asarray(a)
        if a.size == 0:
            continue
        dd = degree_statistics(a, directed)["degree_distribution"]
        dists.append(dd)
        max_len = max(max_len, len(dd))
    if not dists:
        return np.zeros(0)
    padded = np.zeros((len(dists), max_len))
    for i, dd in enumerate(dists):
        padded[i, : len(dd)] = dd
    return padded.mean(axis=0)


def triad_census_series(adjacencies: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-generation directed triad census (one series per connected triad)."""
    series: Dict[str, List[int]] = {t: [] for t in _CONNECTED_TRIADS}
    for a in adjacencies:
        a = np.asarray(a)
        if a.size == 0:
            for t in _CONNECTED_TRIADS:
                series[t].append(0)
            continue
        G = to_networkx((a != 0).astype(int), directed=True)
        census = triad_census(G)
        for t in _CONNECTED_TRIADS:
            series[t].append(int(census.get(t, 0)))
    return {t: np.array(v) for t, v in series.items()}


def compute_measures_evolution(
    models: Sequence[Any],
    n_vars: Optional[int] = None,
    include_motifs_z4: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute network measures for a sequence of models (one per generation).

    Returns
    -------
    dict
        ``'per_generation'`` : list of measure dicts (one per generation).
        ``'series'`` : dict mapping each scalar measure name to a numpy array
        indexed by generation.
        ``'adjacencies'`` : list of adjacency matrices.
    """
    per_generation: List[Dict[str, Any]] = []
    adjacencies: List[np.ndarray] = []
    prev_adj = None

    for model in models:
        try:
            m = compute_network_measures(
                model, n_vars=n_vars, previous_adjacency=prev_adj,
                include_motifs_z4=include_motifs_z4, seed=seed,
            )
        except Exception as exc:  # keep the evolution robust to odd models
            m = {k: 0.0 for k in SCALAR_MEASURE_KEYS}
            m["error"] = str(exc)
            m["adjacency"] = (prev_adj if prev_adj is not None
                              else np.zeros((0, 0), dtype=int))
        per_generation.append(m)
        adjacencies.append(m["adjacency"])
        prev_adj = m["adjacency"]

    series: Dict[str, np.ndarray] = {}
    for key in SCALAR_MEASURE_KEYS:
        series[key] = np.array([float(g.get(key, 0.0)) for g in per_generation])

    return {
        "per_generation": per_generation,
        "series": series,
        "adjacencies": adjacencies,
        "n_generations": len(per_generation),
    }
