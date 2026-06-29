"""
Tests for the network-measures knowledge-extraction module
(:mod:`pateda.knowledge_extraction.network_measures`).
"""
import numpy as np
import pytest

from pateda.knowledge_extraction import network_measures as nm


def _directed_adj():
    # 6-node DAG with one triangle {0,1,2} and a separate component {3,4}.
    A = np.zeros((6, 6), dtype=int)
    A[0, 1] = A[1, 2] = A[0, 2] = 1
    A[3, 4] = A[2, 4] = 1
    return A


def test_cliques_to_adjacency_is_undirected():
    # Two cliques {0,1,2} and {2,3,4} in the [n_nb, n_new, members...] format.
    cliques = np.array([[0, 3, 0, 1, 2], [0, 3, 2, 3, 4]])
    adj = nm.cliques_to_adjacency(cliques, 6)
    assert np.array_equal(adj, adj.T)               # undirected
    assert adj[0, 1] == adj[1, 2] == adj[2, 3] == 1
    assert adj.sum() // 2 == 6                       # 3 + 3 edges


def test_tree_to_adjacency_is_directed():
    # rows: [n_parents, n_new, parent, child]; root has n_parents == 0.
    tree = np.array([[0, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 2], [1, 1, 0, 3]])
    adj = nm.tree_to_adjacency(tree, 4)
    assert adj.sum() == 3                            # n_vars - 1 arcs
    assert adj[0, 1] == 1 and adj[1, 0] == 0         # directed parent -> child
    assert not np.array_equal(adj, adj.T)


def test_model_to_adjacency_square_matrix():
    A = _directed_adj()
    adj, directed = nm.model_to_adjacency(A)
    assert directed is True
    assert np.array_equal(adj, A)


def test_basic_measures():
    A = _directed_adj()
    assert nm.network_density(A, directed=True) == pytest.approx(5 / (36 - 6))
    deg = nm.degree_statistics(A, directed=True)
    assert deg["indegree_mean"] == pytest.approx(5 / 6)
    assert deg["outdegree_mean"] == pytest.approx(5 / 6)
    assert deg["degree_distribution"].sum() == 6     # one bin entry per vertex


def test_clique_and_component_stats():
    G = nm.to_networkx(_directed_adj(), directed=True)
    cl = nm.clique_stats(G)
    assert cl["max_clique_size"] == 3                # the {0,1,2} triangle
    cc = nm.connected_components_stats(G)
    assert cc["n_components"] == 2                    # {0,1,2,4,3} ... actually 2 weak comps


def test_motif_number_and_triads():
    G = nm.to_networkx(_directed_adj(), directed=True)
    census = nm.triad_census(G)
    assert sum(census.values()) == 20                # C(6,3) triads
    assert nm.motif_number(G, size=3) >= 1


def test_modularity_and_participation():
    G = nm.to_networkx(_directed_adj(), directed=True)
    q, comms = nm.max_modularity(G, seed=0)
    assert 0.0 <= q <= 1.0
    part = nm.participation_coefficient(G, comms)
    assert "participation_mean" in part


def test_dagdif():
    A = _directed_adj()
    B = A.copy()
    B[0, 1] = 0
    B[4, 5] = 1
    assert nm.dagdif(A, B) == 2


def test_compute_network_measures_keys():
    A = _directed_adj()
    m = nm.compute_network_measures(A)
    for key in nm.SCALAR_MEASURE_KEYS:
        assert key in m, f"missing measure {key}"
        assert np.isscalar(m[key]) or isinstance(m[key], (int, float))


def test_compute_measures_evolution():
    rng = np.random.default_rng(0)
    adjs = [np.triu(rng.integers(0, 2, (7, 7)), 1) for _ in range(5)]
    ev = nm.compute_measures_evolution(adjs, n_vars=7)
    assert ev["n_generations"] == 5
    assert len(ev["adjacencies"]) == 5
    for key in nm.SCALAR_MEASURE_KEYS:
        assert ev["series"][key].shape == (5,)
    # dagdif of the first generation is 0 by definition.
    assert ev["series"]["dagdif"][0] == 0


def test_edge_frequency_matrix():
    a = np.zeros((4, 4), dtype=int); a[0, 1] = 1
    b = np.zeros((4, 4), dtype=int); b[0, 1] = 1; b[2, 3] = 1
    freq = nm.edge_frequency_matrix([a, b])
    assert freq[0, 1] == pytest.approx(1.0)
    assert freq[2, 3] == pytest.approx(0.5)
