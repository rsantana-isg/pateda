"""
Tests for the HBOA, LFDA and PADA Bayesian-network EDAs.

These three learners follow the same integration contract as EBNA and BOA
(see ``test_bn_edas_bayes_nets.py``): structure and parameter learning are
delegated to ``bayes_nets`` and the result is adapted to the pateda
:class:`~pateda.core.models.BayesianNetworkModel` contract.

Beyond the shared contract, this suite checks the properties that make each
algorithm what it is:

- HBOA uses decision trees / decision graphs as local CPD structure, which is
  what lets it keep more parents per variable than BOA does with tabular CPDs.
- LFDA's BIC penalty weight monotonically controls the density of the learned
  network.
- PADA learns a *polytree*: the skeleton is singly connected, so it has at
  most n-1 edges and no undirected loops.
"""

import numpy as np
import pytest

# Skip the whole module gracefully if bayes_nets is not installed.
bayes_nets = pytest.importorskip("bayes_nets")

from pateda.learning.hboa import LearnHBOA
from pateda.learning.lfda import LearnLFDA
from pateda.learning.pada import LearnPADA
from pateda.learning.boa import LearnBOA
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.core.models import BayesianNetworkModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_data(n_samples, n_vars, flip_prob=0.1, seed=0):
    """Sample data from a Markov chain x0 -> x1 -> ... -> x_{n-1}."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_samples, n_vars), dtype=int)
    data[:, 0] = rng.integers(0, 2, size=n_samples)
    for j in range(1, n_vars):
        flip = rng.random(n_samples) < flip_prob
        data[:, j] = np.where(flip, 1 - data[:, j - 1], data[:, j - 1])
    return data


def _assert_valid_bn_model(model, n_vars, cardinality):
    """Assert the model honours the pateda BayesianNetworkModel contract."""
    assert isinstance(model, BayesianNetworkModel)

    adj = model.structure
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (n_vars, n_vars)
    assert np.all(np.diag(adj) == 0)  # no self-loops

    cpds = model.parameters
    assert isinstance(cpds, dict)
    assert set(cpds.keys()) == set(range(n_vars))

    for var in range(n_vars):
        entry = cpds[var]
        assert set(entry.keys()) >= {"parents", "cpd"}
        parents = entry["parents"]
        cpd = np.asarray(entry["cpd"])
        k = int(cardinality[var])
        assert sorted(parents) == sorted(np.where(adj[:, var] > 0)[0].tolist())
        if len(parents) == 0:
            assert cpd.shape == (k,)
            assert np.isclose(cpd.sum(), 1.0)
        else:
            n_parent_configs = int(np.prod([cardinality[p] for p in parents]))
            assert cpd.shape == (n_parent_configs, k)
            assert np.allclose(cpd.sum(axis=1), 1.0)


def _is_dag(adj):
    """Kahn's algorithm: True when the directed graph has no cycle."""
    adj = np.asarray(adj).copy()
    in_degree = adj.sum(axis=0)
    queue = [v for v in range(adj.shape[0]) if in_degree[v] == 0]
    visited = 0
    while queue:
        v = queue.pop()
        visited += 1
        for child in np.where(adj[v] > 0)[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))
    return visited == adj.shape[0]


def _run_eda_onemax(learner, n_vars=12, pop_size=200, max_gen=30, seed=42):
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations

    def onemax(x):
        return np.sum(x, axis=-1).astype(float)

    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        # Truncation selection is the pateda default for these EDAs.
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    return stats


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("local_structure", ["dt", "dg"])
def test_hboa_model_contract(local_structure):
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(400, n_vars, seed=1)
    fitness = data.sum(axis=1).astype(float)

    model = LearnHBOA(max_parents=3, local_structure=local_structure).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)
    assert model.metadata["model_type"] == "HBOA"
    assert model.metadata["local_structure"] == local_structure
    assert _is_dag(model.structure)


@pytest.mark.parametrize("bic_weight", [0.5, 1.0, 2.0])
def test_lfda_model_contract(bic_weight):
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(400, n_vars, seed=2)
    fitness = data.sum(axis=1).astype(float)

    model = LearnLFDA(max_parents=3, bic_weight=bic_weight, alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)
    assert model.metadata["model_type"] == "LFDA"
    assert model.metadata["bic_weight"] == bic_weight
    assert _is_dag(model.structure)


@pytest.mark.parametrize("dep_mode", ["global", "marginal"])
def test_pada_model_contract(dep_mode):
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(400, n_vars, seed=3)
    fitness = data.sum(axis=1).astype(float)

    model = LearnPADA(dep_mode=dep_mode, alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)
    assert model.metadata["model_type"] == "PADA"
    assert model.metadata["dep_mode"] == dep_mode
    assert _is_dag(model.structure)


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        LearnHBOA(local_structure="tabular")
    with pytest.raises(ValueError):
        LearnLFDA(bic_weight=-1.0)
    with pytest.raises(ValueError):
        LearnPADA(dep_mode="conditional")


# ---------------------------------------------------------------------------
# Algorithm-defining properties
# ---------------------------------------------------------------------------

def test_pada_learns_a_polytree():
    """PADA's skeleton is singly connected: <= n-1 edges and no loops."""
    n_vars = 12
    card = np.full(n_vars, 2)
    data = _chain_data(500, n_vars, seed=4)
    fitness = data.sum(axis=1).astype(float)

    adj = LearnPADA(alpha=1.0).learn(0, n_vars, card, data, fitness).structure

    # Undirected skeleton: at most n-1 edges (a forest).
    skeleton = ((adj + adj.T) > 0).astype(int)
    n_edges = int(skeleton.sum() // 2)
    assert n_edges <= n_vars - 1

    # A graph with <= n-1 edges is loop-free iff its components are trees,
    # i.e. iff n_edges == n_vars - n_components.  Count components by BFS.
    seen = np.zeros(n_vars, dtype=bool)
    n_components = 0
    for start in range(n_vars):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            for u in np.where(skeleton[v] > 0)[0]:
                if not seen[u]:
                    seen[u] = True
                    stack.append(int(u))
    assert n_edges == n_vars - n_components


def test_lfda_bic_weight_controls_density():
    """A heavier BIC penalty yields a network with no more edges."""
    n_vars = 12
    card = np.full(n_vars, 2)
    data = _chain_data(500, n_vars, seed=5)
    fitness = data.sum(axis=1).astype(float)

    def n_edges(weight):
        model = LearnLFDA(max_parents=4, bic_weight=weight, alpha=1.0).learn(
            0, n_vars, card, data, fitness
        )
        return int(np.asarray(model.structure).sum())

    dense, standard, sparse = n_edges(0.3), n_edges(1.0), n_edges(4.0)
    assert dense >= standard >= sparse


def test_hboa_local_structure_allows_more_parents_than_boa():
    """HBOA's decision graphs let it keep parent sets BOA's CPTs cannot afford.

    With a small population and a dense dependency structure, BOA's tabular
    score (and its table-size guard) caps the parent sets well below the
    requested maximum, while HBOA can use them.
    """
    n_vars = 12
    card = np.full(n_vars, 2)
    data = _chain_data(150, n_vars, flip_prob=0.05, seed=6)
    fitness = data.sum(axis=1).astype(float)

    boa_adj = LearnBOA(max_parents=6, metric_alpha=1.0).learn(
        0, n_vars, card, data, fitness
    ).structure
    hboa_adj = LearnHBOA(max_parents=6, local_structure="dg").learn(
        0, n_vars, card, data, fitness
    ).structure

    boa_max_parents = int(np.asarray(boa_adj).sum(axis=0).max())
    hboa_max_parents = int(np.asarray(hboa_adj).sum(axis=0).max())
    assert hboa_max_parents > boa_max_parents


# ---------------------------------------------------------------------------
# End-to-end EDA runs
# ---------------------------------------------------------------------------

def test_hboa_eda_solves_onemax():
    n_vars = 12
    stats = _run_eda_onemax(LearnHBOA(max_parents=3), n_vars=n_vars)
    assert stats.best_fitness_overall == float(n_vars)


def test_lfda_eda_solves_onemax():
    n_vars = 12
    stats = _run_eda_onemax(LearnLFDA(max_parents=3, alpha=0.1), n_vars=n_vars)
    assert stats.best_fitness_overall == float(n_vars)


def test_pada_eda_solves_onemax():
    n_vars = 12
    stats = _run_eda_onemax(LearnPADA(alpha=0.1), n_vars=n_vars)
    assert stats.best_fitness_overall == float(n_vars)
