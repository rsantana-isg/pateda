"""
Tests for the bayes_nets-backed Bayesian-network EDAs.

After migrating ``pateda`` away from pgmpy, the Bayesian-network learning
algorithms (BOA, EBNA) delegate structure and parameter learning to the
``bayes_nets`` library, and Bayesian-network sampling is performed with
ancestral sampling over the learned CPDs.

This suite verifies the *integration contract* between pateda and bayes_nets:

1. ``LearnEBNA`` / ``LearnBOA`` import and use ``bayes_nets.BayesianNetwork``.
2. The returned :class:`~pateda.core.models.BayesianNetworkModel` keeps the
   adjacency-matrix + ``{"parents", "cpd"}`` dict contract that
   ``SampleBayesianNetwork`` (and knowledge_extraction / visualization) rely on.
3. Learned CPDs are valid probability distributions.
4. Structure learning recovers known dependencies (chain model).
5. The fixed-structure EBNA path only estimates parameters.
6. Sampling is reproducible, type-correct, and respects variable cardinalities.
7. End-to-end EDA runs (EBNA, BOA) converge on the separable OneMax problem.

All structure-recovery tests use a positive Dirichlet ``alpha`` so the score
is well defined for zero-count states (see the EBNA ``alpha=0.0`` caveat
documented in ``Extensions_edas_bayes_nets.md``).
"""

import numpy as np
import pytest

# Skip the whole module gracefully if bayes_nets is not installed.
bayes_nets = pytest.importorskip("bayes_nets")

from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.core.models import BayesianNetworkModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_data(n_samples, n_vars, flip_prob=0.1, seed=0):
    """Sample data from a Markov chain x0 -> x1 -> ... -> x_{n-1}.

    x0 ~ Bernoulli(0.5); each subsequent variable copies its predecessor
    with probability ``1 - flip_prob`` and flips otherwise.  This creates a
    strong, easily detectable sequential dependency structure.
    """
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

    # Structure: numpy adjacency matrix (parent -> child), a valid DAG.
    adj = model.structure
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (n_vars, n_vars)
    assert np.all(np.diag(adj) == 0)  # no self-loops

    # Parameters: dict mapping every variable to {"parents", "cpd"}.
    cpds = model.parameters
    assert isinstance(cpds, dict)
    assert set(cpds.keys()) == set(range(n_vars))

    for var in range(n_vars):
        entry = cpds[var]
        assert set(entry.keys()) >= {"parents", "cpd"}
        parents = entry["parents"]
        cpd = np.asarray(entry["cpd"])
        k = int(cardinality[var])
        # Parents must match the adjacency matrix.
        assert sorted(parents) == sorted(np.where(adj[:, var] > 0)[0].tolist())
        if len(parents) == 0:
            assert cpd.shape == (k,)
            assert np.isclose(cpd.sum(), 1.0)
        else:
            n_parent_configs = int(np.prod([cardinality[p] for p in parents]))
            assert cpd.shape == (n_parent_configs, k)
            assert np.allclose(cpd.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Integration: the learners really use bayes_nets
# ---------------------------------------------------------------------------

def test_learners_delegate_to_bayes_nets():
    """The BN learners import bayes_nets.BayesianNetwork (no pgmpy)."""
    import pateda.learning.ebna as ebna_mod
    import pateda.learning.boa as boa_mod
    from bayes_nets import BayesianNetwork

    assert ebna_mod.BayesianNetwork is BayesianNetwork
    assert boa_mod.BayesianNetwork is BayesianNetwork
    # pgmpy must no longer be referenced by these modules.
    assert "pgmpy" not in ebna_mod.__dict__
    assert "pgmpy" not in boa_mod.__dict__


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score_metric", ["bic", "aic", "k2"])
def test_ebna_model_contract(score_metric):
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(400, n_vars, seed=1)
    fitness = data.sum(axis=1).astype(float)

    model = LearnEBNA(max_parents=2, score_metric=score_metric, alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)
    assert model.metadata["model_type"] == "EBNA"
    assert model.metadata["score_metric"] == score_metric


@pytest.mark.parametrize("score_metric", ["k2", "bd", "bic"])
def test_boa_model_contract(score_metric):
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(400, n_vars, seed=2)
    fitness = data.sum(axis=1).astype(float)

    model = LearnBOA(max_parents=3, score_metric=score_metric, metric_alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)
    assert model.metadata["model_type"] == "BOA"
    # "ordering" defaults to the natural order when not supplied.
    assert np.array_equal(model.metadata["ordering"], np.arange(n_vars))


# ---------------------------------------------------------------------------
# Structure recovery
# ---------------------------------------------------------------------------

def test_ebna_recovers_chain_structure():
    """EBNA should detect the strong consecutive dependencies of a chain."""
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(600, n_vars, flip_prob=0.05, seed=3)
    fitness = data.sum(axis=1).astype(float)

    model = LearnEBNA(max_parents=2, score_metric="bic", alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    adj = model.structure
    skeleton = (adj + adj.T) > 0  # undirected skeleton (BIC is score-equivalent)

    # Every consecutive pair (i, i+1) should be connected in the skeleton.
    consecutive_found = sum(skeleton[i, i + 1] for i in range(n_vars - 1))
    assert consecutive_found >= n_vars - 2  # allow one miss for robustness
    assert int(adj.sum()) >= n_vars - 2


def test_boa_respects_ordering_acyclicity():
    """BOA (K2) must return a DAG; edges only go forward in the ordering."""
    n_vars = 6
    card = np.full(n_vars, 2)
    data = _chain_data(600, n_vars, flip_prob=0.05, seed=4)
    fitness = data.sum(axis=1).astype(float)

    ordering = np.arange(n_vars)
    model = LearnBOA(
        max_parents=3, score_metric="k2", metric_alpha=1.0, ordering=ordering
    ).learn(0, n_vars, card, data, fitness)
    adj = model.structure

    # K2 with the natural ordering => parents always have a smaller index.
    parents, children = np.where(adj > 0)
    assert np.all(parents < children)
    assert int(adj.sum()) >= n_vars - 2  # chain dependencies detected


# ---------------------------------------------------------------------------
# Fixed-structure path
# ---------------------------------------------------------------------------

def test_ebna_fixed_structure_only_learns_parameters():
    n_vars = 5
    card = np.full(n_vars, 2)
    data = _chain_data(300, n_vars, seed=5)
    fitness = data.sum(axis=1).astype(float)

    fixed = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        fixed[i, i + 1] = 1  # chain 0->1->2->3->4

    model = LearnEBNA(structure=fixed, alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    # Structure must equal the supplied adjacency exactly.
    assert np.array_equal(model.structure, fixed)
    for i in range(1, n_vars):
        assert model.parameters[i]["parents"] == [i - 1]
    _assert_valid_bn_model(model, n_vars, card)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_shape_range_and_reproducibility():
    n_vars = 6
    card = np.array([2, 3, 2, 4, 2, 3])  # mixed cardinality
    rng = np.random.default_rng(7)
    data = rng.integers(0, card, size=(400, n_vars))
    fitness = data.sum(axis=1).astype(float)

    model = LearnBOA(max_parents=2, score_metric="k2", metric_alpha=1.0).learn(
        0, n_vars, card, data, fitness
    )
    _assert_valid_bn_model(model, n_vars, card)

    sampler = SampleBayesianNetwork(n_samples=200)
    s1 = sampler.sample(n_vars, model, card, rng=np.random.default_rng(123))
    s2 = sampler.sample(n_vars, model, card, rng=np.random.default_rng(123))

    assert s1.shape == (200, n_vars)
    assert s1.dtype.kind in "iu"
    for j in range(n_vars):
        assert s1[:, j].min() >= 0
        assert s1[:, j].max() < card[j]
    # Same seed => identical samples (reproducible).
    assert np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# End-to-end EDA runs
# ---------------------------------------------------------------------------

def _run_eda_onemax(learner, n_vars=12, pop_size=200, max_gen=30, seed=42):
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations

    def onemax(x):
        # Works for both a single individual (1-D) and a population (2-D).
        return np.sum(x, axis=-1).astype(float)

    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleBayesianNetwork(n_samples=pop_size),
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


def test_ebna_eda_solves_onemax():
    n_vars = 12
    stats = _run_eda_onemax(
        LearnEBNA(max_parents=2, score_metric="bic", alpha=0.1), n_vars=n_vars
    )
    assert stats.best_fitness_overall == float(n_vars)


def test_boa_eda_solves_onemax():
    n_vars = 12
    stats = _run_eda_onemax(
        LearnBOA(max_parents=2, score_metric="k2", metric_alpha=1.0), n_vars=n_vars
    )
    assert stats.best_fitness_overall == float(n_vars)
