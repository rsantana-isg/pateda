"""
Tests for Tree-EDA-M (Tree EDA keeping only malign interactions).

Tree-EDA-M (:class:`~pateda.learning.tree_m.LearnTreeModelM`) is Tree-EDA with
a single change: a pairwise interaction contributes an edge only when it is
*malign* -- i.e. the most probable joint configuration disagrees with the joint
mode predicted by the univariate marginals.  Benign pairs are dropped, so the
model is a forest.  The learned model is an ordinary
:class:`~pateda.core.models.FactorizedModel`, sampled with the standard
:class:`~pateda.sampling.fda.SampleFDA`.

The suite checks:

1. The benign/malign detector, on hand-built marginals, for binary and
   non-binary variables.
2. That only malign pairs can become edges, and that an all-benign selected set
   yields an edge-free (UMDA-like) forest.
3. Model contract + reproducibility, including non-binary cardinality.
4. End-to-end: an EDA using Tree-EDA-M solves a simple non-binary problem.
"""

import numpy as np
import pytest

from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_m import LearnTreeModelM
from pateda.learning.utils.marginal_prob import find_marginal_prob
from pateda.sampling.fda import SampleFDA
from pateda.core.models import FactorizedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pop_from_counts(counts):
    rows = []
    for cfg, n in counts.items():
        rows.extend([list(cfg)] * n)
    return np.array(rows, dtype=int)


def _is_malign_pair(counts, cardinality):
    card = np.asarray(cardinality)
    pop = _pop_from_counts(counts)
    univ, biv = find_marginal_prob(pop, 2, card)
    return bool(LearnTreeModelM.detect_malign_mask(2, card, univ, biv)[0, 1])


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------

def test_detects_benign_binary():
    # Both variables favor 1 and are positively correlated: the joint mode (1,1)
    # equals the product-of-marginals mode -> benign.
    counts = {(0, 0): 30, (0, 1): 8, (1, 0): 7, (1, 1): 55}
    assert _is_malign_pair(counts, [2, 2]) is False


def test_detects_malign_binary():
    # Marginals favor (0,0) but the largest joint cell is (0,1) -> malign.
    counts = {(0, 0): 25, (0, 1): 30, (1, 0): 28, (1, 1): 17}
    assert _is_malign_pair(counts, [2, 2]) is True


def test_detects_benign_non_binary():
    counts = {(0, 0): 40, (0, 1): 5, (0, 2): 5, (1, 0): 5, (1, 1): 3,
              (1, 2): 2, (2, 0): 5, (2, 1): 2, (2, 2): 3}
    assert _is_malign_pair(counts, [3, 3]) is False


def test_detects_malign_non_binary():
    # Marginal modes (0, 2) but the joint mode is (1, 2) -> malign.
    counts = {(0, 0): 8, (0, 1): 7, (0, 2): 6, (1, 0): 1, (1, 1): 1,
              (1, 2): 15, (2, 0): 1, (2, 1): 1, (2, 2): 1}
    assert _is_malign_pair(counts, [3, 3]) is True


def test_malign_mask_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    n_vars, card = 6, 4
    pop = rng.integers(0, card, size=(300, n_vars))
    univ, biv = find_marginal_prob(pop, n_vars, np.full(n_vars, card))
    mask = LearnTreeModelM.detect_malign_mask(n_vars, np.full(n_vars, card), univ, biv)
    assert mask.shape == (n_vars, n_vars)
    assert np.array_equal(mask, mask.T)
    assert not np.any(np.diag(mask))


# ---------------------------------------------------------------------------
# 2. Only malign pairs become edges
# ---------------------------------------------------------------------------

def test_edges_are_only_malign_pairs():
    # A chain 0->1->...->5 with deceptive (anti-aligned high-value) structure so
    # that several malign edges exist; every learned edge must be a malign pair.
    rng = np.random.default_rng(3)
    n_vars, N, card = 6, 300, 4
    pop = rng.integers(0, card, size=(N, n_vars))
    # Make each consecutive pair carry a deceptive joint mode.
    for j in range(1, n_vars):
        strong = rng.random(N) < 0.5
        pop[strong, j] = (card - 1) - pop[strong, j - 1]

    model = LearnTreeModelM().learn(0, n_vars, np.full(n_vars, card), pop, np.zeros(N))
    mask = model.metadata["malign_mask"]
    cliques = model.structure
    for c in range(cliques.shape[0]):
        if int(cliques[c, 0]) == 1:                      # an edge parent->child
            parent, child = int(cliques[c, 2]), int(cliques[c, 3])
            assert mask[parent, child], "a benign pair was turned into an edge"


def test_all_benign_gives_forest_without_edges():
    # Independent variables, each strongly favoring value 0: every pair is
    # benign, so Tree-EDA-M must produce an edge-free forest (UMDA-like).
    rng = np.random.default_rng(4)
    n_vars, N, card = 7, 400, 3
    # p(0)=0.7, p(1)=0.2, p(2)=0.1 independently per variable.
    pop = rng.choice(card, size=(N, n_vars), p=[0.7, 0.2, 0.1])

    model = LearnTreeModelM().learn(0, n_vars, np.full(n_vars, card), pop, np.zeros(N))
    assert model.metadata["n_malign_pairs"] == 0
    assert int(np.sum(model.structure[:, 0] > 0)) == 0     # no edges


# ---------------------------------------------------------------------------
# 3. Model contract + sampling (binary and non-binary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("card", [2, 5])
def test_model_contract_and_sampling(card):
    rng = np.random.default_rng(5)
    n_vars, N = 8, 250
    pop = rng.integers(0, card, size=(N, n_vars))
    pop[:, 1] = pop[:, 0]                     # a dependency to (maybe) capture
    cardv = np.full(n_vars, card)

    model = LearnTreeModelM().learn(0, n_vars, cardv, pop, np.zeros(N))
    assert isinstance(model, FactorizedModel)
    assert model.metadata["model_type"] == "Tree-EDA-M"
    assert model.metadata["n_malign_pairs"] + model.metadata["n_benign_pairs"] \
        == n_vars * (n_vars - 1) // 2

    # Sampled with the ordinary FDA sampler; values must be in range.
    sampler = SampleFDA(n_samples=300)
    a = sampler.sample(n_vars, model, cardv, rng=np.random.default_rng(1))
    b = sampler.sample(n_vars, model, cardv, rng=np.random.default_rng(1))
    assert a.shape == (300, n_vars)
    assert a.min() >= 0 and a.max() < card
    assert np.array_equal(a, b)               # reproducible from a seed


def test_reduces_to_treemodel_machinery():
    # Tree-EDA-M is a subclass of LearnTreeModel and reuses its tree/parameter
    # code; a model with some malign edges must be a valid factorization.
    rng = np.random.default_rng(6)
    n_vars, N, card = 5, 200, 3
    pop = rng.integers(0, card, size=(N, n_vars))
    pop[:, 2] = (card - 1) - pop[:, 0]        # deceptive dependency 0 <-> 2
    model = LearnTreeModelM().learn(0, n_vars, np.full(n_vars, card), pop, np.zeros(N))
    assert isinstance(model, FactorizedModel)
    # Every variable must appear exactly once as a "new" variable in the cliques.
    new_vars = sorted(int(r[2]) if int(r[0]) == 0 else int(r[3]) for r in model.structure)
    assert new_vars == list(range(n_vars))


# ---------------------------------------------------------------------------
# 4. End-to-end EDA on a non-binary problem
# ---------------------------------------------------------------------------

def test_tree_eda_m_solves_integer_onemax():
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations

    n_vars, card = 9, 4
    optimum = float(n_vars * (card - 1))

    def integer_onemax(x):
        return float(np.sum(x))

    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnTreeModelM(),
        sampling=SampleFDA(n_samples=300),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=40),
    )
    eda = EDA(
        pop_size=300, n_vars=n_vars, fitness_func=integer_onemax,
        cardinality=np.full(n_vars, card), components=components, random_seed=42,
    )
    stats, _ = eda.run(verbose=False)
    assert stats.best_fitness_overall == optimum
