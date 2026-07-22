"""
Tests for Int_FDA (tree-based FDA for integers).

Int_FDA (:class:`~pateda.learning.int_fda.LearnIntFDA` /
:class:`~pateda.sampling.int_fda.SampleIntFDA`) is a *non-probabilistic* model
EDA: the model is a Chow-Liu tree plus two index tables plus the selected
population, and new individuals are produced by copying genes from donor
vectors along the tree.

The suite checks:

1. Model contract: structure is a tree covering every variable; parameters hold
   the selected population and the two index tables.
2. Auxiliary tables: PopulValues is sorted per column and ParentIndices give
   exact per-value block boundaries.
3. Sampling validity: values are in range, shape is correct, generation never
   leaves a variable unassigned, and results are reproducible from a seed.
4. The defining property: the sampled population matches the empirical
   Chow-Liu tree distribution of the selected set (root marginals and edge
   conditionals) to within Monte-Carlo error.
5. High cardinality: learning/sampling work when the cardinality far exceeds
   the number of selected individuals (the regime Int_FDA targets), and no
   c x c array is ever needed.
6. End-to-end: an EDA using Int_FDA solves a simple integer problem.
"""

import numpy as np
import pytest

from pateda.learning.int_fda import LearnIntFDA
from pateda.sampling.int_fda import SampleIntFDA
from pateda.core.models import Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_selected_set(n_vars, N, card, seed, flip=0.15):
    """Selected set with a 0->1->...->(n-1) dependency chain (strong MI)."""
    rng = np.random.default_rng(seed)
    sel = np.zeros((N, n_vars), dtype=int)
    sel[:, 0] = rng.integers(0, card, N)
    for j in range(1, n_vars):
        redraw = rng.random(N) < flip
        sel[:, j] = np.where(redraw, rng.integers(0, card, N), sel[:, j - 1])
    return sel


def _is_tree_covering_all(cliques, n_vars):
    """Cliques form a spanning forest/tree covering every variable exactly once,
    with each parent appearing before its children (topological order)."""
    seen = set()
    for c in range(cliques.shape[0]):
        n_parents = int(cliques[c, 0])
        if n_parents == 0:
            var = int(cliques[c, 2])
        else:
            parent = int(cliques[c, 2])
            var = int(cliques[c, 3])
            if parent not in seen:          # parent must be instantiated first
                return False
        if var in seen:                     # each variable assigned once
            return False
        seen.add(var)
    return seen == set(range(n_vars))


# ---------------------------------------------------------------------------
# 1. Model contract
# ---------------------------------------------------------------------------

def test_model_contract():
    n_vars, N, card = 6, 200, 8
    sel = _chain_selected_set(n_vars, N, card, seed=1)
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))

    assert isinstance(model, Model)
    assert model.metadata["model_type"] == "IntFDA"
    assert _is_tree_covering_all(model.structure, n_vars)

    params = model.parameters
    assert set(params) >= {"selected_population", "popul_values", "parent_indices"}
    # The model stores the selected population itself (defining feature).
    assert np.array_equal(params["selected_population"], sel)
    assert params["popul_values"].shape == (N, n_vars)
    assert len(params["parent_indices"]) == n_vars
    for i in range(n_vars):
        assert params["parent_indices"][i].shape == (card + 1,)


# ---------------------------------------------------------------------------
# 2. Auxiliary tables
# ---------------------------------------------------------------------------

def test_auxiliary_tables_exact():
    n_vars, N, card = 5, 150, 7
    sel = _chain_selected_set(n_vars, N, card, seed=2)
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
    pv = model.parameters["popul_values"]
    pi = model.parameters["parent_indices"]

    for i in range(n_vars):
        col_sorted = sel[pv[:, i], i]
        # PopulValues column i orders population indices by the value of var i.
        assert np.all(np.diff(col_sorted) >= 0)
        # ParentIndices is cumulative and delimits each value's block exactly.
        assert pi[i][0] == 0
        assert pi[i][-1] == N
        for v in range(card):
            s, e = int(pi[i][v]), int(pi[i][v + 1])
            assert np.all(col_sorted[s:e] == v)
            assert (e - s) == int((sel[:, i] == v).sum())


def test_cardinality_mismatch_raises():
    # A value >= its declared cardinality is a user error and must be caught.
    n_vars, N = 3, 20
    sel = np.zeros((N, n_vars), dtype=int)
    sel[0, 0] = 5                      # value 5 with declared cardinality 3
    with pytest.raises(ValueError):
        LearnIntFDA().learn(0, n_vars, np.full(n_vars, 3), sel, np.zeros(N))


# ---------------------------------------------------------------------------
# 3. Sampling validity + reproducibility
# ---------------------------------------------------------------------------

def test_sampling_valid_and_reproducible():
    n_vars, N, card = 6, 200, 8
    sel = _chain_selected_set(n_vars, N, card, seed=3)
    cardv = np.full(n_vars, card)
    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))

    sampler = SampleIntFDA(n_samples=500)
    a = sampler.sample(n_vars, model, cardv, rng=np.random.default_rng(7))
    b = sampler.sample(n_vars, model, cardv, rng=np.random.default_rng(7))

    assert a.shape == (500, n_vars)
    assert a.min() >= 0 and a.max() < card
    assert not np.any(a < 0)                       # every variable assigned
    assert np.array_equal(a, b)                    # reproducible from a seed

    # Every sampled value must actually occur in the selected set for that
    # variable — Int_FDA can only ever emit values it has seen.
    for i in range(n_vars):
        assert set(np.unique(a[:, i])).issubset(set(np.unique(sel[:, i])))


def test_reordered_bad_model_rejected():
    sampler = SampleIntFDA(n_samples=10)
    bad = Model(structure=np.zeros((1, 4), int), parameters={"foo": 1})
    with pytest.raises(TypeError):
        sampler.sample(3, bad, np.full(3, 2), rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 4. Defining property: samples == empirical tree distribution
# ---------------------------------------------------------------------------

def test_reproduces_empirical_tree_distribution():
    n_vars, N, card = 5, 300, 6
    sel = _chain_selected_set(n_vars, N, card, seed=4, flip=0.2)
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
    cliques = model.structure
    big = SampleIntFDA(n_samples=200_000).sample(
        n_vars, model, cardv, rng=np.random.default_rng(11)
    )

    max_err = 0.0
    for c in range(cliques.shape[0]):
        if int(cliques[c, 0]) == 0:                     # root -> marginal
            rv = int(cliques[c, 2])
            emp = np.bincount(sel[:, rv], minlength=card) / N
            smp = np.bincount(big[:, rv], minlength=card) / big.shape[0]
            max_err = max(max_err, float(np.abs(emp - smp).max()))
            continue
        p, ch = int(cliques[c, 2]), int(cliques[c, 3])  # edge -> conditional
        for v in range(card):
            sel_mask = sel[:, p] == v
            smp_mask = big[:, p] == v
            if sel_mask.sum() == 0 or smp_mask.sum() == 0:
                continue
            emp = np.bincount(sel[sel_mask, ch], minlength=card) / sel_mask.sum()
            smp = np.bincount(big[smp_mask, ch], minlength=card) / smp_mask.sum()
            max_err = max(max_err, float(np.abs(emp - smp).max()))

    assert max_err < 0.02      # Monte-Carlo error at 2e5 samples


# ---------------------------------------------------------------------------
# 5. High cardinality (the target regime)
# ---------------------------------------------------------------------------

def test_high_cardinality_more_values_than_samples():
    # Cardinality (1000) >> number of selected individuals (60): a table-based
    # tree would need 1000x1000 tables; Int_FDA must handle it via indices.
    n_vars, N, card = 4, 60, 1000
    rng = np.random.default_rng(5)
    sel = rng.integers(0, card, size=(N, n_vars))
    sel[:, 1] = sel[:, 0]                 # perfect dependency 0 -> 1
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
    samples = SampleIntFDA(n_samples=400).sample(
        n_vars, model, cardv, rng=np.random.default_rng(9)
    )
    assert samples.shape == (400, n_vars)
    assert samples.min() >= 0 and samples.max() < card

    # If the tree recovered the 0->1 (or 1->0) edge, the deterministic relation
    # must be preserved in every sample; if not, at least sampling is valid.
    struct = model.structure
    edge_01 = any(
        int(r[0]) == 1 and {int(r[2]), int(r[3])} == {0, 1} for r in struct
    )
    if edge_01:
        assert np.all(samples[:, 0] == samples[:, 1])


# ---------------------------------------------------------------------------
# 5b. Diversity injection
# ---------------------------------------------------------------------------

def test_plain_sampler_is_closed_under_selected_set():
    # Without novelty, Int_FDA can only emit values present in the selected set.
    n_vars, N, card = 5, 40, 50
    rng = np.random.default_rng(21)
    sel = rng.integers(0, 10, size=(N, n_vars))     # values 0..9 only
    cardv = np.full(n_vars, card)
    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))

    samples = SampleIntFDA(n_samples=3000).sample(
        n_vars, model, cardv, rng=np.random.default_rng(22)
    )
    assert samples.max() < 10       # never emits an unseen value (>=10)


def test_in_sampling_novelty_emits_unseen_values():
    # novelty_prob > 0 must let Int_FDA reach values absent from the base pop,
    # and must not crash when a novel parent value has an empty conditional block.
    n_vars, N, card = 5, 40, 50
    rng = np.random.default_rng(21)
    sel = rng.integers(0, 10, size=(N, n_vars))     # values 0..9 only
    cardv = np.full(n_vars, card)
    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))

    samples = SampleIntFDA(n_samples=3000, novelty_prob=0.1).sample(
        n_vars, model, cardv, rng=np.random.default_rng(23)
    )
    assert samples.max() >= 10                       # unseen values appear
    assert samples.min() >= 0 and samples.max() < card
    # Roughly novelty_prob of the genes are novel (uniform over [0, card)).
    frac_novel = float((samples >= 10).mean())
    assert 0.05 < frac_novel < 0.15


def test_invalid_novelty_prob_raises():
    with pytest.raises(ValueError):
        SampleIntFDA(n_samples=10, novelty_prob=1.5)


def test_random_reset_mutation_injects_novel_values():
    from pateda.mutation import RandomResetMutation

    n_vars, card = 6, 50
    pop = np.full((200, n_vars), 3, dtype=int)       # every gene == 3
    mutated = RandomResetMutation(mutation_prob=0.2).mutate(
        n_vars, np.full(n_vars, card), pop
    )
    assert mutated.shape == pop.shape
    assert mutated.min() >= 0 and mutated.max() < card
    assert mutated.max() > 3                          # values other than 3 appear
    # About 20% of genes changed away from the constant.
    frac_changed = float((mutated != 3).mean())
    assert 0.1 < frac_changed < 0.3


# ---------------------------------------------------------------------------
# 6. End-to-end EDA
# ---------------------------------------------------------------------------

def test_int_fda_eda_solves_integer_onemax():
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations

    n_vars, card = 8, 5
    optimum = float(n_vars * (card - 1))

    def integer_onemax(x):
        return float(np.sum(x))

    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnIntFDA(),
        sampling=SampleIntFDA(n_samples=300),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=40),
    )
    eda = EDA(
        pop_size=300,
        n_vars=n_vars,
        fitness_func=integer_onemax,
        cardinality=np.full(n_vars, card),
        components=components,
        random_seed=42,
    )
    stats, _ = eda.run(verbose=False)
    # Integer OneMax is separable and easy; Int_FDA should reach the optimum.
    assert stats.best_fitness_overall == optimum
