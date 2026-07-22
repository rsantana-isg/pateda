"""
Tests for the regularized k-order Markov EDA (MkRg-EDA).

Covers the learning method
(:class:`~pateda.learning.regularized_markov.LearnRegularizedMarkov`), the
sampler (:class:`~pateda.sampling.regularized_markov.SampleRegularizedMarkov`),
the three predictor variants (Rgk / BivRgk / AllRgk), and end-to-end use on a
sequential problem and on HP protein folding.
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pateda.learning.regularized_markov import (
    LearnRegularizedMarkov,
    build_markov_features,
    VARIANTS,
)
from pateda.sampling.regularized_markov import SampleRegularizedMarkov
from pateda.core.models import Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sequential_data(n, N, card, seed):
    """Data where X_i = (X_{i-1} + X_{i-2}) mod card (an interaction rule)."""
    rng = np.random.default_rng(seed)
    d = np.zeros((N, n), dtype=int)
    d[:, 0] = rng.integers(0, card, N)
    d[:, 1] = rng.integers(0, card, N)
    for j in range(2, n):
        d[:, j] = (d[:, j - 1] + d[:, j - 2]) % card
    return d


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def test_feature_builder_shapes():
    P = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])   # 2 samples, 3 predictors
    assert build_markov_features(P, "rgk").shape == (2, 3)         # raw only
    assert build_markov_features(P, "bivrgk").shape == (2, 3)      # 3 pairs
    assert build_markov_features(P, "allrgk").shape == (2, 6)      # 3 raw + 3 pairs


def test_feature_builder_products_correct():
    P = np.array([[2.0, 3.0]])                          # one pair: 2*3 = 6
    biv = build_markov_features(P, "bivrgk")
    assert biv.shape == (1, 1)
    assert biv[0, 0] == 6.0
    allf = build_markov_features(P, "allrgk")
    assert np.allclose(allf[0], [2.0, 3.0, 6.0])


def test_feature_builder_single_predictor_fallback():
    P = np.array([[1.0], [2.0]])                        # no pairs possible
    # bivrgk / allrgk gracefully fall back to the raw single predictor.
    assert build_markov_features(P, "bivrgk").shape == (2, 1)
    assert build_markov_features(P, "allrgk").shape == (2, 1)


# ---------------------------------------------------------------------------
# Learning: model contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_model_contract(variant):
    n, N, card = 8, 150, 3
    data = _sequential_data(n, N, card, seed=1)
    model = LearnRegularizedMarkov(k=3, variant=variant).learn(
        0, n, np.full(n, card), data, data.sum(1).astype(float))

    assert isinstance(model, Model)
    assert model.metadata["model_type"] == "RegularizedMarkov"
    assert model.metadata["variant"] == variant
    p = model.parameters
    assert len(p["submodels"]) == n
    assert np.array_equal(p["selected_population"], data)
    # First variable has no predictors -> marginal; later ones -> regression.
    assert p["submodels"][0]["kind"] in ("marginal", "constant")
    assert model.structure.shape[0] == n            # one clique per variable


def test_predictor_counts_respect_order_k():
    n, k = 10, 3
    data = _sequential_data(n, 100, 3, seed=2)
    model = LearnRegularizedMarkov(k=k, variant="rgk").learn(
        0, n, np.full(n, 3), data, data.sum(1).astype(float))
    for i, sm in enumerate(model.parameters["submodels"]):
        assert len(sm["predictors"]) == min(i, k)   # previous min(i, k) variables
        assert sm["predictors"] == list(range(max(0, i - k), i))


def test_constant_variable_handled():
    n, N = 5, 80
    data = np.random.default_rng(3).integers(0, 3, size=(N, n))
    data[:, 2] = 1                                   # variable 2 is constant
    model = LearnRegularizedMarkov(k=2, variant="rgk").learn(
        0, n, np.full(n, 3), data, data.sum(1).astype(float))
    sm = model.parameters["submodels"][2]
    assert sm["kind"] == "constant" and sm["value"] == 1


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["proba", "argmax"])
def test_sampling_valid(variant, mode):
    n, N, card = 8, 150, 3
    data = _sequential_data(n, N, card, seed=4)
    model = LearnRegularizedMarkov(k=3, variant=variant).learn(
        0, n, np.full(n, card), data, data.sum(1).astype(float))
    samples = SampleRegularizedMarkov(n_samples=200, mode=mode).sample(
        n, model, np.full(n, card), rng=np.random.default_rng(1))
    assert samples.shape == (200, n)
    assert samples.min() >= 0 and samples.max() < card


def test_sampling_reproducible():
    n, N, card = 6, 120, 3
    data = _sequential_data(n, N, card, seed=5)
    model = LearnRegularizedMarkov(k=3, variant="allrgk").learn(
        0, n, np.full(n, card), data, data.sum(1).astype(float))
    a = SampleRegularizedMarkov(n_samples=100).sample(
        n, model, np.full(n, card), rng=np.random.default_rng(7))
    b = SampleRegularizedMarkov(n_samples=100).sample(
        n, model, np.full(n, card), rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_allrgk_captures_interaction_rule_best():
    # X_i = (X_{i-1} + X_{i-2}) mod 3 is an interaction rule; the variants with
    # pairwise products should reproduce it better than raw predictors alone.
    n, N, card = 10, 300, 3
    data = _sequential_data(n, N, card, seed=6)

    def consistency(variant):
        model = LearnRegularizedMarkov(k=3, variant=variant).learn(
            0, n, np.full(n, card), data, data.sum(1).astype(float))
        s = SampleRegularizedMarkov(n_samples=1000).sample(
            n, model, np.full(n, card), rng=np.random.default_rng(2))
        return np.mean([(s[:, j] == (s[:, j - 1] + s[:, j - 2]) % card).mean()
                        for j in range(2, n)])

    rgk = consistency("rgk")
    allrgk = consistency("allrgk")
    assert allrgk > rgk                              # products help
    assert allrgk > 0.6                              # and clearly beat random (1/3)


def test_bad_model_rejected():
    with pytest.raises(TypeError):
        SampleRegularizedMarkov(n_samples=5).sample(
            3, Model(structure=None, parameters={"x": 1}), np.full(3, 2),
            rng=np.random.default_rng(0))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        LearnRegularizedMarkov(k=3, variant="bogus")
    with pytest.raises(ValueError):
        LearnRegularizedMarkov(k=0)
    with pytest.raises(ValueError):
        SampleRegularizedMarkov(n_samples=5, mode="bogus")


# ---------------------------------------------------------------------------
# End-to-end EDA
# ---------------------------------------------------------------------------

def test_eda_end_to_end_hp():
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations
    from pateda.functions.discrete_non_binary.problems.hp_protein import (
        create_hp_objective_function,
    )

    seq = np.array([0 if c == "H" else 1 for c in "HPHPPHHPHHPHPHHPPHPH"], dtype=int)
    n = len(seq)
    fitness = create_hp_objective_function(seq)

    comp = EDAComponents(
        seeding=RandomInit(),
        learning=LearnRegularizedMarkov(k=3, variant="allrgk"),
        sampling=SampleRegularizedMarkov(n_samples=80),
        selection=TruncationSelection(ratio=0.15),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=20),
    )
    eda = EDA(pop_size=80, n_vars=n, fitness_func=lambda x: fitness(np.asarray(x, int)),
              cardinality=np.full(n, 3), components=comp, random_seed=1)
    stats, _ = eda.run(verbose=False)
    # The HP fitness (H-H contacts) must be positive: the EDA finds real contacts.
    assert stats.best_fitness_overall > 0


# ---------------------------------------------------------------------------
# HP backtracking repair operator
# ---------------------------------------------------------------------------

def test_hp_repair_produces_self_avoiding_walks():
    from pateda.repairing import HPBacktrackingRepair
    from pateda.functions.discrete_non_binary.problems.hp_protein import eval_chain

    seq = np.array([0 if c == "H" else 1 for c in "HPHPPHHPHHPHPHHPPHPH"], dtype=int)
    n = len(seq)
    rng = np.random.default_rng(0)
    pop = rng.integers(0, 3, size=(300, n))

    before = sum(eval_chain(ind, seq)[1] > 0 for ind in pop)
    repaired = HPBacktrackingRepair().repair(pop, np.full(n, 3))
    after = sum(eval_chain(ind, seq)[1] > 0 for ind in repaired)

    assert before > 0                       # random folds do self-intersect
    assert after == 0                       # all repaired to self-avoiding walks
    assert repaired.min() >= 0 and repaired.max() < 3
    assert np.array_equal(repaired[:, :2], pop[:, :2])   # first two moves untouched


def test_hp_repair_leaves_valid_walk_unchanged():
    from pateda.repairing import repair_hp_self_avoiding
    from pateda.functions.discrete_non_binary.problems.hp_protein import eval_chain

    seq = np.zeros(12, dtype=int)
    straight = np.ones(12, dtype=int)       # all-forward = a straight line (valid)
    assert eval_chain(straight, seq)[1] == 0
    fixed = repair_hp_self_avoiding(straight)
    assert np.array_equal(fixed, straight)  # already self-avoiding -> unchanged


def test_hp_repair_as_eda_component():
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations
    from pateda.repairing import HPBacktrackingRepair
    from pateda.functions.discrete_non_binary.problems.hp_protein import (
        create_hp_objective_function,
    )

    seq = np.array([0 if c == "H" else 1 for c in "HPHPPHHPHHPHPHHPPHPH"], dtype=int)
    n = len(seq)
    fitness = create_hp_objective_function(seq)
    comp = EDAComponents(
        seeding=RandomInit(),
        learning=LearnRegularizedMarkov(k=3, variant="rgk"),
        sampling=SampleRegularizedMarkov(n_samples=80),
        selection=TruncationSelection(ratio=0.15),
        replacement=ElitistReplacement(),
        repairing=HPBacktrackingRepair(),
        stop_condition=MaxGenerations(max_gen=15),
    )
    eda = EDA(pop_size=80, n_vars=n, fitness_func=lambda x: fitness(np.asarray(x, int)),
              cardinality=np.full(n, 3), components=comp, random_seed=1)
    stats, _ = eda.run(verbose=False)
    assert stats.best_fitness_overall > 0
