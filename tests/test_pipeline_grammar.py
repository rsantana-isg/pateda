"""
Tests for the EDA pipeline grammar and the MODMOD / MODCONV model operators.

Covers:
- the model operators (``bn_to_factorized`` MODCONV, ``prune_factorized``,
  ``tree_to_forest``, ``tree_to_malign`` MODMOD) and the ``ModifiedLearning``
  decorator that applies them;
- the context-free grammar: derivations contain only terminals, parse into a
  role-consistent spec, and build into runnable EDAs;
- feasibility of randomly sampled pipelines (most must run);
- coverage of pateda's discrete-EDA component implementations.
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pateda.pipelines import (
    ModifiedLearning, model_type, bn_to_factorized, prune_factorized,
    tree_to_forest, tree_to_malign,
    TERMINALS, RULES, START, sample_derivation, parse_derivation,
    build_pipeline, grammar_terminals_by_category,
)
from pateda.learning import LearnEBNA, LearnTreeModel, LearnBOA, LearnUMDA
from pateda.sampling import SampleFDA


def _chain_data(n, N, seed):
    rng = np.random.default_rng(seed)
    d = np.zeros((N, n), dtype=int)
    d[:, 0] = rng.integers(0, 2, N)
    for j in range(1, n):
        d[:, j] = np.where(rng.random(N) < 0.2, rng.integers(0, 2, N), d[:, j - 1])
    return d


def _onemax(x):
    return float(np.sum(x))


# ---------------------------------------------------------------------------
# Model operators
# ---------------------------------------------------------------------------

def test_bn_to_factorized_is_samplable_by_fda():
    n, N = 10, 300
    data = _chain_data(n, N, 1)
    bn = LearnEBNA(max_parents=3).learn(0, n, np.full(n, 2), data, data.sum(1).astype(float))
    assert model_type(bn) == "bn"
    fac = bn_to_factorized(bn)
    assert model_type(fac) == "factorized"
    pop = SampleFDA(n_samples=100).sample(n, fac, np.full(n, 2), rng=np.random.default_rng(0))
    assert pop.shape == (100, n) and pop.min() >= 0 and pop.max() < 2


def test_prune_factorized_caps_clique_size():
    n, N = 10, 300
    data = _chain_data(n, N, 2)
    bn = LearnBOA(max_parents=4).learn(0, n, np.full(n, 2), data, data.sum(1).astype(float))
    fac = bn_to_factorized(bn)
    fac.metadata["cardinality"] = np.full(n, 2)
    pruned = prune_factorized(fac, K=2)
    # No clique may contain more than K variables (n_overlap + n_new).
    for row in np.asarray(pruned.structure):
        assert int(row[0]) + int(row[1]) <= 2
    pop = SampleFDA(n_samples=80).sample(n, pruned, np.full(n, 2), rng=np.random.default_rng(0))
    assert pop.shape == (80, n)


@pytest.mark.parametrize("op", [tree_to_forest, tree_to_malign])
def test_tree_modmods_produce_samplable_forest(op):
    n, N = 12, 300
    data = _chain_data(n, N, 3)
    tree = LearnTreeModel().learn(0, n, np.full(n, 2), data, data.sum(1).astype(float))
    out = op(tree)
    assert model_type(out) == "factorized"
    n_roots = sum(int(r[0]) == 0 for r in np.asarray(out.structure))
    n_roots_tree = sum(int(r[0]) == 0 for r in np.asarray(tree.structure))
    assert n_roots >= n_roots_tree              # cutting edges adds roots
    pop = SampleFDA(n_samples=80).sample(n, out, np.full(n, 2), rng=np.random.default_rng(0))
    assert pop.shape == (80, n) and pop.max() < 2


def test_modified_learning_chain():
    n, N = 10, 300
    data = _chain_data(n, N, 4)
    ml = ModifiedLearning(LearnBOA(max_parents=3),
                          ["bn_to_factorized", "prune_factorized"],
                          {"prune_factorized": {"K": 2}})
    model = ml.learn(0, n, np.full(n, 2), data, data.sum(1).astype(float))
    assert model_type(model) == "factorized"
    pop = SampleFDA(n_samples=50).sample(n, model, np.full(n, 2), rng=np.random.default_rng(0))
    assert pop.shape == (50, n)


def test_operators_robust_on_wrong_type():
    # A MODMOD on a model it does not apply to returns it unchanged (no crash).
    data = _chain_data(8, 200, 5)
    fac = LearnUMDA().learn(0, 8, np.full(8, 2), data, data.sum(1).astype(float))
    assert bn_to_factorized(fac) is fac         # not a BN -> unchanged


# ---------------------------------------------------------------------------
# Grammar structure
# ---------------------------------------------------------------------------

def test_derivation_yields_only_terminals():
    rng = np.random.default_rng(0)
    for _ in range(50):
        terms = sample_derivation(rng)
        assert all(t in TERMINALS for t in terms)
        assert all(t not in RULES for t in terms)


def test_parse_fills_all_required_roles():
    rng = np.random.default_rng(1)
    for _ in range(50):
        spec = parse_derivation(sample_derivation(rng))
        for role in ("seeding", "selection", "learner", "sampler", "replacement", "stop"):
            assert getattr(spec, role) is not None
        assert TERMINALS[spec.learner].role == "learner"
        assert TERMINALS[spec.sampler].role == "sampler"


def test_model_blocks_are_type_consistent():
    # Every generated learner/sampler pair must be compatible (directly, or the
    # derivation must contain a MODCONV that bridges them).
    compatible = {
        "SampleFDA": {"factorized"},
        "SampleBN": {"bn"},
        "SampleGibbs": {"markovnet"},
        "SampleIntFDA": {"intfda"},
        "SampleRegularizedMarkov": {"regmarkov"},
        "SampleMarkovChain": {"factorized"},
    }
    learner_type = {
        **{n: "factorized" for n in ["LearnUMDA", "LearnPBIL", "LearnFDA", "LearnCFDA",
           "LearnCUMDA", "LearnBMDA", "LearnMIMIC", "LearnMNFDA",
           "LearnMNFDAG", "LearnTreeModel", "LearnTreeModelM",
           "LearnAffinityFactorization", "LearnAffinityFactorizationElim",
           "LearnMarkovChain"]},
        **{n: "bn" for n in ["LearnEBNA", "LearnBOA", "LearnHBOA", "LearnLFDA", "LearnPADA"]},
        "LearnMOA": "markovnet", "LearnIntFDA": "intfda",
        "LearnRegularizedMarkov": "regmarkov",
    }
    rng = np.random.default_rng(2)
    for _ in range(200):
        spec = parse_derivation(sample_derivation(rng))
        t = learner_type[spec.learner]
        if "bn_to_factorized" in spec.operators:
            t = "factorized"                    # MODCONV changes the type
        assert t in compatible[spec.sampler], \
            f"{spec.learner}({t}) incompatible with {spec.sampler}"


# ---------------------------------------------------------------------------
# Build + run
# ---------------------------------------------------------------------------

def test_build_and_run_pipeline():
    n = 12
    eda, spec = build_pipeline(sample_derivation(np.random.default_rng(3)),
                               n, _onemax, np.full(n, 2), pop_size=40, n_gen=3,
                               random_seed=0)
    stats, _ = eda.run(verbose=False)
    assert stats.best_fitness_overall is not None


def test_random_pipelines_mostly_feasible():
    rng = np.random.default_rng(11)
    n = 12
    card = np.full(n, 2)
    ok = 0
    total = 30
    for _ in range(total):
        terms = sample_derivation(rng)
        try:
            eda, _ = build_pipeline(terms, n, _onemax, card, pop_size=40, n_gen=2,
                                    random_seed=0)
            eda.run(verbose=False)
            ok += 1
        except Exception:
            pass
    # The typed grammar makes most sampled pipelines runnable (not required 100%).
    assert ok / total >= 0.7


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_of_learners_and_samplers():
    covered = grammar_terminals_by_category()
    # The grammar must cover a broad set of learners and every model-type sampler.
    assert len(covered["learner"]) >= 20
    for s in ["SampleFDA", "SampleBayesianNetwork", "SampleGibbs", "SampleIntFDA",
              "SampleRegularizedMarkov"]:
        assert s in covered["sampler"]
    for cat in ["selection", "local_opt", "mutation", "replacement", "modop"]:
        assert len(covered[cat]) >= 2
