"""
Pytest tests for discrete EDAs with variables of different cardinalities.

Tests that UMDA, Tree-EDA, Mk-EDA (k-order Markov Chain EDA), and Tree-EDAr
(restricted Tree-EDA) all correctly handle problems where different variables
have different numbers of possible values (mixed cardinality).

Specifically these tests verify:

1. **Cardinality acceptance** – each EDA accepts a per-variable cardinality
   array (not just a scalar constant shared by all variables).

2. **Learn correctness** – the learned model contains probability tables of the
   right size for each variable (matching the variable's cardinality).

3. **Sample validity** – sampled solutions respect each variable's domain
   (``0 <= x[i] < cardinality[i]``).

4. **End-to-end convergence** – a full EDA loop on the Mixed-Cardinality OneMax
   toy problem achieves a fitness close to the known optimum.
"""
from pathlib import Path
import warnings

import numpy as np
import pytest

# Allow running without installing the package.

warnings.filterwarnings("ignore", message="pyvinecopulib not available")

from pateda import EDA, EDAComponents
from pateda.learning.umda import LearnUMDA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.markov import LearnMarkovChain
from pateda.sampling.fda import SampleFDA
from pateda.sampling.markov import SampleMarkovChain
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_cardinality():
    """Mixed-cardinality vector: 4 binary + 4 ternary + 4 quaternary = 12 vars."""
    return np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=int)


@pytest.fixture
def mixed_population(mixed_cardinality):
    """Random population respecting the mixed cardinality fixture."""
    np.random.seed(42)
    n_vars = len(mixed_cardinality)
    pop = np.column_stack(
        [np.random.randint(0, c, size=100) for c in mixed_cardinality]
    )
    return pop


@pytest.fixture
def mixed_fitness(mixed_population):
    """Fitness = sum of all variable values (MC-OneMax)."""
    return np.sum(mixed_population, axis=1).astype(float)


def make_chain_interaction_matrix(n_vars: int) -> np.ndarray:
    """Build a chain (path graph) interaction matrix."""
    R = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        R[i, i + 1] = 1
        R[i + 1, i] = 1
    return R


def mc_onemax(x: np.ndarray) -> np.ndarray:
    """Mixed-Cardinality OneMax fitness function."""
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    return np.sum(x, axis=1).astype(float)


def assert_cardinality_constraints(cardinality, population, label=""):
    """Assert every variable value is in [0, cardinality[i])."""
    for i, card in enumerate(cardinality):
        lo, hi = 0, int(card) - 1
        vals = population[:, i]
        assert np.all(vals >= lo) and np.all(vals <= hi), (
            f"{label}: variable {i} (card={card}) has out-of-range values: "
            f"min={vals.min()}, max={vals.max()}"
        )


# ---------------------------------------------------------------------------
# UMDA
# ---------------------------------------------------------------------------

class TestUMDAMixedCardinality:
    """Tests for UMDA with per-variable cardinalities."""

    def test_learn_accepts_mixed_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """LearnUMDA.learn() must not raise for a mixed-cardinality vector."""
        n_vars = len(mixed_cardinality)
        learner = LearnUMDA(alpha=0.1)
        model = learner.learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        assert model is not None

    def test_probability_tables_match_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Each marginal probability table must have length == cardinality[i]."""
        n_vars = len(mixed_cardinality)
        learner = LearnUMDA(alpha=0.1)
        model = learner.learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        assert len(model.parameters) == n_vars
        for i, table in enumerate(model.parameters):
            assert len(table) == mixed_cardinality[i], (
                f"Variable {i}: expected {mixed_cardinality[i]} probabilities, "
                f"got {len(table)}"
            )

    def test_probability_tables_are_valid(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Every probability table must sum to 1 and have no NaN/Inf values."""
        n_vars = len(mixed_cardinality)
        learner = LearnUMDA(alpha=0.1)
        model = learner.learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), f"Variable {i}: non-finite values"
            assert np.all(table >= 0), f"Variable {i}: negative probability"
            assert abs(np.sum(table) - 1.0) < 1e-9, (
                f"Variable {i}: probabilities sum to {np.sum(table):.6f}"
            )

    def test_sample_respects_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Sampled individuals must have values in [0, cardinality[i])."""
        n_vars = len(mixed_cardinality)
        learner = LearnUMDA(alpha=0.1)
        model = learner.learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        sampler = SampleFDA(n_samples=200)
        new_pop = sampler.sample(n_vars, model, mixed_cardinality)
        assert new_pop.shape == (200, n_vars)
        assert_cardinality_constraints(mixed_cardinality, new_pop, "UMDA")

    def test_convergence_on_mc_onemax(self, mixed_cardinality):
        """UMDA should converge to ≥95 % of optimum on MC-OneMax."""
        n_vars = len(mixed_cardinality)
        opt = float(np.sum(mixed_cardinality - 1))

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnUMDA(alpha=0.1),
            sampling=SampleFDA(n_samples=200),
            replacement=ElitistReplacement(),
            stop_condition=MaxGenerations(max_gen=40),
        )
        eda = EDA(
            pop_size=200,
            n_vars=n_vars,
            fitness_func=mc_onemax,
            cardinality=mixed_cardinality,
            components=components,
            random_seed=42,
        )
        stats, _ = eda.run(verbose=False)
        assert stats.best_fitness_overall >= opt * 0.95, (
            f"UMDA: best={stats.best_fitness_overall:.2f}, required≥{opt * 0.95:.2f}"
        )


# ---------------------------------------------------------------------------
# Tree-EDA
# ---------------------------------------------------------------------------

class TestTreeEDAMixedCardinality:
    """Tests for Tree-EDA with per-variable cardinalities."""

    def test_learn_accepts_mixed_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """LearnTreeModel.learn() must not raise for a mixed-cardinality vector."""
        n_vars = len(mixed_cardinality)
        model = LearnTreeModel(alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        assert model is not None

    def test_no_nan_inf_in_probability_tables(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Conditional probability tables must be finite and non-negative."""
        n_vars = len(mixed_cardinality)
        model = LearnTreeModel(alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), (
                f"Tree-EDA table {i} has NaN/Inf"
            )
            assert np.all(table >= 0), f"Tree-EDA table {i} has negative values"

    def test_sample_respects_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Sampled individuals must have values in [0, cardinality[i])."""
        n_vars = len(mixed_cardinality)
        model = LearnTreeModel(alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        new_pop = SampleFDA(n_samples=200).sample(n_vars, model, mixed_cardinality)
        assert_cardinality_constraints(mixed_cardinality, new_pop, "Tree-EDA")

    def test_convergence_on_mc_onemax(self, mixed_cardinality):
        """Tree-EDA should converge to ≥95 % of optimum on MC-OneMax."""
        n_vars = len(mixed_cardinality)
        opt = float(np.sum(mixed_cardinality - 1))

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnTreeModel(alpha=0.1),
            sampling=SampleFDA(n_samples=200),
            replacement=ElitistReplacement(),
            stop_condition=MaxGenerations(max_gen=50),
        )
        eda = EDA(
            pop_size=200,
            n_vars=n_vars,
            fitness_func=mc_onemax,
            cardinality=mixed_cardinality,
            components=components,
            random_seed=42,
        )
        stats, _ = eda.run(verbose=False)
        assert stats.best_fitness_overall >= opt * 0.95, (
            f"Tree-EDA: best={stats.best_fitness_overall:.2f}, required≥{opt * 0.95:.2f}"
        )


# ---------------------------------------------------------------------------
# Mk-EDA (k-order Markov Chain EDA)
# ---------------------------------------------------------------------------

class TestMkEDAMixedCardinality:
    """Tests for k-order Markov Chain EDA (Mk-EDA) with per-variable cardinalities."""

    def test_learn_accepts_mixed_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """LearnMarkovChain.learn() must not raise for a mixed-cardinality vector."""
        n_vars = len(mixed_cardinality)
        model = LearnMarkovChain(k=1, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        assert model is not None

    def test_probability_tables_are_valid(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Mk-EDA probability tables must be finite and non-negative."""
        n_vars = len(mixed_cardinality)
        model = LearnMarkovChain(k=1, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), (
                f"Mk-EDA table {i} has NaN/Inf"
            )
            assert np.all(table >= 0), f"Mk-EDA table {i} has negative values"

    def test_sample_respects_cardinality_k1(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """First-order Markov samples must respect cardinality constraints."""
        n_vars = len(mixed_cardinality)
        model = LearnMarkovChain(k=1, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        new_pop = SampleMarkovChain(n_samples=200).sample(
            n_vars, model, mixed_cardinality
        )
        assert new_pop.shape == (200, n_vars)
        assert_cardinality_constraints(mixed_cardinality, new_pop, "Mk-EDA(k=1)")

    def test_sample_respects_cardinality_k2(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Second-order Markov samples must respect cardinality constraints."""
        n_vars = len(mixed_cardinality)
        model = LearnMarkovChain(k=2, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        new_pop = SampleMarkovChain(n_samples=200).sample(
            n_vars, model, mixed_cardinality
        )
        assert_cardinality_constraints(mixed_cardinality, new_pop, "Mk-EDA(k=2)")

    def test_convergence_on_mc_onemax(self, mixed_cardinality):
        """Mk-EDA (k=1) should converge to ≥95 % of optimum on MC-OneMax."""
        n_vars = len(mixed_cardinality)
        opt = float(np.sum(mixed_cardinality - 1))

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnMarkovChain(k=1, alpha=0.1),
            sampling=SampleMarkovChain(n_samples=200),
            replacement=ElitistReplacement(),
            stop_condition=MaxGenerations(max_gen=40),
        )
        eda = EDA(
            pop_size=200,
            n_vars=n_vars,
            fitness_func=mc_onemax,
            cardinality=mixed_cardinality,
            components=components,
            random_seed=42,
        )
        stats, _ = eda.run(verbose=False)
        assert stats.best_fitness_overall >= opt * 0.95, (
            f"Mk-EDA: best={stats.best_fitness_overall:.2f}, required≥{opt * 0.95:.2f}"
        )


# ---------------------------------------------------------------------------
# Tree-EDAr
# ---------------------------------------------------------------------------

class TestTreeEDArMixedCardinality:
    """Tests for restricted Tree-EDA (Tree-EDAr) with per-variable cardinalities."""

    def test_learn_accepts_mixed_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """LearnTreeModelR.learn() must not raise for a mixed-cardinality vector."""
        n_vars = len(mixed_cardinality)
        R = make_chain_interaction_matrix(n_vars)
        model = LearnTreeModelR(interaction_matrix=R, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        assert model is not None

    def test_no_nan_inf_in_probability_tables(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Tree-EDAr conditional tables must be finite and non-negative."""
        n_vars = len(mixed_cardinality)
        R = make_chain_interaction_matrix(n_vars)
        model = LearnTreeModelR(interaction_matrix=R, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), (
                f"Tree-EDAr table {i} has NaN/Inf"
            )
            assert np.all(table >= 0), (
                f"Tree-EDAr table {i} has negative values"
            )

    def test_sample_respects_cardinality(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Sampled individuals must have values in [0, cardinality[i])."""
        n_vars = len(mixed_cardinality)
        R = make_chain_interaction_matrix(n_vars)
        model = LearnTreeModelR(interaction_matrix=R, alpha=0.1).learn(
            0, n_vars, mixed_cardinality, mixed_population, mixed_fitness
        )
        new_pop = SampleFDA(n_samples=200).sample(n_vars, model, mixed_cardinality)
        assert new_pop.shape == (200, n_vars)
        assert_cardinality_constraints(mixed_cardinality, new_pop, "Tree-EDAr")

    def test_wrong_interaction_matrix_shape_raises(
        self, mixed_cardinality, mixed_population, mixed_fitness
    ):
        """Providing a wrongly-shaped interaction matrix must raise ValueError."""
        n_vars = len(mixed_cardinality)
        wrong_R = np.zeros((n_vars + 1, n_vars + 1), dtype=int)  # Wrong shape
        learner = LearnTreeModelR(interaction_matrix=wrong_R, alpha=0.1)
        with pytest.raises(ValueError):
            learner.learn(0, n_vars, mixed_cardinality, mixed_population, mixed_fitness)

    def test_convergence_on_mc_onemax(self, mixed_cardinality):
        """Tree-EDAr should converge to ≥95 % of optimum on MC-OneMax."""
        n_vars = len(mixed_cardinality)
        opt = float(np.sum(mixed_cardinality - 1))
        R = make_chain_interaction_matrix(n_vars)

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnTreeModelR(interaction_matrix=R, alpha=0.1),
            sampling=SampleFDA(n_samples=200),
            replacement=ElitistReplacement(),
            stop_condition=MaxGenerations(max_gen=50),
        )
        eda = EDA(
            pop_size=200,
            n_vars=n_vars,
            fitness_func=mc_onemax,
            cardinality=mixed_cardinality,
            components=components,
            random_seed=42,
        )
        stats, _ = eda.run(verbose=False)
        assert stats.best_fitness_overall >= opt * 0.95, (
            f"Tree-EDAr: best={stats.best_fitness_overall:.2f}, required≥{opt * 0.95:.2f}"
        )


# ---------------------------------------------------------------------------
# Cross-algorithm cardinality edge cases
# ---------------------------------------------------------------------------

class TestMixedCardinalityEdgeCases:
    """Edge-case tests shared across algorithms."""

    @pytest.mark.parametrize("cardinality_spec", [
        np.array([2, 5]),            # One binary, one quinary
        np.array([10, 2, 3]),        # High cardinality first
        np.array([2, 2, 3, 3, 4, 4, 5, 5]),  # Pairs with increasing cardinality
    ])
    def test_umda_various_mixed_cardinalities(self, cardinality_spec):
        """UMDA must work for various mixed-cardinality configurations."""
        np.random.seed(0)
        n_vars = len(cardinality_spec)
        pop = np.column_stack(
            [np.random.randint(0, c, size=50) for c in cardinality_spec]
        )
        fitness = np.sum(pop, axis=1).astype(float)
        model = LearnUMDA(alpha=0.1).learn(0, n_vars, cardinality_spec, pop, fitness)
        new_pop = SampleFDA(n_samples=50).sample(n_vars, model, cardinality_spec)
        assert_cardinality_constraints(cardinality_spec, new_pop, "UMDA-edge")

    @pytest.mark.parametrize("cardinality_spec", [
        np.array([2, 5]),
        np.array([10, 2, 3]),
        np.array([2, 2, 3, 3, 4, 4, 5, 5]),
    ])
    def test_tree_eda_various_mixed_cardinalities(self, cardinality_spec):
        """Tree-EDA must work for various mixed-cardinality configurations."""
        np.random.seed(0)
        n_vars = len(cardinality_spec)
        pop = np.column_stack(
            [np.random.randint(0, c, size=80) for c in cardinality_spec]
        )
        fitness = np.sum(pop, axis=1).astype(float)
        model = LearnTreeModel(alpha=0.1).learn(
            0, n_vars, cardinality_spec, pop, fitness
        )
        new_pop = SampleFDA(n_samples=80).sample(n_vars, model, cardinality_spec)
        assert_cardinality_constraints(cardinality_spec, new_pop, "Tree-EDA-edge")

    @pytest.mark.parametrize("cardinality_spec", [
        np.array([2, 5]),
        np.array([10, 2, 3]),
        np.array([2, 2, 3, 3, 4, 4, 5, 5]),
    ])
    def test_mk_eda_various_mixed_cardinalities(self, cardinality_spec):
        """Mk-EDA must work for various mixed-cardinality configurations."""
        np.random.seed(0)
        n_vars = len(cardinality_spec)
        pop = np.column_stack(
            [np.random.randint(0, c, size=80) for c in cardinality_spec]
        )
        fitness = np.sum(pop, axis=1).astype(float)
        model = LearnMarkovChain(k=1, alpha=0.1).learn(
            0, n_vars, cardinality_spec, pop, fitness
        )
        new_pop = SampleMarkovChain(n_samples=80).sample(
            n_vars, model, cardinality_spec
        )
        assert_cardinality_constraints(cardinality_spec, new_pop, "Mk-EDA-edge")

    def test_all_algorithms_produce_valid_samples_for_extreme_cardinality(self):
        """All four algorithms must produce valid samples when cardinality ranges
        from 2 to 10 within the same problem."""
        np.random.seed(1)
        cardinality = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 2], dtype=int)
        n_vars = len(cardinality)
        pop = np.column_stack(
            [np.random.randint(0, c, size=150) for c in cardinality]
        )
        fitness = np.sum(pop, axis=1).astype(float)
        R = make_chain_interaction_matrix(n_vars)

        for name, learner, sampler_cls in [
            ("UMDA", LearnUMDA(alpha=0.1), SampleFDA),
            ("Tree-EDA", LearnTreeModel(alpha=0.1), SampleFDA),
            (
                "Tree-EDAr",
                LearnTreeModelR(interaction_matrix=R, alpha=0.1),
                SampleFDA,
            ),
            ("Mk-EDA", LearnMarkovChain(k=1, alpha=0.1), SampleMarkovChain),
        ]:
            model = learner.learn(0, n_vars, cardinality, pop, fitness)
            sampler = sampler_cls(n_samples=150)
            new_pop = sampler.sample(n_vars, model, cardinality)
            assert_cardinality_constraints(cardinality, new_pop, name)
