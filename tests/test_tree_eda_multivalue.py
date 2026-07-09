"""
Tests for Tree-EDA with multi-value (non-binary) discrete representations.

This module validates:
1. The fix for divide-by-zero in Tree-EDA when parent values are absent from population
2. Non-zero probabilities (prior) for configurations not present in the population
3. Tree-EDA correctness for various cardinalities (binary, ternary, quaternary, etc.)
4. End-to-end Tree-EDA run on non-binary benchmark functions
"""
import warnings
import numpy as np
import pytest
from pathlib import Path

from pateda.learning.tree import LearnTreeModel
from pateda.sampling.fda import SampleFDA
from pateda.core.models import FactorizedModel


class TestTreeEDADivideByZeroFix:
    """Tests that Tree-EDA does not produce divide-by-zero for missing parent values."""

    def test_no_divide_by_zero_missing_parent_value(self):
        """
        When a parent variable's value never appears in the selected population,
        the old code raised RuntimeWarning: divide by zero encountered in divide.
        The fix applies Laplace smoothing to parent marginals before division.
        """
        np.random.seed(42)
        n_vars = 6
        cardinality = np.array([4, 4, 4, 4, 4, 4])
        n_samples = 20

        # Population where value 3 never appears for any variable (cardinality=4 but only 0-2 used)
        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01, mi_threshold=0.01)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # Promote warnings to errors
            # Should NOT raise RuntimeWarning: divide by zero
            model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        assert model is not None
        assert len(model.parameters) == n_vars

    def test_no_nan_in_probability_tables(self):
        """Probability tables must not contain NaN or Inf."""
        np.random.seed(10)
        n_vars = 8
        cardinality = np.array([5, 5, 5, 5, 5, 5, 5, 5])
        n_samples = 15

        # Only values 0-3 appear, value 4 is absent
        population = np.random.randint(0, 4, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.0, mi_threshold=1e-4)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), (
                f"Table {i} contains NaN or Inf values"
            )

    def test_no_negative_probabilities(self):
        """All probability values must be non-negative."""
        np.random.seed(7)
        n_vars = 4
        cardinality = np.array([3, 3, 3, 3])
        n_samples = 10

        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01, mi_threshold=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        for i, table in enumerate(model.parameters):
            assert np.all(table >= 0), f"Table {i} has negative probability values"


class TestTreeEDANonZeroPrior:
    """Tests that configurations absent from the population have non-zero probability."""

    def test_missing_value_has_nonzero_marginal_probability(self):
        """
        Root variable probabilities must be positive even for values not seen.
        Laplace smoothing in the root node computation ensures this.
        """
        np.random.seed(3)
        n_vars = 3
        cardinality = np.array([4, 4, 4])
        n_samples = 30

        # Variable 0 never takes value 3
        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.0, mi_threshold=1e-4)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        # Find root nodes (n_parents == 0) and verify all missing values have non-zero prob.
        # The tree root is chosen dynamically, so we check every root node table.
        cliques = model.structure
        found_root = False
        for j in range(n_vars):
            if cliques[j, 0] == 0:
                found_root = True
                table = model.parameters[j]
                # Value 3 never appeared for any variable; the Laplace prior must give > 0
                assert table[3] > 0, (
                    "Missing value should have non-zero probability due to Laplace prior"
                )
        assert found_root, "Expected at least one root node in tree structure"

    def test_missing_parent_value_has_nonzero_conditional_probability(self):
        """
        Conditional probabilities P(child | parent=v) must be non-zero
        even when parent value v never appears in the population.
        """
        np.random.seed(99)
        n_vars = 4
        cardinality = np.array([4, 4, 4, 4])
        n_samples = 25

        # Only values 0-2 appear
        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01, mi_threshold=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        cliques = model.structure
        for j in range(n_vars):
            if cliques[j, 0] == 1:  # Non-root with one parent
                table = model.parameters[j]
                # All rows (parent configurations) must have non-zero probs
                assert np.all(table > 0), (
                    f"Table {j} has zero entries; absent configurations must have non-zero prior"
                )


class TestTreeEDAValidProbabilityTables:
    """Tests that probability tables form valid distributions."""

    def test_marginal_table_sums_to_one(self):
        """Root node marginal probabilities must sum to 1."""
        np.random.seed(1)
        n_vars = 5
        cardinality = np.array([3, 3, 3, 3, 3])
        n_samples = 40
        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        cliques = model.structure
        for j in range(n_vars):
            if cliques[j, 0] == 0:
                table = model.parameters[j]
                assert np.isclose(table.sum(), 1.0, atol=1e-6), (
                    f"Root table {j} does not sum to 1 (sum={table.sum()})"
                )

    def test_conditional_table_rows_sum_to_one(self):
        """Each row of conditional tables must sum to 1."""
        np.random.seed(2)
        n_vars = 6
        cardinality = np.array([4, 4, 4, 4, 4, 4])
        n_samples = 50
        population = np.random.randint(0, 4, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        cliques = model.structure
        for j in range(n_vars):
            if cliques[j, 0] == 1:
                table = model.parameters[j]
                for row in range(table.shape[0]):
                    assert np.isclose(table[row].sum(), 1.0, atol=1e-6), (
                        f"Table {j} row {row} does not sum to 1 (sum={table[row].sum()})"
                    )

    def test_valid_tables_for_binary_cardinality(self):
        """Verify correctness still holds for binary (cardinality=2) problems."""
        np.random.seed(5)
        n_vars = 8
        cardinality = np.full(n_vars, 2)
        n_samples = 50
        population = np.random.randint(0, 2, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table))
            assert np.all(table >= 0)

    def test_valid_tables_for_high_cardinality(self):
        """Verify correctness for high cardinality (10-valued variables)."""
        np.random.seed(6)
        n_vars = 4
        cardinality = np.full(n_vars, 10)
        n_samples = 30

        # Only values 0-4 appear (half of possible values absent)
        population = np.random.randint(0, 5, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        for i, table in enumerate(model.parameters):
            assert np.all(np.isfinite(table)), f"Table {i} has non-finite values"
            assert np.all(table >= 0), f"Table {i} has negative values"

        cliques = model.structure
        for j in range(n_vars):
            if cliques[j, 0] == 1:
                table = model.parameters[j]
                for row in range(table.shape[0]):
                    assert np.isclose(table[row].sum(), 1.0, atol=1e-6)


class TestTreeEDASamplingWithMultivalue:
    """Tests for end-to-end Tree-EDA (learning + sampling) with non-binary problems."""

    def test_sampling_produces_valid_population(self):
        """Sampled population should contain valid integer values within cardinality."""
        np.random.seed(20)
        n_vars = 6
        cardinality = np.array([4, 4, 4, 4, 4, 4])
        n_samples = 50
        population = np.random.randint(0, 4, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        sampler = SampleFDA(n_samples=n_samples)
        rng = np.random.default_rng(20)
        new_pop = sampler.sample(n_vars, model, cardinality, rng=rng)

        assert new_pop.shape == (n_samples, n_vars)
        for var in range(n_vars):
            assert np.all(new_pop[:, var] >= 0), f"Variable {var} has negative values"
            assert np.all(new_pop[:, var] < cardinality[var]), (
                f"Variable {var} has values >= cardinality"
            )

    def test_sampling_with_missing_parent_values(self):
        """
        Sampling must succeed even when some parent values were absent
        in the training population (the missing-value scenario).
        """
        np.random.seed(21)
        n_vars = 6
        cardinality = np.array([4, 4, 4, 4, 4, 4])
        n_samples = 30

        # Only values 0-2 appear in training population
        population = np.random.randint(0, 3, (n_samples, n_vars))

        learner = LearnTreeModel(alpha=0.01)
        model = learner.learn(0, n_vars, cardinality, population, np.zeros(n_samples))

        sampler = SampleFDA(n_samples=n_samples)
        rng = np.random.default_rng(21)
        new_pop = sampler.sample(n_vars, model, cardinality, rng=rng)

        assert new_pop.shape == (n_samples, n_vars)
        assert not np.any(new_pop == -1), "Some variables were not assigned during sampling"

    def test_full_eda_run_nonbinary_onemax(self):
        """
        Full EDA loop with Tree-EDA on integer OneMax (cardinality=4).
        Fitness should improve over generations.
        """
        from pateda import EDA, EDAComponents
        from pateda.selection import TruncationSelection
        from pateda.stop_conditions import MaxGenerations
        from pateda.seeding import RandomInit
        from pateda.functions.discrete_non_binary.toy_functions.integer_functions import integer_onemax

        np.random.seed(30)
        n_vars = 9
        cardinality_val = 4
        pop_size = 100
        max_gen = 20

        cardinality = np.full(n_vars, cardinality_val)

        def fitness_func(x):
            if x.ndim == 1:
                return np.array([integer_onemax(x, cardinality=cardinality_val)])
            return np.array([integer_onemax(x[i], cardinality=cardinality_val)
                             for i in range(x.shape[0])], dtype=float)

        components = EDAComponents(
            seeding=RandomInit(),
            learning=LearnTreeModel(alpha=0.01, mi_threshold=0.01),
            sampling=SampleFDA(n_samples=pop_size),
            selection=TruncationSelection(ratio=0.5),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=fitness_func,
            cardinality=cardinality,
            components=components,
            random_seed=30,
        )

        stats, _ = eda.run(verbose=False)

        # Fitness should improve
        assert stats.best_fitness_overall > 0
        # With 20 generations and pop=100, should get close to optimal (9*(4-1)=27)
        assert stats.best_fitness_overall >= n_vars * (cardinality_val - 1) * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
