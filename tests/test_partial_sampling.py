"""
Tests for partial sampling (SamplePartialFDA) from discrete EDAs.

Validates that partial sampling:
1. Keeps fixed (non-NaN) positions unchanged.
2. Samples NaN positions from the model.
3. Works with UMDA (independent univariate), Tree-EDA (tree structure),
   MN-FDA / MN-FDAG (Markov-network cliques), and binary / multi-valued variables.
4. Works with a single template (1-D array) and a population of templates (2-D).
5. Handles edge cases: all NaN (equivalent to full sampling), no NaN (all fixed).
"""

import numpy as np
import pytest

from pateda.sampling.partial import SamplePartialFDA
from pateda.learning.umda import LearnUMDA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.mnfda import LearnMNFDA
from pateda.core.models import FactorizedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _learn_umda(population, cardinality):
    n_vars = population.shape[1]
    learner = LearnUMDA(alpha=0.01)
    return learner.learn(0, n_vars, cardinality, population, np.zeros(len(population)))


def _learn_tree(population, cardinality):
    n_vars = population.shape[1]
    learner = LearnTreeModel(alpha=0.01, mi_threshold=0.0)
    return learner.learn(0, n_vars, cardinality, population, np.zeros(len(population)))


def _learn_mnfda(population, cardinality):
    n_vars = population.shape[1]
    learner = LearnMNFDA(max_clique_size=2, threshold=0.0, prior=True)
    return learner.learn(0, n_vars, cardinality, population, np.zeros(len(population)))


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestSamplePartialFDAInterface:
    """Tests for the SamplePartialFDA interface."""

    def test_no_template_equivalent_to_full_sampling(self):
        """With aux_pop=None all positions are NaN → produces full samples."""
        rng = np.random.default_rng(0)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (50, n_vars))
        model = _learn_umda(population, cardinality)

        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=None, rng=rng)

        assert result.shape == (20, n_vars)
        assert np.all((result == 0) | (result == 1))

    def test_single_1d_template_shape(self):
        """A single 1-D template is replicated for every sample."""
        rng = np.random.default_rng(1)
        n_vars = 6
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (40, n_vars))
        model = _learn_umda(population, cardinality)

        template = np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
        sampler = SamplePartialFDA(n_samples=15)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert result.shape == (15, n_vars)

    def test_2d_template_shape(self):
        """A 2-D template population is accepted."""
        rng = np.random.default_rng(2)
        n_vars = 6
        n_samples = 10
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (40, n_vars))
        model = _learn_umda(population, cardinality)

        templates = np.random.choice([0.0, 1.0, np.nan], size=(n_samples, n_vars))
        sampler = SamplePartialFDA(n_samples=n_samples)
        result = sampler.sample(
            n_vars, model, cardinality, aux_pop=templates, rng=rng
        )

        assert result.shape == (n_samples, n_vars)

    def test_wrong_model_type_raises(self):
        """Non-FactorizedModel raises TypeError."""
        from pateda.core.models import GaussianModel

        sampler = SamplePartialFDA(n_samples=5)
        bad_model = GaussianModel(structure=None, parameters={})
        with pytest.raises(TypeError, match="FactorizedModel"):
            sampler.sample(4, bad_model, np.array([2, 2, 2, 2]))

    def test_1d_template_wrong_length_raises(self):
        """A 1-D template with wrong length raises ValueError."""
        rng = np.random.default_rng(3)
        n_vars = 6
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (40, n_vars))
        model = _learn_umda(population, cardinality)

        template = np.array([0.0, 1.0, np.nan])  # too short
        sampler = SamplePartialFDA(n_samples=5)
        with pytest.raises(ValueError):
            sampler.sample(n_vars, model, cardinality, aux_pop=template)

    def test_2d_template_wrong_sample_count_raises(self):
        """A 2-D template with wrong number of rows raises ValueError."""
        rng = np.random.default_rng(4)
        n_vars = 6
        n_samples = 10
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (40, n_vars))
        model = _learn_umda(population, cardinality)

        templates = np.full((5, n_vars), np.nan)  # only 5 rows, not 10
        sampler = SamplePartialFDA(n_samples=n_samples)
        with pytest.raises(ValueError, match="n_samples"):
            sampler.sample(n_vars, model, cardinality, aux_pop=templates)


# ---------------------------------------------------------------------------
# Fixed-position preservation tests
# ---------------------------------------------------------------------------

class TestFixedPositionPreservation:
    """Fixed (non-NaN) positions must be unchanged in the output."""

    def test_umda_fixed_positions_unchanged(self):
        """UMDA: non-NaN positions are preserved exactly."""
        rng = np.random.default_rng(10)
        n_vars = 10
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_umda(population, cardinality)

        # Fix all even positions to 0, odd positions are NaN
        template = np.array(
            [0.0 if i % 2 == 0 else np.nan for i in range(n_vars)]
        )
        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        # Even positions must be 0
        assert np.all(result[:, ::2] == 0)

    def test_umda_all_fixed_no_sampling(self):
        """UMDA: if no positions are NaN, output equals the template."""
        rng = np.random.default_rng(11)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (40, n_vars))
        model = _learn_umda(population, cardinality)

        template = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        sampler = SamplePartialFDA(n_samples=5)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        expected = np.tile(template.astype(int), (5, 1))
        np.testing.assert_array_equal(result, expected)

    def test_tree_fixed_positions_unchanged(self):
        """Tree-EDA: non-NaN positions are preserved exactly."""
        rng = np.random.default_rng(12)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_tree(population, cardinality)

        # Fix first half to 1, second half NaN
        template = np.array([1.0] * 4 + [np.nan] * 4)
        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert np.all(result[:, :4] == 1)

    def test_mnfda_fixed_positions_unchanged(self):
        """MN-FDA: non-NaN positions are preserved exactly."""
        rng = np.random.default_rng(13)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_mnfda(population, cardinality)

        template = np.array([0.0, np.nan, 0.0, np.nan, 1.0, np.nan, 1.0, np.nan])
        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert np.all(result[:, 0] == 0)
        assert np.all(result[:, 2] == 0)
        assert np.all(result[:, 4] == 1)
        assert np.all(result[:, 6] == 1)


# ---------------------------------------------------------------------------
# Sampled positions tests
# ---------------------------------------------------------------------------

class TestSampledPositions:
    """NaN positions must be filled with valid values."""

    def test_umda_sampled_positions_valid(self):
        """UMDA: sampled (NaN) positions hold valid integer values in range."""
        rng = np.random.default_rng(20)
        n_vars = 10
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_umda(population, cardinality)

        # All positions NaN
        template = np.full(n_vars, np.nan)
        sampler = SamplePartialFDA(n_samples=30)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert np.all((result == 0) | (result == 1))

    def test_tree_sampled_positions_valid(self):
        """Tree-EDA: sampled positions hold valid integers."""
        rng = np.random.default_rng(21)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_tree(population, cardinality)

        template = np.full(n_vars, np.nan)
        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert np.all((result == 0) | (result == 1))

    def test_multivalue_sampled_positions_valid(self):
        """Multi-valued variables: sampled positions are within [0, cardinality-1]."""
        rng = np.random.default_rng(22)
        n_vars = 6
        card_value = 4
        cardinality = np.full(n_vars, card_value, dtype=int)
        population = rng.integers(0, card_value, (60, n_vars))
        model = _learn_umda(population, cardinality)

        # Fix first 3 positions, sample the rest
        template = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan])
        sampler = SamplePartialFDA(n_samples=20)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        # Fixed positions intact
        assert np.all(result[:, 0] == 0)
        assert np.all(result[:, 1] == 1)
        assert np.all(result[:, 2] == 2)
        # Sampled positions in range
        assert np.all(result[:, 3:] >= 0)
        assert np.all(result[:, 3:] < card_value)


# ---------------------------------------------------------------------------
# Correctness tests: verify sampling respects model distributions
# ---------------------------------------------------------------------------

class TestDistributionCorrectness:
    """Sampled positions should reflect the learned probability distribution."""

    def test_umda_biased_probability(self):
        """
        With a strongly biased UMDA model (p≈1 for value 1), NaN positions
        should predominantly sample value 1.
        """
        rng = np.random.default_rng(30)
        n_vars = 5
        cardinality = np.full(n_vars, 2, dtype=int)

        # Almost all 1s
        population = np.ones((80, n_vars), dtype=int)
        population[:5] = 0  # a few 0s to avoid degenerate distribution

        model = _learn_umda(population, cardinality)

        template = np.full(n_vars, np.nan)
        sampler = SamplePartialFDA(n_samples=200)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        # With strong bias toward 1, most samples should be 1
        assert np.mean(result) > 0.8

    def test_tree_biased_probability(self):
        """
        With a biased Tree-EDA model, NaN positions reflect the learned conditional
        distributions.
        """
        rng = np.random.default_rng(31)
        n_vars = 6
        cardinality = np.full(n_vars, 2, dtype=int)

        # Strongly biased toward all-zeros
        population = np.zeros((80, n_vars), dtype=int)
        population[:5] = 1

        model = _learn_tree(population, cardinality)

        template = np.full(n_vars, np.nan)
        sampler = SamplePartialFDA(n_samples=200)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        assert np.mean(result) < 0.2

    def test_partial_sampling_does_not_change_fixed_stats(self):
        """
        When only some positions are NaN, the fixed positions remain exactly as
        given and sampled positions obey the model distribution.
        """
        rng = np.random.default_rng(32)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)

        # Model with strong bias toward 1
        population = np.ones((80, n_vars), dtype=int)
        population[:5] = 0
        model = _learn_umda(population, cardinality)

        # Fix first 4 positions to 0 (against the model's bias)
        template = np.array([0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan])
        sampler = SamplePartialFDA(n_samples=100)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=template, rng=rng)

        # Fixed positions must all be 0 regardless of model bias
        assert np.all(result[:, :4] == 0)
        # Sampled positions should be mostly 1 (following model)
        assert np.mean(result[:, 4:]) > 0.7


# ---------------------------------------------------------------------------
# Per-sample template tests
# ---------------------------------------------------------------------------

class TestPerSampleTemplates:
    """Each row in a 2-D template array acts as its own template."""

    def test_different_templates_different_fixed_positions(self):
        """
        Different templates in a 2-D aux_pop produce samples with different
        fixed-position patterns.
        """
        rng = np.random.default_rng(40)
        n_vars = 6
        n_samples = 4
        cardinality = np.full(n_vars, 2, dtype=int)
        population = rng.integers(0, 2, (60, n_vars))
        model = _learn_umda(population, cardinality)

        # Each row has a different fixed value at position 0
        templates = np.full((n_samples, n_vars), np.nan)
        templates[0, 0] = 0.0
        templates[1, 0] = 1.0
        templates[2, 0] = 0.0
        templates[3, 0] = 1.0

        sampler = SamplePartialFDA(n_samples=n_samples)
        result = sampler.sample(n_vars, model, cardinality, aux_pop=templates, rng=rng)

        assert result[0, 0] == 0
        assert result[1, 0] == 1
        assert result[2, 0] == 0
        assert result[3, 0] == 1


# ---------------------------------------------------------------------------
# Integration tests: run a short EDA loop using partial sampling
# ---------------------------------------------------------------------------

class TestPartialSamplingInEDALoop:
    """Integration tests combining learning and partial sampling."""

    def test_umda_partial_sampling_loop(self):
        """
        Run a short UMDA optimization where half the variables are fixed per
        generation.  The non-fixed half should still show learning progress.
        """
        rng = np.random.default_rng(50)
        n_vars = 10
        cardinality = np.full(n_vars, 2, dtype=int)
        pop_size = 60

        population = rng.integers(0, 2, (pop_size, n_vars))

        best_fitness_history = []
        sampler = SamplePartialFDA(n_samples=pop_size)

        for _ in range(10):
            fitness = np.sum(population, axis=1)
            best_fitness_history.append(int(np.max(fitness)))

            # Select top 50 %
            idx = np.argsort(-fitness)[: pop_size // 2]
            selected = population[idx]

            model = _learn_umda(selected, cardinality)

            # Fix first half of positions to best individual's values
            best_ind = population[np.argmax(fitness)]
            template = best_ind.astype(float)
            template[n_vars // 2 :] = np.nan  # sample second half

            population = sampler.sample(
                n_vars, model, cardinality, aux_pop=template, rng=rng
            )

        # First half is always fixed to best → should be good
        assert best_fitness_history[-1] >= best_fitness_history[0]

    def test_tree_eda_partial_sampling_preserves_fixed(self):
        """
        In a Tree-EDA loop with partial sampling, fixed positions are always
        kept intact generation after generation.
        """
        rng = np.random.default_rng(51)
        n_vars = 8
        cardinality = np.full(n_vars, 2, dtype=int)
        pop_size = 50

        population = rng.integers(0, 2, (pop_size, n_vars))

        sampler = SamplePartialFDA(n_samples=pop_size)
        fixed_template = np.array([1.0, 0.0] + [np.nan] * (n_vars - 2))

        for _ in range(5):
            fitness = np.sum(population, axis=1)
            idx = np.argsort(-fitness)[: pop_size // 2]
            model = _learn_tree(population[idx], cardinality)
            population = sampler.sample(
                n_vars, model, cardinality, aux_pop=fixed_template, rng=rng
            )

        # Positions 0 and 1 must always be 1 and 0 respectively
        assert np.all(population[:, 0] == 1)
        assert np.all(population[:, 1] == 0)
