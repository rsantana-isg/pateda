"""
Tests for CUMDA and CFDA (Constraint EDAs for Unitation Problems)

Tests cover:
- CUMDA learning and sampling
- CFDA learning and sampling with different factorizations
- Constraint satisfaction (exact number of ones)
- Integration with EDA framework
"""

import numpy as np
import pytest
from pateda import EDA, EDAComponents
from pateda.learning import (
    LearnCUMDA,
    LearnCFDA,
    create_pairwise_chain_cliques,
    create_block_cliques,
    create_overlapping_windows_cliques,
)
from pateda.sampling import (
    SampleCUMDA,
    SampleCUMDARange,
    SampleCFDA,
    SampleCFDARange,
    SampleCFDAWeighted,
)
from pateda.seeding import SeedingUnitationConstraint
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


class TestCUMDALearning:
    """Tests for CUMDA learning"""

    def test_cumda_learns_probabilities(self):
        """Test that CUMDA learns marginal probabilities correctly"""
        n_vars = 10
        pop_size = 100
        cardinality = np.full(n_vars, 2)

        # Create population where last 5 variables are more often 1
        population = np.zeros((pop_size, n_vars), dtype=int)
        for i in range(pop_size):
            # Randomly set last 5 vars to 1 with 80% probability
            population[i, -5:] = np.random.rand(5) < 0.8
            # First 5 vars to 1 with 20% probability
            population[i, :5] = np.random.rand(5) < 0.2

        # Learn model
        learner = LearnCUMDA()
        model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=population,
            fitness=np.ones(pop_size),
        )

        # Check that p(x_i=1) is higher for last 5 variables
        p_ones = np.array([table[1] for table in model.parameters])
        assert np.mean(p_ones[-5:]) > np.mean(p_ones[:5])

    def test_cumda_requires_binary(self):
        """Test that CUMDA raises error for non-binary variables"""
        learner = LearnCUMDA()
        n_vars = 10
        cardinality = np.full(n_vars, 3)  # Ternary, not binary
        population = np.zeros((10, n_vars), dtype=int)
        fitness = np.ones(10)

        with pytest.raises(ValueError, match="binary"):
            learner.learn(0, n_vars, cardinality, population, fitness)

    def test_cumda_with_laplace_smoothing(self):
        """Test CUMDA with Laplace smoothing"""
        n_vars = 5
        cardinality = np.full(n_vars, 2)

        # Population where var 0 is always 0
        population = np.array([
            [0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ])

        # Without smoothing
        learner_no_smooth = LearnCUMDA(alpha=0.0)
        model_no_smooth = learner_no_smooth.learn(
            0, n_vars, cardinality, population, np.ones(3)
        )
        # p(x_0=1) should be exactly 0
        assert model_no_smooth.parameters[0][1] == 0.0

        # With smoothing
        learner_smooth = LearnCUMDA(alpha=1.0)
        model_smooth = learner_smooth.learn(
            0, n_vars, cardinality, population, np.ones(3)
        )
        # p(x_0=1) should be non-zero
        assert model_smooth.parameters[0][1] > 0.0


class TestCUMDASampling:
    """Tests for CUMDA sampling"""

    def test_cumda_samples_exact_ones(self):
        """Test that CUMDA sampling produces exactly n_ones ones"""
        n_vars = 20
        n_ones = 8
        n_samples = 50
        cardinality = np.full(n_vars, 2)

        # Create uniform model
        population = np.random.randint(0, 2, (100, n_vars))
        learner = LearnCUMDA()
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        # Sample
        sampler = SampleCUMDA(n_samples=n_samples, n_ones=n_ones)
        samples = sampler.sample(n_vars, model, cardinality)

        # Verify shape
        assert samples.shape == (n_samples, n_vars)

        # Verify all samples have exactly n_ones ones
        ones_per_sample = np.sum(samples, axis=1)
        assert np.all(ones_per_sample == n_ones)

    def test_cumda_range_sampling(self):
        """Test CUMDA range sampling (min_ones to max_ones)"""
        n_vars = 20
        min_ones = 5
        max_ones = 10
        n_samples = 50
        cardinality = np.full(n_vars, 2)

        population = np.random.randint(0, 2, (100, n_vars))
        learner = LearnCUMDA()
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        sampler = SampleCUMDARange(
            n_samples=n_samples, min_ones=min_ones, max_ones=max_ones
        )
        samples = sampler.sample(n_vars, model, cardinality)

        # Verify all samples have ones in range
        ones_per_sample = np.sum(samples, axis=1)
        assert np.all(ones_per_sample >= min_ones)
        assert np.all(ones_per_sample <= max_ones)

    def test_cumda_sampling_uses_probabilities(self):
        """Test that CUMDA sampling respects learned probabilities"""
        n_vars = 10
        n_ones = 5
        n_samples = 1000
        cardinality = np.full(n_vars, 2)

        # Create biased population: last 5 vars more likely to be 1
        population = np.zeros((100, n_vars), dtype=int)
        for i in range(100):
            # Always set last 5 to 1, first 5 to 0 (extreme bias)
            population[i, -5:] = 1

        learner = LearnCUMDA()
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        sampler = SampleCUMDA(n_samples=n_samples, n_ones=n_ones)
        samples = sampler.sample(n_vars, model, cardinality)

        # Last 5 vars should be set to 1 more often than first 5
        freq_last_5 = np.mean(samples[:, -5:])
        freq_first_5 = np.mean(samples[:, :5])
        assert freq_last_5 > freq_first_5


class TestCFDALearning:
    """Tests for CFDA learning"""

    def test_cfda_univariate(self):
        """Test CFDA with univariate factorization (like CUMDA)"""
        n_vars = 10
        cardinality = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (50, n_vars))

        learner = LearnCFDA(cliques=None)  # None = univariate
        model = learner.learn(0, n_vars, cardinality, population, np.ones(50))

        # Should have n_vars cliques (one per variable)
        assert model.structure.shape[0] == n_vars
        assert len(model.parameters) == n_vars

    def test_cfda_pairwise_chain(self):
        """Test CFDA with pairwise chain factorization"""
        n_vars = 10
        cardinality = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (50, n_vars))

        cliques = create_pairwise_chain_cliques(n_vars)
        learner = LearnCFDA(cliques=cliques)
        model = learner.learn(0, n_vars, cardinality, population, np.ones(50))

        # Should have n_vars-1 cliques for pairwise chain
        assert model.structure.shape[0] == n_vars - 1

    def test_cfda_block_factorization(self):
        """Test CFDA with block factorization"""
        n_vars = 12
        block_size = 3
        cardinality = np.full(n_vars, 2)
        population = np.random.randint(0, 2, (50, n_vars))

        cliques = create_block_cliques(n_vars, block_size)
        learner = LearnCFDA(cliques=cliques)
        model = learner.learn(0, n_vars, cardinality, population, np.ones(50))

        # Should have n_vars/block_size cliques
        assert model.structure.shape[0] == n_vars // block_size

    def test_cfda_requires_binary(self):
        """Test that CFDA requires binary variables"""
        learner = LearnCFDA()
        n_vars = 10
        cardinality = np.full(n_vars, 3)  # Ternary
        population = np.zeros((10, n_vars), dtype=int)

        with pytest.raises(ValueError, match="binary"):
            learner.learn(0, n_vars, cardinality, population, np.ones(10))

    def test_create_overlapping_windows(self):
        """Test overlapping windows clique creation"""
        n_vars = 10
        window_size = 3
        stride = 2

        cliques = create_overlapping_windows_cliques(n_vars, window_size, stride)

        # Verify structure
        assert cliques.shape[0] > 0
        # First window should have no overlap
        assert cliques[0, 0] == 0  # n_overlap
        assert cliques[0, 1] == window_size  # n_new


class TestCFDASampling:
    """Tests for CFDA sampling"""

    def test_cfda_samples_exact_ones(self):
        """Test that CFDA sampling maintains unitation constraint"""
        n_vars = 20
        n_ones = 8
        n_samples = 50
        cardinality = np.full(n_vars, 2)

        # Create model with pairwise factorization
        population = np.random.randint(0, 2, (100, n_vars))
        cliques = create_pairwise_chain_cliques(n_vars)
        learner = LearnCFDA(cliques=cliques)
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        # Sample
        sampler = SampleCFDA(n_samples=n_samples, n_ones=n_ones)
        samples = sampler.sample(n_vars, model, cardinality)

        # Verify all samples have exactly n_ones ones
        ones_per_sample = np.sum(samples, axis=1)
        assert np.all(ones_per_sample == n_ones)

    def test_cfda_range_sampling(self):
        """Test CFDA range sampling"""
        n_vars = 20
        min_ones = 5
        max_ones = 10
        n_samples = 50
        cardinality = np.full(n_vars, 2)

        population = np.random.randint(0, 2, (100, n_vars))
        learner = LearnCFDA(cliques=None)
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        sampler = SampleCFDARange(
            n_samples=n_samples, min_ones=min_ones, max_ones=max_ones
        )
        samples = sampler.sample(n_vars, model, cardinality)

        ones_per_sample = np.sum(samples, axis=1)
        assert np.all(ones_per_sample >= min_ones)
        assert np.all(ones_per_sample <= max_ones)

    def test_cfda_weighted_sampling(self):
        """Test CFDA weighted sampling"""
        n_vars = 20
        n_ones = 10
        n_samples = 50
        cardinality = np.full(n_vars, 2)

        population = np.random.randint(0, 2, (100, n_vars))
        learner = LearnCFDA(cliques=None)
        model = learner.learn(0, n_vars, cardinality, population, np.ones(100))

        sampler = SampleCFDAWeighted(n_samples=n_samples, n_ones=n_ones, alpha=0.5)
        samples = sampler.sample(n_vars, model, cardinality)

        ones_per_sample = np.sum(samples, axis=1)
        assert np.all(ones_per_sample == n_ones)


class TestCUMDAIntegration:
    """Integration tests for CUMDA in EDA framework"""

    def test_cumda_simple_optimization(self):
        """Test CUMDA on a simple optimization problem"""
        n_vars = 20
        r = 10  # Number of ones
        pop_size = 50
        max_gen = 10

        cardinality = np.full(n_vars, 2)

        # Simple fitness: prefer last r variables to be 1
        def fitness_func(x):
            if x.ndim == 1:
                return np.sum((np.arange(n_vars) + 1) * x)
            else:
                positions = np.arange(n_vars) + 1
                return np.sum(positions * x, axis=1)

        components = EDAComponents(
            seeding=SeedingUnitationConstraint(),
            seeding_params={'num_ones': r},
            learning=LearnCUMDA(),
            sampling=SampleCUMDA(n_samples=pop_size, n_ones=r),
            selection=TruncationSelection(ratio=0.5),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=fitness_func,
            cardinality=cardinality,
            components=components,
        )

        statistics, cache = eda.run(verbose=False)

        # Should improve over generations
        assert statistics.best_fitness_overall >= statistics.best_fitness[0]

        # All solutions should have exactly r ones
        final_pop = cache['population']
        ones_per_solution = np.sum(final_pop, axis=1)
        assert np.all(ones_per_solution == r)


class TestCFDAIntegration:
    """Integration tests for CFDA in EDA framework"""

    def test_cfda_simple_optimization(self):
        """Test CFDA on a simple optimization problem"""
        n_vars = 15
        r = 7
        pop_size = 50
        max_gen = 10

        cardinality = np.full(n_vars, 2)

        # Fitness function that prefers contiguous blocks
        def fitness_func(x):
            if x.ndim == 1:
                return _max_block_length(x)
            else:
                return np.array([_max_block_length(row) for row in x])

        def _max_block_length(x):
            if len(x) == 0 or np.sum(x) == 0:
                return 0
            max_len = 0
            current_len = 0
            for bit in x:
                if bit == 1:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0
            return max_len

        # Use pairwise factorization (better for contiguous blocks)
        cliques = create_pairwise_chain_cliques(n_vars)

        components = EDAComponents(
            seeding=SeedingUnitationConstraint(),
            seeding_params={'num_ones': r},
            learning=LearnCFDA(cliques=cliques),
            sampling=SampleCFDA(n_samples=pop_size, n_ones=r),
            selection=TruncationSelection(ratio=0.5),
            stop_condition=MaxGenerations(max_gen=max_gen),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=fitness_func,
            cardinality=cardinality,
            components=components,
        )

        statistics, cache = eda.run(verbose=False)

        # Should improve
        assert statistics.best_fitness_overall >= statistics.best_fitness[0]

        # All solutions should have exactly r ones
        final_pop = cache['population']
        ones_per_solution = np.sum(final_pop, axis=1)
        assert np.all(ones_per_solution == r)


class TestCliqueCreationHelpers:
    """Tests for clique creation helper functions"""

    def test_pairwise_chain_basic(self):
        """Test basic pairwise chain creation"""
        n_vars = 5
        cliques = create_pairwise_chain_cliques(n_vars)

        # Should have n_vars-1 cliques
        assert cliques.shape[0] == n_vars - 1

        # First clique: (x0, x1)
        assert cliques[0, 0] == 0  # no overlap
        assert cliques[0, 1] == 2  # 2 new vars
        assert cliques[0, 2] == 0  # x0
        assert cliques[0, 3] == 1  # x1

        # Second clique: (x1, x2)
        assert cliques[1, 0] == 1  # 1 overlap (x1)
        assert cliques[1, 1] == 1  # 1 new (x2)
        assert cliques[1, 2] == 1  # overlap: x1
        assert cliques[1, 3] == 2  # new: x2

    def test_block_cliques_basic(self):
        """Test basic block cliques creation"""
        n_vars = 9
        block_size = 3
        cliques = create_block_cliques(n_vars, block_size)

        # Should have 3 blocks
        assert cliques.shape[0] == 3

        # Each block should have no overlap and block_size new vars
        for i in range(3):
            assert cliques[i, 0] == 0  # no overlap
            assert cliques[i, 1] == block_size  # block_size new vars

    def test_block_cliques_invalid_size(self):
        """Test that block cliques raises error for invalid size"""
        with pytest.raises(ValueError):
            create_block_cliques(10, 3)  # 10 not divisible by 3

    def test_overlapping_windows_basic(self):
        """Test overlapping windows creation"""
        n_vars = 10
        window_size = 3
        stride = 2

        cliques = create_overlapping_windows_cliques(n_vars, window_size, stride)

        # Should have multiple windows
        assert cliques.shape[0] > 1

        # First window should have no overlap
        assert cliques[0, 0] == 0
        assert cliques[0, 1] == window_size
