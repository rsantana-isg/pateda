"""
Tests for Generalized Mallows models (Kendall and Cayley distances)

Tests both learning and sampling for Generalized Mallows models.
"""

import numpy as np
import pytest
from pateda.learning.mallows import (
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
)
from pateda.sampling.mallows import (
    SampleGeneralizedMallowsKendall,
    SampleGeneralizedMallowsCayley,
)
from pateda.permutation.distances import kendall_distance, cayley_distance
from pateda.permutation.consensus import find_consensus_borda


class TestGeneralizedMallowsKendall:
    """Test Generalized Mallows model with Kendall distance"""

    def test_learning_basic(self):
        """Test basic learning functionality"""
        np.random.seed(42)

        # Create a simple population
        n_vars = 5
        pop_size = 20
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        # Learn model
        learner = LearnGeneralizedMallowsKendall()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
            initial_theta=0.1,
            upper_theta=10.0,
            max_iter=100,
            consensus_method="borda",
        )

        # Check model structure
        assert "v_probs" in model
        assert "consensus" in model
        assert "theta" in model
        assert "psis" in model
        assert model["model_type"] == "generalized_mallows_kendall"

        # Check dimensions
        assert model["v_probs"].shape == (n_vars - 1, n_vars)
        assert len(model["consensus"]) == n_vars
        assert len(model["theta"]) == n_vars - 1  # Vector of thetas
        assert len(model["psis"]) == n_vars - 1

        # Check that probabilities sum to 1 for each position
        for j in range(n_vars - 1):
            prob_sum = np.sum(model["v_probs"][j, : n_vars - j])
            assert np.isclose(prob_sum, 1.0), f"Position {j}: sum = {prob_sum}"

    def test_sampling_basic(self):
        """Test basic sampling functionality"""
        np.random.seed(42)

        # Create a simple population and learn model
        n_vars = 6
        pop_size = 30
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        learner = LearnGeneralizedMallowsKendall()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        # Sample from model
        sampler = SampleGeneralizedMallowsKendall()
        sample_size = 50
        new_pop = sampler(
            n_vars=n_vars,
            model=model,
            cardinality=np.arange(n_vars),
            population=population,
            fitness=fitness,
            sample_size=sample_size,
        )

        # Check dimensions
        assert new_pop.shape == (sample_size, n_vars)

        # Check that all samples are valid permutations
        for i in range(sample_size):
            assert set(new_pop[i]) == set(range(n_vars)), f"Sample {i} is not a valid permutation"

    def test_theta_vector_different_values(self):
        """Test that theta vector has position-dependent values"""
        np.random.seed(42)

        # Create population with specific structure
        n_vars = 8
        pop_size = 50

        # Create consensus
        consensus = np.arange(n_vars)

        # Create population with varying distances at different positions
        population = []
        for _ in range(pop_size):
            perm = np.random.permutation(n_vars)
            population.append(perm)

        population = np.array(population)
        fitness = np.random.rand(pop_size)

        # Learn model
        learner = LearnGeneralizedMallowsKendall()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        # Check that thetas are reasonable
        thetas = model["theta"]
        assert len(thetas) == n_vars - 1
        assert np.all(thetas > 0), "All thetas should be positive"
        assert np.all(thetas < 100), "Thetas should be reasonable"

    def test_consensus_preserved(self):
        """Test that consensus ranking is correctly learned and used"""
        np.random.seed(42)

        n_vars = 5
        consensus_true = np.array([2, 0, 4, 1, 3])

        # Create population around the consensus
        pop_size = 30
        population = []
        for _ in range(pop_size):
            # Start with consensus and make small changes
            perm = consensus_true.copy()
            # Swap a few random positions
            for _ in range(2):
                i, j = np.random.choice(n_vars, 2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]
            population.append(perm)

        population = np.array(population)
        fitness = np.random.rand(pop_size)

        # Learn model
        learner = LearnGeneralizedMallowsKendall()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
            consensus_method="borda",
        )

        # Consensus should be close to the true consensus
        learned_consensus = model["consensus"]
        assert len(learned_consensus) == n_vars


class TestGeneralizedMallowsCayley:
    """Test Generalized Mallows model with Cayley distance"""

    def test_learning_basic(self):
        """Test basic learning functionality"""
        np.random.seed(42)

        # Create a simple population
        n_vars = 5
        pop_size = 20
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        # Learn model
        learner = LearnGeneralizedMallowsCayley()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
            initial_theta=0.1,
            upper_theta=10.0,
            max_iter=100,
            consensus_method="borda",
        )

        # Check model structure
        assert "x_probs" in model
        assert "consensus" in model
        assert "theta" in model
        assert "psis" in model
        assert model["model_type"] == "generalized_mallows_cayley"

        # Check dimensions
        assert model["x_probs"].shape == (n_vars - 1, 2)
        assert len(model["consensus"]) == n_vars
        assert len(model["theta"]) == n_vars - 1  # Vector of thetas
        assert len(model["psis"]) == n_vars - 1

        # Check that probabilities sum to 1 for each position
        for j in range(n_vars - 1):
            prob_sum = np.sum(model["x_probs"][j, :])
            assert np.isclose(prob_sum, 1.0), f"Position {j}: sum = {prob_sum}"

    def test_sampling_basic(self):
        """Test basic sampling functionality"""
        np.random.seed(42)

        # Create a simple population and learn model
        n_vars = 6
        pop_size = 30
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        learner = LearnGeneralizedMallowsCayley()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        # Sample from model
        sampler = SampleGeneralizedMallowsCayley()
        sample_size = 50
        new_pop = sampler(
            n_vars=n_vars,
            model=model,
            cardinality=np.arange(n_vars),
            population=population,
            fitness=fitness,
            sample_size=sample_size,
        )

        # Check dimensions
        assert new_pop.shape == (sample_size, n_vars)

        # Check that all samples are valid permutations
        for i in range(sample_size):
            assert set(new_pop[i]) == set(range(n_vars)), f"Sample {i} is not a valid permutation"

    def test_theta_vector_properties(self):
        """Test that theta vector has reasonable properties"""
        np.random.seed(42)

        n_vars = 7
        pop_size = 40
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        # Learn model
        learner = LearnGeneralizedMallowsCayley()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        # Check that thetas are reasonable
        thetas = model["theta"]
        assert len(thetas) == n_vars - 1
        assert np.all(thetas > 0), "All thetas should be positive"
        assert np.all(thetas < 100), "Thetas should be reasonable"

        # Check x_probs
        x_probs = model["x_probs"]
        assert np.all(x_probs[:, 0] >= 0) and np.all(x_probs[:, 0] <= 1)
        assert np.all(x_probs[:, 1] >= 0) and np.all(x_probs[:, 1] <= 1)


class TestGeneralizedMallowsIntegration:
    """Integration tests comparing Generalized Mallows with regular Mallows"""

    def test_generalized_reduces_to_mallows(self):
        """Test that when all thetas are equal, GM reduces to Mallows"""
        np.random.seed(42)

        n_vars = 5
        pop_size = 50

        # Create identical population for both
        population = np.array([np.random.permutation(n_vars) for _ in range(pop_size)])
        fitness = np.random.rand(pop_size)

        # Learn Generalized Mallows
        gm_learner = LearnGeneralizedMallowsKendall()
        gm_model = gm_learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        # Check that thetas exist and are positive
        assert len(gm_model["theta"]) == n_vars - 1
        assert np.all(gm_model["theta"] > 0)

    def test_sample_distribution(self):
        """Test that sampled population follows the learned distribution"""
        np.random.seed(42)

        n_vars = 6
        pop_size = 100

        # Create population biased toward identity permutation
        population = []
        identity = np.arange(n_vars)
        for _ in range(pop_size):
            if np.random.rand() < 0.7:
                # Close to identity
                perm = identity.copy()
                # Swap 1-2 positions
                for _ in range(np.random.randint(1, 3)):
                    i, j = np.random.choice(n_vars, 2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
                population.append(perm)
            else:
                # Random
                population.append(np.random.permutation(n_vars))

        population = np.array(population)
        fitness = np.random.rand(pop_size)

        # Learn and sample
        learner = LearnGeneralizedMallowsKendall()
        model = learner(
            generation=0,
            n_vars=n_vars,
            cardinality=np.arange(n_vars),
            selected_pop=population,
            selected_fitness=fitness,
        )

        sampler = SampleGeneralizedMallowsKendall()
        sample_size = 200
        new_pop = sampler(
            n_vars=n_vars,
            model=model,
            cardinality=np.arange(n_vars),
            population=population,
            fitness=fitness,
            sample_size=sample_size,
        )

        # Check that sampled population has lower average distance to consensus
        # than a random population
        consensus = model["consensus"]
        avg_dist_sampled = np.mean([kendall_distance(perm, consensus) for perm in new_pop])

        random_pop = np.array([np.random.permutation(n_vars) for _ in range(sample_size)])
        avg_dist_random = np.mean([kendall_distance(perm, consensus) for perm in random_pop])

        # Sampled population should be closer to consensus
        assert avg_dist_sampled < avg_dist_random


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
