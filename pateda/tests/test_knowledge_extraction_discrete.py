"""
Integration tests for knowledge extraction with discrete EDAs.

Tests the knowledge extraction functionality with various discrete EDAs
including UMDA, BOA, and other probabilistic model-based algorithms.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import pytest
import numpy as np
from typing import Any

# Import knowledge extraction modules
from pateda.knowledge_extraction.fitness_measures import (
    response_to_selection,
    amount_of_selection,
    realized_heritability,
    compute_objective_distribution,
    analyze_fitness_evolution
)
from pateda.knowledge_extraction.dependency_analysis import (
    compute_correlation_matrix,
    learn_bayesian_network,
    learn_gaussian_network
)
from pateda.knowledge_extraction.model_visualizations import (
    view_dendrogram_structure,
    view_glyph_structure
)
from pateda.knowledge_extraction.eda_strategies import (
    extract_bayesian_network_evolution,
    extract_probability_distribution_evolution,
    generate_comprehensive_report
)


class TestFitnessMeasuresDiscrete:
    """Test fitness-related measures with discrete populations."""

    def test_response_to_selection(self):
        """Test response to selection computation."""
        np.random.seed(42)

        # Simulate two populations (before and after selection)
        fitness_before = np.random.rand(100)
        # After selection: better individuals (lower fitness for minimization)
        fitness_after = np.random.rand(50) * 0.5

        result = response_to_selection(fitness_before, fitness_after, minimize=True)

        assert 'response' in result
        assert 'mean_before' in result
        assert 'mean_after' in result
        assert 'improvement' in result

        # For minimization, mean_after should be lower
        assert result['mean_after'] < result['mean_before']
        assert result['improvement'] > 0  # Positive improvement

    def test_amount_of_selection(self):
        """Test selection differential computation."""
        np.random.seed(42)

        fitness_values = np.random.rand(100)
        # Select top 30%
        selected_indices = np.argsort(fitness_values)[-30:]

        result = amount_of_selection(fitness_values, selected_indices)

        assert 'selection_differential' in result
        assert 'selection_intensity' in result
        assert 'proportion_selected' in result
        assert result['proportion_selected'] == 0.3

        # Selected individuals should have higher mean fitness
        assert result['mean_selected'] > result['mean_population']

    def test_realized_heritability(self):
        """Test heritability computation."""
        np.random.seed(42)

        # Simulate correlated parent-offspring fitness
        pop_fitness = np.random.rand(100)
        parent_fitness = np.random.rand(30) + 0.5  # Higher fitness
        # Offspring inherit some parent fitness
        offspring_fitness = parent_fitness * 0.7 + np.random.normal(0, 0.1, 30)

        result = realized_heritability(
            parent_fitness, offspring_fitness, pop_fitness, method='ratio'
        )

        assert 'heritability' in result
        assert 0 <= result['heritability'] <= 1
        assert 'response' in result
        assert 'selection_differential' in result

    def test_objective_distribution(self):
        """Test objective distribution analysis."""
        np.random.seed(42)

        fitness = np.random.normal(0.5, 0.2, 100)

        result = compute_objective_distribution(fitness, n_bins=10)

        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'histogram' in result
        assert result['normality_test'] is not None

    def test_analyze_fitness_evolution(self):
        """Test fitness evolution analysis."""
        # Simulate statistics from multiple generations
        all_statistics = {}

        for gen in range(10):
            fitness_stats = np.array([
                [1.0],  # max
                [0.5 - gen * 0.03],  # mean (improving)
                [0.4 - gen * 0.03],  # median
                [0.2 - gen * 0.02],  # min (best, improving)
                [0.1 / (gen + 1)]   # std (decreasing)
            ])

            all_statistics[gen] = {
                'fitness_stats': fitness_stats,
                'n_unique': 100 - gen * 5  # Diversity decreasing
            }

        result = analyze_fitness_evolution(all_statistics)

        assert 'best_fitness' in result
        assert 'mean_fitness' in result
        assert 'improvements' in result
        assert 'diversity_loss' in result
        assert len(result['best_fitness']) == 10

        # Check improvement trend (fitness should improve)
        assert result['best_fitness'][-1] < result['best_fitness'][0]


class TestDependencyAnalysisDiscrete:
    """Test dependency analysis with discrete populations."""

    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        np.random.seed(42)

        # Create population with some correlations
        pop_size = 100
        x1 = np.random.rand(pop_size)
        x2 = x1 + np.random.normal(0, 0.1, pop_size)  # Correlated with x1
        x3 = np.random.rand(pop_size)  # Independent

        population = np.column_stack([x1, x2, x3])

        result = compute_correlation_matrix(population, method='pearson')

        assert 'correlation_matrix' in result
        assert 'pvalue_matrix' in result
        assert 'significant_pairs' in result

        # Check correlation between x1 and x2
        assert result['correlation_matrix'][0, 1] > 0.5

    def test_learn_bayesian_network_mi(self):
        """Test Bayesian network learning with mutual information."""
        np.random.seed(42)

        # Create discrete population with dependencies
        pop_size = 200
        x1 = np.random.randint(0, 2, pop_size)
        x2 = np.where(x1 == 1, np.random.randint(0, 2, pop_size), 0)  # Depends on x1
        x3 = np.random.randint(0, 2, pop_size)  # Independent

        population = np.column_stack([x1, x2, x3])

        result = learn_bayesian_network(population, method='mi')

        assert 'adjacency_matrix' in result
        assert 'edge_list' in result
        assert 'n_edges' in result

        # Should learn at least one edge
        assert result['n_edges'] > 0

    def test_learn_bayesian_network_k2(self):
        """Test Bayesian network learning with K2 algorithm."""
        np.random.seed(42)

        # Create discrete population
        population = np.random.randint(0, 3, (100, 5))

        result = learn_bayesian_network(population, method='k2', max_parents=2)

        assert 'adjacency_matrix' in result
        assert result['adjacency_matrix'].shape == (5, 5)

    def test_learn_gaussian_network(self):
        """Test Gaussian network learning (for continuous data)."""
        np.random.seed(42)

        # Create continuous population with correlations
        mean = [0, 0, 0]
        cov = [[1, 0.7, 0], [0.7, 1, 0.5], [0, 0.5, 1]]
        population = np.random.multivariate_normal(mean, cov, 100)

        result = learn_gaussian_network(population, threshold=0.4)

        assert 'adjacency_matrix' in result
        assert 'edge_list' in result
        assert 'edge_weights' in result

        # Should detect correlations
        assert result['n_edges'] > 0


class TestVisualizationsDiscrete:
    """Test visualization functions (structure testing)."""

    def test_view_dendrogram_structure(self):
        """Test dendrogram visualization creation."""
        # Create mock run_structures
        n_vars = 10
        n_gens = 5

        # Create some edge matrices
        big_matrix = np.random.randint(0, 2, (n_vars * (n_vars - 1), n_gens))

        run_structures = {
            'all_big_matrices': [big_matrix],
            'index_matrix': np.arange(n_vars * n_vars).reshape(n_vars, n_vars)
        }

        fig, results = view_dendrogram_structure(
            run_structures,
            selected_runs=[0],
            selected_generations=list(range(n_gens))
        )

        assert fig is not None
        assert 'linkage_matrix' in results
        assert 'distance_matrix' in results
        assert 'n_structures' in results
        assert results['n_structures'] == n_gens

    def test_view_glyph_structure(self):
        """Test glyph visualization creation."""
        n_vars = 10
        n_gens = 8

        big_matrix = np.random.rand(n_vars * (n_vars - 1), n_gens)

        run_structures = {
            'all_big_matrices': [big_matrix],
            'index_matrix': np.arange(n_vars * n_vars).reshape(n_vars, n_vars)
        }

        fig, results = view_glyph_structure(
            run_structures,
            glyph_type='star',
            layout='grid',
            max_glyphs=8
        )

        assert fig is not None
        assert 'n_glyphs' in results
        assert 'positions' in results
        assert results['n_glyphs'] <= 8


class TestEDAStrategiesDiscrete:
    """Test EDA-specific extraction strategies."""

    def test_extract_probability_distribution_evolution(self):
        """Test probability distribution extraction for univariate EDAs."""
        # Create mock cache with probability models
        class MockModel:
            def __init__(self, probabilities):
                self.probabilities = probabilities

        class MockCache:
            def __init__(self):
                self.models = []

                # Simulate convergence: probabilities become more extreme
                for gen in range(10):
                    n_vars = 5
                    probs = np.zeros((n_vars, 2))

                    for var in range(n_vars):
                        # Start uniform, converge to 0 or 1
                        p = 0.5 + gen * 0.04 * (1 if var % 2 == 0 else -1)
                        p = np.clip(p, 0.01, 0.99)
                        probs[var, :] = [p, 1 - p]

                    self.models.append(MockModel(probs))

        cache = MockCache()

        result = extract_probability_distribution_evolution(cache)

        assert 'probabilities' in result
        assert 'entropy_per_var' in result
        assert 'convergence_speed' in result
        assert 'converged_variables' in result

        # Entropy should decrease over generations
        assert np.all(result['entropy_per_var'][-1, :] <= result['entropy_per_var'][0, :])

    def test_extract_bayesian_network_evolution(self):
        """Test Bayesian network structure extraction."""
        # Create mock cache with network structures
        class MockModel:
            def __init__(self, adjacency_matrix):
                self.adjacency_matrix = adjacency_matrix

        class MockCache:
            def __init__(self):
                self.models = []
                n_vars = 6

                # Create evolving structures
                for gen in range(8):
                    adj = np.zeros((n_vars, n_vars))

                    # Some stable edges
                    adj[0, 1] = 1
                    adj[1, 2] = 1

                    # Emerging edge (appears later)
                    if gen >= 4:
                        adj[3, 4] = 1

                    # Disappearing edge (disappears later)
                    if gen < 4:
                        adj[2, 3] = 1

                    self.models.append(MockModel(adj))

        cache = MockCache()

        result = extract_bayesian_network_evolution(cache)

        assert 'structures' in result
        assert 'edge_frequencies' in result
        assert 'stable_edges' in result
        assert 'emerging_edges' in result
        assert 'disappearing_edges' in result

        # Should detect stable edges (0,1) and (1,2)
        assert len(result['stable_edges']) >= 2

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create mock cache and statistics
        class MockStatistics:
            def __init__(self):
                self.best_fitness = [1.0, 0.8, 0.6, 0.5, 0.4]
                self.mean_fitness = [0.5, 0.45, 0.4, 0.38, 0.36]
                self.std_fitness = [0.2, 0.15, 0.12, 0.1, 0.08]
                self.best_fitness_overall = 0.4
                self.generation_found = 4

        class MockCache:
            def __init__(self):
                self.populations = [np.random.rand(50, 10) for _ in range(5)]
                self.fitness_values = [np.random.rand(50) for _ in range(5)]
                self.models = []

        cache = MockCache()
        statistics = MockStatistics()

        report = generate_comprehensive_report(cache, statistics, eda_type='discrete_univariate')

        assert 'metadata' in report
        assert 'fitness_evolution' in report
        assert 'population_diversity' in report
        assert 'summary' in report

        # Check metadata
        assert report['metadata']['n_generations'] == 5
        assert report['metadata']['pop_size'] == 50
        assert report['metadata']['n_vars'] == 10


class TestIntegrationScenarios:
    """Integration tests with complete scenarios."""

    def test_complete_knowledge_extraction_workflow(self):
        """Test complete workflow from simulated EDA run to knowledge extraction."""
        np.random.seed(42)

        # Simulate a complete EDA run
        n_gens = 15
        pop_size = 100
        n_vars = 8

        # Create mock data
        all_populations = []
        all_fitness = []

        for gen in range(n_gens):
            # Population converges over time
            convergence_factor = 1 - gen / n_gens
            pop = np.random.rand(pop_size, n_vars) * convergence_factor + \
                  0.5 * (1 - convergence_factor)

            # Fitness improves
            fitness = np.sum(pop, axis=1) + np.random.normal(0, 0.1, pop_size)

            all_populations.append(pop)
            all_fitness.append(fitness)

        # Test 1: Correlation analysis on final population
        corr_result = compute_correlation_matrix(all_populations[-1])
        assert corr_result['correlation_matrix'].shape == (n_vars, n_vars)

        # Test 2: Learn network from final population
        # Discretize for Bayesian network
        discrete_pop = (all_populations[-1] > 0.5).astype(int)
        bn_result = learn_bayesian_network(discrete_pop, method='mi')
        assert 'adjacency_matrix' in bn_result

        # Test 3: Analyze fitness evolution
        # Create statistics dict
        all_statistics = {}
        for gen in range(n_gens):
            fitness = all_fitness[gen]
            fitness_stats = np.array([
                np.max(fitness),
                np.mean(fitness),
                np.median(fitness),
                np.min(fitness),
                np.std(fitness)
            ]).reshape(-1, 1)

            all_statistics[gen] = {
                'fitness_stats': fitness_stats,
                'n_unique': len(np.unique(all_populations[gen], axis=0))
            }

        evolution_result = analyze_fitness_evolution(all_statistics)
        assert len(evolution_result['best_fitness']) == n_gens

        # Test 4: Response to selection
        response = response_to_selection(all_fitness[0], all_fitness[-1])
        assert 'response' in response

        print("✓ Complete knowledge extraction workflow test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
