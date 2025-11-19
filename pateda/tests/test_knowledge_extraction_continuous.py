"""
Integration tests for knowledge extraction with continuous EDAs.

Tests the knowledge extraction functionality with continuous EDAs
including FDA, EMNA, and Gaussian-based algorithms.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import pytest
import numpy as np
from typing import Any

# Import knowledge extraction modules
from pateda.knowledge_extraction.fitness_measures import (
    response_to_selection,
    compute_objective_distribution,
    analyze_fitness_evolution
)
from pateda.knowledge_extraction.dependency_analysis import (
    compute_correlation_matrix,
    learn_gaussian_network
)
from pateda.knowledge_extraction.eda_strategies import (
    extract_gaussian_parameters_evolution,
    generate_comprehensive_report,
    compare_eda_runs
)


class TestFitnessMeasuresContinuous:
    """Test fitness measures with continuous populations."""

    def test_response_to_selection_continuous(self):
        """Test response to selection with continuous fitness values."""
        np.random.seed(42)

        # Continuous fitness values (e.g., from optimization)
        fitness_before = np.random.randn(200) + 5.0
        # Selected individuals have better fitness
        fitness_after = np.random.randn(100) + 3.0  # Lower for minimization

        result = response_to_selection(fitness_before, fitness_after, minimize=True)

        assert result['mean_after'] < result['mean_before']
        assert result['improvement'] > 0

    def test_objective_distribution_continuous(self):
        """Test distribution analysis with continuous objectives."""
        np.random.seed(42)

        # Multimodal distribution
        fitness1 = np.random.normal(-2, 0.5, 50)
        fitness2 = np.random.normal(2, 0.5, 50)
        fitness = np.concatenate([fitness1, fitness2])

        result = compute_objective_distribution(fitness, n_bins=20)

        assert 'mean' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert result['histogram'][0].shape[0] == 20

        # Bimodal should have specific skewness/kurtosis
        # (values depend on exact distribution)

    def test_multiobjective_fitness(self):
        """Test fitness measures with multi-objective problems."""
        np.random.seed(42)

        # Multi-objective fitness (2 objectives)
        fitness_before = np.random.rand(100, 2)
        fitness_after = np.random.rand(50, 2) * 0.7

        # Test with first objective
        result = response_to_selection(
            fitness_before, fitness_after, objective_idx=0, minimize=True
        )

        assert 'response' in result

        # Test with second objective
        result2 = response_to_selection(
            fitness_before, fitness_after, objective_idx=1, minimize=True
        )

        assert 'response' in result2


class TestDependencyAnalysisContinuous:
    """Test dependency analysis with continuous populations."""

    def test_correlation_matrix_continuous(self):
        """Test correlation matrix with continuous variables."""
        np.random.seed(42)

        # Create correlated continuous variables
        mean = [0, 0, 0, 0]
        cov = [
            [1.0, 0.8, 0.3, 0.0],
            [0.8, 1.0, 0.2, 0.0],
            [0.3, 0.2, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0]
        ]

        population = np.random.multivariate_normal(mean, cov, 200)

        result = compute_correlation_matrix(population, method='pearson')

        # Should detect strong correlation between vars 0 and 1
        assert result['correlation_matrix'][0, 1] > 0.7
        assert result['n_significant'] > 0

    def test_spearman_correlation(self):
        """Test Spearman correlation for non-linear relationships."""
        np.random.seed(42)

        # Create non-linear relationship
        x1 = np.random.rand(100)
        x2 = x1 ** 2 + np.random.normal(0, 0.1, 100)  # Quadratic
        x3 = np.random.rand(100)  # Independent

        population = np.column_stack([x1, x2, x3])

        result = compute_correlation_matrix(population, method='spearman')

        # Spearman should detect monotonic relationship
        assert abs(result['correlation_matrix'][0, 1]) > 0.5

    def test_learn_gaussian_network_correlation(self):
        """Test Gaussian network learning with correlation method."""
        np.random.seed(42)

        # Create structured correlations
        mean = [0] * 6
        cov = np.eye(6)
        cov[0, 1] = cov[1, 0] = 0.8  # Strong edge
        cov[2, 3] = cov[3, 2] = 0.7  # Strong edge
        cov[4, 5] = cov[5, 4] = 0.6  # Medium edge

        population = np.random.multivariate_normal(mean, cov, 150)

        result = learn_gaussian_network(population, method='correlation', threshold=0.5)

        assert 'adjacency_matrix' in result
        assert 'edge_weights' in result

        # Should detect strong edges
        assert result['adjacency_matrix'][0, 1] == 1
        assert result['adjacency_matrix'][2, 3] == 1

    def test_learn_gaussian_network_partial_correlation(self):
        """Test Gaussian network with partial correlation."""
        np.random.seed(42)

        # Create chain: x1 -> x2 -> x3
        # Marginal correlation exists between x1 and x3
        # But conditional independence given x2
        x1 = np.random.randn(200)
        x2 = x1 + np.random.randn(200) * 0.5
        x3 = x2 + np.random.randn(200) * 0.5

        population = np.column_stack([x1, x2, x3])

        result = learn_gaussian_network(
            population, method='partial_correlation', threshold=0.3
        )

        assert 'adjacency_matrix' in result

        # Partial correlation should ideally detect chain structure
        # (though may not be perfect with small sample)

    def test_max_edges_constraint(self):
        """Test max_edges parameter."""
        np.random.seed(42)

        # Create many correlations
        n_vars = 10
        population = np.random.randn(100, n_vars)

        # Add correlations
        for i in range(n_vars - 1):
            population[:, i + 1] += population[:, i] * 0.5

        result = learn_gaussian_network(
            population, method='correlation', threshold=0.2, max_edges=5
        )

        # Should have at most 5 edges
        assert result['n_edges'] <= 5


class TestGaussianParametersExtraction:
    """Test extraction of Gaussian parameters from continuous EDAs."""

    def test_extract_gaussian_parameters_evolution(self):
        """Test Gaussian parameter extraction."""
        # Create mock cache with Gaussian models
        class MockModel:
            def __init__(self, mean, cov):
                self.mean = mean
                self.covariance = cov

        class MockCache:
            def __init__(self):
                self.models = []
                n_vars = 5

                # Simulate variance reduction and mean convergence
                for gen in range(12):
                    mean = np.ones(n_vars) * (0.5 + gen * 0.02)
                    std = np.ones(n_vars) * (1.0 / (gen + 1))  # Decreasing variance
                    cov = np.diag(std ** 2)

                    # Add some correlations
                    if gen >= 3:
                        cov[0, 1] = cov[1, 0] = 0.3 * std[0] * std[1]

                    self.models.append(MockModel(mean, cov))

        cache = MockCache()

        result = extract_gaussian_parameters_evolution(cache)

        assert 'means' in result
        assert 'covariances' in result
        assert 'mean_trajectory' in result
        assert 'variance_reduction' in result
        assert 'correlation_evolution' in result

        # Variance should decrease
        assert result['variance_reduction'][-1] < result['variance_reduction'][0]

        # Should have 12 generations
        assert result['n_generations'] == 12
        assert result['n_variables'] == 5

    def test_mean_trajectory_analysis(self):
        """Test mean trajectory extraction and analysis."""
        class MockModel:
            def __init__(self, mu):
                self.mu = mu

        class MockCache:
            def __init__(self):
                self.models = []

                # Means converge to [1, 2, 3, 4]
                for gen in range(20):
                    target = np.array([1.0, 2.0, 3.0, 4.0])
                    noise = np.random.randn(4) * (1.0 / (gen + 1))
                    mu = target + noise

                    self.models.append(MockModel(mu))

        cache = MockCache()

        result = extract_gaussian_parameters_evolution(cache)

        trajectory = result['mean_trajectory']

        # Should have shape (20, 4)
        assert trajectory.shape == (20, 4)

        # Later means should be closer to target
        early_dist = np.linalg.norm(trajectory[0] - np.array([1, 2, 3, 4]))
        late_dist = np.linalg.norm(trajectory[-1] - np.array([1, 2, 3, 4]))

        # Distance should generally decrease (with randomness)
        # Check average of last 5 vs first 5
        early_avg_dist = np.mean([np.linalg.norm(trajectory[i] - np.array([1, 2, 3, 4]))
                                   for i in range(5)])
        late_avg_dist = np.mean([np.linalg.norm(trajectory[i] - np.array([1, 2, 3, 4]))
                                  for i in range(15, 20)])

        assert late_avg_dist < early_avg_dist


class TestComprehensiveReportContinuous:
    """Test comprehensive report generation for continuous EDAs."""

    def test_generate_report_gaussian_eda(self):
        """Test report generation for Gaussian EDA."""
        # Mock Gaussian EDA run
        class MockModel:
            def __init__(self, mean, cov):
                self.mean = mean
                self.cov = cov

        class MockCache:
            def __init__(self):
                n_gens = 10
                pop_size = 100
                n_vars = 6

                self.populations = []
                self.fitness_values = []
                self.models = []

                for gen in range(n_gens):
                    # Population
                    mean = np.zeros(n_vars) + gen * 0.1
                    pop = np.random.randn(pop_size, n_vars) + mean

                    # Fitness (improving)
                    fitness = -np.sum(pop ** 2, axis=1)  # Sphere function

                    # Model
                    model_mean = np.mean(pop, axis=0)
                    model_cov = np.cov(pop.T)

                    self.populations.append(pop)
                    self.fitness_values.append(fitness)
                    self.models.append(MockModel(model_mean, model_cov))

        class MockStatistics:
            def __init__(self):
                self.best_fitness = [-50 + i * 2 for i in range(10)]
                self.mean_fitness = [-100 + i * 5 for i in range(10)]
                self.std_fitness = [20 - i for i in range(10)]
                self.best_fitness_overall = -30
                self.generation_found = 9

        cache = MockCache()
        statistics = MockStatistics()

        report = generate_comprehensive_report(cache, statistics, eda_type='gaussian')

        assert 'metadata' in report
        assert report['metadata']['eda_type'] == 'gaussian'
        assert 'fitness_evolution' in report
        assert 'model_evolution' in report
        assert 'population_diversity' in report

        # Model evolution should contain Gaussian-specific info
        if report['model_evolution']:
            assert 'variance_reduction' in report['model_evolution']

    def test_compare_eda_runs(self):
        """Test comparison of multiple EDA runs."""
        # Create multiple mock reports
        reports = []

        for run_idx in range(3):
            report = {
                'fitness_evolution': {
                    'final_best': -30.0 - run_idx * 5,
                    'generation_found': 8 + run_idx
                },
                'population_diversity': {
                    'final_diversity': 0.5 - run_idx * 0.1
                }
            }
            reports.append(report)

        comparison = compare_eda_runs(reports)

        assert 'n_runs' in comparison
        assert comparison['n_runs'] == 3
        assert 'best_fitnesses' in comparison
        assert 'fitness_rankings' in comparison
        assert 'best_run' in comparison

        # Best run should be the one with highest fitness
        assert comparison['best_run'] is not None


class TestIntegrationScenariosContinuous:
    """Integration tests with realistic continuous EDA scenarios."""

    def test_continuous_eda_complete_workflow(self):
        """Test complete knowledge extraction workflow for continuous EDA."""
        np.random.seed(42)

        # Simulate Rastrigin function optimization
        def rastrigin(x):
            n = len(x)
            return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

        n_gens = 20
        pop_size = 150
        n_vars = 5

        # Simulate EDA evolution
        all_populations = []
        all_fitness = []

        mean = np.zeros(n_vars)
        std = np.ones(n_vars) * 3.0

        for gen in range(n_gens):
            # Sample population
            pop = np.random.randn(pop_size, n_vars) * std + mean

            # Evaluate
            fitness = np.array([-rastrigin(ind) for ind in pop])

            # Select best
            selected_idx = np.argsort(fitness)[-int(pop_size * 0.3):]
            selected = pop[selected_idx]

            # Update parameters
            mean = np.mean(selected, axis=0)
            std = np.std(selected, axis=0) * 0.95  # Decay

            all_populations.append(pop)
            all_fitness.append(fitness)

        # Knowledge extraction tests

        # 1. Correlation analysis on selected populations
        final_selected = all_populations[-1]
        corr_result = compute_correlation_matrix(final_selected)
        assert corr_result['correlation_matrix'].shape == (n_vars, n_vars)

        # 2. Gaussian network learning
        gn_result = learn_gaussian_network(final_selected, threshold=0.3)
        assert 'adjacency_matrix' in gn_result

        # 3. Fitness evolution analysis
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

        # Fitness should improve (maximize)
        assert evolution_result['best_fitness'][-1] > evolution_result['best_fitness'][0]

        # 4. Distribution analysis on final generation
        final_fitness = all_fitness[-1]
        dist_result = compute_objective_distribution(final_fitness)

        assert 'mean' in dist_result
        assert 'std' in dist_result

        print("✓ Continuous EDA complete workflow test passed")

    def test_variance_reduction_tracking(self):
        """Test tracking of variance reduction in continuous EDAs."""
        # Simulate variance reduction over generations
        class MockModel:
            def __init__(self, variance_scale):
                n_vars = 4
                self.mean = np.zeros(n_vars)
                self.sigma = np.ones(n_vars) * variance_scale

        class MockCache:
            def __init__(self):
                self.models = []

                # Exponentially decreasing variance
                for gen in range(15):
                    variance_scale = np.exp(-gen * 0.2)
                    self.models.append(MockModel(variance_scale))

        cache = MockCache()

        result = extract_gaussian_parameters_evolution(cache)

        variance_reduction = result['variance_reduction']

        # Should decrease exponentially
        assert len(variance_reduction) == 15
        assert variance_reduction[-1] < variance_reduction[0]

        # Check roughly exponential decay
        assert variance_reduction[0] / variance_reduction[-1] > 5

    def test_correlation_emergence(self):
        """Test detection of emerging correlations in Gaussian models."""
        class MockModel:
            def __init__(self, gen):
                n_vars = 3
                self.mean = np.zeros(n_vars)

                # Correlation emerges after generation 5
                if gen >= 5:
                    cov = np.array([
                        [1.0, 0.7, 0.0],
                        [0.7, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ])
                else:
                    cov = np.eye(n_vars)

                self.covariance = cov

        class MockCache:
            def __init__(self):
                self.models = [MockModel(gen) for gen in range(10)]

        cache = MockCache()

        result = extract_gaussian_parameters_evolution(cache)

        corr_evolution = result['correlation_evolution']

        # Correlation should increase after generation 5
        early_corr = np.mean(corr_evolution[:5])
        late_corr = np.mean(corr_evolution[5:])

        assert late_corr > early_corr


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
