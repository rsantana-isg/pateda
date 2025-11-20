"""
Tests for GNBG Benchmark

This module tests the GNBG benchmarking functionality for continuous EDAs.
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.gnbg_benchmark import (
    load_gnbg_instance,
    create_gnbg_fitness_wrapper,
    get_eda_configuration,
    run_single_experiment,
    generate_summary_statistics
)

# Path to GNBG instances
INSTANCES_FOLDER = str(Path(__file__).parent.parent / 'pateda' / 'functions' / 'GNBG_Instances.Python-main')


class TestGNBGLoading:
    """Test GNBG instance loading"""

    def test_load_gnbg_instance_f1(self):
        """Test loading GNBG instance f1"""
        gnbg, problem_info = load_gnbg_instance(1, INSTANCES_FOLDER)

        # Check GNBG object
        assert gnbg is not None
        assert hasattr(gnbg, 'fitness')
        assert hasattr(gnbg, 'MaxEvals')
        assert hasattr(gnbg, 'Dimension')

        # Check problem info
        assert problem_info['problem_index'] == 1
        assert problem_info['dimension'] > 0
        assert problem_info['max_evals'] > 0
        assert 'optimum_value' in problem_info
        assert 'acceptance_threshold' in problem_info

    def test_load_multiple_instances(self):
        """Test loading multiple GNBG instances"""
        for i in [1, 5, 10, 15, 20, 24]:
            gnbg, problem_info = load_gnbg_instance(i, INSTANCES_FOLDER)
            assert problem_info['problem_index'] == i
            assert gnbg.Dimension == problem_info['dimension']

    def test_invalid_problem_index(self):
        """Test that invalid problem indices raise errors"""
        with pytest.raises(ValueError):
            load_gnbg_instance(0, INSTANCES_FOLDER)

        with pytest.raises(ValueError):
            load_gnbg_instance(25, INSTANCES_FOLDER)

        with pytest.raises(ValueError):
            load_gnbg_instance(-1, INSTANCES_FOLDER)


class TestFitnessWrapper:
    """Test GNBG fitness wrapper"""

    def test_fitness_wrapper_1d(self):
        """Test fitness wrapper with 1D input"""
        gnbg, problem_info = load_gnbg_instance(1, INSTANCES_FOLDER)
        fitness_func = create_gnbg_fitness_wrapper(gnbg)

        n_vars = problem_info['dimension']
        x = np.random.uniform(-100, 100, n_vars)

        # Evaluate fitness
        fitness = fitness_func(x)

        # Check output
        assert isinstance(fitness, (float, np.floating))
        assert not np.isnan(fitness)
        assert not np.isinf(fitness)

    def test_fitness_wrapper_2d(self):
        """Test fitness wrapper with 2D input (population)"""
        gnbg, problem_info = load_gnbg_instance(1, INSTANCES_FOLDER)
        fitness_func = create_gnbg_fitness_wrapper(gnbg)

        n_vars = problem_info['dimension']
        pop_size = 10
        X = np.random.uniform(-100, 100, (pop_size, n_vars))

        # Evaluate fitness
        fitness = fitness_func(X)

        # Check output
        assert isinstance(fitness, np.ndarray)
        assert fitness.shape == (pop_size,)
        assert not np.any(np.isnan(fitness))
        assert not np.any(np.isinf(fitness))

    def test_fitness_tracking(self):
        """Test that GNBG tracks function evaluations"""
        gnbg, problem_info = load_gnbg_instance(1, INSTANCES_FOLDER)
        fitness_func = create_gnbg_fitness_wrapper(gnbg)

        n_vars = problem_info['dimension']
        x = np.random.uniform(-100, 100, n_vars)

        # Initial state
        initial_fe = gnbg.FE

        # Evaluate fitness
        fitness_func(x)

        # Check tracking
        assert gnbg.FE == initial_fe + 1
        assert len(gnbg.FEhistory) == gnbg.FE
        assert gnbg.BestFoundResult <= fitness_func(x)  # Best should be <= any evaluation


class TestEDAConfigurations:
    """Test EDA configurations"""

    def test_gaussian_umda_config(self):
        """Test Gaussian UMDA configuration"""
        components = get_eda_configuration(
            eda_name='gaussian_umda',
            n_vars=10,
            pop_size=50,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None
        assert components.selection is not None
        assert components.stop_condition is not None

    def test_gaussian_full_config(self):
        """Test Full Gaussian configuration"""
        components = get_eda_configuration(
            eda_name='gaussian_full',
            n_vars=10,
            pop_size=50,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_gaussian_mixture_config(self):
        """Test Gaussian Mixture configuration"""
        components = get_eda_configuration(
            eda_name='gaussian_mixture',
            n_vars=10,
            pop_size=50,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_gmrf_eda_config(self):
        """Test GMRF-EDA configuration"""
        components = get_eda_configuration(
            eda_name='gmrf_eda',
            n_vars=10,
            pop_size=50,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_invalid_eda_name(self):
        """Test that invalid EDA names raise errors"""
        with pytest.raises(ValueError):
            get_eda_configuration(
                eda_name='invalid_eda',
                n_vars=10,
                pop_size=50,
                max_gen=100
            )


class TestSingleExperiment:
    """Test running single experiments"""

    @pytest.mark.slow
    def test_run_gaussian_umda_f1(self):
        """Test running Gaussian UMDA on f1 (slow test)"""
        results = run_single_experiment(
            eda_name='gaussian_umda',
            problem_index=1,
            instances_folder=INSTANCES_FOLDER,
            pop_size=30,  # Small for testing
            selection_ratio=0.5,
            seed=42,
            verbose=False
        )

        # Check required fields
        assert 'eda_name' in results
        assert results['eda_name'] == 'gaussian_umda'
        assert 'problem_index' in results
        assert results['problem_index'] == 1
        assert 'best_fitness' in results
        assert 'error_from_optimum' in results
        assert 'success' in results
        assert 'function_evaluations' in results
        assert 'runtime_seconds' in results

        # Check validity
        assert not np.isnan(results['best_fitness'])
        assert not np.isinf(results['best_fitness'])
        assert results['function_evaluations'] > 0
        assert results['runtime_seconds'] > 0

    @pytest.mark.slow
    def test_run_gaussian_full_f1(self):
        """Test running Full Gaussian on f1 (slow test)"""
        results = run_single_experiment(
            eda_name='gaussian_full',
            problem_index=1,
            instances_folder=INSTANCES_FOLDER,
            pop_size=30,
            selection_ratio=0.5,
            seed=42,
            verbose=False
        )

        assert results['eda_name'] == 'gaussian_full'
        assert not np.isnan(results['best_fitness'])

    @pytest.mark.slow
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        seed = 12345

        results1 = run_single_experiment(
            eda_name='gaussian_umda',
            problem_index=1,
            instances_folder=INSTANCES_FOLDER,
            pop_size=30,
            selection_ratio=0.5,
            seed=seed,
            verbose=False
        )

        results2 = run_single_experiment(
            eda_name='gaussian_umda',
            problem_index=1,
            instances_folder=INSTANCES_FOLDER,
            pop_size=30,
            selection_ratio=0.5,
            seed=seed,
            verbose=False
        )

        # Results should be identical with same seed
        assert results1['best_fitness'] == results2['best_fitness']
        assert results1['function_evaluations'] == results2['function_evaluations']


class TestSummaryStatistics:
    """Test summary statistics generation"""

    def test_generate_summary(self):
        """Test summary statistics generation"""
        import pandas as pd

        # Create mock results
        results = [
            {
                'eda_name': 'gaussian_umda',
                'problem_index': 1,
                'success': True,
                'error_from_optimum': 1e-5,
                'function_evaluations': 1000,
                'runtime_seconds': 2.5
            },
            {
                'eda_name': 'gaussian_umda',
                'problem_index': 1,
                'success': True,
                'error_from_optimum': 2e-5,
                'function_evaluations': 1100,
                'runtime_seconds': 2.7
            },
            {
                'eda_name': 'gaussian_full',
                'problem_index': 1,
                'success': False,
                'error_from_optimum': 1e-3,
                'function_evaluations': 1500,
                'runtime_seconds': 3.2
            }
        ]

        df = pd.DataFrame(results)
        summary = generate_summary_statistics(df)

        # Check summary structure
        assert 'eda_name' in summary.columns
        assert 'problem_index' in summary.columns
        assert 'success_rate' in summary.columns
        assert 'mean_error' in summary.columns
        assert 'mean_fe' in summary.columns

        # Check values
        umda_summary = summary[summary['eda_name'] == 'gaussian_umda']
        assert len(umda_summary) == 1
        assert umda_summary.iloc[0]['success_rate'] == 1.0
        assert umda_summary.iloc[0]['n_runs'] == 2


class TestQuickIntegration:
    """Quick integration tests (fast)"""

    def test_load_and_evaluate(self):
        """Test loading instance and evaluating a few solutions"""
        gnbg, problem_info = load_gnbg_instance(1, INSTANCES_FOLDER)
        fitness_func = create_gnbg_fitness_wrapper(gnbg)

        n_vars = problem_info['dimension']

        # Evaluate a few random solutions
        for _ in range(5):
            x = np.random.uniform(-100, 100, n_vars)
            fitness = fitness_func(x)
            assert not np.isnan(fitness)
            assert not np.isinf(fitness)

    def test_eda_components_creation(self):
        """Test creating all EDA components"""
        eda_names = ['gaussian_umda', 'gaussian_full', 'gaussian_mixture', 'gmrf_eda']

        for eda_name in eda_names:
            components = get_eda_configuration(
                eda_name=eda_name,
                n_vars=5,
                pop_size=20,
                max_gen=10
            )
            assert components is not None


if __name__ == '__main__':
    # Run quick tests only (skip slow tests)
    pytest.main([__file__, '-v', '-m', 'not slow'])
