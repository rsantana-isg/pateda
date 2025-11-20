"""
Tests for Binary Functions Benchmark

This module tests the binary functions benchmarking functionality for discrete EDAs.
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.binary_functions_benchmark import (
    BINARY_FUNCTIONS,
    create_eda_friendly_function,
    get_eda_configuration,
    run_single_experiment,
    generate_summary_statistics
)

# Import binary functions directly for testing
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, hiff, fhtrap1,
    first_polytree3_ochoa, fc2
)


class TestBinaryFunctions:
    """Test binary function registry and wrappers"""

    def test_function_registry_structure(self):
        """Test that function registry has correct structure"""
        assert len(BINARY_FUNCTIONS) > 0

        for name, info in BINARY_FUNCTIONS.items():
            # Check required fields
            assert 'function' in info
            assert 'sizes' in info
            assert 'optimal' in info
            assert 'category' in info

            # Check types
            assert callable(info['function'])
            assert isinstance(info['sizes'], list)
            assert len(info['sizes']) > 0
            assert callable(info['optimal'])
            assert isinstance(info['category'], str)

    def test_function_categories(self):
        """Test that functions are correctly categorized"""
        expected_categories = {
            'k-deceptive', 'deceptive-3', 'hard-deceptive',
            'hierarchical', 'polytree', 'cuban'
        }

        actual_categories = {info['category'] for info in BINARY_FUNCTIONS.values()}
        assert actual_categories.issubset(expected_categories)

    def test_k_deceptive_k3(self):
        """Test k-deceptive-3 function evaluation"""
        func_info = BINARY_FUNCTIONS['k_deceptive_k3']
        func = func_info['function']

        # Test optimal solution (all 1s)
        n = 30
        x_optimal = np.ones(n, dtype=int)
        fitness_optimal = func(x_optimal)

        # Should equal n for all 1s
        assert fitness_optimal == n

        # Test suboptimal solution
        x_suboptimal = np.zeros(n, dtype=int)
        fitness_suboptimal = func(x_suboptimal)

        # Should be less than optimal
        assert fitness_suboptimal < fitness_optimal

    def test_decep3_no_overlap(self):
        """Test decep3 (no overlap) function evaluation"""
        func_info = BINARY_FUNCTIONS['decep3_no_overlap']
        func = func_info['function']

        n = 30
        x_optimal = np.ones(n, dtype=int)
        fitness = func(x_optimal)

        # Should return a finite value
        assert np.isfinite(fitness)
        assert fitness > 0

    def test_hiff_powers_of_2(self):
        """Test HIFF function on different power-of-2 sizes"""
        for size in [16, 32, 64]:
            func_info = BINARY_FUNCTIONS[f'hiff_{size}']
            func = func_info['function']

            # Test all 1s
            x = np.ones(size, dtype=int)
            fitness_ones = func(x)

            # Test all 0s
            x = np.zeros(size, dtype=int)
            fitness_zeros = func(x)

            # Both should give same optimal fitness
            assert fitness_ones == fitness_zeros
            assert fitness_ones > 0

    def test_fc2_cuban_function(self):
        """Test FC2 Cuban function"""
        func_info = BINARY_FUNCTIONS['fc2']
        func = func_info['function']

        n = 50
        x = np.random.randint(0, 2, n)
        fitness = func(x)

        # Should return a finite value
        assert np.isfinite(fitness)


class TestEDAFriendlyWrapper:
    """Test EDA-compatible function wrapper"""

    def test_wrapper_1d_input(self):
        """Test wrapper with 1D input (single solution)"""
        func = lambda x: k_deceptive(x, k=3)
        wrapped_func = create_eda_friendly_function(func, maximize=True)

        n = 30
        x = np.random.randint(0, 2, n)
        result = wrapped_func(x)

        # Should return 1D array with single value
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_wrapper_2d_input(self):
        """Test wrapper with 2D input (population)"""
        func = lambda x: k_deceptive(x, k=3)
        wrapped_func = create_eda_friendly_function(func, maximize=True)

        n = 30
        pop_size = 10
        X = np.random.randint(0, 2, (pop_size, n))
        result = wrapped_func(X)

        # Should return 1D array with pop_size values
        assert isinstance(result, np.ndarray)
        assert result.shape == (pop_size,)
        assert np.all(np.isfinite(result))

    def test_wrapper_maximize_vs_minimize(self):
        """Test maximize vs minimize modes"""
        func = lambda x: k_deceptive(x, k=3)

        wrapped_max = create_eda_friendly_function(func, maximize=True)
        wrapped_min = create_eda_friendly_function(func, maximize=False)

        n = 30
        x = np.random.randint(0, 2, n)

        result_max = wrapped_max(x)[0]
        result_min = wrapped_min(x)[0]

        # Minimize should negate the result
        assert result_max == -result_min


class TestEDAConfigurations:
    """Test EDA configurations"""

    def test_umda_config(self):
        """Test UMDA configuration"""
        components = get_eda_configuration(
            eda_name='umda',
            n_vars=30,
            pop_size=100,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None
        assert components.selection is not None
        assert components.stop_condition is not None

    def test_tree_eda_config(self):
        """Test Tree-EDA configuration"""
        components = get_eda_configuration(
            eda_name='tree_eda',
            n_vars=30,
            pop_size=100,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_mnfda_config(self):
        """Test MN-FDA configuration"""
        components = get_eda_configuration(
            eda_name='mnfda',
            n_vars=30,
            pop_size=100,
            max_gen=100
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_invalid_eda_name(self):
        """Test that invalid EDA names raise errors"""
        with pytest.raises(ValueError):
            get_eda_configuration(
                eda_name='invalid_eda',
                n_vars=30,
                pop_size=100,
                max_gen=100
            )

    def test_target_fitness_config(self):
        """Test configuration with target fitness"""
        components = get_eda_configuration(
            eda_name='umda',
            n_vars=30,
            pop_size=100,
            max_gen=100,
            target_fitness=30.0
        )

        assert components.stop_condition is not None


class TestSingleExperiment:
    """Test running single experiments"""

    @pytest.mark.slow
    def test_run_umda_k_deceptive(self):
        """Test running UMDA on k-deceptive (slow test)"""
        results = run_single_experiment(
            eda_name='umda',
            function_name='k_deceptive_k3',
            n_vars=30,
            pop_size=50,  # Small for testing
            max_gen=20,   # Few generations for speed
            selection_ratio=0.5,
            seed=42,
            verbose=False
        )

        # Check required fields
        assert 'eda_name' in results
        assert results['eda_name'] == 'umda'
        assert 'function_name' in results
        assert results['function_name'] == 'k_deceptive_k3'
        assert 'best_fitness' in results
        assert 'generation_found' in results
        assert 'runtime_seconds' in results

        # Check validity
        assert not np.isnan(results['best_fitness'])
        assert not np.isinf(results['best_fitness'])
        assert results['best_fitness'] > 0
        assert results['generation_found'] >= 0
        assert results['runtime_seconds'] > 0

    @pytest.mark.slow
    def test_run_tree_eda_decep3(self):
        """Test running Tree-EDA on decep3 (slow test)"""
        results = run_single_experiment(
            eda_name='tree_eda',
            function_name='decep3_no_overlap',
            n_vars=30,
            pop_size=50,
            max_gen=20,
            selection_ratio=0.5,
            seed=42,
            verbose=False
        )

        assert results['eda_name'] == 'tree_eda'
        assert not np.isnan(results['best_fitness'])

    def test_invalid_function_name(self):
        """Test that invalid function names raise errors"""
        with pytest.raises(ValueError):
            run_single_experiment(
                eda_name='umda',
                function_name='invalid_function',
                n_vars=30,
                pop_size=50,
                max_gen=20
            )

    def test_invalid_problem_size(self):
        """Test that invalid problem sizes raise errors"""
        with pytest.raises(ValueError):
            run_single_experiment(
                eda_name='umda',
                function_name='k_deceptive_k3',
                n_vars=25,  # Not divisible by 3
                pop_size=50,
                max_gen=20
            )

    @pytest.mark.slow
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        seed = 12345

        results1 = run_single_experiment(
            eda_name='umda',
            function_name='k_deceptive_k3',
            n_vars=30,
            pop_size=50,
            max_gen=20,
            seed=seed,
            verbose=False
        )

        results2 = run_single_experiment(
            eda_name='umda',
            function_name='k_deceptive_k3',
            n_vars=30,
            pop_size=50,
            max_gen=20,
            seed=seed,
            verbose=False
        )

        # Results should be identical with same seed
        assert results1['best_fitness'] == results2['best_fitness']
        assert results1['generation_found'] == results2['generation_found']


class TestSummaryStatistics:
    """Test summary statistics generation"""

    def test_generate_summary(self):
        """Test summary statistics generation"""
        import pandas as pd

        # Create mock results
        results = [
            {
                'eda_name': 'umda',
                'function_name': 'k_deceptive_k3',
                'category': 'k-deceptive',
                'n_vars': 30,
                'success': True,
                'best_fitness': 29.0,
                'generation_found': 50,
                'runtime_seconds': 2.5
            },
            {
                'eda_name': 'umda',
                'function_name': 'k_deceptive_k3',
                'category': 'k-deceptive',
                'n_vars': 30,
                'success': True,
                'best_fitness': 30.0,
                'generation_found': 60,
                'runtime_seconds': 2.7
            },
            {
                'eda_name': 'tree_eda',
                'function_name': 'k_deceptive_k3',
                'category': 'k-deceptive',
                'n_vars': 30,
                'success': False,
                'best_fitness': 25.0,
                'generation_found': 100,
                'runtime_seconds': 3.2
            }
        ]

        df = pd.DataFrame(results)
        summary = generate_summary_statistics(df)

        # Check summary structure
        assert 'eda_name' in summary.columns
        assert 'function_name' in summary.columns
        assert 'category' in summary.columns
        assert 'success_rate' in summary.columns
        assert 'mean_fitness' in summary.columns
        assert 'mean_generations' in summary.columns

        # Check values
        umda_summary = summary[summary['eda_name'] == 'umda']
        assert len(umda_summary) == 1
        assert umda_summary.iloc[0]['success_rate'] == 1.0
        assert umda_summary.iloc[0]['n_runs'] == 2


class TestQuickIntegration:
    """Quick integration tests (fast)"""

    def test_function_evaluation(self):
        """Test evaluating several functions"""
        test_functions = [
            ('k_deceptive_k3', 30),
            ('decep3_no_overlap', 30),
            ('polytree3_no_overlap', 30),
        ]

        for func_name, n_vars in test_functions:
            func_info = BINARY_FUNCTIONS[func_name]
            func = func_info['function']

            x = np.random.randint(0, 2, n_vars)
            fitness = func(x)

            assert np.isfinite(fitness)

    def test_all_eda_configs(self):
        """Test creating all EDA configurations"""
        eda_names = ['umda', 'tree_eda', 'mnfda']

        for eda_name in eda_names:
            components = get_eda_configuration(
                eda_name=eda_name,
                n_vars=30,
                pop_size=100,
                max_gen=100
            )
            assert components is not None

    def test_function_categories_complete(self):
        """Test that all categories are represented"""
        categories = {info['category'] for info in BINARY_FUNCTIONS.values()}

        assert 'k-deceptive' in categories
        assert 'deceptive-3' in categories
        assert 'hierarchical' in categories
        assert 'polytree' in categories
        assert 'cuban' in categories


if __name__ == '__main__':
    # Run quick tests only (skip slow tests)
    pytest.main([__file__, '-v', '-m', 'not slow'])
