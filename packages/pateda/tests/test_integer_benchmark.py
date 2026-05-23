"""
Tests for Integer Functions Benchmark

This module tests the integer representation benchmark functions
and the EDA configurations for integer problems.
"""
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path

from pateda.functions.discrete.integer_functions import (
    integer_onemax,
    integer_leading_blocks,
    integer_max_blocks,
    integer_categorical_match,
    integer_plateau_search,
    integer_dependency_chain,
    integer_multi_level_trap,
    integer_parity_blocks,
    integer_hierarchical_blocks,
    IntegerNKLandscape,
    create_integer_onemax_function,
    create_integer_max_blocks_function,
    create_integer_multi_level_trap_function,
    create_integer_nk_objective_function,
)

from pateda.functions.discrete.additive_decomposable import (
    gen_k_decep,
    gen_k_decep_overlap,
)

from benchmarks.integer_functions_benchmark import (
    create_integer_function_registry,
    get_integer_eda_configuration,
    create_eda_fitness_function,
    run_integer_experiment,
    generate_integer_summary,
)


class TestIntegerFunctions:
    """Test integer benchmark functions"""

    def test_integer_onemax_optimal(self):
        """Test integer_onemax with optimal solution"""
        cardinality = 4
        n = 30
        x_optimal = np.full(n, cardinality - 1)  # All max values
        result = integer_onemax(x_optimal, cardinality=cardinality)
        expected = n * (cardinality - 1)
        assert result == expected

    def test_integer_onemax_zeros(self):
        """Test integer_onemax with all zeros"""
        cardinality = 4
        n = 30
        x_zeros = np.zeros(n, dtype=int)
        result = integer_onemax(x_zeros, cardinality=cardinality)
        assert result == 0

    def test_integer_onemax_mixed(self):
        """Test integer_onemax with mixed values"""
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # Sum = 12
        result = integer_onemax(x, cardinality=4)
        assert result == 12

    def test_integer_leading_blocks_full(self):
        """Test integer_leading_blocks with all max blocks"""
        cardinality = 4
        n = 9
        x = np.full(n, cardinality - 1)  # All max values
        result = integer_leading_blocks(x, cardinality=cardinality, block_size=3)
        assert result == 3  # 3 complete max blocks

    def test_integer_leading_blocks_partial(self):
        """Test integer_leading_blocks with partial max blocks"""
        cardinality = 4
        max_val = cardinality - 1
        x = np.array([max_val, max_val, max_val, max_val, max_val, max_val, 0, 0, 0])
        result = integer_leading_blocks(x, cardinality=cardinality, block_size=3)
        assert result == 2  # 2 complete max blocks before first non-max

    def test_integer_max_blocks(self):
        """Test integer_max_blocks function"""
        cardinality = 4
        max_val = cardinality - 1
        k = 3

        # All max values - optimal
        x_optimal = np.full(9, max_val)
        result = integer_max_blocks(x_optimal, k=k, cardinality=cardinality)

        # Expected: 3 blocks, each with sum=9 plus bonus=3
        expected = 3 * (k * max_val + k)
        assert result == expected

    def test_integer_multi_level_trap_optimal(self):
        """Test integer_multi_level_trap with optimal solution"""
        cardinality = 4
        max_val = cardinality - 1
        k = 3
        n = 9

        x_optimal = np.full(n, max_val)
        result = integer_multi_level_trap(x_optimal, k=k, cardinality=cardinality)

        # Expected: 3 blocks at optimal
        expected = 3 * (k * max_val + k)
        assert result == expected

    def test_integer_multi_level_trap_deceptive(self):
        """Test that all-zeros is deceptive (high but not optimal)"""
        cardinality = 4
        max_val = cardinality - 1
        k = 3
        n = 9

        x_zeros = np.zeros(n, dtype=int)
        result_zeros = integer_multi_level_trap(x_zeros, k=k, cardinality=cardinality)

        x_optimal = np.full(n, max_val)
        result_optimal = integer_multi_level_trap(x_optimal, k=k, cardinality=cardinality)

        # Zeros should be high but less than optimal
        assert result_zeros > 0
        assert result_zeros < result_optimal

    def test_integer_categorical_match(self):
        """Test integer_categorical_match with alternating pattern"""
        cardinality = 4
        n = 8
        # Default pattern: 0, 1, 2, 3, 0, 1, 2, 3
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        result = integer_categorical_match(x, cardinality=cardinality)
        assert result == n  # All match

    def test_integer_categorical_match_partial(self):
        """Test integer_categorical_match with partial match"""
        cardinality = 4
        # Default pattern: 0, 1, 2, 3
        x = np.array([0, 0, 0, 0])  # Only first matches
        result = integer_categorical_match(x, cardinality=cardinality)
        assert result == 1  # Only position 0 matches

    def test_integer_dependency_chain_optimal(self):
        """Test integer_dependency_chain with perfect chain"""
        cardinality = 4
        n = 8
        # Perfect chain: 0, 1, 2, 3, 0, 1, 2, 3
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        result = integer_dependency_chain(x, cardinality=cardinality)

        # 7 correct dependencies * cardinality + sum(x) = 7*4 + 12 = 40
        expected = 7 * cardinality + np.sum(x)
        assert result == expected

    def test_integer_plateau_search(self):
        """Test integer_plateau_search function"""
        cardinality = 4
        max_val = cardinality - 1
        block_size = 3

        # All max values - optimal with bonus
        x_optimal = np.full(9, max_val)
        result = integer_plateau_search(x_optimal, cardinality=cardinality, block_size=block_size)

        # Expected: 3 blocks, each with sum=9 plus bonus=3
        expected = 3 * (block_size * max_val + max_val)
        assert result == expected

    def test_integer_parity_blocks(self):
        """Test integer_parity_blocks with even sum"""
        cardinality = 4
        max_val = cardinality - 1
        block_size = 4

        # Block with even sum
        x = np.array([1, 1, 0, 0])  # Sum = 2 (even)
        result = integer_parity_blocks(x, block_size=block_size, cardinality=cardinality)

        # Even parity bonus: sum + max_val = 2 + 3 = 5
        expected = 2 + max_val
        assert result == expected

    def test_integer_hierarchical_blocks(self):
        """Test integer_hierarchical_blocks basic functionality"""
        cardinality = 4
        n = 8
        max_val = cardinality - 1

        # All same value - should have good hierarchical score
        x_uniform = np.full(n, max_val)
        result_uniform = integer_hierarchical_blocks(x_uniform, cardinality=cardinality)

        # Mixed values - lower score
        x_mixed = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        result_mixed = integer_hierarchical_blocks(x_mixed, cardinality=cardinality)

        # Uniform should have higher hierarchical fitness
        assert result_uniform > result_mixed


class TestIntegerNKLandscape:
    """Test IntegerNKLandscape class"""

    def test_nk_creation(self):
        """Test NK landscape creation"""
        n_vars = 10
        k = 2
        cardinality = 4

        nk = IntegerNKLandscape(n_vars, k, cardinality, random_seed=42)

        assert nk.n_vars == n_vars
        assert nk.k == k
        assert nk.cardinality == cardinality
        assert len(nk.structure) == n_vars
        assert len(nk.tables) == n_vars

    def test_nk_evaluation(self):
        """Test NK landscape evaluation"""
        n_vars = 10
        k = 2
        cardinality = 4

        nk = IntegerNKLandscape(n_vars, k, cardinality, random_seed=42)

        x = np.random.randint(0, cardinality, n_vars)
        result = nk.evaluate(x)

        assert np.isfinite(result)
        assert result >= 0

    def test_nk_reproducibility(self):
        """Test NK landscape reproducibility with same seed"""
        n_vars = 10
        k = 2
        cardinality = 4

        nk1 = IntegerNKLandscape(n_vars, k, cardinality, random_seed=42)
        nk2 = IntegerNKLandscape(n_vars, k, cardinality, random_seed=42)

        x = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        result1 = nk1.evaluate(x)
        result2 = nk2.evaluate(x)

        assert result1 == result2

    def test_nk_population_evaluation(self):
        """Test NK landscape population evaluation"""
        n_vars = 10
        k = 2
        cardinality = 4
        pop_size = 5

        nk = IntegerNKLandscape(n_vars, k, cardinality, random_seed=42)

        population = np.random.randint(0, cardinality, (pop_size, n_vars))
        fitness = nk.evaluate_population(population)

        assert fitness.shape == (pop_size,)
        assert np.all(np.isfinite(fitness))

    def test_nk_objective_function(self):
        """Test create_integer_nk_objective_function"""
        n_vars = 10
        k = 2
        cardinality = 4

        obj_func = create_integer_nk_objective_function(n_vars, k, cardinality, random_seed=42)

        x = np.random.randint(0, cardinality, n_vars)
        result = obj_func(x)

        assert np.isfinite(result)


class TestGeneralizedKDeceptive:
    """Test generalized k-deceptive for integers"""

    def test_gen_k_decep_optimal(self):
        """Test gen_k_decep with optimal solution"""
        k = 3
        cardinality = 4
        max_val = cardinality - 1
        n = 9

        x_optimal = np.full(n, max_val)  # All at max value
        result = gen_k_decep(x_optimal, k=k, cardinality=cardinality)

        # 3 optimal partitions
        expected = 3 * (max_val * k)
        assert result == expected

    def test_gen_k_decep_zeros(self):
        """Test gen_k_decep with all zeros (second best)"""
        k = 3
        cardinality = 4
        max_val = cardinality - 1
        n = 9

        x_zeros = np.zeros(n, dtype=int)
        result = gen_k_decep(x_zeros, k=k, cardinality=cardinality)

        # All zeros gives second-best per partition
        expected = 3 * (max_val * k - 1)
        assert result == expected

    def test_gen_k_decep_overlap(self):
        """Test gen_k_decep_overlap function"""
        k = 3
        cardinality = 4
        overlap = 1
        n = 9

        x = np.random.randint(0, cardinality, n)
        result = gen_k_decep_overlap(x, k=k, cardinality=cardinality, overlap=overlap)

        assert np.isfinite(result)


class TestIntegerFunctionRegistry:
    """Test function registry and configurations"""

    def test_registry_structure(self):
        """Test that registry has correct structure"""
        registry = create_integer_function_registry(cardinality=4)

        assert len(registry) > 0

        for name, info in registry.items():
            assert 'function' in info
            assert 'sizes' in info
            assert 'optimal' in info
            assert 'category' in info
            assert 'description' in info

            assert callable(info['function'])
            assert isinstance(info['sizes'], list)
            assert len(info['sizes']) > 0
            assert callable(info['optimal'])
            assert isinstance(info['category'], str)

    def test_registry_categories(self):
        """Test that registry has expected categories"""
        registry = create_integer_function_registry(cardinality=4)

        expected_categories = {'simple', 'block', 'deceptive', 'dependency', 'hierarchical'}
        actual_categories = {info['category'] for info in registry.values()}

        assert actual_categories.issubset(expected_categories)

    def test_all_functions_evaluate(self):
        """Test that all registered functions can be evaluated"""
        cardinality = 4
        registry = create_integer_function_registry(cardinality=cardinality)

        for name, info in registry.items():
            func = info['function']
            n_vars = info['sizes'][0]  # Use first valid size

            x = np.random.randint(0, cardinality, n_vars)
            result = func(x)

            assert np.isfinite(result), f"Function {name} returned non-finite value"


class TestEDAConfigurations:
    """Test EDA configurations for integer problems"""

    def test_umda_config(self):
        """Test UMDA configuration"""
        components = get_integer_eda_configuration(
            eda_name='umda',
            n_vars=30,
            pop_size=100,
            max_gen=100,
            cardinality=4
        )

        assert components.learning is not None
        assert components.sampling is not None
        assert components.selection is not None
        assert components.stop_condition is not None

    def test_tree_eda_config(self):
        """Test Tree-EDA configuration"""
        components = get_integer_eda_configuration(
            eda_name='tree_eda',
            n_vars=30,
            pop_size=100,
            max_gen=100,
            cardinality=4
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_mnfda_config(self):
        """Test MN-FDA configuration"""
        components = get_integer_eda_configuration(
            eda_name='mnfda',
            n_vars=30,
            pop_size=100,
            max_gen=100,
            cardinality=4
        )

        assert components.learning is not None
        assert components.sampling is not None

    def test_invalid_eda_name(self):
        """Test that invalid EDA names raise errors"""
        with pytest.raises(ValueError):
            get_integer_eda_configuration(
                eda_name='invalid_eda',
                n_vars=30,
                pop_size=100,
                max_gen=100,
                cardinality=4
            )


class TestEDAFitnessWrapper:
    """Test EDA fitness function wrapper"""

    def test_wrapper_1d_input(self):
        """Test wrapper with 1D input"""
        func = lambda x: integer_onemax(x, cardinality=4)
        wrapped = create_eda_fitness_function(func, maximize=True)

        x = np.array([0, 1, 2, 3])
        result = wrapped(x)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 6  # Sum of 0+1+2+3

    def test_wrapper_2d_input(self):
        """Test wrapper with 2D input (population)"""
        func = lambda x: integer_onemax(x, cardinality=4)
        wrapped = create_eda_fitness_function(func, maximize=True)

        pop = np.array([
            [0, 0, 0, 0],
            [3, 3, 3, 3],
            [0, 1, 2, 3],
        ])
        result = wrapped(pop)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 0   # All zeros
        assert result[1] == 12  # All max
        assert result[2] == 6   # Mixed


class TestSingleExperiment:
    """Test running single experiments"""

    @pytest.mark.slow
    def test_run_umda_integer_onemax(self):
        """Test running UMDA on integer_onemax"""
        results = run_integer_experiment(
            eda_name='umda',
            function_name='integer_onemax',
            n_vars=30,
            cardinality=4,
            pop_size=50,
            max_gen=30,
            seed=42,
            verbose=False
        )

        assert 'eda_name' in results
        assert results['eda_name'] == 'umda'
        assert 'function_name' in results
        assert results['function_name'] == 'integer_onemax'
        assert 'best_fitness' in results
        assert not np.isnan(results['best_fitness'])
        assert results['best_fitness'] > 0

    @pytest.mark.slow
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        seed = 12345

        results1 = run_integer_experiment(
            eda_name='umda',
            function_name='integer_onemax',
            n_vars=30,
            cardinality=4,
            pop_size=50,
            max_gen=30,
            seed=seed,
            verbose=False
        )

        results2 = run_integer_experiment(
            eda_name='umda',
            function_name='integer_onemax',
            n_vars=30,
            cardinality=4,
            pop_size=50,
            max_gen=30,
            seed=seed,
            verbose=False
        )

        assert results1['best_fitness'] == results2['best_fitness']

    def test_invalid_function_name(self):
        """Test that invalid function names raise errors"""
        with pytest.raises(ValueError):
            run_integer_experiment(
                eda_name='umda',
                function_name='invalid_function',
                n_vars=30,
                cardinality=4,
                pop_size=50,
                max_gen=20
            )


class TestSummaryStatistics:
    """Test summary statistics generation"""

    def test_generate_summary(self):
        """Test summary statistics generation"""
        import pandas as pd

        # Create mock results
        results = [
            {
                'eda_name': 'umda',
                'function_name': 'integer_onemax',
                'category': 'simple',
                'n_vars': 30,
                'cardinality': 4,
                'success': True,
                'best_fitness': 85.0,
                'generation_found': 50,
                'runtime_seconds': 2.5
            },
            {
                'eda_name': 'umda',
                'function_name': 'integer_onemax',
                'category': 'simple',
                'n_vars': 30,
                'cardinality': 4,
                'success': True,
                'best_fitness': 90.0,
                'generation_found': 45,
                'runtime_seconds': 2.3
            },
        ]

        df = pd.DataFrame(results)
        summary = generate_integer_summary(df)

        assert 'eda_name' in summary.columns
        assert 'function_name' in summary.columns
        assert 'cardinality' in summary.columns
        assert 'success_rate' in summary.columns
        assert 'mean_fitness' in summary.columns

        umda_summary = summary[summary['eda_name'] == 'umda']
        assert len(umda_summary) == 1
        assert umda_summary.iloc[0]['n_runs'] == 2


class TestQuickIntegration:
    """Quick integration tests"""

    def test_all_eda_configs(self):
        """Test creating all EDA configurations"""
        eda_names = ['umda', 'tree_eda', 'mnfda']

        for eda_name in eda_names:
            components = get_integer_eda_configuration(
                eda_name=eda_name,
                n_vars=30,
                pop_size=100,
                max_gen=100,
                cardinality=4
            )
            assert components is not None

    def test_different_cardinalities(self):
        """Test functions with different cardinalities"""
        for cardinality in [2, 3, 4, 5]:
            registry = create_integer_function_registry(cardinality=cardinality)
            assert len(registry) > 0

            # Test onemax
            func = registry['integer_onemax']['function']
            x = np.random.randint(0, cardinality, 30)
            result = func(x)
            assert np.isfinite(result)


if __name__ == '__main__':
    # Run quick tests only (skip slow tests)
    pytest.main([__file__, '-v', '-m', 'not slow'])
