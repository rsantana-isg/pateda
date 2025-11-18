"""
Tests for additively decomposable benchmark functions
"""

import numpy as np
import pytest
from pateda.functions.discrete.additive_decomposable import (
    # K-Deceptive functions
    k_deceptive,
    gen_k_decep,
    gen_k_decep_overlap,
    # Deceptive-3 variants
    decep3,
    decep_marta3,
    decep_marta3_new,
    decep3_mh,
    two_peaks_decep3,
    decep_venturini,
    # Hard deceptive-5
    hard_decep5,
    # Hierarchical functions
    hiff,
    fhtrap1,
    # Polytree functions
    first_polytree3_ochoa,
    first_polytree5_ochoa,
    # Cuban functions
    fc2,
    fc3,
    fc4,
    fc5,
    # Factory functions
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
)


class TestKDeceptive:
    """Tests for K-Deceptive functions"""

    def test_k_deceptive_k3_all_ones(self):
        """Test K=3 deceptive with all ones (optimal)"""
        x = np.array([1, 1, 1, 1, 1, 1])
        result = k_deceptive(x, k=3)
        assert result == 6.0  # 2 partitions * 3 each

    def test_k_deceptive_k3_all_zeros(self):
        """Test K=3 deceptive with all zeros"""
        x = np.array([0, 0, 0, 0, 0, 0])
        result = k_deceptive(x, k=3)
        assert result == 4.0  # 2 partitions * 2 each

    def test_k_deceptive_k3_mixed(self):
        """Test K=3 deceptive with mixed values"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = k_deceptive(x, k=3)
        assert result == 5.0  # First: 3, Second: 2

    def test_k_deceptive_k5(self):
        """Test K=5 deceptive"""
        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result = k_deceptive(x, k=5)
        assert result == 9.0  # First: 5, Second: 4

    def test_gen_k_decep_binary(self):
        """Test generalized K-deceptive with binary values"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = gen_k_decep(x, k=3, cardinality=2)
        # First partition sum=3, optimal=3*1=3, returns 3
        # Second partition sum=0, returns 3-1=2
        assert result == 5.0

    def test_gen_k_decep_overlap(self):
        """Test generalized K-deceptive with overlap"""
        x = np.array([1, 1, 1, 0, 0])
        result = gen_k_decep_overlap(x, k=3, cardinality=2, overlap=1)
        # Should create overlapping partitions


class TestDeceptive3:
    """Tests for Deceptive-3 variants"""

    def test_decep3_no_overlap(self):
        """Test Decep3 without overlap"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = decep3(x, overlap=False)
        # First partition [1,1,1]: idx=7, table[7]=1.0
        # Second partition [0,0,0]: idx=0, table[0]=0.9
        assert result == pytest.approx(1.9)

    def test_decep3_with_overlap(self):
        """Test Decep3 with overlap"""
        x = np.array([1, 1, 1, 0, 0])
        result = decep3(x, overlap=True)
        # Overlapping partitions with step=2

    def test_decep_marta3(self):
        """Test Marta's Decep3"""
        x = np.array([0, 0, 0, 1, 1, 1])
        result = decep_marta3(x)
        assert isinstance(result, float)

    def test_decep_marta3_new(self):
        """Test new Marta's Decep3"""
        x = np.array([0, 0, 0, 1, 1, 1])
        result = decep_marta3_new(x)
        # [0,0,0]: idx=0, table[0]=1.5
        # [1,1,1]: idx=7, table[7]=1.5
        assert result == 3.0

    def test_decep3_mh(self):
        """Test MH's Decep3"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = decep3_mh(x)
        # [1,1,1]: idx=7, table[7]=3.0
        # [0,0,0]: idx=0, table[0]=2.0
        assert result == 5.0

    def test_two_peaks_decep3_first_bit_0(self):
        """Test Two Peaks Decep3 with first bit = 0"""
        x = np.array([0, 1, 1, 1, 0, 0, 0])
        result = two_peaks_decep3(x)
        # Should use standard deceptive-3
        assert isinstance(result, float)

    def test_two_peaks_decep3_first_bit_1(self):
        """Test Two Peaks Decep3 with first bit = 1"""
        x = np.array([1, 1, 1, 1, 0, 0, 0])
        result = two_peaks_decep3(x)
        # Should use inverted deceptive-3
        assert isinstance(result, float)

    def test_decep_venturini(self):
        """Test Venturini's Deceptive"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = decep_venturini(x)
        assert isinstance(result, float)


class TestHardDecep5:
    """Tests for Hard Deceptive-5"""

    def test_hard_decep5_all_ones(self):
        """Test hard decep5 with all ones"""
        x = np.array([1, 1, 1, 1, 1])
        result = hard_decep5(x)
        # sum=5, table[5]=1.0
        assert result == 1.0

    def test_hard_decep5_all_zeros(self):
        """Test hard decep5 with all zeros"""
        x = np.array([0, 0, 0, 0, 0])
        result = hard_decep5(x)
        # sum=0, table[0]=0.9
        assert result == 0.9

    def test_hard_decep5_mixed(self):
        """Test hard decep5 with mixed values"""
        x = np.array([1, 1, 1, 1, 0])
        result = hard_decep5(x)
        # sum=4, table[4]=0.0
        assert result == 0.0


class TestHierarchical:
    """Tests for hierarchical functions"""

    def test_hiff_size_16(self):
        """Test HIFF with size 16"""
        x = np.ones(16, dtype=int)
        result = hiff(x)
        assert result > 0

    def test_hiff_size_32(self):
        """Test HIFF with size 32"""
        x = np.ones(32, dtype=int)
        result = hiff(x)
        assert result > 0

    def test_hiff_all_zeros(self):
        """Test HIFF with all zeros"""
        x = np.zeros(16, dtype=int)
        result = hiff(x)
        assert result > 0

    def test_hiff_mixed(self):
        """Test HIFF with mixed values"""
        x = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        result = hiff(x)
        assert result > 0

    def test_fhtrap1_size_9(self):
        """Test fhtrap1 with size 9"""
        x = np.ones(9, dtype=int)
        result = fhtrap1(x)
        assert result > 0

    def test_fhtrap1_size_27(self):
        """Test fhtrap1 with size 27"""
        x = np.ones(27, dtype=int)
        result = fhtrap1(x)
        assert result > 0

    def test_fhtrap1_all_zeros(self):
        """Test fhtrap1 with all zeros"""
        x = np.zeros(27, dtype=int)
        result = fhtrap1(x)
        assert result > 0


class TestPolytree:
    """Tests for Polytree functions"""

    def test_first_polytree3_no_overlap(self):
        """Test Polytree-3 without overlap"""
        x = np.array([1, 1, 1, 0, 0, 0])
        result = first_polytree3_ochoa(x, overlap=False)
        assert isinstance(result, float)

    def test_first_polytree3_with_overlap(self):
        """Test Polytree-3 with overlap"""
        x = np.array([1, 1, 1, 0, 0])
        result = first_polytree3_ochoa(x, overlap=True)
        assert isinstance(result, float)

    def test_first_polytree5(self):
        """Test Polytree-5"""
        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result = first_polytree5_ochoa(x)
        assert isinstance(result, float)


class TestCuban:
    """Tests for Cuban functions"""

    def test_fc2(self):
        """Test Fc2 function"""
        x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = fc2(x)
        assert isinstance(result, float)

    def test_fc3(self):
        """Test Fc3 function"""
        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result = fc3(x)
        assert isinstance(result, float)

    def test_fc4(self):
        """Test Fc4 function"""
        x = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0])
        result = fc4(x)
        assert isinstance(result, float)

    def test_fc5(self):
        """Test Fc5 function"""
        x = np.array([1, 0, 0, 0, 0] + [0] * 16)
        result = fc5(x)
        assert isinstance(result, float)


class TestFactoryFunctions:
    """Tests for factory functions that create objectives for EDAs"""

    def test_create_k_deceptive_function(self):
        """Test creating K-deceptive objective"""
        obj_func = create_k_deceptive_function(k=3)

        # Test with single solution
        x = np.array([1, 1, 1])
        fitness = obj_func(x)
        assert fitness.shape == (1,)
        assert fitness[0] == 3.0

        # Test with population
        pop = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0]])
        fitness = obj_func(pop)
        assert fitness.shape == (3,)

    def test_create_hiff_function(self):
        """Test creating HIFF objective"""
        obj_func = create_hiff_function()

        # Test with single solution
        x = np.ones(16, dtype=int)
        fitness = obj_func(x)
        assert fitness.shape == (1,)

        # Test with population
        pop = np.ones((5, 16), dtype=int)
        fitness = obj_func(pop)
        assert fitness.shape == (5,)

    def test_create_decep3_function(self):
        """Test creating Decep3 objective"""
        obj_func = create_decep3_function(overlap=True)

        x = np.array([1, 1, 1, 0, 0])
        fitness = obj_func(x)
        assert fitness.shape == (1,)

    def test_create_polytree3_function(self):
        """Test creating Polytree-3 objective"""
        obj_func = create_polytree3_function(overlap=False)

        x = np.array([1, 1, 1, 0, 0, 0])
        fitness = obj_func(x)
        assert fitness.shape == (1,)


class TestEdgeCases:
    """Tests for edge cases and special inputs"""

    def test_2d_input_flattening(self):
        """Test that 2D inputs are properly flattened"""
        x_2d = np.array([[1, 1, 1]])
        x_1d = np.array([1, 1, 1])

        assert k_deceptive(x_2d, k=3) == k_deceptive(x_1d, k=3)
        assert decep3(x_2d) == decep3(x_1d)

    def test_power_of_two_requirement(self):
        """Test that HIFF requires power of 2 sizes"""
        # HIFF is designed for power-of-2 problem sizes
        # Test with valid power-of-2 size
        x = np.array([1, 1, 1, 1])  # Size 4 = 2^2
        result = hiff(x)
        assert isinstance(result, float)
        assert result > 0

    def test_consistency_across_calls(self):
        """Test that functions are deterministic"""
        x = np.array([1, 0, 1, 0, 1, 0])

        # Call each function twice and verify consistency
        assert k_deceptive(x, k=3) == k_deceptive(x, k=3)
        assert decep3(x) == decep3(x)
        assert hiff(np.ones(16)) == hiff(np.ones(16))


class TestKnownOptima:
    """Tests for known optimal solutions"""

    def test_k_deceptive_optima(self):
        """Test that all-ones is optimal for K-deceptive"""
        n = 30
        all_ones = np.ones(n, dtype=int)
        all_zeros = np.zeros(n, dtype=int)
        random_solution = np.random.randint(0, 2, n)

        f_ones = k_deceptive(all_ones, k=3)
        f_zeros = k_deceptive(all_zeros, k=3)
        f_random = k_deceptive(random_solution, k=3)

        # All ones should be optimal
        assert f_ones >= f_zeros
        assert f_ones >= f_random

    def test_hiff_optima(self):
        """Test that uniform solutions are good for HIFF"""
        n = 16
        all_ones = np.ones(n, dtype=int)
        all_zeros = np.zeros(n, dtype=int)
        alternating = np.array([1, 0] * (n // 2))

        f_ones = hiff(all_ones)
        f_zeros = hiff(all_zeros)
        f_alt = hiff(alternating)

        # Uniform solutions should be better than alternating
        assert f_ones > f_alt
        assert f_zeros > f_alt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
