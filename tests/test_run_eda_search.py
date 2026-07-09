"""
Tests for examples/run_eda_search.py
"""

import os
import math
import tempfile

import numpy as np
import pytest

# Allow running with or without the package installed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from run_eda_search import (
    get_function_subfunction_vars,
    get_interaction_matrix,
    save_structure_file,
    save_samples_file,
    parse_problem,
    build_eda,
    run_eda_search,
)


# ---------------------------------------------------------------------------
# get_function_subfunction_vars
# ---------------------------------------------------------------------------

class TestGetFunctionSubfunctionVars:

    def test_onemax_no_interactions(self):
        subfs = get_function_subfunction_vars("OneMax", 6)
        assert subfs == []

    def test_kdeceptive3(self):
        subfs = get_function_subfunction_vars("KDeceptive3", 6)
        assert subfs == [[0, 1, 2], [3, 4, 5]]

    def test_kdeceptive5(self):
        subfs = get_function_subfunction_vars("KDeceptive5", 10)
        assert subfs == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_deceptive3_non_overlap(self):
        subfs = get_function_subfunction_vars("Deceptive3", 6)
        assert subfs == [[0, 1, 2], [3, 4, 5]]

    def test_deceptive3_overlap_step2(self):
        subfs = get_function_subfunction_vars("Deceptive3Overlap", 6)
        # decep3 uses range(0, n-2, step=2) → starts at 0, 2 (not 4, since 4 >= n-2=4)
        assert subfs == [[0, 1, 2], [2, 3, 4]]

    def test_hiff_level_structure(self):
        subfs = get_function_subfunction_vars("HIFF", 8)
        # Level 1 pairs: [0,1],[2,3],[4,5],[6,7]
        # Level 2 quads: [0,1,2,3],[4,5,6,7]
        # Level 3 octet: [0,1,2,3,4,5,6,7]
        assert [0, 1] in subfs
        assert [2, 3] in subfs
        assert [0, 1, 2, 3] in subfs
        assert [0, 1, 2, 3, 4, 5, 6, 7] in subfs

    def test_hard_decep5(self):
        subfs = get_function_subfunction_vars("HardDecep5", 10)
        assert subfs == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_function_subfunction_vars("UnknownFunction", 10)


# ---------------------------------------------------------------------------
# get_interaction_matrix
# ---------------------------------------------------------------------------

class TestGetInteractionMatrix:

    def test_onemax_empty_matrix(self):
        R = get_interaction_matrix("OneMax", 4)
        assert R.shape == (4, 4)
        assert np.all(R == 0)

    def test_kdeceptive3_block_structure(self):
        R = get_interaction_matrix("KDeceptive3", 6)
        assert R.shape == (6, 6)
        # Within first block: 0-1, 0-2, 1-2 should interact
        assert R[0, 1] == 1 and R[1, 0] == 1
        assert R[0, 2] == 1 and R[2, 0] == 1
        assert R[1, 2] == 1 and R[2, 1] == 1
        # Across blocks: no interaction
        assert R[0, 3] == 0 and R[0, 5] == 0

    def test_matrix_symmetric(self):
        for fname in ["KDeceptive3", "Deceptive3", "HIFF"]:
            n = 6 if fname != "HIFF" else 8
            R = get_interaction_matrix(fname, n)
            assert np.array_equal(R, R.T), f"{fname} matrix not symmetric"

    def test_diagonal_zero(self):
        R = get_interaction_matrix("KDeceptive3", 6)
        assert np.all(np.diag(R) == 0)


# ---------------------------------------------------------------------------
# save_structure_file
# ---------------------------------------------------------------------------

class TestSaveStructureFile:

    def test_empty_structure(self):
        R = np.zeros((4, 4), dtype=int)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".dat", delete=False) as f:
            path = f.name
        try:
            save_structure_file(R, path)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            assert lines[0] == "0"
            assert len(lines) == 1
        finally:
            os.unlink(path)

    def test_kdeceptive3_file_format(self):
        R = get_interaction_matrix("KDeceptive3", 6)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".dat", delete=False) as f:
            path = f.name
        try:
            save_structure_file(R, path)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            # 6 variables, 2 blocks of 3 → 2 * C(3,2) = 6 edges
            assert lines[0] == "6"
            assert len(lines) == 7  # 1 header + 6 edge lines
            # All edges must satisfy i < j
            for line in lines[1:]:
                i, j = map(int, line.split())
                assert i < j
        finally:
            os.unlink(path)

    def test_edges_upper_triangle_only(self):
        R = get_interaction_matrix("Deceptive3", 9)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".dat", delete=False) as f:
            path = f.name
        try:
            save_structure_file(R, path)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            n_edges = int(lines[0])
            for line in lines[1:]:
                i, j = map(int, line.split())
                assert i < j, f"Edge ({i},{j}) violates i < j constraint"
            assert len(lines) - 1 == n_edges
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# save_samples_file
# ---------------------------------------------------------------------------

class TestSaveSamplesFile:

    def test_sorted_descending(self):
        solutions = [np.array([1, 0, 1]), np.array([0, 0, 0]), np.array([1, 1, 1])]
        fitnesses = [2.0, 0.0, 3.0]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".dat", delete=False) as f:
            path = f.name
        try:
            save_samples_file(solutions, fitnesses, path)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            read_fits = [float(line.split()[0]) for line in lines]
            assert read_fits == sorted(read_fits, reverse=True)
            assert read_fits[0] == 3.0
        finally:
            os.unlink(path)

    def test_solution_values_preserved(self):
        sol = np.array([1, 0, 1, 1])
        with tempfile.NamedTemporaryFile(mode="r", suffix=".dat", delete=False) as f:
            path = f.name
        try:
            save_samples_file([sol], [5.0], path)
            with open(path) as fh:
                line = fh.readline().strip()
            parts = list(map(float, line.split()))
            assert parts[0] == 5.0
            assert list(map(int, parts[1:])) == [1, 0, 1, 1]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# parse_problem
# ---------------------------------------------------------------------------

class TestParseProblem:

    def test_onemax_returns_callable(self):
        func, n_vars, opt = parse_problem("OneMax", 5)
        assert callable(func)
        assert n_vars == 5
        assert opt == 5.0
        assert func(np.ones(5)) == 5.0
        assert func(np.zeros(5)) == 0.0

    def test_kdeceptive3_optimal(self):
        func, n_vars, opt = parse_problem("KDeceptive3", 6)
        assert opt == 6.0
        # All ones is the global optimum
        assert func(np.ones(6, dtype=int)) == 6.0

    def test_hiff_optimal(self):
        _, n_vars, opt = parse_problem("HIFF", 8)
        assert n_vars == 8
        expected_opt = 8.0 * (1 + math.log2(8))
        assert abs(opt - expected_opt) < 1e-9

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            parse_problem("NotAFunction", 10)

    def test_kdeceptive_bad_n_raises(self):
        with pytest.raises(ValueError):
            parse_problem("KDeceptive3", 7)  # 7 not divisible by 3


# ---------------------------------------------------------------------------
# build_eda
# ---------------------------------------------------------------------------

class TestBuildEda:

    def test_umda_runs(self):
        func, n_vars, _ = parse_problem("OneMax", 10)
        eda = build_eda("UMDA", func, n_vars, pop_size=50, max_generations=5,
                        random_seed=0)
        stats, _ = eda.run(verbose=False)
        assert stats.best_fitness_overall is not None

    def test_unknown_eda_raises(self):
        func, n_vars, _ = parse_problem("OneMax", 10)
        with pytest.raises(ValueError):
            build_eda("NotAnEDA", func, n_vars, pop_size=50, max_generations=5)


# ---------------------------------------------------------------------------
# run_eda_search (integration)
# ---------------------------------------------------------------------------

class TestRunEdaSearch:

    def test_creates_both_files(self, tmp_path):
        solutions, fitnesses = run_eda_search(
            obj_func="KDeceptive3",
            n_vars=6,
            eda_name="UMDA",
            pop_size=50,
            n_gen=5,
            n_reps=2,
            samp=3,
            base_seed=0,
            verbose=False,
            output_dir=str(tmp_path),
        )
        struct_file = tmp_path / "KDeceptive3_6_structure.dat"
        samp_file = tmp_path / "KDeceptive3_6_UMDA_samples.dat"
        assert struct_file.exists(), "Structure file not created"
        assert samp_file.exists(), "Samples file not created"

    def test_structure_file_content(self, tmp_path):
        run_eda_search(
            obj_func="KDeceptive3",
            n_vars=6,
            eda_name="UMDA",
            pop_size=50,
            n_gen=5,
            n_reps=2,
            samp=3,
            base_seed=0,
            verbose=False,
            output_dir=str(tmp_path),
        )
        path = tmp_path / "KDeceptive3_6_structure.dat"
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        n_edges = int(lines[0])
        assert n_edges == 6  # 2 blocks × C(3,2)
        assert len(lines) == 7
        for line in lines[1:]:
            i, j = map(int, line.split())
            assert i < j

    def test_samples_file_content(self, tmp_path):
        run_eda_search(
            obj_func="OneMax",
            n_vars=8,
            eda_name="UMDA",
            pop_size=50,
            n_gen=10,
            n_reps=3,
            samp=2,
            base_seed=1,
            verbose=False,
            output_dir=str(tmp_path),
        )
        path = tmp_path / "OneMax_8_UMDA_samples.dat"
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        # At most SAMP=2 lines
        assert 1 <= len(lines) <= 2
        for line in lines:
            parts = line.split()
            # First token = fitness, rest = solution variables
            float(parts[0])  # must be a number
            assert len(parts) == 1 + 8  # fitness + 8 variables

    def test_samp_limits_returned_solutions(self, tmp_path):
        solutions, fitnesses = run_eda_search(
            obj_func="OneMax",
            n_vars=10,
            eda_name="UMDA",
            pop_size=80,
            n_gen=10,
            n_reps=5,
            samp=3,
            base_seed=0,
            verbose=False,
            output_dir=str(tmp_path),
        )
        assert len(solutions) <= 3
        assert len(fitnesses) <= 3

    def test_fitnesses_descending_in_file(self, tmp_path):
        run_eda_search(
            obj_func="OneMax",
            n_vars=8,
            eda_name="UMDA",
            pop_size=60,
            n_gen=10,
            n_reps=4,
            samp=4,
            base_seed=7,
            verbose=False,
            output_dir=str(tmp_path),
        )
        path = tmp_path / "OneMax_8_UMDA_samples.dat"
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        fits = [float(line.split()[0]) for line in lines]
        assert fits == sorted(fits, reverse=True)
