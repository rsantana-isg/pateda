"""
Quick tests for the fitness-based distribution utilities.

The ``benchmark_dendiff_distributions`` module lives in the top-level
``benchmarks/`` directory of the pateda repository (it is a benchmarking
script, not part of the installed ``pateda_nn`` package).  These tests locate
it relative to this file and skip cleanly when it cannot be found, so the
package test suite stays green even outside the source checkout.
"""
import os
import sys

import numpy as np
import pytest

# Locate the repository-level benchmarks directory:
#   <repo>/packages/pateda_nn/tests/this_file
#   <repo>/benchmarks/benchmark_dendiff_distributions.py
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_BENCH_DIR = os.path.join(_REPO_ROOT, "benchmarks")

if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

bdd = pytest.importorskip(
    "benchmark_dendiff_distributions",
    reason="benchmark_dendiff_distributions not available (repo benchmarks/ dir)",
)


def test_objective_functions_basic():
    x = np.random.randn(10, 5)
    f_sphere = bdd.sphere_function(x)
    assert f_sphere.shape == (10,)
    assert np.all(f_sphere >= 0)

    f_rastrigin = bdd.rastrigin_function(x)
    assert f_rastrigin.shape == (10,)


def test_objective_functions_registry():
    expected = [
        "sphere", "ellipsoid", "rastrigin", "rosenbrock",
        "ackley", "griewank", "schwefel",
    ]
    for name in expected:
        assert name in bdd.OBJECTIVE_FUNCTIONS
        entry = bdd.OBJECTIVE_FUNCTIONS[name]
        for key in ("function", "bounds", "optimum"):
            assert key in entry, f"Missing '{key}' for {name}"


def test_empirical_fitness_distribution():
    MAT, MAT_fitness, metadata = bdd.generate_empirical_fitness_distribution(
        "sphere", n_initial=100, n_selected=50, n_vars=5, seed=42
    )
    assert MAT.shape == (50, 5)
    assert MAT_fitness.shape == (50,)
    # Fitness sorted best-first (ascending for a minimisation objective).
    assert np.all(MAT_fitness[:-1] <= MAT_fitness[1:])
    assert "best_fitness" in metadata


def test_different_objectives_differ():
    results = {}
    for obj_name in ["sphere", "rastrigin", "rosenbrock"]:
        _, _, metadata = bdd.generate_empirical_fitness_distribution(
            obj_name, n_initial=100, n_selected=50, n_vars=5, seed=42
        )
        results[obj_name] = metadata["best_fitness"]
    assert len(set(results.values())) > 1
