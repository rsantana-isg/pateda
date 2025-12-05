"""
Exhaustive Benchmark Test Suite for Permutation-based EDAs

This module provides comprehensive benchmarks for testing permutation-based
Estimation of Distribution Algorithms (EDAs) on various optimization problems:
- Traveling Salesman Problem (TSP)
- Quadratic Assignment Problem (QAP)
- Linear Ordering Problem (LOP)

The benchmark suite tests different EDA models:
- Mallows Model with Kendall distance
- Edge Histogram Model (EHM)
- Node Histogram Model (NHM)

References:
    [1] J. Ceberio, A. Mendiburu, J.A Lozano: A review of distances for the
        Mallows and Generalized Mallows estimation of distribution algorithms.
        Computational Optimization and Applications, 2015
    [2] P. Larranaga et al.: A review on probabilistic graphical models in
        evolutionary computation. Journal of Heuristics, 2012
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple, Callable
import time

# Import EDA components
from pateda.core.eda import EDA
from pateda.seeding.random_init import random_permutation_init
from pateda.selection.truncation import TruncationSelection
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.learning.mallows import learn_mallows_kendall
from pateda.learning.histogram import learn_ehm, learn_nhm
from pateda.sampling.mallows import sample_mallows_kendall
from pateda.sampling.histogram import sample_ehm, sample_nhm

# Import benchmark problems
from pateda.functions.permutation import (
    # TSP
    TSP,
    create_random_tsp,
    create_tsp_from_coordinates,
    # QAP
    QAP,
    create_random_qap,
    create_uniform_qap,
    create_grid_qap,
    # LOP
    LOP,
    create_random_lop,
    create_tournament_lop,
    create_triangular_lop,
    create_sparse_lop,
)


class BenchmarkResult:
    """Container for benchmark results"""

    def __init__(
        self,
        problem_name: str,
        eda_name: str,
        problem_size: int,
        best_fitness: float,
        mean_fitness: float,
        std_fitness: float,
        best_solution: np.ndarray,
        generations: int,
        time_elapsed: float,
        convergence_curve: List[float],
    ):
        self.problem_name = problem_name
        self.eda_name = eda_name
        self.problem_size = problem_size
        self.best_fitness = best_fitness
        self.mean_fitness = mean_fitness
        self.std_fitness = std_fitness
        self.best_solution = best_solution
        self.generations = generations
        self.time_elapsed = time_elapsed
        self.convergence_curve = convergence_curve

    def __repr__(self):
        return (
            f"BenchmarkResult("
            f"problem={self.problem_name}, "
            f"eda={self.eda_name}, "
            f"n={self.problem_size}, "
            f"best_fitness={self.best_fitness:.2f}, "
            f"time={self.time_elapsed:.2f}s)"
        )


def run_eda_benchmark(
    problem: Callable,
    problem_name: str,
    eda_config: Dict,
    eda_name: str,
    n_vars: int,
    pop_size: int = 100,
    max_gen: int = 100,
    n_runs: int = 10,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Run a single EDA benchmark experiment.

    Args:
        problem: Optimization problem (fitness function)
        problem_name: Name of the problem
        eda_config: EDA configuration dictionary
        eda_name: Name of the EDA variant
        n_vars: Problem size
        pop_size: Population size
        max_gen: Maximum generations
        n_runs: Number of independent runs
        seed: Random seed

    Returns:
        BenchmarkResult object
    """
    np.random.seed(seed)

    best_fitnesses = []
    best_solutions = []
    convergence_curves = []
    times = []

    for run in range(n_runs):
        # Run EDA
        start_time = time.time()

        # Create EDA
        eda = EDA(
            fitness_func=problem,
            n_vars=n_vars,
            cardinality=np.array([n_vars] * n_vars),  # Permutation representation
            seeding_method=random_permutation_init,
            selection_method=TruncationSelection(selection_ratio=0.5),
            learning_method=eda_config["learning"],
            sampling_method=eda_config["sampling"],
            stop_condition=MaxGenerations(max_generations=max_gen),
            pop_size=pop_size,
            representation="permutation",
        )

        # Run optimization
        result = eda.run()

        elapsed_time = time.time() - start_time

        # Extract results
        best_fitnesses.append(result["best_fitness"])
        best_solutions.append(result["best_solution"])
        convergence_curves.append(result.get("convergence_curve", []))
        times.append(elapsed_time)

    # Compute statistics
    best_idx = np.argmax(best_fitnesses)
    best_fitness = best_fitnesses[best_idx]
    best_solution = best_solutions[best_idx]
    mean_fitness = np.mean(best_fitnesses)
    std_fitness = np.std(best_fitnesses)
    mean_time = np.mean(times)
    mean_convergence = np.mean(convergence_curves, axis=0).tolist()

    return BenchmarkResult(
        problem_name=problem_name,
        eda_name=eda_name,
        problem_size=n_vars,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        std_fitness=std_fitness,
        best_solution=best_solution,
        generations=max_gen,
        time_elapsed=mean_time,
        convergence_curve=mean_convergence,
    )


# ============================================================================
# TSP Benchmarks
# ============================================================================


def test_tsp_small_mallows():
    """Test TSP with Mallows model - small instance"""
    tsp = create_random_tsp(n_cities=10, seed=42)

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    result = run_eda_benchmark(
        problem=tsp,
        problem_name="TSP-10",
        eda_config=eda_config,
        eda_name="Mallows-Kendall",
        n_vars=10,
        pop_size=50,
        max_gen=50,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    assert result.best_fitness < 0  # TSP returns negative distances
    print(f"\n{result}")


def test_tsp_medium_ehm():
    """Test TSP with Edge Histogram Model - medium instance"""
    tsp = create_random_tsp(n_cities=20, seed=42)

    eda_config = {
        "learning": learn_ehm,
        "sampling": sample_ehm,
    }

    result = run_eda_benchmark(
        problem=tsp,
        problem_name="TSP-20",
        eda_config=eda_config,
        eda_name="EHM",
        n_vars=20,
        pop_size=100,
        max_gen=100,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


def test_tsp_nhm():
    """Test TSP with Node Histogram Model"""
    tsp = create_random_tsp(n_cities=15, seed=42)

    eda_config = {
        "learning": learn_nhm,
        "sampling": sample_nhm,
    }

    result = run_eda_benchmark(
        problem=tsp,
        problem_name="TSP-15",
        eda_config=eda_config,
        eda_name="NHM",
        n_vars=15,
        pop_size=75,
        max_gen=75,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


# ============================================================================
# QAP Benchmarks
# ============================================================================


def test_qap_small_mallows():
    """Test QAP with Mallows model - small instance"""
    qap = create_random_qap(n=10, seed=42)

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    result = run_eda_benchmark(
        problem=qap,
        problem_name="QAP-Random-10",
        eda_config=eda_config,
        eda_name="Mallows-Kendall",
        n_vars=10,
        pop_size=50,
        max_gen=50,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    assert result.best_fitness < 0  # QAP returns negative costs
    print(f"\n{result}")


def test_qap_uniform_ehm():
    """Test QAP (uniform) with Edge Histogram Model"""
    qap = create_uniform_qap(n=12, seed=42)

    eda_config = {
        "learning": learn_ehm,
        "sampling": sample_ehm,
    }

    result = run_eda_benchmark(
        problem=qap,
        problem_name="QAP-Uniform-12",
        eda_config=eda_config,
        eda_name="EHM",
        n_vars=12,
        pop_size=60,
        max_gen=60,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


def test_qap_grid_nhm():
    """Test QAP (grid layout) with Node Histogram Model"""
    qap = create_grid_qap(grid_size=3, seed=42)  # 3x3 = 9 facilities

    eda_config = {
        "learning": learn_nhm,
        "sampling": sample_nhm,
    }

    result = run_eda_benchmark(
        problem=qap,
        problem_name="QAP-Grid-9",
        eda_config=eda_config,
        eda_name="NHM",
        n_vars=9,
        pop_size=50,
        max_gen=50,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


def test_qap_sparse_mallows():
    """Test QAP (sparse) with Mallows model"""
    qap = create_random_qap(n=15, seed=42, sparse=True)

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    result = run_eda_benchmark(
        problem=qap,
        problem_name="QAP-Sparse-15",
        eda_config=eda_config,
        eda_name="Mallows-Kendall",
        n_vars=15,
        pop_size=75,
        max_gen=75,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


# ============================================================================
# LOP Benchmarks
# ============================================================================


def test_lop_random_mallows():
    """Test LOP (random) with Mallows model"""
    lop = create_random_lop(n=10, seed=42)

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    result = run_eda_benchmark(
        problem=lop,
        problem_name="LOP-Random-10",
        eda_config=eda_config,
        eda_name="Mallows-Kendall",
        n_vars=10,
        pop_size=50,
        max_gen=50,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    assert result.best_fitness > 0  # LOP maximizes by default
    print(f"\n{result}")


def test_lop_tournament_ehm():
    """Test LOP (tournament) with Edge Histogram Model"""
    lop = create_tournament_lop(n=12, seed=42)

    eda_config = {
        "learning": learn_ehm,
        "sampling": sample_ehm,
    }

    result = run_eda_benchmark(
        problem=lop,
        problem_name="LOP-Tournament-12",
        eda_config=eda_config,
        eda_name="EHM",
        n_vars=12,
        pop_size=60,
        max_gen=60,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


def test_lop_triangular_nhm():
    """Test LOP (triangular) with Node Histogram Model"""
    lop = create_triangular_lop(n=15, seed=42)

    eda_config = {
        "learning": learn_nhm,
        "sampling": sample_nhm,
    }

    result = run_eda_benchmark(
        problem=lop,
        problem_name="LOP-Triangular-15",
        eda_config=eda_config,
        eda_name="NHM",
        n_vars=15,
        pop_size=75,
        max_gen=75,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


def test_lop_sparse_mallows():
    """Test LOP (sparse) with Mallows model"""
    lop = create_sparse_lop(n=20, density=0.3, seed=42)

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    result = run_eda_benchmark(
        problem=lop,
        problem_name="LOP-Sparse-20",
        eda_config=eda_config,
        eda_name="Mallows-Kendall",
        n_vars=20,
        pop_size=100,
        max_gen=100,
        n_runs=3,
        seed=42,
    )

    assert result.best_fitness is not None
    print(f"\n{result}")


# ============================================================================
# Comparative Benchmarks
# ============================================================================


def test_comparative_benchmark_tsp():
    """Compare all EDA models on TSP"""
    tsp = create_random_tsp(n_cities=15, seed=42)

    eda_configs = {
        "Mallows-Kendall": {
            "learning": learn_mallows_kendall,
            "sampling": sample_mallows_kendall,
        },
        "EHM": {
            "learning": learn_ehm,
            "sampling": sample_ehm,
        },
        "NHM": {
            "learning": learn_nhm,
            "sampling": sample_nhm,
        },
    }

    results = []
    for eda_name, eda_config in eda_configs.items():
        result = run_eda_benchmark(
            problem=tsp,
            problem_name="TSP-15-Comparison",
            eda_config=eda_config,
            eda_name=eda_name,
            n_vars=15,
            pop_size=75,
            max_gen=75,
            n_runs=5,
            seed=42,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("TSP-15 Comparative Benchmark Results")
    print("=" * 80)
    for r in results:
        print(f"{r.eda_name:20s} | Best: {r.best_fitness:10.2f} | "
              f"Mean: {r.mean_fitness:10.2f} | Std: {r.std_fitness:8.2f} | "
              f"Time: {r.time_elapsed:6.2f}s")
    print("=" * 80)

    # All should produce valid results
    for r in results:
        assert r.best_fitness is not None


def test_comparative_benchmark_qap():
    """Compare all EDA models on QAP"""
    qap = create_random_qap(n=12, seed=42)

    eda_configs = {
        "Mallows-Kendall": {
            "learning": learn_mallows_kendall,
            "sampling": sample_mallows_kendall,
        },
        "EHM": {
            "learning": learn_ehm,
            "sampling": sample_ehm,
        },
        "NHM": {
            "learning": learn_nhm,
            "sampling": sample_nhm,
        },
    }

    results = []
    for eda_name, eda_config in eda_configs.items():
        result = run_eda_benchmark(
            problem=qap,
            problem_name="QAP-12-Comparison",
            eda_config=eda_config,
            eda_name=eda_name,
            n_vars=12,
            pop_size=60,
            max_gen=60,
            n_runs=5,
            seed=42,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("QAP-12 Comparative Benchmark Results")
    print("=" * 80)
    for r in results:
        print(f"{r.eda_name:20s} | Best: {r.best_fitness:10.2f} | "
              f"Mean: {r.mean_fitness:10.2f} | Std: {r.std_fitness:8.2f} | "
              f"Time: {r.time_elapsed:6.2f}s")
    print("=" * 80)

    for r in results:
        assert r.best_fitness is not None


def test_comparative_benchmark_lop():
    """Compare all EDA models on LOP"""
    lop = create_random_lop(n=12, seed=42)

    eda_configs = {
        "Mallows-Kendall": {
            "learning": learn_mallows_kendall,
            "sampling": sample_mallows_kendall,
        },
        "EHM": {
            "learning": learn_ehm,
            "sampling": sample_ehm,
        },
        "NHM": {
            "learning": learn_nhm,
            "sampling": sample_nhm,
        },
    }

    results = []
    for eda_name, eda_config in eda_configs.items():
        result = run_eda_benchmark(
            problem=lop,
            problem_name="LOP-12-Comparison",
            eda_config=eda_config,
            eda_name=eda_name,
            n_vars=12,
            pop_size=60,
            max_gen=60,
            n_runs=5,
            seed=42,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("LOP-12 Comparative Benchmark Results")
    print("=" * 80)
    for r in results:
        print(f"{r.eda_name:20s} | Best: {r.best_fitness:10.2f} | "
              f"Mean: {r.mean_fitness:10.2f} | Std: {r.std_fitness:8.2f} | "
              f"Time: {r.time_elapsed:6.2f}s")
    print("=" * 80)

    for r in results:
        assert r.best_fitness is not None


# ============================================================================
# Scalability Benchmarks
# ============================================================================


@pytest.mark.slow
def test_scalability_tsp():
    """Test scalability of EDAs on TSP with increasing problem sizes"""
    problem_sizes = [10, 15, 20, 25, 30]

    eda_config = {
        "learning": learn_mallows_kendall,
        "sampling": sample_mallows_kendall,
    }

    print("\n" + "=" * 80)
    print("TSP Scalability Benchmark (Mallows-Kendall)")
    print("=" * 80)

    for n in problem_sizes:
        tsp = create_random_tsp(n_cities=n, seed=42)

        result = run_eda_benchmark(
            problem=tsp,
            problem_name=f"TSP-{n}",
            eda_config=eda_config,
            eda_name="Mallows-Kendall",
            n_vars=n,
            pop_size=n * 10,
            max_gen=n * 5,
            n_runs=3,
            seed=42,
        )

        print(f"n={n:3d} | Best: {result.best_fitness:12.2f} | "
              f"Mean: {result.mean_fitness:12.2f} | "
              f"Time: {result.time_elapsed:8.2f}s")

    print("=" * 80)


@pytest.mark.slow
def test_scalability_qap():
    """Test scalability of EDAs on QAP with increasing problem sizes"""
    problem_sizes = [8, 10, 12, 15, 20]

    eda_config = {
        "learning": learn_ehm,
        "sampling": sample_ehm,
    }

    print("\n" + "=" * 80)
    print("QAP Scalability Benchmark (EHM)")
    print("=" * 80)

    for n in problem_sizes:
        qap = create_random_qap(n=n, seed=42)

        result = run_eda_benchmark(
            problem=qap,
            problem_name=f"QAP-{n}",
            eda_config=eda_config,
            eda_name="EHM",
            n_vars=n,
            pop_size=n * 8,
            max_gen=n * 5,
            n_runs=3,
            seed=42,
        )

        print(f"n={n:3d} | Best: {result.best_fitness:12.2f} | "
              f"Mean: {result.mean_fitness:12.2f} | "
              f"Time: {result.time_elapsed:8.2f}s")

    print("=" * 80)


@pytest.mark.slow
def test_scalability_lop():
    """Test scalability of EDAs on LOP with increasing problem sizes"""
    problem_sizes = [10, 15, 20, 25, 30]

    eda_config = {
        "learning": learn_nhm,
        "sampling": sample_nhm,
    }

    print("\n" + "=" * 80)
    print("LOP Scalability Benchmark (NHM)")
    print("=" * 80)

    for n in problem_sizes:
        lop = create_random_lop(n=n, seed=42)

        result = run_eda_benchmark(
            problem=lop,
            problem_name=f"LOP-{n}",
            eda_config=eda_config,
            eda_name="NHM",
            n_vars=n,
            pop_size=n * 8,
            max_gen=n * 5,
            n_runs=3,
            seed=42,
        )

        print(f"n={n:3d} | Best: {result.best_fitness:12.2f} | "
              f"Mean: {result.mean_fitness:12.2f} | "
              f"Time: {result.time_elapsed:8.2f}s")

    print("=" * 80)


if __name__ == "__main__":
    # Run quick tests
    print("Running Permutation EDA Benchmark Suite")
    print("=" * 80)

    # TSP tests
    test_tsp_small_mallows()
    test_tsp_medium_ehm()
    test_tsp_nhm()

    # QAP tests
    test_qap_small_mallows()
    test_qap_uniform_ehm()
    test_qap_grid_nhm()
    test_qap_sparse_mallows()

    # LOP tests
    test_lop_random_mallows()
    test_lop_tournament_ehm()
    test_lop_triangular_nhm()
    test_lop_sparse_mallows()

    # Comparative tests
    test_comparative_benchmark_tsp()
    test_comparative_benchmark_qap()
    test_comparative_benchmark_lop()

    print("\n" + "=" * 80)
    print("Benchmark suite completed successfully!")
    print("=" * 80)
