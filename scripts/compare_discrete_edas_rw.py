"""
Compare all discrete EDAs on packaged real-world benchmark instances.

Problems and default instances follow ``packages/pateda/examples/discrete_EDA_RW.py``:
  1. SAT   -> uf20-01
  2. Ising -> SG_16_1
  3. UBQP  -> bqp50

Usage:
    python scripts/compare_discrete_edas_rw.py
"""

import time
import traceback
import numpy as np

from pateda import (
    UMDA, BMDA, TreeEDA, TreeEDAR, MIMIC, PBIL,
    EBNA, BOA, AffEDA, MKEDA, MTED,
    MNFDA, MNFDAR, MNFDAG, MNFDAGR, MOA,
    FDA, BSC,
)
from pateda.functions.discrete.ising import (
    eval_ising,
    load_ising_benchmark_instance,
    build_ising_interaction_matrix,
)
from pateda.functions.discrete.sat import (
    evaluate_sat,
    load_sat_benchmark_instance,
    build_sat_interaction_matrix,
)
from pateda.functions.discrete.ubqp import (
    evaluate_ubqp,
    load_ubqp_benchmark_instance,
    build_ubqp_interaction_matrix,
)


ALGORITHMS = [
    ("UMDA", UMDA),
    ("BMDA", BMDA),
    ("TreeEDA", TreeEDA),
    ("TreeEDA-r", TreeEDAR),
    ("MIMIC", MIMIC),
    ("PBIL", PBIL),
    ("EBNA", EBNA),
    ("BOA", BOA),
    ("AffEDA", AffEDA),
    ("MKEDA", MKEDA),
    ("MTED", MTED),
    ("MNFDA", MNFDA),
    ("MNFDAR", MNFDAR),
 #   ("MNFDAG", MNFDAG),
 #   ("MNFDAGR", MNFDAGR),
 #   ("MOA", MOA),
    ("FDA", FDA),
    ("BSC", BSC),
]

PROBLEMS = [
    ("SAT", "uf20-01"),
    ("Ising", "SG_16_1"),
    ("UBQP", "bqp50"),
]

RESTRICTED_ALGORITHMS = {"TreeEDA-r", "MNFDAR", "MNFDAGR"}
POP_SIZE = 200
N_GEN = 50
SEL_RATIO = 0.5
SEED = 42
UBQP_THRESHOLD_RATIO = 0.5


def _single_objective(values):
    """Convert scalar-like outputs to scalar/1-D single-objective outputs."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type, instance_name):
    """Load one real-world benchmark and its prior interaction structure."""
    problem_key = problem_type.upper()

    if problem_key == "SAT":
        sat_instance, optimal = load_sat_benchmark_instance(instance_name)
        interaction_matrix = build_sat_interaction_matrix(sat_instance)

        def fitness_func(solution):
            return _single_objective(evaluate_sat(np.asarray(solution), sat_instance))

        return fitness_func, sat_instance.n_vars, 2, interaction_matrix, optimal

    if problem_key == "ISING":
        n_vars, lattice, inter, optimal = load_ising_benchmark_instance(instance_name)
        interaction_matrix = build_ising_interaction_matrix(lattice)

        def fitness_func(solution):
            solution_array = np.asarray(solution)
            if solution_array.ndim == 1:
                return -eval_ising(solution_array, lattice, inter)
            return np.array([-eval_ising(sol, lattice, inter) for sol in solution_array])

        return fitness_func, n_vars, 2, interaction_matrix, optimal

    if problem_key == "UBQP":
        ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
        interaction_matrix = build_ubqp_interaction_matrix(
            ubqp_instance,
            threshold_ratio=UBQP_THRESHOLD_RATIO,
        )

        def fitness_func(solution):
            return _single_objective(evaluate_ubqp(np.asarray(solution), ubqp_instance))

        return fitness_func, ubqp_instance.n_vars, 2, interaction_matrix, optimal

    raise ValueError(f"Unsupported problem type: {problem_type}")


def run_one(alg_name, alg_cls, n_vars, cardinality, fitness_func, interaction_matrix):
    """Run one algorithm on one benchmark."""
    kwargs = {}
    if alg_name in RESTRICTED_ALGORITHMS:
        kwargs["interaction_matrix"] = interaction_matrix

    alg = alg_cls(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=SEED,
        **kwargs,
    )

    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    mean_last = stats.mean_fitness[-1] if stats.mean_fitness else float("nan")
    return stats.best_fitness_overall, mean_last, elapsed


def main():
    """Run the comparison table for all configured real-world benchmarks."""
    header_width = 14
    col_w = 14

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_vars, cardinality, interaction_matrix, optimal = load_problem(
            problem_type,
            instance_name,
        )
        n_edges = int(np.sum(np.triu(interaction_matrix, k=1)))

        print(f"\n{'=' * 84}")
        print(
            f"Problem: {problem_type}  Instance: {instance_name}  "
            f"(n_vars={n_vars}, cardinality={cardinality}, prior_edges={n_edges}, optimal={optimal})"
        )
        print(f"{'=' * 84}")
        print(
            f"{'Algorithm':<{header_width}} {'Best':>{col_w}} "
            f"{'MeanLast':>{col_w}} {'Time(s)':>{col_w}}"
        )
        print("-" * (header_width + 3 * col_w + 4))

        for alg_name, alg_cls in ALGORITHMS:
            try:
                best, mean_last, elapsed = run_one(
                    alg_name,
                    alg_cls,
                    n_vars,
                    cardinality,
                    fitness_func,
                    interaction_matrix,
                )
                print(
                    f"{alg_name:<{header_width}} {best:>{col_w}.4f} "
                    f"{mean_last:>{col_w}.4f} {elapsed:>{col_w}.2f}"
                )
            except Exception as exc:
                print(
                    f"{alg_name:<{header_width}} {'ERROR':>{col_w}} "
                    f"{str(exc)[:30]:>{col_w}} {'':>{col_w}}"
                )
                traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
