"""
Compare all discrete EDAs on packaged real-world benchmark instances.

Each algorithm is executed ``N_RUNS`` times on every instance (the seeds
``[1, 2, ..., N_RUNS]``).  For every (algorithm, instance) pair we print
the vector of best fitnesses across runs together with the mean fitness
and mean wall-clock time, formatted as::

    UMDA     UBQP bqp100: [3874.0, 3758.0, 3829.0]  mean=3820.33  time=12.34s

The benchmark uses harder, larger n=100 instances of all three problems:

  1. SAT   -> uf100-01     (100 binary vars, 430 clauses)
  2. Ising -> SG_100_1     (100-spin spin glass)
  3. UBQP  -> bqp100       (100-var unconstrained binary QP)

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
    #("MNFDAG", MNFDAG),
    #("MNFDAGR", MNFDAGR),
    #("MOA", MOA),
    ("FDA", FDA),
    ("BSC", BSC),
]

# Harder, n=100 benchmark instances of each problem type.
PROBLEMS = [
    ("UBQP", "bqp100"),
    ("SAT", "uf100-01"),
    ("Ising", "SG_100_1"),
  
]

RESTRICTED_ALGORITHMS = {"TreeEDA-r", "MNFDAR", "MNFDAGR"}
POP_SIZE = 200
N_GEN = 50
SEL_RATIO = 0.5
N_RUNS = 3                 # repetitions per (algorithm, instance) pair
SEEDS = list(range(1, N_RUNS + 1))   # seeds [1, 2, 3]
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


def run_single_seed(alg_name, alg_cls, n_vars, cardinality, fitness_func,
                    interaction_matrix, seed):
    """Run one algorithm on one benchmark for one seed."""
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
        random_seed=seed,
        **kwargs,
    )

    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    return float(stats.best_fitness_overall), elapsed


def run_all_seeds(alg_name, alg_cls, n_vars, cardinality, fitness_func,
                  interaction_matrix):
    """Repeat one algorithm ``N_RUNS`` times; return per-seed bests and times."""
    bests = []
    times = []
    for seed in SEEDS:
        best, elapsed = run_single_seed(
            alg_name, alg_cls, n_vars, cardinality,
            fitness_func, interaction_matrix, seed,
        )
        bests.append(best)
        times.append(elapsed)
    return bests, times


def main():
    """Run the comparison table for all configured real-world benchmarks."""
    name_w = 10

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_vars, cardinality, interaction_matrix, optimal = load_problem(
            problem_type,
            instance_name,
        )
        n_edges = int(np.sum(np.triu(interaction_matrix, k=1)))

        print(f"\n{'=' * 84}")
        print(
            f"Problem: {problem_type}  Instance: {instance_name}  "
            f"(n_vars={n_vars}, cardinality={cardinality}, "
            f"prior_edges={n_edges}, optimal={optimal})"
        )
        print(
            f"pop_size={POP_SIZE}  n_gen={N_GEN}  selection_ratio={SEL_RATIO}  "
            f"n_runs={N_RUNS}  seeds={SEEDS}"
        )
        print(f"{'=' * 84}")

        problem_tag = f"{problem_type} {instance_name}"

        for alg_name, alg_cls in ALGORITHMS:
            try:
                bests, times = run_all_seeds(
                    alg_name, alg_cls, n_vars, cardinality,
                    fitness_func, interaction_matrix,
                )
                mean_best = float(np.mean(bests))
                mean_time = float(np.mean(times))
                # Format the per-seed bests as a vector with up to 4 decimals.
                bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                print(
                    f"{alg_name:<{name_w}} {problem_tag:<16}: "
                    f"{bests_str}  mean={mean_best:.4f}  time={mean_time:.2f}s"
                )
            except Exception as exc:
                print(
                    f"{alg_name:<{name_w}} {problem_tag:<16}: ERROR -- {exc}"
                )
                traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
