"""
Single-instance runner for the real-world discrete EDA benchmark.

Built from ``scripts/compare_discrete_edas_RW.py`` but designed for one
``(algorithm, problem, instance, seed)`` combination at a time so the SLURM
launcher in ``slurm/launch_compare_EDAs_RW.sh`` can fan out the experiment
matrix across the cluster.

Positional arguments (no ``--flags``):

    1. n_vars           number of variables (must match the instance size)
    2. pop_size         population size
    3. n_gen            number of generations
    4. selection_ratio  truncation ratio in (0, 1]
    5. random_seed      RNG seed
    6. alg              one of the algorithms in ALGORITHMS below
    7. problem          SAT | Ising | UBQP
    8. instance         full path to the instance file
                        (e.g. ``packages/pateda/src/pateda/functions/SAT_instances/uf20-01.cnf``)

Output (single line on stdout, space-separated):

    <best_fitness>  <mean_fitness>  <time_seconds>

A header line starting with ``#`` is also printed first so the file is
self-describing when redirected to a ``.dat`` file by SLURM.

Usage::

    python3.11 scripts/execute_discrete_edas_RW.py \
        20 100 50 0.5 1 UMDA SAT \
        packages/pateda/src/pateda/functions/SAT_instances/uf20-01.cnf
"""

import os
import sys
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


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "UMDA": UMDA,
    "BMDA": BMDA,
    "TreeEDA": TreeEDA,
    "TreeEDA-r": TreeEDAR,
    "MIMIC": MIMIC,
    "PBIL": PBIL,
    "EBNA": EBNA,
    "BOA": BOA,
    "AffEDA": AffEDA,
    "MKEDA": MKEDA,
    "MTED": MTED,
    "MNFDA": MNFDA,
    "MNFDAR": MNFDAR,
    "MNFDAG": MNFDAG,
    "MNFDAGR": MNFDAGR,
    "MOA": MOA,
    "FDA": FDA,
    "BSC": BSC,
}

# Wrappers that consume an interaction-matrix prior.
RESTRICTED_ALGORITHMS = {"TreeEDA-r", "MNFDAR", "MNFDAGR"}

# Threshold-ratio used to derive a sparse interaction matrix for UBQP.
UBQP_THRESHOLD_RATIO = 0.5


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def _single_objective(values):
    """Coerce scalar/array fitness outputs to a Python float."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type, instance_path):
    """
    Load a packaged real-world benchmark plus its prior interaction matrix.

    The ``instance_path`` argument is the *full* path to the instance file;
    we split it into directory + basename so the existing pateda loaders
    can find both.
    """
    instances_dir = os.path.dirname(instance_path) or None
    instance_name = os.path.basename(instance_path)
    problem_key = problem_type.upper()

    if problem_key == "SAT":
        sat_instance, optimal = load_sat_benchmark_instance(
            instance_name, instances_dir=instances_dir,
        )
        interaction_matrix = build_sat_interaction_matrix(sat_instance)

        def fitness_func(solution):
            return _single_objective(evaluate_sat(np.asarray(solution), sat_instance))

        return fitness_func, sat_instance.n_vars, 2, interaction_matrix, optimal

    if problem_key == "ISING":
        n_vars_inst, lattice, inter, optimal = load_ising_benchmark_instance(
            instance_name, instances_dir=instances_dir,
        )
        interaction_matrix = build_ising_interaction_matrix(lattice)

        def fitness_func(solution):
            solution_array = np.asarray(solution)
            if solution_array.ndim == 1:
                return -eval_ising(solution_array, lattice, inter)
            return np.array([-eval_ising(sol, lattice, inter) for sol in solution_array])

        return fitness_func, n_vars_inst, 2, interaction_matrix, optimal

    if problem_key == "UBQP":
        ubqp_instance, optimal = load_ubqp_benchmark_instance(
            instance_name, instances_dir=instances_dir,
        )
        interaction_matrix = build_ubqp_interaction_matrix(
            ubqp_instance, threshold_ratio=UBQP_THRESHOLD_RATIO,
        )

        def fitness_func(solution):
            return _single_objective(evaluate_ubqp(np.asarray(solution), ubqp_instance))

        return fitness_func, ubqp_instance.n_vars, 2, interaction_matrix, optimal

    raise ValueError(f"Unsupported problem type: {problem_type}")


# ---------------------------------------------------------------------------
# Algorithm runner
# ---------------------------------------------------------------------------

def run_algorithm(
    alg_name, alg_cls, n_vars, cardinality, fitness_func,
    interaction_matrix, pop_size, n_gen, selection_ratio, random_seed,
):
    """Construct, run, and time one EDA wrapper."""
    kwargs = {}
    if alg_name in RESTRICTED_ALGORITHMS:
        kwargs["interaction_matrix"] = interaction_matrix

    alg = alg_cls(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        pop_size=pop_size,
        n_gen=n_gen,
        selection_ratio=selection_ratio,
        random_seed=random_seed,
        **kwargs,
    )

    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0

    best = float(stats.best_fitness_overall)
    mean_last = float(stats.mean_fitness[-1]) if stats.mean_fitness else float("nan")
    return best, mean_last, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

USAGE = (
    "Usage: python execute_discrete_edas_RW.py "
    "n_vars pop_size n_gen selection_ratio random_seed "
    "alg problem instance_path"
)


def parse_args(argv):
    if len(argv) != 9:
        raise SystemExit(USAGE)

    n_vars = int(argv[1])
    pop_size = int(argv[2])
    n_gen = int(argv[3])
    selection_ratio = float(argv[4])
    random_seed = int(argv[5])
    alg_name = argv[6]
    problem = argv[7]
    instance_path = argv[8]

    if alg_name not in ALGORITHMS:
        raise SystemExit(
            f"Unknown algorithm '{alg_name}'. "
            f"Choose one of: {', '.join(sorted(ALGORITHMS))}"
        )

    return dict(
        n_vars=n_vars,
        pop_size=pop_size,
        n_gen=n_gen,
        selection_ratio=selection_ratio,
        random_seed=random_seed,
        alg_name=alg_name,
        problem=problem,
        instance_path=instance_path,
    )


def main(argv):
    args = parse_args(argv)

    print(
        f"# alg={args['alg_name']}  problem={args['problem']}  "
        f"instance={args['instance_path']}  "
        f"n_vars={args['n_vars']}  pop_size={args['pop_size']}  "
        f"n_gen={args['n_gen']}  selection_ratio={args['selection_ratio']}  "
        f"seed={args['random_seed']}"
    )
    print("# columns: best_fitness  mean_fitness  time_seconds")

    try:
        fitness_func, n_vars_inst, cardinality, interaction_matrix, optimal = load_problem(
            args["problem"], args["instance_path"],
        )

        if n_vars_inst != args["n_vars"]:
            print(
                f"# WARNING: n_vars argument ({args['n_vars']}) does not match "
                f"instance size ({n_vars_inst}). Using instance size."
            )

        print(f"# known_optimum={optimal}")

        best, mean_last, elapsed = run_algorithm(
            alg_name=args["alg_name"],
            alg_cls=ALGORITHMS[args["alg_name"]],
            n_vars=n_vars_inst,
            cardinality=cardinality,
            fitness_func=fitness_func,
            interaction_matrix=interaction_matrix,
            pop_size=args["pop_size"],
            n_gen=args["n_gen"],
            selection_ratio=args["selection_ratio"],
            random_seed=args["random_seed"],
        )
    except Exception:
        traceback.print_exc()
        # Emit a sentinel data row so downstream parsers can detect failures.
        print("NaN NaN NaN")
        return 1

    print(f"{best:.6f} {mean_last:.6f} {elapsed:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
