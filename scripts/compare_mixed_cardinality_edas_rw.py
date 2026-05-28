"""
Compare all discrete EDAs on real-world benchmarks with mixed-cardinality
super-variable encoding.

The 100 binary variables of each benchmark are randomly partitioned into
non-overlapping groups of size uniformly drawn from {1,...,6}.  Group i is
represented as a single super-variable y_i with cardinality 2^|group_i|, so
cardinalities fall in {2, 4, 8, 16, 32, 64}.

Each wrapped fitness function:
  1. Receives a solution in super-variable space (integers 0 .. card_i - 1).
  2. Decodes each y_i back to its |group_i| binary variables via LSB-first
     encoding:  b_{g[k]} = (y_i >> k) & 1.
  3. Evaluates the original binary objective on the decoded representation.

The interaction matrix used by restricted algorithms (TreeEDA-r, MNFDAR) is
lifted from binary to super-variable space: super_interaction[i, j] = 1 iff
any binary variable in group_i interacts with any binary variable in group_j.

All EDA hyper-parameters (pop_size, n_gen, selection_ratio, seeds) are
identical to compare_discrete_edas_rw.py.  The purpose is to verify that
every algorithm is robust to a mixed-cardinality encoding of the same problems.

Benchmarks (all have n=100 binary variables):
  1. UBQP  -> bqp100      (unconstrained binary quadratic programming)
  2. SAT   -> uf100-01    (100-var, 430-clause random 3-SAT)
  3. Ising -> SG_100_1    (100-spin spin-glass)

Usage:
    python scripts/compare_mixed_cardinality_edas_rw.py
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
    ("UMDA",     UMDA),
    ("BMDA",     BMDA),
    ("TreeEDA",  TreeEDA),
    ("TreeEDA-r", TreeEDAR),
    ("MIMIC",    MIMIC),
    ("PBIL",     PBIL),
    ("EBNA",     EBNA),
  #  ("BOA",      BOA),
    ("AffEDA",   AffEDA),
    ("MKEDA",    MKEDA),
    ("MTED",     MTED),
    ("MNFDA",    MNFDA),
    ("MNFDAR",   MNFDAR),
    #("MNFDAG",  MNFDAG),
    #("MNFDAGR", MNFDAGR),
    #("MOA",     MOA),
    ("FDA",      FDA),
    ("BSC",      BSC),
]

PROBLEMS = [
    ("UBQP",  "bqp100"),
    ("SAT",   "uf100-01"),
    ("Ising", "SG_100_1"),
]

RESTRICTED_ALGORITHMS = {"TreeEDA-r", "MNFDAR", "MNFDAGR"}
POP_SIZE  = 200
N_GEN     = 50
SEL_RATIO = 0.5
N_RUNS    = 3
SEEDS     = list(range(1, N_RUNS + 1))
UBQP_THRESHOLD_RATIO = 0.5

# Super-variable encoding parameters — fixed across all problems and seeds.
N_BINARY_VARS  = 100   # all three benchmarks use exactly 100 binary variables
MAX_GROUP_SIZE = 6     # group sizes in {1,...,6} → cardinalities in {2,4,8,16,32,64}
GROUPING_SEED  = 42    # reproducible random partition


# ---------------------------------------------------------------------------
# Super-variable partitioning
# ---------------------------------------------------------------------------

def make_grouping(n_binary_vars: int, max_group_size: int, seed: int) -> list:
    """
    Randomly partition binary variable indices into non-overlapping groups.

    Group sizes are drawn uniformly from {1, ..., max_group_size} until all
    n_binary_vars indices are assigned.  The final group may be smaller than
    max_group_size if fewer variables remain.

    Returns:
        list of np.ndarray, each holding the binary variable indices for one
        super-variable.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_binary_vars)
    groups = []
    i = 0
    while i < n_binary_vars:
        remaining = n_binary_vars - i
        size = int(rng.integers(1, min(max_group_size, remaining) + 1))
        groups.append(perm[i : i + size])
        i += size
    return groups


def decode_super_to_binary(
    super_solution: np.ndarray, groups: list, n_binary_vars: int
) -> np.ndarray:
    """
    Decode one individual from super-variable space to binary space.

    LSB-first encoding:
        y_i = b_{g[0]}·2^0 + b_{g[1]}·2^1 + ... + b_{g[k-1]}·2^{k-1}
    Inverse:
        b_{g[j]} = (y_i >> j) & 1

    Args:
        super_solution: 1-D integer array of length n_sv.
        groups:         list of binary-variable index arrays per super-variable.
        n_binary_vars:  total number of binary variables.

    Returns:
        1-D binary array of length n_binary_vars.
    """
    binary = np.zeros(n_binary_vars, dtype=int)
    for sv_idx, group in enumerate(groups):
        val = int(super_solution[sv_idx])
        for bit_pos, bin_idx in enumerate(group):
            binary[int(bin_idx)] = (val >> bit_pos) & 1
    return binary


def lift_interaction_matrix(
    binary_interaction: np.ndarray, groups: list
) -> np.ndarray:
    """
    Project a binary-variable interaction matrix onto the super-variable space.

    super_interaction[i, j] = 1 iff binary_interaction[gi, gj] contains any
    nonzero entry (where gi, gj are the variable indices of groups i and j).
    Diagonal entries are always 1.
    """
    n_sv = len(groups)
    matrix = np.zeros((n_sv, n_sv), dtype=int)
    for i in range(n_sv):
        matrix[i, i] = 1
        for j in range(i + 1, n_sv):
            gi = groups[i].astype(int)
            gj = groups[j].astype(int)
            if np.any(binary_interaction[np.ix_(gi, gj)]):
                matrix[i, j] = 1
                matrix[j, i] = 1
    return matrix


# ---------------------------------------------------------------------------
# Problem loading with super-variable wrapping
# ---------------------------------------------------------------------------

def _single_objective(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type: str, instance_name: str, groups: list):
    """
    Load a benchmark instance and wrap its fitness for super-variable inputs.

    Returns:
        fitness_func      : callable(1-D super-var array) → scalar fitness.
        n_sv              : number of super-variables.
        cardinality_vec   : np.ndarray (n_sv,) of per-variable cardinalities.
        super_interaction : np.ndarray (n_sv, n_sv) lifted interaction matrix.
        optimal           : known optimal fitness value (or None).
    """
    n_sv = len(groups)
    cardinality_vec = np.array([2 ** len(g) for g in groups], dtype=int)
    problem_key = problem_type.upper()

    if problem_key == "SAT":
        sat_instance, optimal = load_sat_benchmark_instance(instance_name)
        n_binary = sat_instance.n_vars
        binary_interaction = build_sat_interaction_matrix(sat_instance)
        super_interaction = lift_interaction_matrix(binary_interaction, groups)

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return _single_objective(evaluate_sat(binary, sat_instance))

    elif problem_key == "ISING":
        n_binary, lattice, inter, optimal = load_ising_benchmark_instance(instance_name)
        binary_interaction = build_ising_interaction_matrix(lattice)
        super_interaction = lift_interaction_matrix(binary_interaction, groups)

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return -eval_ising(binary, lattice, inter)

    elif problem_key == "UBQP":
        ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
        n_binary = ubqp_instance.n_vars
        binary_interaction = build_ubqp_interaction_matrix(
            ubqp_instance, threshold_ratio=UBQP_THRESHOLD_RATIO
        )
        super_interaction = lift_interaction_matrix(binary_interaction, groups)

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return _single_objective(evaluate_ubqp(binary, ubqp_instance))

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return fitness_func, n_sv, cardinality_vec, super_interaction, optimal


# ---------------------------------------------------------------------------
# EDA runners
# ---------------------------------------------------------------------------

def run_single_seed(
    alg_name, alg_cls, n_vars, cardinality, fitness_func,
    interaction_matrix, seed,
):
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
    bests, times = [], []
    for seed in SEEDS:
        best, elapsed = run_single_seed(
            alg_name, alg_cls, n_vars, cardinality,
            fitness_func, interaction_matrix, seed,
        )
        bests.append(best)
        times.append(elapsed)
    return bests, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build one fixed grouping shared across all three problems.
    groups = make_grouping(N_BINARY_VARS, MAX_GROUP_SIZE, GROUPING_SEED)
    cardinality_vec = np.array([2 ** len(g) for g in groups], dtype=int)
    group_sizes = sorted([len(g) for g in groups], reverse=True)
    card_counts: dict = {}
    for c in cardinality_vec:
        card_counts[c] = card_counts.get(c, 0) + 1

    print(f"\n{'=' * 90}")
    print("Super-variable encoding (shared across all problems and seeds)")
    print(f"  {N_BINARY_VARS} binary variables  →  {len(groups)} super-variables")
    print(f"  Group sizes (desc):  {group_sizes}")
    print(
        "  Cardinality counts:  "
        + "  ".join(f"card={c}×{cnt}" for c, cnt in sorted(card_counts.items()))
    )
    print(f"  Encoding: LSB-first  (y_i = b_{{g[0]}}·1 + b_{{g[1]}}·2 + ...)")
    print(f"{'=' * 90}")

    name_w = 10

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_sv, card_vec, super_interaction, optimal = load_problem(
            problem_type, instance_name, groups
        )
        n_edges = int(np.sum(np.triu(super_interaction, k=1)))

        print(f"\n{'=' * 90}")
        print(
            f"Problem: {problem_type}  Instance: {instance_name}  "
            f"(n_supervars={n_sv}, prior_edges={n_edges}, optimal={optimal})"
        )
        print(
            f"pop_size={POP_SIZE}  n_gen={N_GEN}  selection_ratio={SEL_RATIO}  "
            f"n_runs={N_RUNS}  seeds={SEEDS}"
        )
        print(f"{'=' * 90}")

        problem_tag = f"{problem_type} {instance_name}"

        for alg_name, alg_cls in ALGORITHMS:
            try:
                bests, times = run_all_seeds(
                    alg_name, alg_cls, n_sv, card_vec,
                    fitness_func, super_interaction,
                )
                mean_best = float(np.mean(bests))
                mean_time = float(np.mean(times))
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
