"""
Run EDA Search - Problem Exploration with Structure and Sample File Output
==========================================================================

Given an objective function, an EDA, a number of repetitions and a number
``SAMP`` of solutions to store, this program runs the EDA on the function and
writes two files describing the search.

1. ``<func>_<n>_structure.dat`` -- the problem (interaction) structure, given
   as the list of edges of the symmetric interaction matrix.  The first line is
   the number of edges; each subsequent line is ``i j`` with ``i < j`` (0-based
   variable indices) meaning variables ``i`` and ``j`` are related in the
   structure.  For a function with no interactions (e.g. OneMax) the file
   contains a single ``0``.

2. ``<func>_<n>_<eda>_samples.dat`` -- ``SAMP`` rows and ``n + 2`` columns.
   Each row is a solution (``n`` variable values), followed by its objective
   value, followed by a number in ``{0, 1, 2}`` giving the third of the
   generation range where the solution was collected (``0`` = first third,
   ``1`` = second third, ``2`` = last third).  Roughly ``SAMP / 3`` solutions
   are drawn for each third; they are sampled at random, without repetition,
   from the pool of solutions collected in that third across *all* runs.  If a
   third does not contain ``SAMP / 3`` distinct solutions (e.g. the EDA
   converged early), repeated solutions are kept to fill the quota.

Supported objective functions / problems (with the sizes used for testing):

    OneMax          36, 64, 100, 256      (no interactions)
    Trap            36, 64, 100, 256      (non-overlapping blocks of 4)
    Deceptive3      39, 66, 102, 258      (non-overlapping blocks of 3)
    Checkerboard    36, 64, 100, 256      (2-D grid neighbourhood)
    EqualProducts   36, 64, 100, 256      (fully connected)
    Ising           36, 64, 100, 256      (spin-glass lattice)
    UBQP            50, 100               (quadratic interaction graph)
    MaxClique       30, 60, 125           (problem graph)
    Braid           36, 64, 100, 256      (sequence / chain structure)

Usage
-----
::

    python run_eda_search.py <obj_func> <n> <eda> <pop_size> <n_gen> \\
                             <n_reps> <SAMP> [seed]

    python run_eda_search.py OneMax 64 UMDA 200 30 10 300
    python run_eda_search.py Ising 100 TreeEDA 500 50 10 300 7
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import CacheConfig
from pateda.learning.ebna import LearnEBNA
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.moa import LearnMOA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.umda import LearnUMDA
from pateda.replacement import ElitistReplacement
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.fda import SampleFDA
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.markov import SampleMarkovChain
from pateda.sampling.mixture_trees import SampleMixtureTrees
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations

from pateda.learning.interaction_learning import (
    find_matrix_interactions_additive_decomposable,
)

# Objective functions / problem loaders
from pateda.functions.discrete_binary.toy_functions.onemax import onemax
from pateda.functions.discrete_binary.toy_functions.trap import trap_n
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.checkerboard import checkerboard
from pateda.functions.discrete_binary.problems.equal_products import (
    EqualProductsInstance,
    eval_equal_products,
)
from pateda.functions.discrete_binary.problems.ising import (
    load_ising,
    eval_ising,
    build_ising_interaction_matrix,
)
from pateda.functions.discrete_binary.problems.ubqp import (
    load_ubqp_benchmark_instance,
    evaluate_ubqp,
    build_ubqp_interaction_matrix,
)
from pateda.functions.discrete_binary.problems.max_clique import (
    MaxCliqueInstance,
    eval_max_clique,
)
from pateda.functions.discrete_non_binary.problems.braid import (
    make_icosahedral_benchmark_problem,
)
from pateda.functions.graph_utils import graph_instances_dir


# Trap uses non-overlapping blocks of this size (divides 36, 64, 100, 256).
TRAP_BLOCK = 4
# Deceptive3 uses non-overlapping blocks of size 3.
DECEP_BLOCK = 3
# Fixed seed used to build the (run-independent) EqualProducts instance.
EQUAL_PRODUCTS_INSTANCE_SEED = 12345
# Ising instance number used for every size.
ISING_INSTANCE = 1
# Icosahedral target used for every Braid size.
BRAID_TARGET = 0
# Cardinality of the Braid representation (4 Fibonacci-anyon generators).
BRAID_CARDINALITY = 4

# Maximum Clique benchmark files by number of nodes.
CLIQUE_INSTANCES = {
    30: "gnp_30_60.clq",
    60: "gnp_60_60.clq",
    125: "C125.9.clq",
}


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Everything the search loop needs to run and describe a problem."""

    name: str
    n_vars: int
    cardinality: np.ndarray
    #: Maximisation fitness passed to the EDA (1-D vector -> scalar).
    fitness_func: Callable[[np.ndarray], float]
    #: True objective value written to the samples file (1-D vector -> scalar).
    objective_func: Callable[[np.ndarray], float]
    #: Symmetric 0/1 interaction matrix (diagonal ignored).
    interaction_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Interaction-structure helpers
# ---------------------------------------------------------------------------

def _block_matrix(n_vars: int, block: int) -> np.ndarray:
    """Interaction matrix of a non-overlapping additively decomposable function."""
    if n_vars % block != 0:
        raise ValueError(f"n ({n_vars}) must be a multiple of the block size {block}")
    subfs = [list(range(i, i + block)) for i in range(0, n_vars, block)]
    return find_matrix_interactions_additive_decomposable(subfs, n_vars)


def _checkerboard_matrix(n_vars: int) -> np.ndarray:
    """Interaction matrix of the checkerboard function on an N x N grid.

    Each interior cell interacts with its four primary (von-Neumann) neighbours,
    reflecting the terms that actually appear in the fitness.
    """
    side = int(round(math.sqrt(n_vars)))
    if side * side != n_vars:
        raise ValueError(f"Checkerboard requires n to be a perfect square, got {n_vars}")

    matrix = np.zeros((n_vars, n_vars), dtype=int)

    def idx(i: int, j: int) -> int:
        return i * side + j

    for i in range(1, side - 1):
        for j in range(1, side - 1):
            a = idx(i, j)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                b = idx(i + di, j + dj)
                matrix[a, b] = 1
                matrix[b, a] = 1
    return matrix


def _complete_matrix(n_vars: int) -> np.ndarray:
    """Fully connected interaction matrix (every pair of variables interacts)."""
    matrix = np.ones((n_vars, n_vars), dtype=int)
    np.fill_diagonal(matrix, 0)
    return matrix


def _chain_matrix(n_vars: int) -> np.ndarray:
    """Sequence structure: variable i interacts with variable i+1."""
    matrix = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        matrix[i, i + 1] = 1
        matrix[i + 1, i] = 1
    return matrix


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

def _maxclique_instance_path(n_vars: int) -> str:
    directory = graph_instances_dir("maximum_clique")
    filename = CLIQUE_INSTANCES.get(n_vars, f"gnp_{n_vars}_60.clq")
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Maximum Clique instance not found: {path}")
    return str(path)


def parse_problem(obj_func: str, n: int) -> Problem:
    """Build the :class:`Problem` for *obj_func* with *n* variables."""
    binary_card = np.full(n, 2)

    if obj_func == "OneMax":
        obj = lambda x: float(np.sum(x))
        return Problem(obj_func, n, binary_card, obj, obj, np.zeros((n, n), dtype=int))

    if obj_func == "Trap":
        obj = lambda x: float(trap_n(np.asarray(x), TRAP_BLOCK))
        return Problem(obj_func, n, binary_card, obj, obj, _block_matrix(n, TRAP_BLOCK))

    if obj_func == "Deceptive3":
        obj = lambda x: float(deceptive3(np.asarray(x)))
        return Problem(obj_func, n, binary_card, obj, obj, _block_matrix(n, DECEP_BLOCK))

    if obj_func == "Checkerboard":
        obj = lambda x: float(checkerboard(np.asarray(x)))
        return Problem(obj_func, n, binary_card, obj, obj, _checkerboard_matrix(n))

    if obj_func == "EqualProducts":
        rng = np.random.default_rng(EQUAL_PRODUCTS_INSTANCE_SEED)
        numbers = rng.uniform(1e-4, 5.0, size=n)
        instance = EqualProductsInstance(numbers)
        # Minimisation problem -> maximise the negated difference.
        obj = lambda x, _inst=instance: float(eval_equal_products(np.asarray(x), _inst))
        fit = lambda x, _obj=obj: -_obj(x)
        return Problem(obj_func, n, binary_card, fit, obj, _complete_matrix(n))

    if obj_func == "Ising":
        lattice, inter = load_ising(n, ISING_INSTANCE)
        obj = lambda x, _l=lattice, _i=inter: float(eval_ising(np.asarray(x), _l, _i))
        return Problem(obj_func, n, binary_card, obj, obj,
                       build_ising_interaction_matrix(lattice))

    if obj_func == "UBQP":
        instance, _ = load_ubqp_benchmark_instance(f"bqp{n}")
        if instance.n_vars != n:
            raise ValueError(
                f"UBQP instance bqp{n} has {instance.n_vars} variables, expected {n}"
            )
        obj = lambda x, _inst=instance: float(evaluate_ubqp(np.asarray(x), _inst)[0])
        return Problem(obj_func, n, binary_card, obj, obj,
                       build_ubqp_interaction_matrix(instance))

    if obj_func == "MaxClique":
        instance = MaxCliqueInstance.from_file(_maxclique_instance_path(n))
        if instance.n_nodes != n:
            raise ValueError(
                f"MaxClique instance has {instance.n_nodes} nodes, expected {n}"
            )
        obj = lambda x, _inst=instance: float(eval_max_clique(np.asarray(x), _inst))
        return Problem(obj_func, n, binary_card, obj, obj,
                       np.asarray(instance.adj_matrix, dtype=int))

    if obj_func == "Braid":
        problem = make_icosahedral_benchmark_problem(BRAID_TARGET, n)
        card = np.full(n, BRAID_CARDINALITY)
        obj = lambda x, _p=problem: float(_p.fitness(np.asarray(x), lam=0.0))
        return Problem(obj_func, n, card, obj, obj, _chain_matrix(n))

    raise ValueError(f"Unknown objective function: '{obj_func}'")


# ---------------------------------------------------------------------------
# EDA builder
# ---------------------------------------------------------------------------

def build_eda(alg: str, problem: Problem, pop_size: int, max_generations: int,
              random_seed: int = None) -> EDA:
    """Assemble a configured :class:`~pateda.core.eda.EDA` for *problem*."""
    if alg == "UMDA":
        learning = LearnUMDA(alpha=1.0)
        sampling = SampleFDA(n_samples=pop_size)
    elif alg == "TreeEDA":
        learning = LearnTreeModel(alpha=0.1)
        sampling = SampleFDA(n_samples=pop_size)
    elif alg == "EBNA":
        learning = LearnEBNA(max_parents=3, score_metric="bic")
        sampling = SampleBayesianNetwork(n_samples=pop_size)
    elif alg == "MOA":
        learning = LearnMOA(k_neighbors=5, threshold_factor=1.5)
        sampling = SampleGibbs(n_samples=pop_size, IT=4, temperature=1.0)
    elif alg == "MN-FDA":
        learning = LearnMNFDA(max_clique_size=3, threshold=0.05, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)
    elif alg == "MN-FDAG":
        learning = LearnMNFDAG(max_clique_size=5, alpha=0.01, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)
    elif alg == "MK-EDA1":
        learning = LearnMarkovChain(k=1, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    elif alg == "MK-EDA2":
        learning = LearnMarkovChain(k=2, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    elif alg == "MK-EDA3":
        learning = LearnMarkovChain(k=3, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    elif alg == "MT-EDA2":
        learning = LearnMixtureTrees(n_components=2, component_learning="tree",
                                     alpha=0.1, weight_learning="uniform",
                                     random_seed=random_seed)
        sampling = SampleMixtureTrees(n_samples=pop_size)
    elif alg == "MT-EDA3":
        learning = LearnMixtureTrees(n_components=3, component_learning="tree",
                                     alpha=0.1, weight_learning="uniform",
                                     random_seed=random_seed)
        sampling = SampleMixtureTrees(n_samples=pop_size)
    else:
        raise ValueError(
            f"Unknown EDA: '{alg}'.  Supported: UMDA, TreeEDA, EBNA, MOA, "
            "MN-FDA, MN-FDAG, MK-EDA1, MK-EDA2, MK-EDA3, MT-EDA2, MT-EDA3"
        )

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_generations),
    )

    return EDA(
        pop_size=pop_size,
        n_vars=problem.n_vars,
        fitness_func=problem.fitness_func,
        cardinality=problem.cardinality,
        components=components,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_structure_file(interaction_matrix: np.ndarray, filepath: str) -> int:
    """Write the interaction structure as an edge list. Returns the edge count."""
    n_vars = interaction_matrix.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if interaction_matrix[i, j]:
                edges.append((i, j))

    with open(filepath, "w") as fh:
        fh.write(f"{len(edges)}\n")
        for i, j in edges:
            fh.write(f"{i} {j}\n")
    return len(edges)


def save_samples_file(rows: List[Tuple[np.ndarray, int]],
                      objective_func: Callable[[np.ndarray], float],
                      filepath: str) -> None:
    """Write ``<x_0 ... x_{n-1}> <objective> <third>`` for every sampled row."""
    with open(filepath, "w") as fh:
        for solution, third in rows:
            obj = objective_func(solution)
            var_str = " ".join(str(int(v)) for v in solution)
            fh.write(f"{var_str} {obj:.10g} {third}\n")


# ---------------------------------------------------------------------------
# Generation-third sampling
# ---------------------------------------------------------------------------

def _generation_third(generation: int, n_generations: int) -> int:
    """Map a 0-based generation index to its third {0, 1, 2}."""
    b1 = n_generations // 3
    b2 = (2 * n_generations) // 3
    if generation < b1:
        return 0
    if generation < b2:
        return 1
    return 2


def _third_quotas(samp: int) -> List[int]:
    """Split *samp* into three near-equal quotas summing to *samp*."""
    base = samp // 3
    rem = samp - 3 * base
    return [base + (1 if t < rem else 0) for t in range(3)]


def _select_samples(pools: Dict[int, Dict[bytes, np.ndarray]], samp: int,
                    rng: np.random.Generator) -> List[Tuple[np.ndarray, int]]:
    """Draw the per-third quotas from the collected pools of distinct solutions."""
    quotas = _third_quotas(samp)
    rows: List[Tuple[np.ndarray, int]] = []

    for third in (0, 1, 2):
        need = quotas[third]
        unique = list(pools[third].values())
        if need <= 0 or not unique:
            continue

        if len(unique) >= need:
            idx = rng.choice(len(unique), size=need, replace=False)
            chosen = [unique[i] for i in idx]
        else:
            # Not enough distinct solutions: keep all and fill with repeats.
            chosen = list(unique)
            deficit = need - len(unique)
            idx = rng.choice(len(unique), size=deficit, replace=True)
            chosen.extend(unique[i] for i in idx)

        rows.extend((sol, third) for sol in chosen)

    return rows


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_eda_search(obj_func: str, n_vars: int, eda_name: str, pop_size: int,
                   n_gen: int, n_reps: int, samp: int, base_seed: int = 0,
                   verbose: bool = True, output_dir: str = ".") -> None:
    """Run *eda_name* on *obj_func* for *n_reps* runs and write both files."""
    problem = parse_problem(obj_func, n_vars)

    # -- Structure file ----------------------------------------------------
    struct_path = os.path.join(output_dir, f"{obj_func}_{n_vars}_structure.dat")
    n_edges = save_structure_file(problem.interaction_matrix, struct_path)
    if verbose:
        print(f"Structure saved to '{struct_path}'  ({n_edges} edges)")

    # -- Run the EDA and collect solutions per generation-third ------------
    # pools[third] maps solution-bytes -> solution vector (distinct solutions).
    pools: Dict[int, Dict[bytes, np.ndarray]] = {0: {}, 1: {}, 2: {}}
    cache_config = CacheConfig(cache_populations=True)

    for rep in range(n_reps):
        seed = base_seed + rep
        eda = build_eda(eda_name, problem, pop_size, n_gen, seed)
        stats, cache = eda.run(cache_config=cache_config, verbose=False)

        n_generations = len(cache.populations)
        for generation, population in enumerate(cache.populations):
            third = _generation_third(generation, n_generations)
            bucket = pools[third]
            for individual in population:
                sol = np.asarray(individual, dtype=int)
                bucket[sol.tobytes()] = sol

        if verbose:
            print(f"  Rep {rep + 1}/{n_reps}: best fitness = "
                  f"{float(stats.best_fitness_overall):.6f}  "
                  f"({n_generations} generations)")

    if verbose:
        print(f"  Distinct solutions per third: "
              f"{[len(pools[t]) for t in (0, 1, 2)]}")

    # -- Sample and write the samples file ---------------------------------
    rng = np.random.default_rng(base_seed)
    rows = _select_samples(pools, samp, rng)

    samples_path = os.path.join(output_dir,
                                f"{obj_func}_{n_vars}_{eda_name}_samples.dat")
    save_samples_file(rows, problem.objective_func, samples_path)
    if verbose:
        counts = [sum(1 for _, t in rows if t == third) for third in (0, 1, 2)]
        print(f"Samples saved to  '{samples_path}'  "
              f"({len(rows)} rows; per third {counts})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    usage = (
        "Usage: python run_eda_search.py <obj_func> <n> <eda> <pop_size> "
        "<n_gen> <n_reps> <SAMP> [seed]\n"
        "\n"
        "Supported obj_func values:\n"
        "  OneMax, Trap, Deceptive3, Checkerboard, EqualProducts,\n"
        "  Ising, UBQP, MaxClique, Braid\n"
        "\n"
        "Supported eda values:\n"
        "  UMDA, TreeEDA, EBNA, MOA, MN-FDA, MN-FDAG,\n"
        "  MK-EDA1, MK-EDA2, MK-EDA3, MT-EDA2, MT-EDA3\n"
        "\n"
        "Example:\n"
        "  python run_eda_search.py OneMax 64 UMDA 200 30 10 300\n"
    )

    if len(sys.argv) < 8:
        print(usage)
        sys.exit(1)

    obj_func = sys.argv[1]
    n_vars = int(sys.argv[2])
    eda_name = sys.argv[3]
    pop_size = int(sys.argv[4])
    n_gen = int(sys.argv[5])
    n_reps = int(sys.argv[6])
    samp = int(sys.argv[7])
    base_seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0

    print("=" * 70)
    print("EDA Search")
    print("=" * 70)
    print(f"  Function   : {obj_func} (n={n_vars})")
    print(f"  EDA        : {eda_name}")
    print(f"  Pop size   : {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Repetitions: {n_reps}")
    print(f"  SAMP       : {samp}")
    print(f"  Base seed  : {base_seed}")
    print("=" * 70)

    try:
        run_eda_search(
            obj_func=obj_func,
            n_vars=n_vars,
            eda_name=eda_name,
            pop_size=pop_size,
            n_gen=n_gen,
            n_reps=n_reps,
            samp=samp,
            base_seed=base_seed,
            verbose=True,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
