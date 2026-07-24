"""
Self-contained problem registry for the BN-EDA study.

Builds the objective, cardinality and true interaction structure for every
problem in ``eda_cluster_results.csv`` using only the **installed** ``pateda``
package (``pateda.functions`` / ``pateda.learning``).  It is a vendored copy of
``examples/run_eda_search.py``'s ``parse_problem`` so that the cluster runner does
not depend on the ``examples/`` folder (which is not part of the pip-installed
pateda wheel).

The instance definitions here MUST match the ones that generated the offline
benchmark datasets (same block sizes, instance seeds, instance numbers, targets).
"""
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pateda.learning.interaction_learning import (
    find_matrix_interactions_additive_decomposable,
)
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


# --- instance constants (must match the benchmark dataset generation) --------
TRAP_BLOCK = 4
DECEP_BLOCK = 3
EQUAL_PRODUCTS_INSTANCE_SEED = 12345
ISING_INSTANCE = 1
BRAID_TARGET = 0
BRAID_CARDINALITY = 4
CLIQUE_INSTANCES = {
    30: "gnp_30_60.clq",
    60: "gnp_60_60.clq",
    125: "C125.9.clq",
}


@dataclass
class Problem:
    """Everything the search loop needs to run and describe a problem."""
    name: str
    n_vars: int
    cardinality: np.ndarray
    fitness_func: Callable[[np.ndarray], float]       # maximisation
    objective_func: Callable[[np.ndarray], float]
    interaction_matrix: np.ndarray                    # symmetric 0/1


# --- interaction-structure helpers -------------------------------------------
def _block_matrix(n_vars: int, block: int) -> np.ndarray:
    if n_vars % block != 0:
        raise ValueError(f"n ({n_vars}) must be a multiple of the block size {block}")
    subfs = [list(range(i, i + block)) for i in range(0, n_vars, block)]
    return find_matrix_interactions_additive_decomposable(subfs, n_vars)


def _checkerboard_matrix(n_vars: int) -> np.ndarray:
    side = int(round(math.sqrt(n_vars)))
    if side * side != n_vars:
        raise ValueError(f"Checkerboard requires n to be a perfect square, got {n_vars}")
    matrix = np.zeros((n_vars, n_vars), dtype=int)

    def idx(i, j):
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
    matrix = np.ones((n_vars, n_vars), dtype=int)
    np.fill_diagonal(matrix, 0)
    return matrix


def _chain_matrix(n_vars: int) -> np.ndarray:
    matrix = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        matrix[i, i + 1] = 1
        matrix[i + 1, i] = 1
    return matrix


def _maxclique_instance_path(n_vars: int) -> str:
    directory = graph_instances_dir("maximum_clique")
    filename = CLIQUE_INSTANCES.get(n_vars, f"gnp_{n_vars}_60.clq")
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Maximum Clique instance not found: {path}")
    return str(path)


# --- registry ----------------------------------------------------------------
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
                f"UBQP instance bqp{n} has {instance.n_vars} variables, expected {n}")
        obj = lambda x, _inst=instance: float(evaluate_ubqp(np.asarray(x), _inst)[0])
        return Problem(obj_func, n, binary_card, obj, obj,
                       build_ubqp_interaction_matrix(instance))

    if obj_func == "MaxClique":
        instance = MaxCliqueInstance.from_file(_maxclique_instance_path(n))
        if instance.n_nodes != n:
            raise ValueError(
                f"MaxClique instance has {instance.n_nodes} nodes, expected {n}")
        obj = lambda x, _inst=instance: float(eval_max_clique(np.asarray(x), _inst))
        return Problem(obj_func, n, binary_card, obj, obj,
                       np.asarray(instance.adj_matrix, dtype=int))

    if obj_func == "Braid":
        problem = make_icosahedral_benchmark_problem(BRAID_TARGET, n)
        card = np.full(n, BRAID_CARDINALITY)
        obj = lambda x, _p=problem: float(_p.fitness(np.asarray(x), lam=0.0))
        return Problem(obj_func, n, card, obj, obj, _chain_matrix(n))

    raise ValueError(f"Unknown objective function: '{obj_func}'")
