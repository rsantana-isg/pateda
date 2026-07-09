import numpy as np

from pateda import MNFDAGR, MNFDAR, TreeEDAR
from pateda.functions.discrete_binary.problems.ising import (
    build_ising_interaction_matrix,
    load_ising_benchmark_instance,
)
from pateda.functions.discrete_binary.problems.sat import (
    build_sat_interaction_matrix,
    load_sat_benchmark_instance,
)
from pateda.functions.discrete_binary.problems.ubqp import (
    build_ubqp_interaction_matrix,
    load_ubqp_benchmark_instance,
)


def _onemax(x):
    arr = np.asarray(x)
    if arr.ndim == 1:
        return float(np.sum(arr))
    return np.sum(arr, axis=1)


def test_load_sat_benchmark_instance_builds_clause_interactions():
    sat_instance, optimal = load_sat_benchmark_instance("uf20-01")

    assert sat_instance.n_vars == 20
    assert sat_instance.n_objectives == 1
    assert optimal == "Unknown"

    interaction_matrix = build_sat_interaction_matrix(sat_instance)
    assert interaction_matrix.shape == (20, 20)
    assert np.all(np.diag(interaction_matrix) == 1)
    assert interaction_matrix[3, 17] == 1
    assert interaction_matrix[3, 18] == 1
    assert interaction_matrix[17, 18] == 1


def test_load_ising_benchmark_instance_builds_lattice_interactions():
    n_vars, lattice, inter, optimal = load_ising_benchmark_instance("SG_16_1")

    assert n_vars == 16
    assert lattice.shape[0] == n_vars
    assert inter.shape[0] == n_vars
    assert optimal == "Unknown"

    interaction_matrix = build_ising_interaction_matrix(lattice)
    assert interaction_matrix.shape == (16, 16)
    assert np.all(np.diag(interaction_matrix) == 1)
    assert interaction_matrix[0, 12] == 1
    assert interaction_matrix[12, 0] == 1


def test_load_ubqp_benchmark_instance_builds_thresholded_interactions():
    ubqp_instance, optimal = load_ubqp_benchmark_instance("bqp50")

    assert ubqp_instance.n_vars == 50
    assert optimal == "Unknown"

    interaction_matrix = build_ubqp_interaction_matrix(ubqp_instance, threshold_ratio=0.0)
    assert interaction_matrix.shape == (50, 50)
    assert np.all(np.diag(interaction_matrix) == 1)
    assert interaction_matrix[0, 3] == 1
    assert interaction_matrix[3, 0] == 1


def test_restricted_wrappers_accept_custom_interaction_matrices():
    interaction_matrix = np.eye(6, dtype=int)
    interaction_matrix[0, 1] = 1
    interaction_matrix[1, 0] = 1

    for alg_cls in (TreeEDAR, MNFDAR, MNFDAGR):
        alg = alg_cls(
            n_vars=6,
            cardinality=2,
            fitness_func=_onemax,
            pop_size=10,
            n_gen=2,
            interaction_matrix=interaction_matrix,
            random_seed=7,
        )
        np.testing.assert_array_equal(
            alg._eda.components.learning.interaction_matrix,
            interaction_matrix,
        )
