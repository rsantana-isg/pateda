"""Tests for the quasiparticle braid quantum-gate approximation problem.

Covers: generator/target algebra (unitarity, icosahedral group closure), the
error and effective-length definitions, the single-objective and bi-objective
fitness, and a short EDA run.
"""

import itertools
import numpy as np
import pytest

from pateda.functions.discrete_non_binary.problems.braid import (
    fibonacci_anyon_generators, su2_from_axis_angle, icosahedral_group,
    default_inverse_index, braid_matrix, braid_error, effective_length,
    braid_fitness, make_fibonacci_braid_problem, make_icosahedral_benchmark_problem,
    load_icosahedral_targets, load_anyon_generators,
)
from pateda.functions.discrete_non_binary.multiobjective.braid_biobjective import (
    braid_raw_objectives, braid_biobjective, make_icosahedral_braid_biobjective,
)


def _is_unitary(m):
    return np.allclose(m @ m.conj().T, np.eye(2), atol=1e-10)


# --------------------------------------------------------------------------- #
# Generators and targets
# --------------------------------------------------------------------------- #

def test_generators_are_unitary_and_inverses():
    g = fibonacci_anyon_generators()
    assert len(g) == 4
    assert all(_is_unitary(m) for m in g)
    # index 2 and 3 are the inverses of 0 and 1
    assert np.allclose(g[0] @ g[2], np.eye(2), atol=1e-10)
    assert np.allclose(g[1] @ g[3], np.eye(2), atol=1e-10)


def test_icosahedral_group_size_and_unitarity():
    ico = icosahedral_group()
    assert len(ico) == 60
    assert all(_is_unitary(m) for m in ico)
    assert all(abs(np.linalg.det(m) - 1.0) < 1e-9 for m in ico)


def test_icosahedral_group_is_closed_up_to_sign():
    ico = icosahedral_group()

    def in_group(M):
        return any(np.linalg.norm(M - g) < 1e-8 or np.linalg.norm(M + g) < 1e-8 for g in ico)

    rng = np.random.default_rng(0)
    for _ in range(100):
        i, j = int(rng.integers(60)), int(rng.integers(60))
        assert in_group(ico[i] @ ico[j])


def test_identity_element_is_first():
    ico = icosahedral_group()
    assert np.allclose(ico[0], np.eye(2), atol=1e-12)


# --------------------------------------------------------------------------- #
# Braid algebra
# --------------------------------------------------------------------------- #

def test_braid_matrix_matches_manual_product():
    g = fibonacci_anyon_generators()
    x = [0, 1, 3, 2]
    expected = g[0] @ g[1] @ g[3] @ g[2]
    assert np.allclose(braid_matrix(x, g), expected)


def test_error_zero_when_braid_equals_target():
    g = fibonacci_anyon_generators()
    # sigma_1 sigma_1^{-1} = identity; use identity target (icosahedral[0]).
    target = np.eye(2, dtype=complex)
    assert braid_error([0, 2], g, target) == pytest.approx(0.0, abs=1e-10)


def test_effective_length_free_reduction():
    inv = default_inverse_index(4)
    assert effective_length([0, 2], inv) == 0            # s1 s1^-1 -> identity
    assert effective_length([0, 1, 3], inv) == 1         # s1 s2 s2^-1 -> s1
    assert effective_length([0, 0, 0, 0, 2], inv) == 3   # s1^4 s1^-1 -> s1^3
    assert effective_length([0, 1, 2, 3], inv) == 4      # no adjacent inverse pair


def test_fitness_decreases_with_error():
    g = fibonacci_anyon_generators()
    target = icosahedral_group()[5]
    good = braid_error([3, 0, 3, 0, 3, 3, 3, 3], g, target)  # known decent braid
    bad = braid_error([0, 0, 0, 0, 0, 0, 0, 0], g, target)
    f_good = braid_fitness([3, 0, 3, 0, 3, 3, 3, 3], g, target, lam=0.0)
    f_bad = braid_fitness([0, 0, 0, 0, 0, 0, 0, 0], g, target, lam=0.0)
    assert good < bad
    assert f_good > f_bad
    assert f_good == pytest.approx(1.0 / (1.0 + good))


def test_brute_force_optimum_reachable():
    g = fibonacci_anyon_generators()
    target = icosahedral_group()[5]
    best = min(braid_error(c, g, target) for c in itertools.product(range(4), repeat=6))
    assert best < 0.6  # a length-6 braid gets a rough approximation


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #

def test_packaged_instances_match_generated():
    ico_file = load_icosahedral_targets()
    ico_gen = np.array(icosahedral_group())
    assert ico_file.shape == (60, 2, 2)
    assert np.allclose(ico_file, ico_gen)
    gens_file = load_anyon_generators()
    assert gens_file.shape == (4, 2, 2)
    assert np.allclose(gens_file, np.array(fibonacci_anyon_generators()))


def test_make_icosahedral_benchmark_problem():
    prob = make_icosahedral_benchmark_problem(10, 20)
    assert prob.cardinality == 4
    assert prob.n_matrices == 20
    assert np.allclose(prob.target, icosahedral_group()[10])
    with pytest.raises(ValueError):
        make_icosahedral_benchmark_problem(60, 20)


# --------------------------------------------------------------------------- #
# Bi-objective
# --------------------------------------------------------------------------- #

def test_biobjective_vector_and_directions():
    obj, prob = make_icosahedral_braid_biobjective(5, 12)
    x = np.zeros(12, dtype=int)  # all sigma_1: short (elen 12) but poor approx
    vec = obj(x)
    assert vec.shape == (2,)
    assert np.all((vec > 0) & (vec <= 1.0))
    eps, length = braid_raw_objectives(x, prob)
    assert vec[0] == pytest.approx(1.0 / (1.0 + eps))
    assert vec[1] == pytest.approx(1.0 / length)


def test_biobjective_population_shape():
    obj, _ = make_icosahedral_braid_biobjective(5, 12)
    pop = np.random.randint(0, 4, size=(7, 12))
    out = obj(pop)
    assert out.shape == (7, 2)


# --------------------------------------------------------------------------- #
# EDA integration
# --------------------------------------------------------------------------- #

def test_umda_improves_braid_fitness():
    from pateda import UMDA
    from pateda.functions.discrete_non_binary.problems.braid import create_braid_objective_function

    prob = make_icosahedral_benchmark_problem(5, 12)
    objective = create_braid_objective_function(prob, lam=0.0)
    alg = UMDA(n_vars=12, cardinality=4, fitness_func=objective, pop_size=200,
               n_gen=25, selection_ratio=0.15, elitism=True, alpha=1.0, random_seed=1)
    stats, _ = alg.run(verbose=False)
    # A decent braid beats a random start: fitness > 1/(1+2) (error < 2).
    assert stats.best_fitness_overall > 1.0 / (1.0 + 2.0)
    assert stats.best_fitness[-1] >= stats.best_fitness[0]
