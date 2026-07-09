"""Tests for the multi-objective instance generators and objective functions.

Covers the three models implemented from ``functions/Multi_Objective_Code``:
mNM (truncated-Walsh), MNK landscapes, and mUBQP (incl. hard instances).
"""

import itertools
import numpy as np
import pytest

from pateda.functions.discrete_binary.multiobjective.mnm import (
    MNMModel, MNMInstance, generate_mnm, create_mnm_objective_function,
)
from pateda.functions.discrete_binary.multiobjective.mnk_landscape import (
    MNKLandscape, generate_mnk, create_mnk_objective_function,
)
from pateda.functions.discrete_binary.multiobjective.mubqp import (
    MUBQPInstance, generate_mubqp, create_artificial_mubqp,
    create_mubqp_objective_function, enumerate_order5_chunks, chunk_pair_metrics,
    select_hard_chunk_pairs, create_mubqp_from_chunk, create_heavy_mubqp_from_chunks,
)


# --------------------------------------------------------------------------- #
# mNM model
# --------------------------------------------------------------------------- #

def test_mnm_walsh_value_matches_manual():
    model = MNMModel.create(n_vars=6, max_order=3, sigma=2.0, seed=1)
    x01 = np.array([1, 0, 1, 1, 0, 0])
    spins = 2 * x01 - 1
    ncomp = len(model.components)
    manual = 0.0
    for comp, beta in zip(model.components, model.betas):
        manual += beta if len(comp) == 0 else beta * np.prod(spins[list(comp)])
    manual /= ncomp
    assert model.evaluate_raw(x01) == pytest.approx(manual)


def test_mnm_truncation_reduces_components():
    model = MNMModel.create(n_vars=6, max_order=4, sigma=1.0, seed=2)
    x = np.array([1, 1, 0, 0, 1, 0])
    v_full = model.evaluate_raw(x)
    v_ord1 = model.evaluate_raw(x, max_order=1)
    # order-1 truncation uses fewer components, so the value generally differs
    assert v_full != pytest.approx(v_ord1)


def test_mnm_sign_transform_flips_odd_orders():
    # order-1 only model: f(-x) = -f(x) on the non-constant part
    model = MNMModel.create(n_vars=5, max_order=1, sigma=1.0, seed=3)
    x = np.array([1, 0, 1, 0, 1])
    ncomp = len(model.components)
    beta0 = model.betas[0]
    plus = model.evaluate_raw(x, sign=1)
    minus = model.evaluate_raw(x, sign=-1)
    # (plus - beta0/ncomp) == -(minus - beta0/ncomp)
    assert (plus - beta0 / ncomp) == pytest.approx(-(minus - beta0 / ncomp))


def test_mnm_instance_evaluate_and_shape():
    inst = generate_mnm(8, max_order=3, sigma=5.0, objective_orders=[2, 3], seed=4)
    obj = create_mnm_objective_function(inst)
    assert obj(np.zeros(8, dtype=int)).shape == (2,)
    assert obj(np.random.randint(0, 2, (7, 8))).shape == (7, 2)


def test_mnm_save_load_roundtrip(tmp_path):
    inst = generate_mnm(8, max_order=3, sigma=3.0, seed=5)
    p = tmp_path / "mnm.npz"
    inst.save(str(p))
    loaded = MNMInstance.load(str(p))
    pop = np.random.randint(0, 2, (6, 8))
    assert np.allclose(inst.evaluate(pop), loaded.evaluate(pop))


# --------------------------------------------------------------------------- #
# MNK landscape
# --------------------------------------------------------------------------- #

def test_mnk_values_in_unit_range():
    inst = generate_mnk(20, k=3, n_objectives=2, seed=1)
    F = create_mnk_objective_function(inst)(np.random.randint(0, 2, (200, 20)))
    assert F.shape == (200, 2)
    assert F.min() >= 0.0 and F.max() < 1.0


def test_mnk_heterogeneous_objectives():
    inst = generate_mnk(15, k=[1, 3, 5], n_objectives=3, seed=2)
    assert inst.ks == [1, 3, 5]
    assert inst.n_objectives == 3
    # objective o depends on k_o + 1 variables per subfunction
    assert inst.objectives[0].lattice.shape[1] == 2
    assert inst.objectives[2].lattice.shape[1] == 6


def test_mnk_neighbourhoods_valid():
    inst = generate_mnk(12, k=3, n_objectives=2, seed=3)
    for obj in inst.objectives:
        for i in range(inst.n_vars):
            row = obj.lattice[i]
            assert row[0] == i                      # self first
            assert len(set(row)) == len(row)        # distinct neighbours


def test_mnk_save_load_roundtrip(tmp_path):
    inst = generate_mnk(15, k=[2, 4], seed=6)
    p = tmp_path / "mnk.npz"
    inst.save(str(p))
    loaded = MNKLandscape.load(str(p))
    pop = np.random.randint(0, 2, (10, 15))
    assert np.allclose(inst.evaluate(pop), loaded.evaluate(pop))
    assert loaded.ks == inst.ks


# --------------------------------------------------------------------------- #
# mUBQP
# --------------------------------------------------------------------------- #

def test_mubqp_evaluation_matches_quadratic_form():
    inst = generate_mubqp(25, n_objectives=2, density=0.5, rho=-0.4, seed=1)
    x = np.random.randint(0, 2, 25).astype(float)
    manual = np.array([x @ Q @ x for Q in inst.matrices])
    assert np.allclose(inst.evaluate(x), manual)


def test_mubqp_diagonal_is_linear_term():
    # single objective with only a diagonal term w on variable 0
    Q = np.zeros((3, 3)); Q[0, 0] = 5.0
    inst = MUBQPInstance(3, [Q])
    assert inst.evaluate_single(np.array([1, 0, 0]))[0] == pytest.approx(5.0)
    assert inst.evaluate_single(np.array([0, 1, 1]))[0] == pytest.approx(0.0)


def test_mubqp_artificial_types_build():
    for itype in [1, 2, 3, 4]:
        inst = create_artificial_mubqp(16, itype)
        assert inst.n_objectives == 2
    inst5 = create_artificial_mubqp(16, 5)
    assert len(inst5.edges(0)) == 5 * (16 // 4)
    with pytest.raises(ValueError):
        create_artificial_mubqp(16, 99)


def test_mubqp_save_load_roundtrip(tmp_path):
    inst = generate_mubqp(20, n_objectives=2, density=0.4, rho=-0.5, seed=7)
    p = tmp_path / "mubqp.dat"
    inst.save(str(p))
    loaded = MUBQPInstance.load(str(p))
    pop = np.random.randint(0, 2, (8, 20))
    assert np.allclose(inst.evaluate(pop), loaded.evaluate(pop))


def test_order5_chunk_enumeration():
    params, edges = enumerate_order5_chunks()
    assert params.shape == (1024, 10)
    assert len(edges) == 10
    assert set(np.unique(params)) <= {-1.0, 1.0}


def test_chunk_pair_metrics_fields():
    params, _ = enumerate_order5_chunks()
    m = chunk_pair_metrics(params[10], params[500])
    assert set(m) == {"pareto_size", "deception_1", "deception_2", "fdc"}
    assert 1 <= m["pareto_size"] <= 32
    assert -1.0 <= m["fdc"] <= 1.0


def test_select_hard_chunks_and_compose():
    hard = select_hard_chunk_pairs(max_pairs=5, n_candidates=1500, min_pareto=3, seed=11)
    assert len(hard) >= 1
    w1, w2, m = hard[0]
    assert m["pareto_size"] >= 3
    tiled = create_mubqp_from_chunk(w1, w2, n_vars=50, k=5)
    assert tiled.n_vars == 50
    # tiled instance is block separable: 10 edges per block * 10 blocks
    assert len(tiled.edges(0)) == 10 * (50 // 5)
    heavy = create_heavy_mubqp_from_chunks([(w1, w2) for w1, w2, _ in hard],
                                           n_vars=50, k=5, n_chunks=20, seed=1)
    assert heavy.n_vars == 50 and heavy.n_objectives == 2


# --------------------------------------------------------------------------- #
# Integration: instances are usable by a multi-objective EDA
# --------------------------------------------------------------------------- #

def test_instances_run_with_mo_eda():
    from pateda.core.eda import EDA, EDAComponents
    from pateda.learning import LearnUMDA
    from pateda.sampling import SampleFDA
    from pateda.selection import TruncationSelection
    from pateda.replacement import GenerationalReplacement
    from pateda.seeding import RandomInit
    from pateda.stop_conditions import MaxGenerations

    inst = generate_mnk(15, k=2, n_objectives=2, seed=1)
    obj = create_mnk_objective_function(inst)
    comp = EDAComponents(
        seeding=RandomInit(), selection=TruncationSelection(ratio=0.5),
        learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=80),
        replacement=GenerationalReplacement(), stop_condition=MaxGenerations(8))
    eda = EDA(80, 15, obj, 2 * np.ones(15, dtype=int), comp, random_seed=1)
    stats, _ = eda.run(verbose=False)
    assert len(np.atleast_1d(stats.best_fitness[-1])) == 2
