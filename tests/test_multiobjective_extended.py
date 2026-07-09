"""
Tests for the extended multi-objective toolkit:

* core utilities (crowding distance, weights, scalarization, archive),
* quality indicators (hypervolume, contributions, IBEA fitness, IGD),
* the new selection methods (crowding, indicator-based),
* the MOEA/D decomposition driver,
* the multi-objective discrete benchmark functions.

Run with pytest, or directly: ``python test_multiobjective_extended.py``.
"""

import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import CrowdingDistanceSelection, IndicatorBasedSelection
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.multiobjective import (
    crowding_distance, ParetoArchive, generate_weights, das_dennis_weights,
    weight_neighbourhoods, scalarize, hypervolume, hypervolume_contributions,
    ibea_fitness, additive_epsilon_matrix, igd, reference_point_from, MOEAD,
    find_pareto_set,
)
from pateda.functions.discrete_binary.multiobjective import (
    mo_onemax_zeromax, make_mo_deceptive, make_mubqp,
    mo_pareto_front_onemax_zeromax,
)


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def test_crowding_distance_boundaries_infinite():
    objs = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    cd = crowding_distance(objs)
    # The two extreme points (min/max on f1) get infinite crowding distance.
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    assert np.all(np.isfinite(cd[[1, 2]]))


def test_weight_generation():
    w2 = generate_weights(2, 6)
    assert w2.shape == (6, 2)
    assert np.allclose(w2.sum(axis=1), 1.0)

    w3 = generate_weights(3, 15)
    assert w3.shape[1] == 3
    assert np.allclose(w3.sum(axis=1), 1.0)

    dd = das_dennis_weights(3, 4)  # C(6, 2) = 15 vectors
    assert dd.shape == (15, 3)
    assert np.allclose(dd.sum(axis=1), 1.0)


def test_weight_neighbourhoods():
    w = generate_weights(2, 10)
    nb = weight_neighbourhoods(w, 3)
    assert nb.shape == (10, 3)
    # First neighbour of each sub-problem is itself.
    assert np.array_equal(nb[:, 0], np.arange(10))


def test_scalarization_directions():
    obj = np.array([2.0, 2.0])
    w = np.array([0.5, 0.5])
    # Tchebycheff cost is zero at the ideal point, positive otherwise.
    ideal = np.array([2.0, 2.0])
    assert scalarize(obj, w, ideal, "tchebycheff", maximize=True) == 0.0
    worse = scalarize(np.array([1.0, 1.0]), w, ideal, "tchebycheff", maximize=True)
    assert worse > 0.0
    # weighted_sum is lower-is-better; better objectives -> lower cost (maximise).
    s_good = scalarize(np.array([3.0, 3.0]), w, None, "weighted_sum", maximize=True)
    s_bad = scalarize(np.array([1.0, 1.0]), w, None, "weighted_sum", maximize=True)
    assert s_good < s_bad


def test_pareto_archive():
    arc = ParetoArchive(maximize=True)
    assert arc.add(np.array([0, 0]), np.array([2.0, 1.0]))
    assert arc.add(np.array([1, 1]), np.array([1.0, 2.0]))
    # Dominated solution rejected.
    assert not arc.add(np.array([0, 1]), np.array([1.0, 1.0]))
    # A dominating solution removes the dominated member.
    assert arc.add(np.array([1, 0]), np.array([3.0, 3.0]))
    sols, objs = arc.get_front()
    assert len(objs) == 1
    assert np.allclose(objs[0], [3.0, 3.0])


def test_archive_capacity_prune():
    arc = ParetoArchive(maximize=True, capacity=3)
    for k in range(6):
        arc.add(np.array([k]), np.array([float(k), float(5 - k)]))
    assert arc.size <= 3


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def test_hypervolume_exact_2d():
    pts = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
    hv = hypervolume(pts, reference=np.array([0.0, 0.0]), maximize=True)
    assert abs(hv - 6.0) < 1e-9


def test_hypervolume_contributions():
    pts = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
    contrib = hypervolume_contributions(pts, reference=np.array([0.0, 0.0]),
                                        maximize=True)
    assert np.allclose(contrib, [1.0, 1.0, 1.0])


def test_dominated_point_zero_contribution():
    pts = np.array([[3.0, 3.0], [1.0, 1.0]])  # second dominated (maximise)
    contrib = hypervolume_contributions(pts, reference=np.array([0.0, 0.0]),
                                        maximize=True)
    assert contrib[1] == 0.0
    assert contrib[0] > 0.0


def test_ibea_fitness_prefers_nondominated():
    pts = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 0.5]])
    fit = ibea_fitness(pts, maximize=True)
    # The dominating solution should have the largest IBEA fitness.
    assert np.argmax(fit) == 0
    assert additive_epsilon_matrix(pts, maximize=True).shape == (3, 3)


def test_igd_zero_when_matching_front():
    front = mo_pareto_front_onemax_zeromax(10)
    assert igd(front, front, maximize=True) == 0.0


def test_reference_point_is_worse_than_all():
    pts = np.array([[3.0, 1.0], [1.0, 3.0]])
    ref = reference_point_from(pts, maximize=True)
    assert np.all(ref < pts.min(axis=0))


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def test_mo_onemax_zeromax_sum_constant():
    x = np.array([1, 0, 1, 1, 0, 0])
    f = mo_onemax_zeromax(x)
    assert f.sum() == len(x)


def test_make_mo_deceptive_optima():
    f = make_mo_deceptive(10, block_size=5)
    all_ones = f(np.ones(10, dtype=int))
    all_zeros = f(np.zeros(10, dtype=int))
    assert all_ones[0] == 10.0 and all_zeros[1] == 10.0


def test_make_mubqp_returns_vector():
    f, inst = make_mubqp(12, n_objectives=2, seed=0)
    val = f(np.random.default_rng(0).integers(0, 2, 12))
    assert val.shape == (2,)


# ---------------------------------------------------------------------------
# Selection methods inside the EDA loop
# ---------------------------------------------------------------------------

def _run_eda(selection, n_vars=15, pop=80, ngen=10, seed=3):
    f = make_mo_deceptive(n_vars, 5)
    comp = EDAComponents(
        seeding=RandomInit(), selection=selection,
        learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=pop),
        stop_condition=MaxGenerations(ngen))
    eda = EDA(pop, n_vars, f, np.full(n_vars, 2), comp, random_seed=seed)
    eda.run(verbose=False)
    return eda.fitness


def test_crowding_selection_runs():
    fit = _run_eda(CrowdingDistanceSelection(ratio=0.5, maximize=True))
    assert fit.shape[1] == 2


def test_indicator_selection_epsilon_runs():
    fit = _run_eda(IndicatorBasedSelection(ratio=0.5, maximize=True,
                                           indicator="epsilon"))
    assert fit.shape[1] == 2


def test_indicator_selection_hypervolume_runs():
    fit = _run_eda(IndicatorBasedSelection(ratio=0.5, maximize=True,
                                           indicator="hypervolume"),
                   n_vars=10, pop=40, ngen=5)
    assert fit.shape[1] == 2


def test_selection_returns_requested_count():
    pop = np.random.default_rng(0).integers(0, 2, (50, 10))
    fit = np.random.default_rng(1).random((50, 2))
    for sel in (CrowdingDistanceSelection(n_select=20, maximize=True),
                IndicatorBasedSelection(n_select=20, maximize=True)):
        sp, sf = sel.select(pop, fit, rng=np.random.default_rng(2))
        assert len(sp) == 20 and len(sf) == 20


# ---------------------------------------------------------------------------
# MOEA/D driver
# ---------------------------------------------------------------------------

def test_moead_onemax_zeromax_spread():
    f = mo_onemax_zeromax
    comp = EDAComponents(
        seeding=RandomInit(), selection=CrowdingDistanceSelection(),
        learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=100),
        stop_condition=MaxGenerations(10))
    moead = MOEAD(20, np.full(20, 2), f, comp, n_obj=2, n_weights=30,
                  neighbourhood_size=8, scalarization="tchebycheff",
                  maximize=True, n_gen=15, random_seed=5)
    res = moead.run(verbose=False)
    assert res.pareto_objectives.shape[0] >= 5
    # Every archived point must lie on f1 + f2 == 20.
    assert np.allclose(res.pareto_objectives.sum(axis=1), 20.0)


def test_moead_global_scope_runs():
    f = make_mo_deceptive(15, 5)
    comp = EDAComponents(
        seeding=RandomInit(), selection=CrowdingDistanceSelection(),
        learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=60),
        stop_condition=MaxGenerations(8))
    moead = MOEAD(15, np.full(15, 2), f, comp, n_obj=2, n_weights=20,
                  scalarization="weighted_sum", maximize=True, n_gen=8,
                  model_scope="global", random_seed=1)
    res = moead.run(verbose=False)
    assert res.pareto_objectives.shape[0] >= 1


def test_moead_continuous_bounds():
    # Simple continuous bi-objective: maximise -(x)^2 and -(x-1)^2 componentwise.
    def f(x):
        return np.array([-np.sum(x ** 2), -np.sum((x - 1.0) ** 2)])
    bounds = np.array([[-2.0] * 5, [2.0] * 5])  # (2, n_vars)
    from pateda.learning import LearnGaussianUnivariate
    from pateda.sampling.basic_gaussian import SampleGaussianUnivariate
    comp = EDAComponents(
        seeding=None, selection=CrowdingDistanceSelection(),
        learning=LearnGaussianUnivariate(),
        sampling=SampleGaussianUnivariate(n_samples=40),
        stop_condition=MaxGenerations(8))
    moead = MOEAD(5, bounds, f, comp, n_obj=2, n_weights=20,
                  scalarization="tchebycheff", maximize=True, n_gen=8,
                  random_seed=2)
    res = moead.run(verbose=False)
    assert res.pareto_objectives.shape[0] >= 1


if __name__ == "__main__":
    import sys
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
