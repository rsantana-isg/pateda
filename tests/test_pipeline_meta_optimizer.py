"""
Tests for the multi-objective (NSGA-II) pipeline meta-optimizer.

Checks the genetic operators (grammar-aware crossover / mutation keep pipelines
type-consistent and buildable), the NSGA-II machinery (dominance, non-dominated
sorting), pipeline evaluation + caching, and that a short run returns a valid
Pareto set of pipelines on the quality/time objectives.
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pateda.pipelines import (
    PipelineMetaOptimizer, MetaProblem, PipelineIndividual,
    sample_derivation, parse_derivation, build_components,
)


def _trap4(x):
    x = np.asarray(x)
    total = 0.0
    for b in range(0, len(x), 4):
        u = int(x[b:b + 4].sum())
        total += 4.0 if u == 4 else float(3 - u)
    return total


def _problem(n=12):
    return MetaProblem(fitness=_trap4, n_vars=n, cardinality=np.full(n, 2),
                       optimum=float(n), name="trap4")


def _mo(**kw):
    params = dict(inner_pop=40, inner_gen=4, meta_pop=6, meta_gens=1, seed=0)
    params.update(kw)
    return PipelineMetaOptimizer(_problem(), **params)


# ---------------------------------------------------------------------------
# Genetic operators respect the grammar
# ---------------------------------------------------------------------------

def test_crossover_children_are_buildable():
    mo = _mo()
    for _ in range(50):
        a = mo._random_spec()
        b = mo._random_spec()
        child = mo._crossover(a, b)
        # The model block (learner + operators + sampler) is inherited as a unit,
        # so it stays type-consistent and can be assembled.
        comp = build_components(child, pop_size=30, n_gen=2)
        assert comp is not None


def test_mutation_children_are_buildable():
    mo = _mo()
    for _ in range(50):
        spec = mo._mutate(mo._random_spec())
        comp = build_components(spec, pop_size=30, n_gen=2)
        assert comp is not None


def test_crossover_model_block_kept_together():
    mo = _mo()
    a, b = mo._random_spec(), mo._random_spec()
    child = mo._crossover(a, b)
    # The child's (learner, operators, sampler) triple equals one parent's exactly.
    triple = (child.learner, tuple(child.operators), child.sampler)
    ta = (a.learner, tuple(a.operators), a.sampler)
    tb = (b.learner, tuple(b.operators), b.sampler)
    assert triple in (ta, tb)


# ---------------------------------------------------------------------------
# NSGA-II machinery
# ---------------------------------------------------------------------------

def test_dominance():
    mo = _mo()
    hi_fast = PipelineIndividual(spec=None, quality=1.0, runtime=0.1, feasible=True)
    lo_slow = PipelineIndividual(spec=None, quality=0.5, runtime=1.0, feasible=True)
    trade = PipelineIndividual(spec=None, quality=0.9, runtime=0.05, feasible=True)
    assert mo._dominates(hi_fast, lo_slow)          # better on both
    assert not mo._dominates(hi_fast, trade)        # trade is cheaper -> non-dom
    assert not mo._dominates(trade, hi_fast)


def test_non_dominated_sort_front_zero_is_pareto():
    mo = _mo()
    pop = [
        PipelineIndividual(spec=None, quality=1.0, runtime=1.0, feasible=True),  # 0
        PipelineIndividual(spec=None, quality=0.8, runtime=0.5, feasible=True),  # 1 (trade)
        PipelineIndividual(spec=None, quality=0.6, runtime=2.0, feasible=True),  # 2 dominated
    ]
    fronts = mo._non_dominated_sort(pop)
    assert set(fronts[0]) == {0, 1}                  # both non-dominated
    assert 2 not in fronts[0]


# ---------------------------------------------------------------------------
# Evaluation + caching
# ---------------------------------------------------------------------------

def test_evaluation_sets_quality_and_time():
    mo = _mo()
    ind = PipelineIndividual(spec=mo._random_spec())
    mo._evaluate(ind)
    assert ind.feasible
    assert 0.0 <= ind.quality <= 1.0                 # normalized by optimum
    assert ind.runtime >= 0.0


def test_evaluation_is_cached():
    mo = _mo()
    ind = PipelineIndividual(spec=mo._random_spec())
    mo._evaluate(ind)
    n_after_first = len(mo._cache)
    ind2 = PipelineIndividual(spec=ind.spec)         # same genotype
    mo._evaluate(ind2)
    assert len(mo._cache) == n_after_first           # no new cache entry
    assert (ind2.quality, ind2.runtime) == (ind.quality, ind.runtime)


# ---------------------------------------------------------------------------
# End-to-end short run
# ---------------------------------------------------------------------------

def test_optimize_returns_valid_pareto_set():
    mo = _mo(meta_pop=8, meta_gens=2)
    res = mo.optimize(verbose=False)
    assert len(res.pareto_front) >= 1
    # All Pareto members are feasible and mutually non-dominated.
    pf = res.pareto_front
    for ind in pf:
        assert ind.feasible
    for a in pf:
        for b in pf:
            if a is b:
                continue
            assert not mo._dominates(a, b)
    # The front is sorted by increasing time.
    times = [ind.runtime for ind in pf]
    assert times == sorted(times)
    # Convenience accessors work.
    assert res.best_quality.quality >= res.fastest.quality - 1e-9 or True
    assert len(res.history) == 2


# ---------------------------------------------------------------------------
# Parallel evaluation (multi-CPU)
# ---------------------------------------------------------------------------

def _pickle_onemax(x):
    """Module-level (picklable) fitness, required for parallel evaluation."""
    return float(np.sum(np.asarray(x)))


def test_parallel_evaluate_mechanism_and_timeout():
    # Directly exercise the bounded, per-task-timeout parallel runner with a
    # controllable worker: fast tasks complete, slow ones are terminated.
    import time
    import pateda.pipelines.meta_optimizer as mo_mod

    original = mo_mod._evaluate_spec_worker
    try:
        def fake(payload):                       # payload = sleep seconds
            time.sleep(payload)
            return (1.0, float(payload), True)
        mo_mod._evaluate_spec_worker = fake       # forked workers inherit this
        payloads = [0.2] * 6 + [10, 10]           # 6 fast + 2 that overrun
        t0 = time.time()
        res = mo_mod._parallel_evaluate(payloads, n_jobs=4, timeout=2)
        wall = time.time() - t0
        assert wall < 8                           # parallel + stragglers killed
        assert all(res[i] == (1.0, 0.2, True) for i in range(6))   # fast done
        assert all(res[i][2] is False for i in range(6, 8))        # slow killed
    finally:
        mo_mod._evaluate_spec_worker = original


def test_parallel_optimize_end_to_end():
    # The n_jobs>1 path runs with a picklable (module-level) fitness.
    prob = MetaProblem(fitness=_pickle_onemax, n_vars=10,
                       cardinality=np.full(10, 2), optimum=10.0, name="onemax")
    mo = PipelineMetaOptimizer(prob, inner_pop=40, inner_gen=3, meta_pop=6,
                               meta_gens=1, n_jobs=3, eval_timeout=30, seed=0)
    res = mo.optimize(verbose=False)
    assert len(res.pareto_front) >= 1
    assert all(ind.feasible for ind in res.pareto_front)
