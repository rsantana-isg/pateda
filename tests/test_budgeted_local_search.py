"""
Tests for the budget/subset-aware local search components.

Covered optimizers (all share
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`):

    DeterministicHillClimber, FirstImprovementHillClimber, StochasticHillClimber,
    SimulatedAnnealing, VariableNeighborhoodSearch,
    ReducedVariableNeighborhoodSearch.

The suite checks the shared contract that makes the optimizers interchangeable:

1. The evaluation budget is never exceeded (total, shared across the subset).
2. ``subset_fraction`` controls exactly how many solutions are optimized, and
   ``subset_selection`` chooses which ones.
3. No solution is ever replaced by a worse one; the ``optimize`` shape contract
   holds; non-selected solutions are untouched.
4. Each optimizer improves an easy separable problem (binary and non-binary).
5. End-to-end use as an EDA ``local_opt`` component (binary and non-binary).
"""

import numpy as np
import pytest

from pateda.local_optimization import (
    DeterministicHillClimber,
    FirstImprovementHillClimber,
    StochasticHillClimber,
    SimulatedAnnealing,
    VariableNeighborhoodSearch,
    ReducedVariableNeighborhoodSearch,
)
from pateda.local_optimization.budgeted_search import (
    _BudgetEvaluator,
    random_neighbor,
    shake,
    best_improvement_descent,
)


ALL_OPTIMIZERS = [
    DeterministicHillClimber,
    FirstImprovementHillClimber,
    StochasticHillClimber,
    SimulatedAnnealing,
    VariableNeighborhoodSearch,
    ReducedVariableNeighborhoodSearch,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onemax(x):
    return float(np.sum(x))


class _Counter:
    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return self.f(x)


def _make_pop(n_ind, n_vars, card, seed):
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, card, size=(n_ind, n_vars))
    fit = np.array([[_onemax(ind)] for ind in pop], dtype=float)   # 2D like EDA
    return pop, fit


# ---------------------------------------------------------------------------
# 1. Budget adherence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
def test_total_budget_not_exceeded(Optimizer):
    n_ind, n_vars, card = 30, 15, 4
    pop, fit = _make_pop(n_ind, n_vars, card, seed=1)
    budget = 2000
    opt = Optimizer(subset_fraction=0.5, evaluation_budget=budget, seed=7)

    counter = _Counter(_onemax)
    opt.optimize(pop, fit, counter, np.full(n_vars, card))
    assert counter.n <= budget


@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
def test_per_solution_budget_used_when_no_total(Optimizer):
    n_ind, n_vars, card = 10, 12, 3
    pop, fit = _make_pop(n_ind, n_vars, card, seed=2)
    opt = Optimizer(subset_fraction=1.0, evaluation_budget=None,
                    per_solution_budget=50, seed=7)
    counter = _Counter(_onemax)
    opt.optimize(pop, fit, counter, np.full(n_vars, card))
    # Every one of the 10 solutions may use up to 50 evaluations.
    assert counter.n <= 10 * 50


# ---------------------------------------------------------------------------
# 2. Subset control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
@pytest.mark.parametrize("fraction,expected", [(0.0, 0), (0.5, 15), (1.0, 30)])
def test_subset_fraction_counts(Optimizer, fraction, expected):
    n_ind, n_vars, card = 30, 10, 3
    pop, fit = _make_pop(n_ind, n_vars, card, seed=3)
    opt = Optimizer(subset_fraction=fraction, evaluation_budget=1500, seed=7)
    new_pop, new_fit = opt.optimize(pop, fit, _onemax, np.full(n_vars, card))
    changed = int(np.sum(np.any(new_pop != pop, axis=1)))
    assert changed <= expected                 # at most the selected fraction
    if fraction == 0.0:
        assert np.array_equal(new_pop, pop)    # disabled -> untouched


def test_subset_selection_best_targets_top_fitness():
    # With subset_selection='best', only the highest-fitness individuals may
    # change; the lowest-fitness ones must be left untouched.
    n_ind, n_vars, card = 20, 10, 2
    pop, fit = _make_pop(n_ind, n_vars, card, seed=4)
    opt = DeterministicHillClimber(subset_fraction=0.25, evaluation_budget=1000,
                                   subset_selection="best", seed=7)
    new_pop, _ = opt.optimize(pop, fit, _onemax, np.full(n_vars, card))
    changed = np.where(np.any(new_pop != pop, axis=1))[0]
    order = np.argsort(fit[:, 0])[::-1]
    top5 = set(order[:5].tolist())
    assert set(changed.tolist()).issubset(top5)


# ---------------------------------------------------------------------------
# 3. Never worsen + shape + reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
def test_never_worsens_and_shape(Optimizer):
    n_ind, n_vars, card = 25, 12, 4
    pop, fit = _make_pop(n_ind, n_vars, card, seed=5)
    opt = Optimizer(subset_fraction=0.8, evaluation_budget=3000, seed=7)
    new_pop, new_fit = opt.optimize(pop, fit, _onemax, np.full(n_vars, card))
    assert new_pop.shape == pop.shape
    assert new_fit.shape == fit.shape
    assert np.all(new_fit >= fit)                       # monotone (never worse)
    # Returned fitness must equal the true fitness of the returned solutions.
    recomputed = np.array([[_onemax(ind)] for ind in new_pop])
    assert np.allclose(recomputed, new_fit)


@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
def test_reproducible_with_seed(Optimizer):
    n_ind, n_vars, card = 20, 10, 3
    pop, fit = _make_pop(n_ind, n_vars, card, seed=6)
    a_pop, a_fit = Optimizer(subset_fraction=0.5, evaluation_budget=1500,
                             seed=123).optimize(pop, fit, _onemax, np.full(n_vars, card))
    b_pop, b_fit = Optimizer(subset_fraction=0.5, evaluation_budget=1500,
                             seed=123).optimize(pop, fit, _onemax, np.full(n_vars, card))
    assert np.array_equal(a_pop, b_pop)
    assert np.array_equal(a_fit, b_fit)


# ---------------------------------------------------------------------------
# 4. Actually improves (binary and non-binary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
@pytest.mark.parametrize("card", [2, 5])
def test_improves_separable_problem(Optimizer, card):
    n_ind, n_vars = 10, 15
    pop, fit = _make_pop(n_ind, n_vars, card, seed=8)
    opt = Optimizer(subset_fraction=1.0, evaluation_budget=6000, seed=7)
    _, new_fit = opt.optimize(pop, fit, _onemax, np.full(n_vars, card))
    # On OneMax the mean fitness of the population must strictly increase.
    assert new_fit.mean() > fit.mean()


# ---------------------------------------------------------------------------
# 5. Neighborhood primitives
# ---------------------------------------------------------------------------

def test_random_neighbor_changes_one_variable_to_different_value():
    rng = np.random.default_rng(0)
    card = np.array([2, 5, 3, 4])
    x = np.array([0, 2, 1, 3])
    for _ in range(50):
        y, var = random_neighbor(x, card, rng)
        assert np.sum(y != x) == 1                    # exactly one change
        assert y[var] != x[var]                       # to a different value
        assert 0 <= y[var] < card[var]


def test_shake_changes_exactly_k_variables():
    rng = np.random.default_rng(0)
    card = np.full(10, 4)
    x = rng.integers(0, 4, size=10)
    for k in (1, 3, 5):
        y = shake(x, k, card, rng)
        assert np.sum(y != x) == k


def test_best_improvement_descent_reaches_local_optimum():
    # On OneMax the descent must reach the all-max solution given enough budget.
    n_vars, card = 8, 3
    x = np.zeros(n_vars, dtype=int)
    ev = _BudgetEvaluator(_onemax, 10_000)
    bx, bf = best_improvement_descent(x, _onemax(x), ev, np.full(n_vars, card),
                                      np.random.default_rng(0))
    assert bf == n_vars * (card - 1)
    assert np.all(bx == card - 1)


# ---------------------------------------------------------------------------
# 6. End-to-end as EDA component (binary + non-binary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
@pytest.mark.parametrize("card", [2, 4])
def test_as_eda_component(Optimizer, card):
    from pateda import EDA, EDAComponents
    from pateda.seeding import RandomInit
    from pateda.learning import LearnUMDA
    from pateda.sampling import SampleFDA
    from pateda.selection import TruncationSelection
    from pateda.replacement import ElitistReplacement
    from pateda.stop_conditions import MaxGenerations

    n_vars = 12
    optimum = float(n_vars * (card - 1))

    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=120),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        local_opt=Optimizer(subset_fraction=0.5, evaluation_budget=1500, seed=1),
        stop_condition=MaxGenerations(max_gen=25),
    )
    eda = EDA(
        pop_size=120, n_vars=n_vars, fitness_func=_onemax,
        cardinality=np.full(n_vars, card), components=components, random_seed=42,
    )
    stats, _ = eda.run(verbose=False)
    # The hybrid must reach the optimum on this easy separable problem.
    assert stats.best_fitness_overall == optimum
