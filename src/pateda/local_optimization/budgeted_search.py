"""
Budgeted local search for EDAs — shared infrastructure

This module provides the common machinery for the memetic (hybrid) local search
components in pateda: hill climbers, simulated annealing and variable
neighborhood search.  They are meant to be used as the ``local_opt`` component
of an :class:`~pateda.core.eda.EDA`, i.e. applied to the population that was
just sampled, before replacement.

All optimizers share one interface, deliberately, so that they are
interchangeable in an EDA pipeline.  Two parameters, common to every optimizer,
control the *intensity* of the local search exactly as requested by the memetic
EDA literature (see e.g. Radetic, Pelikan & Goldberg, 2009, where a
deterministic hill climber is applied to each sampled string with probability
``p_dhc`` and a bounded number of flips):

- ``subset_fraction`` -- the fraction of the sampled solutions the local search
  is applied to (0 = disabled, 1 = whole population).  Which solutions are
  chosen is controlled by ``subset_selection`` (``"best"``, ``"random"`` or
  ``"worst"``).
- ``evaluation_budget`` -- the *total* number of fitness evaluations the local
  search may spend in one call (one generation), shared equally among the
  selected solutions.  If ``None``, ``per_solution_budget`` is used per selected
  solution instead.

Because every optimizer honours the same two knobs and the same
:class:`~pateda.core.components.LocalOptMethod` contract, switching optimizer in
an EDA is a one-line change, and the cost of the local search is directly
comparable across methods.

Design
------
:class:`BudgetedLocalSearch` implements :meth:`optimize` once: it selects the
subset, splits the evaluation budget, wraps the fitness function in a
:class:`_BudgetEvaluator` that stops the search when the per-solution budget is
exhausted, and writes the (never worsened) results back into the population.
Each concrete optimizer only implements :meth:`_optimize_one`, the search
kernel for a single solution.  The neighborhood primitives shared by the
kernels (random neighbor, shake, best-/first-improvement descent) are provided
as module-level functions so that, for instance, variable neighborhood search
can reuse the very same descent as the hill climber.

The problems targeted are **discrete** (binary or integer): a solution is a
vector of integers with ``x[i]`` in ``[0, cardinality[i])`` and a *move* changes
one variable to another value of its domain.
"""

from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple
import numpy as np

from pateda.core.components import LocalOptMethod


# ---------------------------------------------------------------------------
# Evaluation-budget bookkeeping
# ---------------------------------------------------------------------------

class _BudgetEvaluator:
    """Fitness wrapper that counts evaluations and enforces a hard budget.

    A kernel calls the evaluator like the plain fitness function; it must check
    :attr:`exhausted` *before* every call so the budget is never exceeded.
    """

    __slots__ = ("func", "budget", "n_evals")

    def __init__(self, func: Callable, budget: int):
        self.func = func
        self.budget = int(budget)
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        return float(self.func(x))

    @property
    def exhausted(self) -> bool:
        return self.n_evals >= self.budget

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_evals)


# ---------------------------------------------------------------------------
# Neighborhood primitives (discrete moves) — shared by every kernel
# ---------------------------------------------------------------------------

def random_neighbor(
    x: np.ndarray, cardinality: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    """Return a copy of ``x`` with one random variable set to a *different*
    value of its domain, together with the index of the changed variable."""
    y = x.copy()
    n = y.shape[0]
    var = int(rng.integers(0, n))
    card = int(cardinality[var])
    if card > 1:
        # Draw uniformly among the card-1 values different from the current one.
        offset = int(rng.integers(1, card))
        y[var] = (int(y[var]) + offset) % card
    return y, var


def shake(
    x: np.ndarray, k: int, cardinality: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return a random point of the ``k``-th neighborhood of ``x``: exactly
    ``k`` distinct variables changed to a different value each (Hamming
    distance ``k``).  Used by variable neighborhood search."""
    y = x.copy()
    n = y.shape[0]
    k = int(min(k, n))
    if k <= 0:
        return y
    variables = rng.choice(n, size=k, replace=False)
    for var in variables:
        card = int(cardinality[var])
        if card > 1:
            offset = int(rng.integers(1, card))
            y[var] = (int(y[var]) + offset) % card
    return y


def best_improvement_descent(
    x: np.ndarray,
    fx: float,
    evaluator: _BudgetEvaluator,
    cardinality: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Steepest-ascent descent in the single-change neighborhood.

    At each step every neighbor that differs from the current solution in one
    variable is evaluated and the single *best-improving* change is applied;
    the search stops at a local optimum or when the budget runs out.  For
    binary variables this is exactly the deterministic hill climber (DHC) of
    Radetic et al. (2009): "flip the bit that produces the maximum improvement".
    """
    best_x = x.copy()
    best_f = float(fx)
    n = best_x.shape[0]

    while not evaluator.exhausted:
        move_var, move_val, move_f = -1, -1, best_f
        interrupted = False
        for var in rng.permutation(n):            # random var order: fair ties
            card = int(cardinality[var])
            current = int(best_x[var])
            for val in range(card):
                if val == current:
                    continue
                if evaluator.exhausted:
                    interrupted = True
                    break
                cand = best_x.copy()
                cand[var] = val
                f = evaluator(cand)
                if f > move_f:
                    move_f, move_var, move_val = f, int(var), val
            if interrupted:
                break

        if move_var >= 0 and move_f > best_f:
            best_x[move_var] = move_val
            best_f = move_f
        else:
            break                                 # local optimum (or no gain)

    return best_x, best_f


def first_improvement_descent(
    x: np.ndarray,
    fx: float,
    evaluator: _BudgetEvaluator,
    cardinality: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """First-improvement descent in the single-change neighborhood.

    Neighbors are scanned in random order and the *first* improving change is
    applied immediately (Algorithm 6, "first descent", of Hansen &
    Mladenović).  Cheaper per improvement than steepest ascent."""
    best_x = x.copy()
    best_f = float(fx)
    n = best_x.shape[0]

    while not evaluator.exhausted:
        # All single-variable changes away from the current solution.
        moves = [
            (var, val)
            for var in range(n)
            for val in range(int(cardinality[var]))
            if val != int(best_x[var])
        ]
        if not moves:
            break
        order = rng.permutation(len(moves))
        improved = False
        for idx in order:
            if evaluator.exhausted:
                break
            var, val = moves[idx]
            cand = best_x.copy()
            cand[var] = val
            f = evaluator(cand)
            if f > best_f:
                best_x, best_f, improved = cand, f, True
                break
        if not improved:
            break

    return best_x, best_f


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BudgetedLocalSearch(LocalOptMethod):
    """
    Base class for the memetic local-search components.

    Subclasses implement :meth:`_optimize_one` (the per-solution kernel); this
    class provides the uniform :meth:`optimize` that every optimizer shares:
    subset selection, evaluation-budget splitting and write-back.

    All optimizers therefore accept the same intensity-control parameters
    (documented on :meth:`__init__`), which makes them fully interchangeable in
    an EDA pipeline and their costs directly comparable.
    """

    def __init__(
        self,
        subset_fraction: float = 1.0,
        evaluation_budget: Optional[int] = None,
        per_solution_budget: int = 100,
        subset_selection: str = "best",
        seed: Optional[int] = None,
    ):
        """
        Args:
            subset_fraction: Fraction of the sampled solutions the local search
                is applied to, in ``[0, 1]``.  ``0`` disables the search.
            evaluation_budget: Total number of fitness evaluations the search
                may spend per call, shared equally among the selected solutions.
                ``None`` falls back to ``per_solution_budget`` per solution.
            per_solution_budget: Evaluations allowed per selected solution when
                ``evaluation_budget`` is ``None``.
            subset_selection: Which solutions to optimize -- ``"best"`` (elitist
                intensification, default), ``"random"`` (reproduces the
                ``p_dhc`` semantics of the DHC paper) or ``"worst"`` (repair).
            seed: Seed for this optimizer's private random generator (``None``
                for a fresh non-deterministic generator).  Kept separate from
                the EDA's generator, matching the existing local searches.
        """
        if not 0.0 <= subset_fraction <= 1.0:
            raise ValueError(
                f"subset_fraction must be in [0, 1], got {subset_fraction}"
            )
        if subset_selection not in ("best", "random", "worst"):
            raise ValueError(
                "subset_selection must be 'best', 'random' or 'worst', "
                f"got {subset_selection!r}"
            )
        self.subset_fraction = subset_fraction
        self.evaluation_budget = evaluation_budget
        self.per_solution_budget = per_solution_budget
        self.subset_selection = subset_selection
        self.rng = np.random.default_rng(seed)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _coerce_cardinality(cardinality: np.ndarray, n_vars: int) -> np.ndarray:
        card = np.asarray(cardinality)
        if card.ndim == 2:                        # continuous [min; max] format
            card = (card[1, :] - card[0, :])
        return card.astype(int).reshape(n_vars)

    @staticmethod
    def _select_indices(
        fitness_1d: np.ndarray,
        n_select: int,
        selection: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if selection == "best":
            return np.argsort(fitness_1d)[::-1][:n_select]
        if selection == "worst":
            return np.argsort(fitness_1d)[:n_select]
        return rng.choice(fitness_1d.shape[0], size=n_select, replace=False)

    # -- the shared optimize() -----------------------------------------
    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the local search to a subset of ``population``.

        Args:
            population: Sampled population ``(n_individuals, n_vars)`` of
                integers with ``x[i] in [0, cardinality[i])``.
            fitness: Current fitness values, shape ``(n_individuals,)`` or
                ``(n_individuals, 1)`` (single-objective).
            fitness_func: Scalar fitness function evaluated on a 1-D solution.
            cardinality: Variable cardinalities ``(n_vars,)``.
            **params: Per-call overrides of any of the intensity parameters
                (``subset_fraction``, ``evaluation_budget``,
                ``per_solution_budget``, ``subset_selection``).

        Returns:
            ``(new_population, new_fitness)`` with the optimized solutions
            written back; non-selected solutions are unchanged, and no solution
            is ever replaced by a worse one.
        """
        pop = np.asarray(population)
        fit = np.asarray(fitness, dtype=float)
        n_ind, n_vars = pop.shape

        subset_fraction = params.get("subset_fraction", self.subset_fraction)
        evaluation_budget = params.get("evaluation_budget", self.evaluation_budget)
        per_solution_budget = params.get("per_solution_budget", self.per_solution_budget)
        subset_selection = params.get("subset_selection", self.subset_selection)

        card = self._coerce_cardinality(cardinality, n_vars)
        fit_1d = fit[:, 0] if fit.ndim == 2 else fit

        n_select = int(round(subset_fraction * n_ind))
        n_select = max(0, min(n_ind, n_select))
        new_pop = pop.copy()
        new_fit = fit.copy()
        if n_select == 0:
            return new_pop, new_fit

        indices = self._select_indices(fit_1d, n_select, subset_selection, self.rng)

        if evaluation_budget is not None:
            per_budget = max(1, int(evaluation_budget) // n_select)
        else:
            per_budget = max(1, int(per_solution_budget))

        for i in indices:
            x = pop[i].astype(int).copy()
            fx = float(fit_1d[i])
            evaluator = _BudgetEvaluator(fitness_func, per_budget)
            best_x, best_f = self._optimize_one(x, fx, evaluator, card, self.rng)
            if best_f >= fx:                       # guard: never worsen
                new_pop[i] = best_x
                if new_fit.ndim == 2:
                    new_fit[i, 0] = best_f
                else:
                    new_fit[i] = best_f

        return new_pop, new_fit

    # -- kernel to implement -------------------------------------------
    @abstractmethod
    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        """Optimize a single solution ``x`` (fitness ``fx``) within the budget
        carried by ``evaluator``; return the best solution found and its
        fitness (never worse than ``fx``)."""
        raise NotImplementedError
