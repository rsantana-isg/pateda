"""
Simulated annealing local search for EDAs (discrete problems)

A budget/subset-aware simulated annealing optimizer sharing the common
interface of
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.

Simulated annealing (Kirkpatrick, Gelatt & Vecchi, 1983) extends local search
with the Metropolis acceptance rule: a proposed single-change neighbor is
accepted with probability ``min(1, exp(Delta / T))`` (for maximization,
``Delta = f(neighbor) - f(current)``), where the temperature ``T`` is lowered
along a cooling schedule.  High temperatures accept many worsening moves
(exploration); as ``T`` falls the search behaves like a hill climber
(exploitation).  The best solution ever visited is returned.

Cooling is stretched over the *whole per-solution evaluation budget* so that
the temperature reaches ``final_temp`` exactly as the budget is spent,
regardless of how large the budget is.  This makes the ``evaluation_budget``
knob (shared with every other optimizer) the single control of search length.

Variants (selected with the ``cooling`` argument):

- ``"geometric"`` -- ``T = T0 * (Tf / T0) ** (step / total_steps)`` (the
  classic geometric / exponential schedule);
- ``"linear"`` -- ``T = T0 - (T0 - Tf) * step / total_steps``.

``auto_temp`` (default ``True``) sets the initial temperature automatically from
the problem: a handful of random moves are sampled and ``T0`` is chosen so that
a worsening move of average magnitude is accepted with probability
``init_accept_prob`` at the start, following the standard warm-up rule
``T0 = mean|Delta| / -ln(init_accept_prob)``.

References
----------
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by
  Simulated Annealing." Science 220(4598), 671-680.
- Delahaye, D., Chaimatanan, S., & Mongeau, M. (2019). "Simulated Annealing:
  From Basics to Applications." Handbook of Metaheuristics.
"""

from typing import Optional, Tuple
import numpy as np

from pateda.local_optimization.budgeted_search import (
    BudgetedLocalSearch,
    _BudgetEvaluator,
    random_neighbor,
)


class SimulatedAnnealing(BudgetedLocalSearch):
    """Metropolis simulated annealing over the single-change neighborhood.

    Shares the subset / evaluation-budget interface of
    :class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.
    """

    def __init__(
        self,
        initial_temp: Optional[float] = None,
        final_temp: float = 1e-3,
        cooling: str = "geometric",
        auto_temp: bool = True,
        init_accept_prob: float = 0.8,
        subset_fraction: float = 1.0,
        evaluation_budget: Optional[int] = None,
        per_solution_budget: int = 200,
        subset_selection: str = "best",
        seed: Optional[int] = None,
    ):
        """
        Args:
            initial_temp: Initial temperature ``T0``.  Ignored when
                ``auto_temp`` is ``True``; required (``> final_temp``) otherwise.
            final_temp: Final temperature ``Tf`` reached as the budget is spent.
            cooling: Cooling schedule, ``"geometric"`` or ``"linear"``.
            auto_temp: If ``True`` (default) estimate ``T0`` from the problem.
            init_accept_prob: Target acceptance probability of an
                average-magnitude worsening move at the start, used by
                ``auto_temp``.
            subset_fraction, evaluation_budget, per_solution_budget,
            subset_selection, seed: See
                :class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.
        """
        super().__init__(
            subset_fraction=subset_fraction,
            evaluation_budget=evaluation_budget,
            per_solution_budget=per_solution_budget,
            subset_selection=subset_selection,
            seed=seed,
        )
        if cooling not in ("geometric", "linear"):
            raise ValueError(
                f"cooling must be 'geometric' or 'linear', got {cooling!r}"
            )
        if not auto_temp:
            if initial_temp is None:
                raise ValueError("initial_temp is required when auto_temp=False")
            if initial_temp <= final_temp:
                raise ValueError(
                    f"initial_temp ({initial_temp}) must be > final_temp ({final_temp})"
                )
        if not 0.0 < init_accept_prob < 1.0:
            raise ValueError(
                f"init_accept_prob must be in (0, 1), got {init_accept_prob}"
            )
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling = cooling
        self.auto_temp = auto_temp
        self.init_accept_prob = init_accept_prob

    def _estimate_initial_temp(
        self, x, fx, evaluator, cardinality, rng, n_probe
    ) -> Tuple[float, np.ndarray, float]:
        """Warm-up: sample up to ``n_probe`` random moves, return an initial
        temperature plus the best (solution, fitness) seen during probing."""
        deltas = []
        best_x, best_f = x.copy(), float(fx)
        cur_f = float(fx)
        for _ in range(n_probe):
            if evaluator.exhausted:
                break
            neighbor, _ = random_neighbor(x, cardinality, rng)
            f = evaluator(neighbor)
            deltas.append(abs(f - cur_f))
            if f > best_f:
                best_x, best_f = neighbor.copy(), f
        mean_abs = float(np.mean(deltas)) if deltas else 0.0
        if mean_abs <= 0.0:
            mean_abs = 1.0                          # flat probe -> neutral T0
        t0 = mean_abs / (-np.log(self.init_accept_prob))
        return max(t0, self.final_temp * 10.0), best_x, best_f

    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        best_x = x.copy()
        best_f = float(fx)

        if self.auto_temp:
            n_probe = max(1, min(evaluator.budget // 10, 30))
            t0, best_x, best_f = self._estimate_initial_temp(
                x, fx, evaluator, cardinality, rng, n_probe
            )
        else:
            t0 = float(self.initial_temp)
        tf = float(self.final_temp)

        cur_x = best_x.copy()
        cur_f = best_f
        # Number of annealing steps left after any warm-up probing.
        total_steps = max(1, evaluator.remaining)
        step = 0
        while not evaluator.exhausted:
            frac = step / max(1, total_steps - 1)
            frac = min(1.0, frac)
            if self.cooling == "geometric":
                temp = t0 * (tf / t0) ** frac
            else:
                temp = t0 - (t0 - tf) * frac
            temp = max(temp, 1e-12)

            neighbor, _ = random_neighbor(cur_x, cardinality, rng)
            f = evaluator(neighbor)
            delta = f - cur_f
            if delta > 0 or rng.random() < np.exp(delta / temp):
                cur_x, cur_f = neighbor, f
                if cur_f > best_f:
                    best_x, best_f = neighbor.copy(), f
            step += 1

        return best_x, best_f
