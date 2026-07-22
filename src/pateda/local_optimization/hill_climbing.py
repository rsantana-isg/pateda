"""
Hill-climbing local search variants for EDAs (discrete problems)

Three budget/subset-aware hill climbers sharing the common interface of
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.  They
differ only in how they explore the single-change neighborhood (change one
variable to another value of its domain):

- :class:`DeterministicHillClimber` -- *steepest ascent*.  Every single-change
  neighbor is evaluated and the best-improving change is applied.  For binary
  variables this is exactly the deterministic hill climber (DHC) of
  Radetic, Pelikan & Goldberg (2009): "flip the bit that produces the maximum
  improvement in fitness", iterated until no single change improves.
- :class:`FirstImprovementHillClimber` -- *first improvement*.  Neighbors are
  scanned in random order and the first improving change is applied at once
  (cheaper per step; "first descent" of Hansen & Mladenović).
- :class:`StochasticHillClimber` -- *random-mutation hill climbing* (RMHC,
  Forrest & Mitchell, 1993).  A single random change is proposed each step and
  accepted iff it does not worsen the fitness (equal fitness is accepted so
  plateaus can be traversed).

All three optimize only a fraction of the sampled population and share a total
evaluation budget; see
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.

References
----------
- Radetic, E., Pelikan, M., & Goldberg, D. E. (2009). "Effects of a
  Deterministic Hill Climber on hBOA." GECCO 2009, pp. 437-444.
- Hansen, P., & Mladenović, N. (2003). "Variable Neighborhood Search."
  (best/first improvement descent.)
- Forrest, S., & Mitchell, M. (1993). "Relative building-block fitness and the
  building-block hypothesis." FOGA 2 (random-mutation hill climbing).
"""

from typing import Tuple
import numpy as np

from pateda.local_optimization.budgeted_search import (
    BudgetedLocalSearch,
    _BudgetEvaluator,
    best_improvement_descent,
    first_improvement_descent,
    random_neighbor,
)


class DeterministicHillClimber(BudgetedLocalSearch):
    """Steepest-ascent single-change hill climber (the DHC of Radetic et al.).

    At each step the whole single-change neighborhood is evaluated and the
    best-improving move is applied; the search stops at a local optimum or when
    the budget is exhausted.  Deterministic given the current solution."""

    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        return best_improvement_descent(x, fx, evaluator, cardinality, rng)


class FirstImprovementHillClimber(BudgetedLocalSearch):
    """First-improvement single-change hill climber.

    Scans neighbors in random order and applies the first improving move,
    which reaches a local optimum with fewer evaluations per step than steepest
    ascent when improving moves are common."""

    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        return first_improvement_descent(x, fx, evaluator, cardinality, rng)


class StochasticHillClimber(BudgetedLocalSearch):
    """Random-mutation hill climbing (RMHC).

    Each step proposes a single random change and accepts it iff the fitness
    does not decrease.  Unlike the descent climbers it does not scan the whole
    neighborhood, so it spends its whole budget taking many cheap steps; equal
    fitness is accepted, allowing neutral drift across plateaus."""

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
        while not evaluator.exhausted:
            neighbor, _ = random_neighbor(best_x, cardinality, rng)
            f = evaluator(neighbor)
            if f >= best_f:                        # accept non-worsening moves
                best_x, best_f = neighbor, f
        return best_x, best_f
