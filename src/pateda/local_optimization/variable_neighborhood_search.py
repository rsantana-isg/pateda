"""
Variable neighborhood search (VNS) local search for EDAs (discrete problems)

Two VNS variants sharing the common interface of
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.

VNS (Hansen & Mladenović) is built on a *systematic change of neighborhood*.
For discrete problems the ``k``-th neighborhood ``N_k(x)`` is the set of
solutions at Hamming distance ``k`` from ``x`` (``k`` variables changed).  A
*shake* draws a random point of ``N_k(x)`` to escape the current basin, after
which the neighborhood index ``k`` grows when no improvement is found and is
reset to 1 whenever a better solution is reached (the ``NeighborhoodChange``
rule).

- :class:`VariableNeighborhoodSearch` -- **Basic VNS** (BVNS): each cycle
  shakes the incumbent in ``N_k`` and then runs a local descent from the shaken
  point; the result replaces the incumbent only if it is better, otherwise
  ``k`` is incremented (up to ``k_max``).  The descent reuses the very same
  single-change hill climber as
  :class:`~pateda.local_optimization.hill_climbing.DeterministicHillClimber`
  (best improvement) or its first-improvement variant.

- :class:`ReducedVariableNeighborhoodSearch` -- **Reduced VNS** (RVNS): the
  same scheme *without* the descent -- only the shaken point is compared to the
  incumbent.  Useful when evaluations are expensive, as a cheap stochastic
  search closer to Monte-Carlo with growing step size.

Both optimize only a fraction of the sampled population and share a total
evaluation budget; see
:class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.

References
----------
- Hansen, P., & Mladenović, N. (2003, 2025). "Variable Neighborhood Search."
  Handbook of Metaheuristics (basic VNS, reduced VNS, neighborhood change).
- Mladenović, N., & Hansen, P. (1997). "Variable neighborhood search."
  Computers & Operations Research 24(11), 1097-1100.
"""

from typing import Optional, Tuple
import numpy as np

from pateda.local_optimization.budgeted_search import (
    BudgetedLocalSearch,
    _BudgetEvaluator,
    best_improvement_descent,
    first_improvement_descent,
    shake,
)


class VariableNeighborhoodSearch(BudgetedLocalSearch):
    """Basic VNS (BVNS): shake in growing neighborhoods + local descent."""

    def __init__(
        self,
        k_max: int = 3,
        local_search: str = "best",
        subset_fraction: float = 1.0,
        evaluation_budget: Optional[int] = None,
        per_solution_budget: int = 300,
        subset_selection: str = "best",
        seed: Optional[int] = None,
    ):
        """
        Args:
            k_max: Largest neighborhood index used for shaking (the maximum
                number of variables perturbed at once).  ``k`` cycles from 1 to
                ``k_max``.
            local_search: Descent used after each shake -- ``"best"``
                (steepest ascent) or ``"first"`` (first improvement).
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
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")
        if local_search not in ("best", "first"):
            raise ValueError(
                f"local_search must be 'best' or 'first', got {local_search!r}"
            )
        self.k_max = k_max
        self.local_search = local_search

    def _descend(self, x, fx, evaluator, cardinality, rng):
        if self.local_search == "best":
            return best_improvement_descent(x, fx, evaluator, cardinality, rng)
        return first_improvement_descent(x, fx, evaluator, cardinality, rng)

    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        # Initial descent to a local optimum (as in the basic VNS scheme).
        best_x, best_f = self._descend(x, fx, evaluator, cardinality, rng)

        k_max = min(self.k_max, best_x.shape[0])
        while not evaluator.exhausted:
            k = 1
            while k <= k_max and not evaluator.exhausted:
                shaken = shake(best_x, k, cardinality, rng)
                shaken_f = evaluator(shaken)
                cand_x, cand_f = self._descend(
                    shaken, shaken_f, evaluator, cardinality, rng
                )
                if cand_f > best_f:                 # NeighborhoodChange: improve
                    best_x, best_f = cand_x, cand_f
                    k = 1
                else:
                    k += 1                          # ... else widen the shake
        return best_x, best_f


class ReducedVariableNeighborhoodSearch(BudgetedLocalSearch):
    """Reduced VNS (RVNS): shake in growing neighborhoods, no local descent."""

    def __init__(
        self,
        k_max: int = 3,
        subset_fraction: float = 1.0,
        evaluation_budget: Optional[int] = None,
        per_solution_budget: int = 300,
        subset_selection: str = "best",
        seed: Optional[int] = None,
    ):
        """
        Args:
            k_max: Largest neighborhood index used for shaking.
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
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")
        self.k_max = k_max

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
        k_max = min(self.k_max, best_x.shape[0])
        while not evaluator.exhausted:
            k = 1
            while k <= k_max and not evaluator.exhausted:
                shaken = shake(best_x, k, cardinality, rng)
                shaken_f = evaluator(shaken)
                if shaken_f > best_f:
                    best_x, best_f = shaken, shaken_f
                    k = 1
                else:
                    k += 1
        return best_x, best_f
