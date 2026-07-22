"""
Stagnation-based stopping condition.

Stops the EDA when the best fitness has not improved by more than a tolerance
over a number of consecutive generations. This is a natural budget saver for
EDAs: once the model has converged, further generations rarely improve the
incumbent.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import StopCondition


def _scalar_best(fitness: np.ndarray) -> float:
    """Best (maximum) scalar value of a fitness array.

    For multi-objective fitness the mean over objectives is used as a scalar
    proxy, consistent with the aggregation used elsewhere in pateda.
    """
    fitness = np.asarray(fitness, dtype=float)
    if fitness.ndim == 2 and fitness.shape[1] > 1:
        return float(np.max(np.mean(fitness, axis=1)))
    return float(np.max(fitness))


class NoImprovement(StopCondition):
    """
    Stop after ``k`` generations without significant improvement.

    A generation counts as an improvement when the best fitness exceeds the best
    fitness seen so far by more than ``epsilon``. When ``k`` consecutive
    generations pass without such an improvement, :meth:`should_stop` returns
    ``True``.

    Parameters
    ----------
    k : int
        Number of consecutive stagnating generations tolerated.
    epsilon : float
        Minimum increase in best fitness that counts as an improvement.
    max_gen : int, optional
        Optional hard cap on generations, combined with the stagnation test
        (stop if either fires). Useful as a safety net.
    """

    def __init__(self, k: int = 20, epsilon: float = 1e-6,
                 max_gen: Optional[int] = None):
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.max_gen = max_gen
        self._best: Optional[float] = None
        self._stall = 0

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        if self.max_gen is not None and generation >= self.max_gen:
            return True

        current = _scalar_best(fitness)
        if self._best is None or current > self._best + self.epsilon:
            self._best = current
            self._stall = 0
        else:
            self._stall += 1

        return self._stall >= self.k

    def reset(self) -> None:
        self._best = None
        self._stall = 0
