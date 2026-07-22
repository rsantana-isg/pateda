"""
Composite stopping condition.

Combines several stopping conditions with an ``any`` (stop as soon as one fires)
or ``all`` (stop only when every condition fires) rule. This lets, for example,
a fixed generation budget be combined with stagnation or convergence tests.
"""

from typing import Any, List
import numpy as np

from pateda.core.components import StopCondition


class CompositeStop(StopCondition):
    """
    Combine several :class:`StopCondition` objects.

    Parameters
    ----------
    conditions : list of StopCondition
        The wrapped stopping conditions.
    mode : {"any", "all"}
        ``"any"`` stops when at least one condition is satisfied (logical OR),
        ``"all"`` stops only when every condition is satisfied (logical AND).

    Notes
    -----
    Every wrapped condition is evaluated on each call (no short-circuiting), so
    stateful conditions such as :class:`NoImprovement` keep an accurate history
    regardless of ``mode``.
    """

    def __init__(self, conditions: List[StopCondition], mode: str = "any"):
        if mode not in ("any", "all"):
            raise ValueError("mode must be 'any' or 'all'")
        if not conditions:
            raise ValueError("CompositeStop requires at least one condition")
        self.conditions = list(conditions)
        self.mode = mode

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        # evaluate all (no short-circuit) so stateful conditions stay consistent
        results = [
            c.should_stop(generation, population, fitness, **params)
            for c in self.conditions
        ]
        if self.mode == "any":
            return any(results)
        return all(results)

    def reset(self) -> None:
        for c in self.conditions:
            c.reset()
