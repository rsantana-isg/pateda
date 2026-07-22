"""
Multi-objective statistics tracker.

A :class:`~pateda.core.components.StatisticsMethod` that records set-quality
indicators of the current population every generation, so that multi-objective
runs report hypervolume (and optionally IGD) evolution without any change to the
EDA core. Pass it as ``EDAComponents(statistics=MultiObjectiveTracker(...))``;
the values land in ``stats.custom["hypervolume"]`` (and ``["igd"]``).
"""

from typing import Any, Dict, Optional
import numpy as np

from pateda.core.components import StatisticsMethod
from pateda.core.models import Model
from pateda.multiobjective.indicators import (
    hypervolume,
    igd,
    reference_point_from,
)


class MultiObjectiveTracker(StatisticsMethod):
    """
    Track hypervolume and IGD across generations.

    Parameters
    ----------
    reference_point : array-like, optional
        Fixed hypervolume reference point in the original objective space. If not
        given, it is derived once from the first generation's objective vectors
        (placed slightly worse than the observed nadir) and reused for every
        subsequent generation, so the hypervolume values are comparable across
        generations.
    reference_front : array-like, optional
        Known Pareto front, shape ``(r, m)``. When provided, the inverted
        generational distance (IGD) to it is also recorded; otherwise IGD is
        stored as ``nan``.
    maximize : bool
        Direction of optimisation (pateda maximizes, so the default is ``True``).
    """

    def __init__(self, reference_point=None, reference_front=None,
                 maximize: bool = True):
        self.reference_point = (
            None if reference_point is None else np.asarray(reference_point, float)
        )
        self.reference_front = (
            None if reference_front is None else np.asarray(reference_front, float)
        )
        self.maximize = maximize
        self._ref = self.reference_point

    def collect(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        model: Optional[Model] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        fit = np.atleast_2d(np.asarray(fitness, dtype=float))
        if fit.shape[1] < 2:
            # not a multi-objective run; nothing meaningful to track
            return {"hypervolume": float("nan"), "igd": float("nan")}

        if self._ref is None:
            self._ref = reference_point_from(fit, maximize=self.maximize)

        hv = hypervolume(fit, reference=self._ref, maximize=self.maximize)
        if self.reference_front is not None:
            ind = igd(fit, self.reference_front, maximize=self.maximize)
        else:
            ind = float("nan")
        return {"hypervolume": float(hv), "igd": float(ind)}
