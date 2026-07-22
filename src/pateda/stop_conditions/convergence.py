"""
Population-convergence stopping condition.

Stops the EDA when the population has lost most of its diversity, i.e. when the
learned model can no longer produce meaningfully different samples. Detecting
this avoids spending evaluations on a converged model.
"""

from typing import Any
import numpy as np

from pateda.core.components import StopCondition


def population_diversity(population: np.ndarray) -> float:
    """A single scalar diversity measure in ``[0, 1]`` (roughly).

    * Discrete (integer) populations: ``1 - mean_i(freq of the most common value
      of variable i)``. It is ``0`` when every variable is fixed and grows as the
      per-variable value distributions spread out.
    * Continuous (float) populations: mean over variables of the standard
      deviation divided by the current per-variable range, giving a
      scale-free dispersion that tends to ``0`` as the population collapses.
    """
    pop = np.asarray(population)
    pop_size = pop.shape[0]
    if pop_size == 0:
        return 0.0

    if np.issubdtype(pop.dtype, np.integer):
        agreements = []
        for j in range(pop.shape[1]):
            _, counts = np.unique(pop[:, j], return_counts=True)
            agreements.append(counts.max() / pop_size)
        return float(1.0 - np.mean(agreements))

    # continuous
    pop = pop.astype(float)
    ranges = pop.max(axis=0) - pop.min(axis=0)
    stds = pop.std(axis=0)
    # avoid division by zero on collapsed variables
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ranges > 1e-12, stds / ranges, 0.0)
    return float(np.mean(rel))


class PopulationConvergence(StopCondition):
    """
    Stop when the population diversity drops below ``tol``.

    Parameters
    ----------
    tol : float
        Diversity threshold (see :func:`population_diversity`). Typical values
        are small, e.g. ``1e-2``.
    patience : int
        Require the diversity to stay below ``tol`` for this many consecutive
        generations before stopping (default 1, i.e. stop immediately).
    max_gen : int, optional
        Optional hard generation cap combined with the convergence test.
    """

    def __init__(self, tol: float = 1e-2, patience: int = 1,
                 max_gen=None):
        self.tol = float(tol)
        self.patience = int(patience)
        self.max_gen = max_gen
        self._below = 0
        self.last_diversity = None

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        if self.max_gen is not None and generation >= self.max_gen:
            return True

        div = population_diversity(population)
        self.last_diversity = div
        if div < self.tol:
            self._below += 1
        else:
            self._below = 0
        return self._below >= self.patience

    def reset(self) -> None:
        self._below = 0
        self.last_diversity = None
