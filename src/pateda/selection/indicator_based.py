"""
Indicator-based selection (the metric / indicator paradigm).

This selection method scores solutions with a quality *indicator* rather than
with raw dominance, so a single scalar fitness reflects both convergence and
diversity.  It plugs into the standard EDA loop and therefore turns any pateda
probabilistic model into an indicator-based MOEDA (IBEA / SMS-EMOA style).

Two indicators are supported:

* ``"epsilon"`` (default) -- the binary additive epsilon indicator with the
  adaptive IBEA fitness assignment and iterative worst-removal environmental
  selection (Zitzler & Kunzli, 2004).
* ``"hypervolume"`` -- greedy selection by exclusive hypervolume contribution
  (SMS-EMOA style); the least-contributing solution is removed repeatedly until
  ``n_select`` remain.
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod
from pateda.multiobjective.indicators import (
    additive_epsilon_matrix,
    hypervolume_contributions,
    reference_point_from,
)

__all__ = ["IndicatorBasedSelection"]


class IndicatorBasedSelection(SelectionMethod):
    """Select individuals by a quality indicator (IBEA / SMS-EMOA style)."""

    def __init__(
        self,
        ratio: float = 0.5,
        n_select: Optional[int] = None,
        maximize: bool = True,
        indicator: str = "epsilon",
        kappa: float = 0.05,
        reference: Optional[np.ndarray] = None,
    ):
        """
        Args:
            ratio: Fraction of the population to select (used if ``n_select`` is
                ``None``).
            n_select: Exact number to select (overrides ``ratio``).
            maximize: Direction of optimisation.
            indicator: ``"epsilon"`` (IBEA) or ``"hypervolume"`` (SMS-EMOA).
            kappa: Scaling factor for the IBEA exponential fitness.
            reference: Optional HV reference point (``"hypervolume"`` only).
        """
        if indicator not in ("epsilon", "hypervolume"):
            raise ValueError("indicator must be 'epsilon' or 'hypervolume'")
        self.ratio = ratio
        self.n_select = n_select
        self.maximize = maximize
        self.indicator = indicator
        self.kappa = kappa
        self.reference = reference

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pop_size = population.shape[0]
        maximize = params.get("maximize", self.maximize)

        if n_select is None:
            n_select = self.n_select
        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))
        n_select = min(n_select, pop_size)

        if fitness.ndim == 1:
            fitness = fitness.reshape(-1, 1)

        # Single objective: simple fitness ordering.
        if fitness.shape[1] == 1:
            order = np.argsort(fitness[:, 0])
            if maximize:
                order = order[::-1]
            selected = order[:n_select]
            return population[selected], fitness[selected]

        if self.indicator == "epsilon":
            keep = self._ibea_environmental_selection(fitness, n_select, maximize)
        else:
            keep = self._hv_greedy_selection(fitness, n_select, maximize)

        return population[keep], fitness[keep]

    # ------------------------------------------------------------------
    # IBEA (additive epsilon)
    # ------------------------------------------------------------------

    def _ibea_environmental_selection(
        self, fitness: np.ndarray, n_select: int, maximize: bool
    ) -> np.ndarray:
        """Iterative worst-removal with adaptive IBEA fitness updates."""
        indicator = additive_epsilon_matrix(fitness, maximize=maximize)
        c = np.max(np.abs(indicator))
        if c < 1e-12:
            c = 1.0
        # contrib[j, i] = -exp(-I(j, i) / (kappa * c))  (j's effect on i)
        contrib = -np.exp(-indicator / (self.kappa * c))
        np.fill_diagonal(contrib, 0.0)
        fit = contrib.sum(axis=0)  # larger is better

        alive = np.ones(fitness.shape[0], dtype=bool)
        n_alive = alive.sum()
        while n_alive > n_select:
            alive_idx = np.where(alive)[0]
            worst = alive_idx[np.argmin(fit[alive_idx])]
            alive[worst] = False
            # Remove worst's contribution from the remaining solutions' fitness.
            fit[alive] -= contrib[worst, alive]
            n_alive -= 1
        return np.where(alive)[0]

    # ------------------------------------------------------------------
    # Hypervolume contribution (SMS-EMOA style)
    # ------------------------------------------------------------------

    def _hv_greedy_selection(
        self, fitness: np.ndarray, n_select: int, maximize: bool
    ) -> np.ndarray:
        """Repeatedly drop the least-contributing solution."""
        reference = self.reference
        if reference is None:
            reference = reference_point_from(fitness, maximize=maximize)

        alive = np.ones(fitness.shape[0], dtype=bool)
        while alive.sum() > n_select:
            alive_idx = np.where(alive)[0]
            contrib = hypervolume_contributions(
                fitness[alive_idx], reference=reference, maximize=maximize
            )
            worst = alive_idx[int(np.argmin(contrib))]
            alive[worst] = False
        return np.where(alive)[0]
