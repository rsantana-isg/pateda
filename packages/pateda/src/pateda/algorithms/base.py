"""
Base classes and adapter classes for plug-and-play EDA wrappers.

This module provides:
- _BaseEDA: thin wrapper around a configured EDA instance
- _GMRFLearner / _GMRFSampler: adapters for the function-based GMRF-EDA API
- _VineLearner / _VineSampler: adapters for the function-based Vine copula API
"""

from typing import Any, Dict, Optional
import numpy as np

from pateda.core.components import LearningMethod, SamplingMethod
from pateda.core.models import Model
from pateda.learning.gmrf_eda import learn_gmrf_eda
from pateda.sampling.gmrf_eda import sample_gmrf_eda


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class _BaseEDA:
    """Thin wrapper that delegates run() to the underlying EDA instance."""

    def __init__(self, eda_instance):
        self._eda = eda_instance

    def set_weighting(self, weighting: str, beta: float = 1.0) -> "_BaseEDA":
        """
        Configure *customized selection* for this algorithm.

        Controls how the selected individuals are weighted when the
        probabilistic model is learned each generation:

        - ``"uniform"``      : every selected solution has weight ``1/N``
          (classic EDA behaviour, the default).
        - ``"proportional"`` : weight proportional to (shifted) fitness.
        - ``"boltzmann"``    : weight proportional to ``exp(beta * z)`` with
          ``z`` the standardised fitness; ``beta`` is the inverse-temperature.

        Returns ``self`` to allow chaining (e.g. ``UMDA(...).set_weighting(...)``).
        """
        self._eda.selection_weighting = weighting
        self._eda.weighting_beta = beta
        return self

    def run(self, verbose=False):
        return self._eda.run(verbose=verbose)


# ---------------------------------------------------------------------------
# GMRF adapters
# ---------------------------------------------------------------------------

class _GMRFLearner(LearningMethod):
    """Adapter wrapping learn_gmrf_eda as a LearningMethod."""

    def __init__(self, regularization: str = 'lasso'):
        self.regularization = regularization

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Dict[str, Any]:
        gmrf_params = {'regularization': self.regularization}
        gmrf_params.update(params)
        return learn_gmrf_eda(population, fitness, gmrf_params)


class _GMRFSampler(SamplingMethod):
    """Adapter wrapping sample_gmrf_eda as a SamplingMethod."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def sample(
        self,
        n_vars: int,
        model: Any,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        bounds = cardinality  # shape (2, n_vars)
        return sample_gmrf_eda(model, self.n_samples, bounds=bounds)


# ---------------------------------------------------------------------------
# Vine copula adapters
# ---------------------------------------------------------------------------

class _VineLearner(LearningMethod):
    """Adapter wrapping learn_vine_copula_auto as a LearningMethod."""

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Dict[str, Any]:
        try:
            from pateda.learning.vine_copula import learn_vine_copula_auto
        except ImportError:
            raise ImportError(
                "pyvinecopulib is required for VineEDA. "
                "Install it with: pip install pyvinecopulib"
            )
        return learn_vine_copula_auto(population, fitness, params or None)


class _VineSampler(SamplingMethod):
    """Adapter wrapping sample_vine_copula as a SamplingMethod."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def sample(
        self,
        n_vars: int,
        model: Any,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        try:
            from pateda.sampling.vine_copula import sample_vine_copula
        except ImportError:
            raise ImportError(
                "pyvinecopulib is required for VineEDA. "
                "Install it with: pip install pyvinecopulib"
            )
        return sample_vine_copula(model, self.n_samples, bounds=cardinality, rng=rng)
