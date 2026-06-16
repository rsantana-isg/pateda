"""
Adapter classes bridging permutation learners/samplers that expose only
``__call__`` into the pateda ``LearningMethod`` / ``SamplingMethod`` contract.

``LearnMallowsKendall`` already has ``.learn()``, but ``LearnMallowsCayley``,
``LearnGeneralizedMallowsKendall`` and ``LearnGeneralizedMallowsCayley`` (and
their samplers) are simpler call-only objects.  These adapters let any of them
be plugged into a generic :class:`pateda.core.eda.EDA` component pipeline.
"""

from typing import Any, Dict, Optional
import numpy as np

from pateda.core.components import LearningMethod, SamplingMethod


class _MallowsLearnerAdapter(LearningMethod):
    """
    Wraps a Mallows learner that only has ``__call__`` into a LearningMethod.

    ``LearnMallowsKendall`` already has ``.learn()``, but ``LearnMallowsCayley``,
    ``LearnGeneralizedMallowsKendall``, and ``LearnGeneralizedMallowsCayley`` do
    not.  This adapter handles all of them uniformly.
    """

    def __init__(self, learner_instance):
        self._learner = learner_instance

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Dict[str, Any]:
        # Check if .learn() is defined on the instance
        if hasattr(self._learner, "learn"):
            return self._learner.learn(
                generation=generation,
                n_vars=n_vars,
                cardinality=cardinality,
                population=population,
                fitness=fitness,
                **params,
            )
        # Fall back to __call__
        return self._learner(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_pop=population,
            selected_fitness=fitness,
        )


class _MallowsSamplerAdapter(SamplingMethod):
    """
    Wraps a Mallows sampler that only has ``__call__`` into a SamplingMethod.

    ``SampleMallowsKendall`` already has ``.sample()``, but ``SampleMallowsCayley``,
    ``SampleGeneralizedMallowsKendall``, ``SampleGeneralizedMallowsCayley`` do not.
    """

    def __init__(self, sampler_instance, n_samples: int):
        self._sampler = sampler_instance
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
        pop = aux_pop if aux_pop is not None else np.array([])
        fit = aux_fitness if aux_fitness is not None else np.array([])
        # Check if .sample() is defined
        if hasattr(self._sampler, "sample"):
            return self._sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                population=pop,
                fitness=fit,
                sample_size=self.n_samples,
                rng=rng,
            )
        # Fall back to __call__
        return self._sampler(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=pop,
            fitness=fit,
            sample_size=self.n_samples,
            rng=rng,
        )
