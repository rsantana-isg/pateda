"""
Sampling from regularized k-order Markov models (MkRg-EDA)

Generation step of the EDA with regularized k-order Markov models (Santana,
Karshenas, Bielza & Larrañaga, 2011); the model is produced by
:class:`~pateda.learning.regularized_markov.LearnRegularizedMarkov`.

New solutions are generated variable by variable in the model ordering
(Algorithm 4 of the paper):

1. Seed the first ``k`` variables of every new solution by copying them from a
   random individual of the previous *selected* population (the paper's chosen
   initialization; carried inside the model).
2. For each remaining variable ``X_i`` in order, feed its already-assigned
   predictors ``x_{i-1}, ..., x_{i-k}`` to the regularized regression learned for
   ``X_i`` and draw its value from the predicted conditional distribution
   ``Pr(X_i = l | predictors)`` (probabilistic logic sampling); ``"argmax"``
   sampling (most probable value) is also available.

Because the predictors of ``X_i`` are sampled before ``X_i``, the conditional
distribution is always defined.  With ``init="sample"`` even the first ``k``
variables are drawn from their (partial-predictor) regressions instead of being
seeded, so the model is used end to end.

Consistency: this is an ordinary
:class:`~pateda.core.components.SamplingMethod`, sampled like every other pateda
sampler; the EDA passes the current population as ``aux_pop`` (unused here, as
the selected population needed for seeding travels inside the model).

References
----------
- Santana, R., Karshenas, H., Bielza, C., & Larrañaga, P. (2011). "Regularized
  k-order Markov models in EDAs." GECCO 2011, pp. 593-600.
"""

from typing import Any, Dict, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model
from pateda.learning.regularized_markov import build_markov_features


class SampleRegularizedMarkov(SamplingMethod):
    """
    Sample a new population from a regularized k-order Markov model.

    Consumes the model of
    :class:`~pateda.learning.regularized_markov.LearnRegularizedMarkov`.
    """

    def __init__(self, n_samples: int, mode: str = "proba", init: str = "seed"):
        """
        Args:
            n_samples: Number of individuals to generate.
            mode: ``"proba"`` (default) draws each variable from its predicted
                conditional distribution; ``"argmax"`` takes the most probable
                value.
            init: How to fix the first ``k`` variables -- ``"seed"`` (default)
                copies them from a random selected individual (the paper's
                choice), ``"sample"`` draws them from their partial-predictor
                regressions.
        """
        if mode not in ("proba", "argmax"):
            raise ValueError(f"mode must be 'proba' or 'argmax', got {mode!r}")
        if init not in ("seed", "sample"):
            raise ValueError(f"init must be 'seed' or 'sample', got {init!r}")
        self.n_samples = n_samples
        self.mode = mode
        self.init = init

    def _draw(self, submodel: Dict[str, Any], predictor_vals: np.ndarray,
              rng: np.random.Generator) -> int:
        """Draw one value for a variable from its sub-model."""
        kind = submodel["kind"]
        if kind == "constant":
            return int(submodel["value"])
        if kind == "marginal":
            probs = submodel["probs"]
            if self.mode == "argmax":
                return int(np.argmax(probs))
            return int(rng.choice(len(probs), p=probs))

        # Regression sub-model: build the 1-row feature vector, predict, draw.
        feats = build_markov_features(predictor_vals.reshape(1, -1), submodel["variant"])
        if submodel["scaler"] is not None:
            feats = submodel["scaler"].transform(feats)
        model = submodel["model"]
        proba = model.predict_proba(feats)[0]
        classes = model.classes_
        if self.mode == "argmax":
            return int(classes[int(np.argmax(proba))])
        return int(classes[int(rng.choice(len(classes), p=proba))])

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Generate a new population from the regularized Markov model.

        Args:
            n_vars: Number of variables.
            model: Regularized Markov model (see
                :class:`~pateda.learning.regularized_markov.LearnRegularizedMarkov`).
            cardinality: Variable cardinalities.
            aux_pop: Unused (the selected population for seeding is in the model).
            aux_fitness: Unused.
            rng: Random number generator (``None`` creates a fresh one).
            **params: ``n_samples`` overrides the instance value.

        Returns:
            Sampled population of shape ``(n_samples, n_vars)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        p = model.parameters
        if not isinstance(p, dict) or "submodels" not in p:
            raise TypeError(
                "SampleRegularizedMarkov expects a model from LearnRegularizedMarkov."
            )

        n_samples = int(params.get("n_samples", self.n_samples))
        submodels = p["submodels"]
        k = int(p["k"])
        selected = np.asarray(p["selected_population"], dtype=int)
        n_selected = selected.shape[0]

        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Optional seeding of the first k variables from the selected population.
        n_seed = k if self.init == "seed" else 0
        n_seed = min(n_seed, n_vars)
        if n_seed > 0:
            donors = rng.integers(0, n_selected, size=n_samples)
            new_pop[:, :n_seed] = selected[donors][:, :n_seed]

        # Sample the remaining variables in order using their sub-models.
        for j in range(n_samples):
            for i in range(n_seed, n_vars):
                sm = submodels[i]
                preds = sm["predictors"]
                pvals = new_pop[j, preds] if preds else np.empty(0)
                new_pop[j, i] = self._draw(sm, np.asarray(pvals, dtype=float), rng)

        return new_pop
