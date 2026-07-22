"""
Regularized k-order Markov model learning (MkRg-EDA)

Implements the learning step of the EDA with regularized k-order Markov models
of Santana, Karshenas, Bielza & Larrañaga (2011).

A k-order Markov model makes each variable ``X_i`` depend on its previous ``k``
variables in a fixed ordering,

    p_MK(x) = p(x_1, ..., x_{k+1}) * prod_{i=k+2}^{n} p(x_i | x_{i-1}, ..., x_{i-k}),

which the classic :class:`~pateda.learning.markov.LearnMarkovChain` estimates
with a conditional probability table (CPT) per variable -- exponential in ``k``.
The regularized variant replaces each CPT by a **regularized multinomial
(multi-logit) regression** that predicts ``X_i`` from (a function of) its
previous ``k`` variables, fitted with the **elastic net** (Zou & Hastie, 2005).
This keeps the number of parameters polynomial in ``k`` (so larger ``k`` becomes
affordable) and lets regularization prune the previous variables that do not
contribute.

Three predictor variants are provided (Section 3.2 of the paper), of increasing
complexity:

- ``"rgk"``     -- predictors are the ``k`` previous variables directly
                   (``y = (x_{i-1}, ..., x_{i-k})``); ``O(k)`` parameters.
- ``"bivrgk"``  -- predictors are all pairwise products of the ``k`` previous
                   variables; ``O(k^2)`` parameters.
- ``"allrgk"``  -- predictors are the ``k`` previous variables *and* their
                   pairwise products; ``O(k^2)`` parameters.

The companion sampler is
:class:`~pateda.sampling.regularized_markov.SampleRegularizedMarkov`.

Reused pateda components: the k-order Markov chain structure and clique format of
:class:`~pateda.learning.markov.LearnMarkovChain`; the customized-selection
weighting utilities (:func:`~pateda.learning.utils.weights.count_weights_from_p`);
and scikit-learn's elastic-net multinomial logistic regression (already used by
:mod:`pateda.learning.gmrf_eda`).

References
----------
- Santana, R., Karshenas, H., Bielza, C., & Larrañaga, P. (2011). "Regularized
  k-order Markov models in EDAs." GECCO 2011, pp. 593-600.
- Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the
  elastic net." J. R. Stat. Soc. B, 67(2):301-320.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization paths for
  generalized linear models via coordinate descent." J. Stat. Softw. 33(1).
"""

import warnings
from typing import Any, Dict, List, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import Model
from pateda.learning.utils.weights import count_weights_from_p

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


VARIANTS = ("rgk", "bivrgk", "allrgk")


def build_markov_features(predictor_values: np.ndarray, variant: str) -> np.ndarray:
    """
    Build the regression feature matrix from the predictor variable columns.

    Parameters
    ----------
    predictor_values : np.ndarray of shape (n_samples, n_pred)
        The values of the ``n_pred`` previous variables used as predictors,
        ordered as ``[x_{i-k}, ..., x_{i-1}]``.
    variant : str
        ``"rgk"`` (raw predictors), ``"bivrgk"`` (pairwise products only) or
        ``"allrgk"`` (raw predictors + pairwise products).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Feature matrix.  With fewer than two predictors there are no pairwise
        products, so ``"bivrgk"`` / ``"allrgk"`` gracefully fall back to the raw
        predictors.
    """
    P = np.asarray(predictor_values, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    n_pred = P.shape[1]

    # Pairwise products of the predictor columns.
    products = []
    for a in range(n_pred):
        for b in range(a + 1, n_pred):
            products.append(P[:, a] * P[:, b])
    products = np.stack(products, axis=1) if products else np.empty((P.shape[0], 0))

    if variant == "rgk":
        return P
    if variant == "bivrgk":
        return products if products.shape[1] > 0 else P
    if variant == "allrgk":
        return np.hstack([P, products]) if products.shape[1] > 0 else P
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


class LearnRegularizedMarkov(LearningMethod):
    """
    Learn a regularized k-order Markov model (MkRg-EDA).

    Fits, for every variable, a regularized multinomial logistic regression that
    predicts it from a function of its previous ``k`` variables (see the module
    docstring for the ``variant`` choices).  The learned
    :class:`~pateda.core.models.Model` stores, in ``parameters``:

    - ``"submodels"``: one entry per variable (a constant, a marginal, or a
      fitted regression together with its feature scaler and predictor list);
    - ``"selected_population"``: the selected set, used by the sampler to seed
      the first ``k`` variables;
    - ``"k"`` and ``"variant"``.

    ``structure`` holds the k-order Markov cliques (one factor
    ``[n_pred, 1, predictors..., i]`` per variable), so the model is compatible
    with the structural tools in :mod:`pateda.knowledge_extraction`.
    """

    def __init__(
        self,
        k: int = 3,
        variant: str = "rgk",
        l1_ratio: float = 0.5,
        C: float = 1.0,
        max_iter: int = 100,
        standardize: bool = True,
    ):
        """
        Args:
            k: Markov order (number of previous variables each variable may
                depend on).
            variant: Predictor variant, ``"rgk"``, ``"bivrgk"`` or ``"allrgk"``.
            l1_ratio: Elastic-net mixing (the paper's ``alpha``): 0 = ridge,
                1 = lasso.  ``0.5`` is a balanced elastic net.
            C: Inverse regularization strength (``1/lambda``) of the elastic net.
            max_iter: Maximum saga iterations per variable regression.
            standardize: Standardize the regression features before fitting
                (recommended for the elastic net's scale sensitivity).
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
        self.k = k
        self.variant = variant
        self.l1_ratio = l1_ratio
        self.C = C
        self.max_iter = max_iter
        self.standardize = standardize

    def _predictors(self, i: int) -> List[int]:
        """Previous ``min(i, k)`` variable indices for variable ``i``,
        ordered ``[i-k, ..., i-1]``."""
        return list(range(max(0, i - self.k), i))

    def _fit_variable(
        self,
        i: int,
        population: np.ndarray,
        cardinality: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Fit the sub-model that predicts variable ``i`` from its predictors."""
        target = population[:, i].astype(int)
        classes = np.unique(target)
        predictors = self._predictors(i)

        # Constant variable: nothing to regress, always output the single value.
        if classes.size <= 1:
            return {"kind": "constant", "value": int(classes[0]) if classes.size else 0,
                    "predictors": predictors}

        # No predictors (first variable): the marginal class distribution.
        if len(predictors) == 0:
            if weights is None:
                counts = np.bincount(target, minlength=int(cardinality[i])).astype(float)
            else:
                counts = np.bincount(target, weights=weights,
                                     minlength=int(cardinality[i])).astype(float)
            probs = counts / counts.sum()
            return {"kind": "marginal", "probs": probs, "predictors": predictors}

        # Regularized multinomial logistic regression on the predictor features.
        feats = build_markov_features(population[:, predictors], self.variant)
        scaler = None
        if self.standardize:
            scaler = StandardScaler()
            feats = scaler.fit_transform(feats)

        # Elastic-net multinomial logistic regression.  The saga solver with an
        # explicit l1_ratio realizes the elastic-net penalty (l1_ratio=0 ridge,
        # 1 lasso); the deprecated ``penalty="elasticnet"`` is intentionally not
        # passed (scikit-learn >= 1.8).
        model = LogisticRegression(
            solver="saga", l1_ratio=self.l1_ratio,
            C=self.C, max_iter=self.max_iter,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(feats, target,
                      sample_weight=None if weights is None else weights)
        return {"kind": "regression", "model": model, "scaler": scaler,
                "predictors": predictors, "variant": self.variant}

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Model:
        """
        Learn a regularized k-order Markov model from the selected population.

        Args:
            generation: Current generation number.
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            population: Selected population to learn from.
            fitness: Fitness values (unused).
            **params: Additional parameters (``p`` weights for customized
                selection are consumed as sklearn ``sample_weight``).

        Returns:
            A :class:`~pateda.core.models.Model` with the per-variable regularized
            sub-models (see the class docstring).
        """
        cardinality = np.asarray(cardinality, dtype=int)
        data = np.asarray(population, dtype=int)
        weights = count_weights_from_p(params.get("p"), data.shape[0])

        submodels = [self._fit_variable(i, data, cardinality, weights)
                     for i in range(n_vars)]

        # k-order Markov clique structure: one factor per variable.
        cliques = []
        for i in range(n_vars):
            preds = submodels[i]["predictors"]
            row = [len(preds), 1] + list(preds) + [i]
            cliques.append(row)
        width = max(len(r) for r in cliques)
        structure = np.zeros((n_vars, width), dtype=int)
        for r, row in enumerate(cliques):
            structure[r, :len(row)] = row

        return Model(
            structure=structure,
            parameters={
                "submodels": submodels,
                "selected_population": data,
                "k": self.k,
                "variant": self.variant,
                "cardinality": cardinality,
            },
            metadata={
                "generation": generation,
                "model_type": "RegularizedMarkov",
                "k": self.k,
                "variant": self.variant,
                "l1_ratio": self.l1_ratio,
                "C": self.C,
            },
        )
