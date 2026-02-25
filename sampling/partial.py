"""
Partial Sampling for Discrete EDAs

Implements partial sampling for factorized distribution models (FDA, UMDA, Tree-EDA,
MN-FDA, MN-FDAG, MK-EDA, etc.).

In partial sampling, a template solution is provided where some positions are fixed
(non-NaN) and others are marked with NaN to indicate they should be sampled from the
model.

Algorithm:
1. Receive a template (or population of templates) where NaN positions are to be
   sampled and non-NaN positions are kept fixed.
2. Process cliques in topological order (root nodes first):
   a. If a position's template value is non-NaN: keep it as-is.
   b. If a position's template value is NaN: sample it from the conditional
      distribution given any already-fixed or previously-sampled parent variables.

This method works for:
- UMDA: each variable is independent, NaN positions sampled from univariate marginals.
- Tree-EDA / MK-EDA: sampling follows tree order from root; root NaN → marginal,
  child NaN → conditional on parent (whether fixed from template or sampled earlier).
- MN-FDA / MN-FDAG: same logic applies via the FactorizedModel clique structure.

References:
- Mühlenbein, H., & Paass, G. (1996). "From recombination of genes to the
  estimation of distributions I. Binary parameters." PPSN IV, pp. 178-187.
- Santana, R. (2003). "A Markov network based factorized distribution algorithm
  for optimization." ECML 2003.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, FactorizedModel
from pateda.learning.utils.conversions import (
    find_acc_card,
    index_convert_card,
    num_convert_card,
)


class SamplePartialFDA(SamplingMethod):
    """
    Partial sampling from a Factorized Distribution Algorithm (FDA) model.

    A template solution (or population of templates) is provided where NaN values
    mark positions to be sampled from the model.  Non-NaN positions are kept fixed.

    Cliques are processed in topological order so that parent values — whether
    fixed by the template or sampled in an earlier clique — are always available
    when a child is sampled.

    Compatible with any EDA that produces a FactorizedModel:
    - UMDA (univariate independent cliques, n_overlap=0, n_new=1 per clique)
    - Tree-EDA / MK-EDA (tree cliques, n_overlap=1, n_new=1 per non-root clique)
    - MN-FDA / MN-FDAG (Markov-network cliques, possibly n_new>1)

    Attributes:
        n_samples: Default number of individuals to generate.
    """

    def __init__(self, n_samples: int):
        """
        Initialize partial FDA sampling.

        Args:
            n_samples: Default number of individuals to sample.
        """
        self.n_samples = n_samples

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
        Partially sample a new population from an FDA model.

        Args:
            n_vars: Number of variables.
            model: FactorizedModel to sample from.
            cardinality: Variable cardinalities, shape (n_vars,).
            aux_pop: Template(s) containing NaN at positions to be sampled.
                     Accepted shapes:
                     - ``(n_vars,)``  – a single template reused for every sample.
                     - ``(n_samples, n_vars)`` – one template per output individual.
                     - ``None`` – all positions are NaN (equivalent to standard
                       full sampling via :class:`SampleFDA`).
            aux_fitness: Not used; present for interface compatibility.
            rng: Optional random-number generator.
            **params: Keyword overrides:
                      - ``n_samples`` (int): override ``self.n_samples``.

        Returns:
            Sampled population of shape ``(n_samples, n_vars)``.

        Raises:
            TypeError: If *model* is not a :class:`FactorizedModel`.
            ValueError: If ``aux_pop`` has an incompatible shape.
            RuntimeError: If any variable remains unassigned after processing all
                          cliques (indicates a mis-ordered or incomplete model).
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)
        cliques = model.structure
        tables = model.parameters
        n_cliques = cliques.shape[0]

        # ------------------------------------------------------------------
        # Build float templates: NaN = sample, non-NaN = keep fixed
        # ------------------------------------------------------------------
        if aux_pop is None:
            # All positions NaN → standard full sampling
            templates = np.full((n_samples, n_vars), np.nan)
        elif aux_pop.ndim == 1:
            if aux_pop.shape[0] != n_vars:
                raise ValueError(
                    f"1-D aux_pop must have length n_vars={n_vars}, "
                    f"got {aux_pop.shape[0]}"
                )
            templates = np.tile(aux_pop.astype(float), (n_samples, 1))
        else:
            if aux_pop.shape[1] != n_vars:
                raise ValueError(
                    f"2-D aux_pop must have {n_vars} columns, "
                    f"got {aux_pop.shape[1]}"
                )
            if aux_pop.shape[0] != n_samples:
                raise ValueError(
                    f"Number of templates ({aux_pop.shape[0]}) "
                    f"does not match n_samples ({n_samples})"
                )
            templates = aux_pop.astype(float)

        # ------------------------------------------------------------------
        # Initialize output: fixed values copied from template, NaN → -1
        # ------------------------------------------------------------------
        new_pop = np.full((n_samples, n_vars), -1, dtype=int)
        fixed_mask = ~np.isnan(templates)          # shape (n_samples, n_vars)
        new_pop[fixed_mask] = templates[fixed_mask].astype(int)

        # ------------------------------------------------------------------
        # Process cliques in topological order
        # ------------------------------------------------------------------
        for c in range(n_cliques):
            n_overlap = int(cliques[c, 0])
            n_new = int(cliques[c, 1])

            overlap_vars = (
                cliques[c, 2 : 2 + n_overlap].astype(int)
                if n_overlap > 0
                else np.array([], dtype=int)
            )
            new_vars = cliques[c, 2 + n_overlap : 2 + n_overlap + n_new].astype(int)
            table = tables[c]

            new_card = cardinality[new_vars]
            new_acc_card = find_acc_card(n_new, new_card)

            # Pre-compute overlap lookup structures (only once per clique)
            if n_overlap > 0:
                overlap_card = cardinality[overlap_vars]
                overlap_acc_card = find_acc_card(n_overlap, overlap_card)
            else:
                overlap_acc_card = np.array([], dtype=int)

            for j in range(n_samples):
                # Identify which of the new_vars need to be sampled
                nan_positions = new_pop[j, new_vars] == -1  # True → needs sampling

                if not np.any(nan_positions):
                    # All new_vars already fixed by template; nothing to do
                    continue

                # Get overlap variable values (must all be assigned by now)
                if n_overlap > 0:
                    overlap_vals = new_pop[j, overlap_vars]
                    if np.any(overlap_vals == -1):
                        # Overlap vars not yet assigned — clique ordering issue;
                        # skip and leave unsampled (will raise RuntimeError below)
                        continue
                    k = num_convert_card(overlap_vals, n_overlap, overlap_acc_card)
                    joint_probs = table[k, :].copy()
                else:
                    joint_probs = table.ravel().copy()

                if np.all(nan_positions):
                    # --------------------------------------------------------
                    # All new_vars are NaN: standard conditional sampling
                    # --------------------------------------------------------
                    prob_sum = np.sum(joint_probs)
                    if prob_sum > 0:
                        joint_probs /= prob_sum
                    else:
                        joint_probs = np.ones(len(joint_probs)) / len(joint_probs)

                    cumsum = np.cumsum(joint_probs)
                    idx = int(np.searchsorted(cumsum, rng.random()))
                    idx = min(idx, len(joint_probs) - 1)

                    var_values = index_convert_card(idx, n_new, new_acc_card)
                    new_pop[j, new_vars] = var_values

                else:
                    # --------------------------------------------------------
                    # Mixed case: some new_vars fixed, some NaN
                    # Marginalise over fixed positions to sample only NaN ones.
                    # --------------------------------------------------------
                    # Reshape joint probability to per-variable tensor
                    shape = tuple(new_card)
                    probs_tensor = joint_probs.reshape(shape)

                    # Slice out the fixed-variable dimensions
                    slices = [slice(None)] * n_new
                    for pos in np.where(~nan_positions)[0]:
                        slices[pos] = int(new_pop[j, new_vars[pos]])

                    marginal = probs_tensor[tuple(slices)].ravel()

                    prob_sum = np.sum(marginal)
                    if prob_sum > 0:
                        marginal /= prob_sum
                    else:
                        marginal = np.ones(len(marginal)) / len(marginal)

                    cumsum = np.cumsum(marginal)
                    idx = int(np.searchsorted(cumsum, rng.random()))
                    idx = min(idx, len(marginal) - 1)

                    nan_new_vars = new_vars[nan_positions]
                    nan_new_card = cardinality[nan_new_vars]
                    nan_acc_card = find_acc_card(len(nan_new_vars), nan_new_card)
                    var_values = index_convert_card(idx, len(nan_new_vars), nan_acc_card)
                    new_pop[j, nan_new_vars] = var_values

        # ------------------------------------------------------------------
        # Sanity check: all positions must be assigned
        # ------------------------------------------------------------------
        if np.any(new_pop == -1):
            unassigned = np.where(new_pop == -1)
            raise RuntimeError(
                f"Some variables were not assigned during partial sampling. "
                f"Check clique ordering in the model. Unassigned: {unassigned}"
            )

        return new_pop
