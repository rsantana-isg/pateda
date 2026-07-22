"""
Random-Reset Mutation for discrete / integer EDAs

Implements the standard random-reset (a.k.a. uniform, or "random resetting")
mutation for variables of arbitrary cardinality.  For every variable of every
individual:

- with probability ``mutation_prob``: replace the value by one drawn uniformly
  at random from that variable's full range ``[0, cardinality)``;
- otherwise: leave the value unchanged.

This is the multi-value generalization of :func:`~pateda.mutation.bitflip.bit_flip_mutation`
(which is limited to binary variables).  It is the natural way to *infuse
diversity* into EDAs whose sampler is closed under the current population — in
particular :class:`~pateda.sampling.int_fda.SampleIntFDA`, which copies genes
from donor vectors and therefore never emits an unseen value on its own.  A
random reset draws from the whole value space, so at high cardinality it
introduces, with high probability, values that are absent from the base
population.

Because the EDA main loop applies the mutation component *after* sampling and
*before* fitness evaluation, dropping a ``RandomResetMutation`` into
``EDAComponents(mutation=...)`` is all that is needed; no change to the sampler
is required.

References
----------
- Eiben, A. E., & Smith, J. E. (2015). "Introduction to Evolutionary
  Computing" (2nd ed.), Section 4.3 (random resetting mutation).
- MATEDA-2.0 User Guide (mutation operators).
"""

from typing import Any, Dict
import numpy as np

from pateda.core.components import MutationMethod


def random_reset_mutation(
    n_vars: int,
    cardinality: np.ndarray,
    population: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Apply random-reset mutation to a discrete population.

    Args:
        n_vars: Number of variables.
        cardinality: Per-variable cardinalities (scalar or length-``n_vars``
            array).  Reset values are drawn from ``[0, cardinality[i])``.
        population: Population to mutate ``(n_individuals, n_vars)``.
        params: Dictionary with ``"mutation_prob"`` — the per-gene reset
            probability (required).

    Returns:
        A mutated copy of ``population``.

    Raises:
        ValueError: If ``mutation_prob`` is missing or outside ``[0, 1]``.
    """
    if "mutation_prob" not in params:
        raise ValueError("mutation_prob is required in params")

    mutation_prob = params["mutation_prob"]
    if not 0 <= mutation_prob <= 1:
        raise ValueError(f"mutation_prob must be in [0, 1], got {mutation_prob}")

    new_pop = population.copy()
    if mutation_prob == 0:
        return new_pop

    n_individuals = population.shape[0]
    card = np.broadcast_to(np.asarray(cardinality), (n_vars,))

    # Positions to reset.
    mask = np.random.rand(n_individuals, n_vars) < mutation_prob
    rows, cols = np.where(mask)
    if rows.size:
        # Uniform value in [0, cardinality[var]) for each selected position.
        col_card = card[cols]
        rand_vals = (np.random.rand(rows.size) * col_card).astype(int)
        rand_vals = np.minimum(rand_vals, col_card - 1)  # guard against rand == 1.0
        new_pop[rows, cols] = rand_vals

    return new_pop


class RandomResetMutation(MutationMethod):
    """
    Random-reset mutation component for discrete / integer EDAs.

    Each gene is, with probability ``mutation_prob``, redrawn uniformly from its
    full range ``[0, cardinality)``.  Suitable for any cardinality; the standard
    tool for injecting diversity into donor-copy samplers such as
    :class:`~pateda.sampling.int_fda.SampleIntFDA`.
    """

    def __init__(self, mutation_prob: float = 0.0):
        """
        Args:
            mutation_prob: Per-gene probability of a random reset.  ``0.0``
                (default) leaves the population unchanged.
        """
        if not 0 <= mutation_prob <= 1:
            raise ValueError(f"mutation_prob must be in [0, 1], got {mutation_prob}")
        self.mutation_prob = mutation_prob

    def mutate(
        self,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply random-reset mutation (see :func:`random_reset_mutation`)."""
        mutation_prob = params.get("mutation_prob", self.mutation_prob)
        return random_reset_mutation(
            n_vars, cardinality, population, {"mutation_prob": mutation_prob}
        )
