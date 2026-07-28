"""
PLS sampling augmented with the most-probable configuration (MN-FDA-P).

``SampleFDAWithMPC`` samples a new population exactly as MN-FDA does with PLS
(:class:`SampleFDA`), except that **one** individual is replaced by the
*most-probable configuration* (MPC) of the learned factorized model, and the
remaining ``n_samples - 1`` individuals are drawn with PLS.

Most-probable configuration
---------------------------
The MPC is computed **exactly** by max-product on the junction tree
(:func:`pateda.inference.max_product_mpc.max_product_mpc`): an inward
max-marginalisation pass followed by an argmax back-tracking pass (the
generalized-Viterbi / two-pass junction-tree algorithm; see ``docs/MPC``).  MN-FDA
models are acyclic forests, so max-product is exact and always tractable.

``mpc_method="single_pass"`` selects the older greedy single-forward-pass
approximation (the mode of each factor given the running assignment); it is kept
only for comparison because forward greedy argmax can be strictly suboptimal
(it differs from the exact MPE on a noticeable fraction of models).
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
from pateda.sampling.fda import SampleFDA
from pateda.inference.max_product_mpc import max_product_mpc, MPCIntractable


def compute_mpc_factorized(
    n_vars: int, model: FactorizedModel, cardinality: np.ndarray
) -> np.ndarray:
    """Most-probable configuration of an MN-FDA :class:`FactorizedModel`.

    Single forward pass over the clique tables taking the ``argmax`` of each
    (marginal / conditional) factor given the running assignment -- the
    deterministic, greedy counterpart of :class:`SampleFDA`.  Returns a 1-D int
    array of length ``n_vars``.
    """
    structure = model.structure
    tables = model.parameters
    n_cliques = structure.shape[0]
    mpc = -np.ones(n_vars, dtype=int)

    for c in range(n_cliques):
        n_overlap = int(structure[c, 0])
        n_new = int(structure[c, 1])
        new_vars = structure[c, 2 + n_overlap : 2 + n_overlap + n_new].astype(int)
        new_card = cardinality[new_vars]
        new_acc_card = find_acc_card(n_new, new_card)
        table = np.asarray(tables[c])

        if n_overlap == 0:
            # Root clique: mode of the marginal p(new).
            best = int(np.argmax(table.ravel()))
        else:
            # Non-root clique: mode of p(new | overlap = assigned).
            overlap_vars = structure[c, 2 : 2 + n_overlap].astype(int)
            overlap_card = cardinality[overlap_vars]
            overlap_acc_card = find_acc_card(n_overlap, overlap_card)
            k = num_convert_card(mpc[overlap_vars], n_overlap, overlap_acc_card)
            best = int(np.argmax(table[k, :]))

        mpc[new_vars] = index_convert_card(best, n_new, new_acc_card)

    if np.any(mpc < 0):        # should not happen: every variable is "new" once
        mpc[mpc < 0] = 0
    return mpc


def compute_mpc(n_vars, model, cardinality, method="max_product"):
    """Most-probable configuration of a FactorizedModel.

    ``method="max_product"`` (default) computes the *exact* MPC by junction-tree
    max-product (:func:`max_product_mpc`), trying the efficient ``min_degree``
    elimination order first and falling back to ``min_fill`` (lower width) only
    if the safety cap is hit.  ``method="single_pass"`` uses the greedy
    single-forward-pass approximation.
    """
    if method == "single_pass":
        return compute_mpc_factorized(n_vars, model, cardinality)
    try:
        x, _ = max_product_mpc(model.structure, model.parameters, cardinality,
                               order_method="min_degree")
    except MPCIntractable:
        x, _ = max_product_mpc(model.structure, model.parameters, cardinality,
                               order_method="min_fill")
    return x


class SampleFDAWithMPC(SamplingMethod):
    """
    PLS sampling with the most-probable configuration inserted (MN-FDA-P).

    Produces ``n_samples`` individuals: the first is the model's MPC, the other
    ``n_samples - 1`` are sampled with PLS (:class:`SampleFDA`), exactly as
    MN-FDA.  The MPC is computed exactly by junction-tree max-product; pass
    ``mpc_method="single_pass"`` for the greedy approximation.
    """

    def __init__(self, n_samples: int, mpc_method: str = "max_product"):
        self.n_samples = n_samples
        self.mpc_method = mpc_method

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
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)
        mpc = compute_mpc(n_vars, model, cardinality, self.mpc_method)

        if n_samples <= 1:
            return mpc.reshape(1, n_vars)

        # Remaining individuals via PLS, exactly as MN-FDA.
        pls = SampleFDA(n_samples=n_samples - 1).sample(
            n_vars=n_vars, model=model, cardinality=cardinality, rng=rng)
        return np.vstack([mpc.reshape(1, n_vars), pls])
