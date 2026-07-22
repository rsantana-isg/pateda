"""
Tree-EDA-M: Tree EDA that keeps only *malign* interactions

Variant of Tree-EDA (:class:`~pateda.learning.tree.LearnTreeModel`) introduced
by Santana, Larrañaga & Lozano (2005).  It builds the same maximum-weight
spanning tree of pairwise mutual information, but it first discards every pair
of variables whose interaction is *benign*, keeping only *malign* interactions.
The resulting model is therefore usually a forest (some variables end up with
no parent).

Benign vs. malign interactions
------------------------------
Following the paper, the type of a bivariate interaction is read directly from
the estimated marginals of the selected set:

- Let ``(a*, b*) = argmax_{a,b} p(X_i = a, X_j = b)`` be the most probable
  *joint* configuration of the pair.
- Let ``(a_u, b_u) = (argmax_a p(X_i = a), argmax_b p(X_j = b))`` be the
  configuration that maximizes the *product of the univariate marginals*
  (the joint mode a univariate/independent model would predict).

If the two coincide (``a* == a_u`` and ``b* == b_u``) the interaction is
**benign**: the dependency merely reinforces the message already carried by the
main (univariate) effects, so it can be recovered by sampling the two variables
independently.  Its mutual information is *not* computed and no edge is added.

Otherwise the interaction is **malign**: the joint mode disagrees with what the
main effects alone would predict (the situation the GA community calls
deception).  Only these dependencies cannot be reconstructed from the univariate
marginals, so only for them is the mutual information computed and an edge
allowed in the tree.

This criterion is defined purely in terms of ``argmax`` over the marginal
tables, so it applies unchanged to **discrete non-binary** variables of any
cardinality.

Relationship to the other tree learners in pateda
-------------------------------------------------
Tree-EDA-M reuses the tree construction and parameter estimation of
:class:`~pateda.learning.tree.LearnTreeModel` verbatim; the *only* difference is
the mutual-information step, where benign pairs are skipped (their MI is left at
zero, so the maximum-weight spanning tree never selects them).  It is the exact
analogue of :class:`~pateda.learning.tree_r.LearnTreeModelR`, which skips pairs
according to a fixed interaction matrix; here the pairs to skip are *detected*
from the data every generation instead of being supplied a priori.

Because the learned model is an ordinary
:class:`~pateda.core.models.FactorizedModel` (tree cliques + conditional
tables), it is sampled with the standard
:class:`~pateda.sampling.fda.SampleFDA`, exactly like Tree-EDA.

References
----------
- Santana, R., Larrañaga, P., & Lozano, J. A. (2005). "Interactions and
  Dependencies in Estimation of Distribution Algorithms." Proceedings of the
  2005 Congress on Evolutionary Computation (CEC-2005), pp. 1418-1425.
  (Introduces Tree-EDA-M; see its Algorithm 2 and the benign/malign definition
  in Section 4.)
- Baluja, S., & Davies, S. (1997). "Using optimal dependency-trees for
  combinatorial optimization." ICML 1997.
- Kallel, L., Naudts, B., & Reeves, R. (2000). "Properties of fitness functions
  and search landscapes." In Theoretical Aspects of Evolutionary Computing.
"""

from typing import Any, List, Optional
import numpy as np

from pateda.core.models import FactorizedModel
from pateda.learning.tree import LearnTreeModel
from pateda.learning.utils.marginal_prob import find_marginal_prob
from pateda.learning.utils.weights import count_weights_from_p


class LearnTreeModelM(LearnTreeModel):
    """
    Learn a Tree-EDA-M model (malign interactions only).

    Same as :class:`~pateda.learning.tree.LearnTreeModel` except that, before
    building the tree, every *benign* pairwise interaction is removed: its
    mutual information is not computed, so the maximum-weight spanning tree can
    only pick *malign* edges.  Variables whose interactions are all benign
    become isolated roots, i.e. the model becomes a forest of independent
    components (and, in the limit where every interaction is benign, reduces to
    UMDA).

    Works for binary and non-binary discrete variables.  Sampling uses the
    standard :class:`~pateda.sampling.fda.SampleFDA`.
    """

    def __init__(self, alpha: float = 1.0, mi_threshold: float = 1e-4):
        """
        Initialize Tree-EDA-M learning.

        Args:
            alpha: Smoothing parameter for Laplace estimation (default: 0.0).
            mi_threshold: Minimum mutual information threshold for creating
                edges (default: 1e-4).  Benign pairs are excluded independently
                of this threshold.
        """
        super().__init__(alpha=alpha, mi_threshold=mi_threshold)

    # ------------------------------------------------------------------
    # Malign-interaction detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_malign_mask(
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Classify every pair of variables as malign (``True``) or benign.

        A pair ``(i, j)`` is *benign* when the most probable joint configuration
        ``argmax p(x_i, x_j)`` coincides with the configuration that maximizes
        the product of univariate marginals ``(argmax p(x_i), argmax p(x_j))``;
        it is *malign* otherwise.  See the module docstring for the rationale.

        Args:
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            univ_prob: Univariate marginal distributions (``univ_prob[i]`` has
                length ``cardinality[i]``).
            biv_prob: Bivariate marginals; ``biv_prob[i][j]`` (for ``i < j``) is
                a flat array of length ``card_i * card_j`` indexed by
                ``card_j * a + b`` for ``X_i = a, X_j = b``.

        Returns:
            Symmetric boolean matrix ``(n_vars, n_vars)``; entry ``(i, j)`` is
            ``True`` iff the interaction between ``X_i`` and ``X_j`` is malign.
            The diagonal is ``False``.
        """
        malign = np.zeros((n_vars, n_vars), dtype=bool)

        # Mode of each univariate marginal.  The product p_i(a)*p_j(b) is
        # separable, so its joint argmax is (argmax_a p_i, argmax_b p_j).
        uni_mode = [int(np.argmax(univ_prob[i])) for i in range(n_vars)]

        for i in range(n_vars - 1):
            for j in range(i + 1, n_vars):
                card_j = int(cardinality[j])
                flat_idx = int(np.argmax(biv_prob[i][j]))
                a_star, b_star = divmod(flat_idx, card_j)  # joint mode (a*, b*)

                benign = (a_star == uni_mode[i]) and (b_star == uni_mode[j])
                malign[i, j] = malign[j, i] = not benign

        return malign

    def _compute_malign_mi_matrix(
        self,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
        malign_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Mutual-information matrix computed *only* for malign pairs.

        Benign pairs (``malign_mask[i, j] == False``) are skipped and left at
        zero, so the spanning tree cannot select them.  For malign pairs the
        normalized mutual information is computed exactly as in
        :class:`~pateda.learning.tree.LearnTreeModel`.

        Args:
            n_vars: Number of variables.
            cardinality: Variable cardinalities.
            univ_prob: Univariate marginal distributions.
            biv_prob: Bivariate marginal distributions.
            malign_mask: Boolean matrix marking malign pairs.

        Returns:
            Symmetric normalized-MI matrix with zeros on benign pairs.
        """
        mi_matrix = np.zeros((n_vars, n_vars))

        entropies = np.array([
            -sum(p * np.log(p) for p in univ_prob[i] if p > 0)
            for i in range(n_vars)
        ])

        for i in range(n_vars - 1):
            for j in range(i + 1, n_vars):
                # Step 6/7 of Algorithm 2: only malign interactions contribute.
                if not malign_mask[i, j]:
                    continue

                mi = 0.0
                card_i = int(cardinality[i])
                card_j = int(cardinality[j])

                for k in range(card_i):
                    for l in range(card_j):
                        idx = card_j * k + l
                        p_ij = biv_prob[i][j][idx]
                        p_i = univ_prob[i][k]
                        p_j = univ_prob[j][l]

                        if p_ij > 0 and p_i > 0 and p_j > 0:
                            mi += p_ij * np.log(p_ij / (p_i * p_j))

                denom = entropies[i] + entropies[j]
                nmi = max(0.0, min(1.0, 2.0 * mi / denom)) if denom > 0 else 0.0
                mi_matrix[i, j] = nmi
                mi_matrix[j, i] = nmi

        return mi_matrix

    # ------------------------------------------------------------------
    # Learning entry point (Algorithm 2)
    # ------------------------------------------------------------------
    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn a Tree-EDA-M model from the selected population.

        Implements Algorithm 2 of Santana et al. (2005): estimate the univariate
        and bivariate marginals, detect malign interactions, compute the mutual
        information of the malign pairs only, and build the maximum-weight
        spanning tree/forest from that matrix.

        Args:
            generation: Current generation number.
            n_vars: Number of variables.
            cardinality: Variable cardinalities (binary or non-binary).
            population: Selected population to learn from.
            fitness: Fitness values (unused for tree learning).
            **params: Additional parameters (``alpha`` override; ``p`` weights
                for customized selection, handled like the other tree learners).

        Returns:
            A :class:`~pateda.core.models.FactorizedModel` (tree cliques +
            conditional tables) sampled with
            :class:`~pateda.sampling.fda.SampleFDA`.
        """
        alpha = params.get("alpha", self.alpha)
        cardinality = np.asarray(cardinality, dtype=int)

        # Customized selection: count-scale weights (N * p) or None for uniform.
        weights = count_weights_from_p(params.get("p"), population.shape[0])

        # Univariate and bivariate marginal frequencies of the selected set.
        univ_prob, biv_prob = find_marginal_prob(
            population, n_vars, cardinality, alpha=alpha, weights=weights
        )

        # Detect malign interactions and compute MI only for them.
        malign_mask = self.detect_malign_mask(
            n_vars, cardinality, univ_prob, biv_prob
        )
        mi_matrix = self._compute_malign_mi_matrix(
            n_vars, cardinality, univ_prob, biv_prob, malign_mask
        )

        # Maximum-weight spanning tree/forest over the malign MI (inherited).
        cliques = self._create_tree_structure(mi_matrix, n_vars)

        # Conditional probability tables (inherited).
        tables = self._learn_parameters(
            cliques, population, n_vars, cardinality, univ_prob, biv_prob, weights=weights
        )

        n_malign = int(np.sum(np.triu(malign_mask, k=1)))
        n_pairs = n_vars * (n_vars - 1) // 2

        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "Tree-EDA-M",
                "alpha": alpha,
                "mi_threshold": self.mi_threshold,
                "mi_matrix": mi_matrix,
                "malign_mask": malign_mask,
                "n_malign_pairs": n_malign,
                "n_benign_pairs": n_pairs - n_malign,
            },
        )

        return model
