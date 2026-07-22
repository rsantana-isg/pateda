"""
Probabilistic (categorical) population initialization.

Generalizes :class:`~pateda.seeding.bias_init.BiasInit` from binary variables
to arbitrary discrete variables: each variable is initialized by sampling from a
user-supplied categorical distribution over its ``cardinality`` values. This is
the discrete counterpart of biasing the initial population toward configurations
believed to be good (a form of a-priori knowledge, see the user guide).

``BiasInit`` is recovered as the binary special case with
``probs = [[1 - p, p], ...]``.
"""

from typing import Any, List, Optional, Sequence, Union
import numpy as np

from pateda.core.components import SeedingMethod


class ProbabilisticInit(SeedingMethod):
    """
    Initialize each variable from a per-variable categorical distribution.

    The distribution for variable :math:`X_i` is a probability vector of length
    ``cardinality[i]`` giving :math:`P(X_i = 0), \\ldots, P(X_i = r_i - 1)`.
    Variables are sampled independently, so the initial population follows the
    product distribution :math:`\\prod_i P(X_i)`.

    The distributions can be supplied at construction time or at run time through
    the ``probs`` parameter of :meth:`seed`. They may be given either as

    * a list of ``n_vars`` 1-D arrays, each of length ``cardinality[i]``, or
    * a 2-D array of shape ``(n_vars, max_cardinality)`` whose row ``i`` is
      padded with zeros beyond ``cardinality[i]``.

    If no distribution is provided the method falls back to the uniform
    distribution over each variable's values (equivalent to ``RandomInit``).

    Example:
        >>> import numpy as np
        >>> # 3 variables with cardinalities 2, 3, 2
        >>> probs = [np.array([0.1, 0.9]),
        ...          np.array([0.2, 0.3, 0.5]),
        ...          np.array([0.5, 0.5])]
        >>> seeder = ProbabilisticInit(probs)
        >>> pop = seeder.seed(3, 100, np.array([2, 3, 2]))
    """

    def __init__(
        self,
        probs: Optional[Union[Sequence[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            probs: Per-variable categorical distributions (see class docstring).
                   May be omitted here and passed to :meth:`seed` instead.
        """
        self.probs = probs

    @staticmethod
    def _normalize_probs(
        probs: Union[Sequence[np.ndarray], np.ndarray],
        n_vars: int,
        cardinality: np.ndarray,
    ) -> List[np.ndarray]:
        """Validate and coerce ``probs`` into a list of 1-D probability vectors."""
        as_list: List[np.ndarray] = []
        for i in range(n_vars):
            r_i = int(cardinality[i])
            row = np.asarray(probs[i], dtype=float)
            # allow *zero-padded* rows coming from a 2-D (n_vars, max_card) array,
            # but reject genuinely wrong-length distributions
            if row.shape[0] > r_i:
                if np.allclose(row[r_i:], 0.0):
                    row = row[:r_i]
                else:
                    raise ValueError(
                        f"Distribution for variable {i} has length {row.shape[0]} "
                        f"with non-zero entries beyond cardinality {r_i}"
                    )
            if row.shape[0] != r_i:
                raise ValueError(
                    f"Distribution for variable {i} has length {row.shape[0]}, "
                    f"expected cardinality {r_i}"
                )
            total = row.sum()
            if total <= 0 or not np.isfinite(total):
                raise ValueError(
                    f"Distribution for variable {i} must have a positive, finite sum"
                )
            as_list.append(row / total)
        return as_list

    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Generate an initial population from per-variable categorical distributions.

        Args:
            n_vars: Number of variables.
            pop_size: Population size.
            cardinality: 1-D integer array of variable cardinalities.
            rng: Random number generator (None = create a new generator).
            **params: Additional parameters.
                probs: Overrides the distributions given at construction time.

        Returns:
            Integer population of shape ``(pop_size, n_vars)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        cardinality = np.asarray(cardinality)
        if cardinality.ndim != 1:
            raise ValueError(
                "ProbabilisticInit only supports discrete problems "
                "(1-D cardinality array)"
            )

        probs = params.get("probs", self.probs)

        population = np.empty((pop_size, n_vars), dtype=int)
        if probs is None:
            # uniform fallback (== RandomInit)
            for i in range(n_vars):
                population[:, i] = rng.integers(0, int(cardinality[i]), size=pop_size)
            return population

        dists = self._normalize_probs(probs, n_vars, cardinality)
        for i in range(n_vars):
            population[:, i] = rng.choice(
                int(cardinality[i]), size=pop_size, p=dists[i]
            )
        return population
