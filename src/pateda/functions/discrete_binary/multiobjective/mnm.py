"""
mNM model: multi-objective Markov-network / Walsh benchmark.

The mNM model (``Multi_Objective_Code/mNM-Model``; Santana et al., *"Computing
factorized approximations of Pareto-fronts"*, CAEPIA 2015) defines a
pseudo-boolean function as a truncated Walsh expansion over ``+/-1`` variables::

    f(x) = beta_0 + sum_{c in components} beta_c * prod_{i in c} x_i ,   x in {-1, +1}

The components are all variable subsets of size ``1..M`` (plus the empty set for
``beta_0``), and the coefficients are ``beta_c = exp(-|sigma * z_c|)`` with
``z_c ~ N(0, 1)`` (``Create_mNM_model.m``).  ``sigma`` controls how fast the
coefficients decay: small ``sigma`` keeps high-order interactions relevant,
large ``sigma`` concentrates weight on low-order terms.

Truncating the expansion to interactions of order ``<= k`` (``Save_mNM_model.m``)
yields functions of controllable complexity.  A multi-objective instance builds
each objective from the *same* base model but with its own maximum interaction
order and a sign transform ``x -> s * x`` (``EvaluatePop_nNM`` is evaluated on
``+x`` for one objective and ``-x`` for another), which tunes the correlation /
conflict between objectives.  All objectives are **maximised**.
"""

from pathlib import Path
from itertools import combinations
from typing import List, Optional, Sequence, Tuple
import numpy as np


class MNMModel:
    """Base mNM (Walsh) model shared by all objectives of an instance.

    Attributes
    ----------
    n_vars : int
    max_order : int
        Maximum interaction order ``M`` present in the model.
    components : list of tuple
        Variable subsets (0-based), ordered by increasing size; the first is the
        empty tuple (constant term).
    betas : (n_components,) ndarray
        Coefficient of each component.
    orders : (n_components,) ndarray
        Size of each component (cached for fast truncation).
    """

    def __init__(self, n_vars: int, max_order: int, components: List[Tuple[int, ...]],
                 betas: np.ndarray, seed: int = 0):
        self.n_vars = int(n_vars)
        self.max_order = int(max_order)
        self.components = components
        self.betas = np.asarray(betas, dtype=float)
        self.orders = np.array([len(c) for c in components], dtype=int)
        self.seed = int(seed)

    @classmethod
    def create(cls, n_vars: int, max_order: int, sigma: float,
               seed: Optional[int] = None) -> "MNMModel":
        """Sample a model (``Create_mNM_model.m``).

        Components are the empty set followed by all subsets of size ``1..M``;
        ``beta_c = exp(-|sigma * N(0,1)|)``.
        """
        rng = np.random.default_rng(seed)
        components: List[Tuple[int, ...]] = [()]
        for order in range(1, max_order + 1):
            components.extend(combinations(range(n_vars), order))
        betas = np.exp(-np.abs(sigma * rng.standard_normal(len(components))))
        return cls(n_vars, max_order, components, betas,
                   seed=seed if seed is not None else 0)

    def evaluate_raw(self, population: np.ndarray, max_order: Optional[int] = None,
                     sign: int = 1) -> np.ndarray:
        """Raw Walsh value ``beta_0 + sum beta_c prod (sign * x_i)`` (per solution).

        The returned value is normalised by the number of active components
        (``fv / ncomp`` in ``EvaluatePop_nNM``) but *not* min-max scaled, so it is
        deterministic per solution (independent of any population).

        Args:
            population: ``{0,1}`` or ``{-1,+1}`` individual (1-D) or population (2-D).
            max_order: truncate to interactions of order ``<= max_order``
                (default: full model).
            sign: ``+1`` or ``-1``; evaluates on ``sign * x``.
        """
        P = np.asarray(population)
        single = P.ndim == 1
        if single:
            P = P.reshape(1, -1)
        # map {0,1} -> {-1,+1}; leave {-1,+1} untouched
        if P.min() >= 0:
            spins = (2 * P - 1).astype(float)
        else:
            spins = P.astype(float)
        spins = sign * spins

        if max_order is None:
            active = np.ones(len(self.components), dtype=bool)
        else:
            active = self.orders <= max_order
        ncomp = int(active.sum())

        vals = np.full(P.shape[0], 0.0)
        for c_idx in np.flatnonzero(active):
            comp = self.components[c_idx]
            beta = self.betas[c_idx]
            if len(comp) == 0:
                vals += beta
            else:
                vals += beta * np.prod(spins[:, comp], axis=1)
        vals = vals / ncomp
        return float(vals[0]) if single else vals


class MNMInstance:
    """A multi-objective mNM instance.

    Each objective is the base :class:`MNMModel` truncated to its own maximum
    interaction order and evaluated on ``sign * x``.

    Parameters
    ----------
    model : MNMModel
    objective_orders : sequence of int
        Maximum interaction order for each objective.
    objective_signs : sequence of int, optional
        ``+1`` / ``-1`` sign transform for each objective (default all ``+1``,
        except a bi-objective default of ``[+1, -1]``).
    """

    def __init__(self, model: MNMModel, objective_orders: Sequence[int],
                 objective_signs: Optional[Sequence[int]] = None):
        self.model = model
        self.objective_orders = [int(o) for o in objective_orders]
        self.n_objectives = len(self.objective_orders)
        if objective_signs is None:
            if self.n_objectives == 2:
                self.objective_signs = [1, -1]
            else:
                self.objective_signs = [1] * self.n_objectives
        else:
            self.objective_signs = [int(s) for s in objective_signs]
        self.n_vars = model.n_vars

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Objective values for a 1-D individual or a 2-D population."""
        P = np.asarray(population)
        single = P.ndim == 1
        cols = [self.model.evaluate_raw(P, max_order=o, sign=s)
                for o, s in zip(self.objective_orders, self.objective_signs)]
        if single:
            return np.array([float(c) for c in cols])
        return np.column_stack(cols)

    # -- I/O ---------------------------------------------------------------- #
    def save(self, filepath: str) -> None:
        """Save the instance to a ``.npz`` file."""
        # store components as a padded (-1) integer matrix for round-tripping
        maxlen = max((len(c) for c in self.model.components), default=0)
        comp_mat = -np.ones((len(self.model.components), max(1, maxlen)), dtype=int)
        for i, c in enumerate(self.model.components):
            if c:
                comp_mat[i, :len(c)] = c
        np.savez(filepath,
                 n_vars=self.model.n_vars, max_order=self.model.max_order,
                 betas=self.model.betas, components=comp_mat, seed=self.model.seed,
                 objective_orders=np.array(self.objective_orders),
                 objective_signs=np.array(self.objective_signs))

    @classmethod
    def load(cls, filepath: str) -> "MNMInstance":
        d = np.load(filepath)
        comp_mat = d["components"]
        components = [tuple(int(v) for v in row if v >= 0) for row in comp_mat]
        model = MNMModel(int(d["n_vars"]), int(d["max_order"]), components,
                         d["betas"], seed=int(d["seed"]))
        return cls(model, [int(o) for o in d["objective_orders"]],
                   [int(s) for s in d["objective_signs"]])


def create_mnm_objective_function(instance: MNMInstance):
    """Return an objective ``f(pop) -> (pop, n_obj)`` maximising all objectives."""
    def objective(population: np.ndarray) -> np.ndarray:
        return instance.evaluate(population)
    return objective


def generate_mnm(n_vars: int, max_order: int = 3, sigma: float = 5.0,
                 objective_orders: Optional[Sequence[int]] = None,
                 objective_signs: Optional[Sequence[int]] = None,
                 seed: Optional[int] = None) -> MNMInstance:
    """Convenience builder for a multi-objective mNM instance.

    Args:
        n_vars: number of variables.
        max_order: maximum interaction order ``M`` of the base model.
        sigma: coefficient-decay parameter.
        objective_orders: per-objective truncation orders (default
            ``[max_order, max_order]`` for a bi-objective instance).
        objective_signs: per-objective sign transforms (default ``[+1, -1]``).
        seed: random seed.
    """
    model = MNMModel.create(n_vars, max_order, sigma, seed=seed)
    if objective_orders is None:
        objective_orders = [max_order, max_order]
    return MNMInstance(model, objective_orders, objective_signs)
