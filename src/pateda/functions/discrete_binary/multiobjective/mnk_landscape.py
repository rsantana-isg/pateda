"""
Multi-objective NK landscapes (MNK).

An MNK landscape defines ``m`` objectives over the same ``n`` binary variables,
where each objective is an independent NK landscape (Aguirre & Tanaka).  For
objective ``o``:

    f_o(x) = (1 / n) * sum_{i=1}^{n} C^o_i( x_i, x_{N^o_i(1)}, ..., x_{N^o_i(k_o)} )

Each variable ``i`` contributes a subfunction ``C^o_i`` that depends on ``i``
itself plus ``k_o`` neighbours ``N^o_i``; subfunction values are drawn i.i.d.
uniformly in ``[0, 1)`` and stored in a table of size ``2^{k_o + 1}``.  Every
objective is **maximised** and lies in ``[0, 1)``.

This ports ``Multi_Objective_Code/Multi-NK`` (``NKModel.cpp`` random-neighbour
instances, ``mainMultiNK.cpp`` objective vector) and supports the
*heterogeneous-objective* setting (a different ``k_o`` per objective) from the
paper *"Multi-objective NK Landscapes with Heterogeneous Objectives"*.

The single-objective circular NK landscape already lives in
:mod:`pateda.functions.discrete_binary.problems.nk_landscape`; this module adds
the multi-objective, random-neighbourhood construction.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union
import numpy as np


def create_random_nk_neighbourhoods(n_vars: int, k: int,
                                    rng: np.random.Generator) -> np.ndarray:
    """Random NK neighbourhood structure (``NKModel.cpp::RandomInstance``).

    Returns an ``(n_vars, k + 1)`` integer array where row ``i`` lists variable
    ``i`` followed by ``k`` distinct random neighbours (``!= i``).
    """
    lattice = np.empty((n_vars, k + 1), dtype=int)
    for i in range(n_vars):
        lattice[i, 0] = i
        others = np.delete(np.arange(n_vars), i)
        lattice[i, 1:] = rng.choice(others, size=k, replace=False)
    return lattice


class NKObjective:
    """A single NK landscape (one MNK objective).

    Attributes
    ----------
    n_vars, k : int
    lattice : (n_vars, k+1) int ndarray
        Neighbourhood of each variable (self first).
    tables : (n_vars, 2**(k+1)) float ndarray
        Sub-function value tables, entries in ``[0, 1)``.
    """

    def __init__(self, n_vars: int, k: int, lattice: np.ndarray, tables: np.ndarray):
        self.n_vars = int(n_vars)
        self.k = int(k)
        self.lattice = lattice
        self.tables = tables
        # weights to convert (k+1) bits (MSB = self) into a table index
        self._pow = (1 << np.arange(k, -1, -1)).astype(int)

    @classmethod
    def random(cls, n_vars: int, k: int, rng: np.random.Generator) -> "NKObjective":
        lattice = create_random_nk_neighbourhoods(n_vars, k, rng)
        tables = rng.random((n_vars, 1 << (k + 1)))
        return cls(n_vars, k, lattice, tables)

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate a 1-D individual (-> float) or 2-D population (-> 1-D)."""
        population = np.asarray(population)
        single = population.ndim == 1
        P = population.reshape(1, -1) if single else population
        # gather neighbour bits for every variable: shape (pop, n_vars, k+1)
        neigh = P[:, self.lattice]                       # (pop, n_vars, k+1)
        idx = neigh.astype(int) @ self._pow              # (pop, n_vars) table index
        rows = np.arange(self.n_vars)
        vals = self.tables[rows, idx]                    # (pop, n_vars)
        f = vals.mean(axis=1)
        return float(f[0]) if single else f


class MNKLandscape:
    """A multi-objective NK landscape (``m`` NK objectives over ``n`` variables).

    Parameters
    ----------
    n_vars : int
        Number of binary variables.
    k : int or sequence of int
        Epistasis level.  A scalar gives homogeneous objectives; a length-``m``
        sequence gives heterogeneous objectives (per-objective ``k_o``).
    n_objectives : int
        Number of objectives (ignored if ``k`` is a sequence).
    seed : int or None
        Random seed.
    """

    def __init__(self, n_vars: int, k: Union[int, Sequence[int]] = 2,
                 n_objectives: int = 2, seed: Optional[int] = None):
        self.n_vars = int(n_vars)
        if np.isscalar(k):
            self.ks = [int(k)] * int(n_objectives)
        else:
            self.ks = [int(v) for v in k]
        self.n_objectives = len(self.ks)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.objectives = [NKObjective.random(self.n_vars, kk, rng) for kk in self.ks]

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Objective values for a 1-D individual or a 2-D population."""
        population = np.asarray(population)
        if population.ndim == 1:
            return np.array([obj.evaluate(population) for obj in self.objectives])
        cols = [obj.evaluate(population) for obj in self.objectives]
        return np.column_stack(cols)

    # -- I/O ---------------------------------------------------------------- #
    def save(self, filepath: str) -> None:
        """Save the instance to a ``.npz`` file (structure + tables per objective)."""
        data = {"n_vars": self.n_vars, "ks": np.array(self.ks),
                "seed": -1 if self.seed is None else self.seed}
        for o, obj in enumerate(self.objectives):
            data[f"lattice_{o}"] = obj.lattice
            data[f"tables_{o}"] = obj.tables
        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str) -> "MNKLandscape":
        d = np.load(filepath)
        n_vars = int(d["n_vars"]); ks = [int(v) for v in d["ks"]]
        inst = cls.__new__(cls)
        inst.n_vars = n_vars
        inst.ks = ks
        inst.n_objectives = len(ks)
        inst.seed = int(d["seed"]) if int(d["seed"]) >= 0 else None
        inst.objectives = [
            NKObjective(n_vars, ks[o], d[f"lattice_{o}"], d[f"tables_{o}"])
            for o in range(len(ks))
        ]
        return inst


def create_mnk_objective_function(instance: MNKLandscape):
    """Return an objective ``f(pop) -> (pop, n_obj)`` maximising all objectives."""
    def objective(population: np.ndarray) -> np.ndarray:
        return instance.evaluate(population)
    return objective


def generate_mnk(n_vars: int, k: Union[int, Sequence[int]] = 2,
                 n_objectives: int = 2, seed: Optional[int] = None) -> MNKLandscape:
    """Convenience builder for an :class:`MNKLandscape` instance."""
    return MNKLandscape(n_vars, k=k, n_objectives=n_objectives, seed=seed)
