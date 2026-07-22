"""
Structural transfer learning between EDA runs.

The models learned during an EDA run encode which variables interact. This
module lets that information be *aggregated* over a source run, *persisted*, and
*injected* into a subsequent run on a related problem, so that structure learning
does not have to be rediscovered from scratch. Three forms of transfer are
supported (see the user guide, "Transfer Learning"):

* **Structural transfer** -- turn the aggregated edge frequencies into a binary
  ``interaction_matrix`` that restricts (hard) or biases (soft) structure
  learning in the target run. The restricting variant is consumed directly by
  ``TreeEDAR``, ``MNFDAR`` and ``MNFDAGR``.
* **Parametric transfer** -- warm-start the target population from the best
  solutions of the source run (used with ``SeedThisPop``).

Reference: Santana, Mendiburu & Lozano, "Structural transfer using EDAs" (2012).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
import numpy as np

from pateda.core.models import Model
from pateda.knowledge_extraction.model_structure import model_to_linkage_graph


def aggregate_edge_frequencies(
    models: Sequence[Model],
    n_vars: Optional[int] = None,
) -> np.ndarray:
    """
    Average, over a sequence of learned models, the presence of each edge.

    Parameters
    ----------
    models : sequence of Model
        The models learned across the generations of a run (e.g. ``cache.models``).
    n_vars : int, optional
        Number of variables (inferred from the first model when omitted).

    Returns
    -------
    np.ndarray
        Symmetric ``(n_vars, n_vars)`` matrix whose ``(i, j)`` entry is the
        fraction of models in which the undirected edge ``(i, j)`` was present.
    """
    if len(models) == 0:
        raise ValueError("aggregate_edge_frequencies requires at least one model")
    graphs = [model_to_linkage_graph(m, n_vars) for m in models]
    n = graphs[0].shape[0]
    freq = np.zeros((n, n), dtype=float)
    for g in graphs:
        freq += (g > 0).astype(float)
    freq /= len(graphs)
    np.fill_diagonal(freq, 0.0)
    return freq


def structure_to_interaction_matrix(
    freq: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Threshold an edge-frequency matrix into a binary interaction matrix.

    Parameters
    ----------
    freq : np.ndarray
        Symmetric edge-frequency matrix (e.g. from
        :func:`aggregate_edge_frequencies`).
    threshold : float
        Edges whose frequency is ``>= threshold`` are allowed (``1``); the rest
        are forbidden (``0``).

    Returns
    -------
    np.ndarray
        Symmetric 0/1 interaction matrix with zero diagonal, suitable as the
        ``interaction_matrix`` argument of the restricted EDAs.
    """
    freq = np.asarray(freq, dtype=float)
    M = (freq >= threshold).astype(int)
    M = ((M + M.T) > 0).astype(int)
    np.fill_diagonal(M, 0)
    return M


@dataclass
class TransferredStructure:
    """
    Persistable container for the structural knowledge extracted from a run.

    Attributes
    ----------
    edge_frequencies : np.ndarray
        The aggregated edge-frequency matrix of the source run.
    n_vars : int
        Number of variables the structure is defined over.
    metadata : dict
        Free-form provenance information (source problem, generations, ...).
    """

    edge_frequencies: np.ndarray
    n_vars: int
    metadata: dict

    @classmethod
    def from_models(cls, models: Sequence[Model], n_vars: Optional[int] = None,
                    **metadata: Any) -> "TransferredStructure":
        freq = aggregate_edge_frequencies(models, n_vars)
        return cls(edge_frequencies=freq, n_vars=freq.shape[0], metadata=dict(metadata))

    def interaction_matrix(self, threshold: float = 0.5) -> np.ndarray:
        """Hard structural transfer: a binary interaction matrix."""
        return structure_to_interaction_matrix(self.edge_frequencies, threshold)

    def soft_bias(self, strength: float = 1.0) -> np.ndarray:
        """
        Soft structural transfer: a per-edge score bias in ``[0, strength]``.

        The bias is ``strength * edge_frequencies`` and is intended to be added
        to a structure-learning score to favour edges that were useful on the
        source problem (a soft, distance-free analogue of the soft distance-based
        bias of Hauschild & Pelikan, 2012). It is returned for use by score-based
        learners that accept an additive edge prior.
        """
        return float(strength) * self.edge_frequencies

    def save(self, path: str) -> None:
        """Persist to a ``.npz`` file."""
        np.savez(
            path,
            edge_frequencies=self.edge_frequencies,
            n_vars=np.array([self.n_vars]),
        )

    @classmethod
    def load(cls, path: str) -> "TransferredStructure":
        data = np.load(path, allow_pickle=True)
        return cls(
            edge_frequencies=data["edge_frequencies"],
            n_vars=int(data["n_vars"][0]),
            metadata={},
        )


def warm_start_population(
    source_populations: Union[np.ndarray, List[np.ndarray]],
    source_fitness: Union[np.ndarray, List[np.ndarray]],
    pop_size: int,
    fraction: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a warm-start population for parametric transfer.

    The best ``fraction`` of the source solutions are copied into the new
    population and the remaining slots are filled by resampling (with
    replacement) from those elites, giving a target population seeded around the
    source's best regions. Use with :class:`~pateda.seeding.seed_thispop.SeedThisPop`.

    Parameters
    ----------
    source_populations, source_fitness :
        Arrays (or lists of per-generation arrays, e.g. ``cache.populations`` /
        ``cache.fitness_values``) from the source run.
    pop_size : int
        Size of the population to build.
    fraction : float
        Fraction of ``pop_size`` taken as elites from the source.
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(source_populations, list):
        pop = np.vstack(source_populations)
    else:
        pop = np.asarray(source_populations)
    if isinstance(source_fitness, list):
        fit = np.vstack([np.asarray(f).reshape(len(np.asarray(f)), -1)
                         for f in source_fitness])
    else:
        fit = np.asarray(source_fitness).reshape(pop.shape[0], -1)

    s = fit.mean(axis=1) if fit.shape[1] > 1 else fit.reshape(-1)
    n_elite = max(1, int(round(pop_size * fraction)))
    n_elite = min(n_elite, pop.shape[0])
    elite_idx = np.argsort(-s)[:n_elite]
    elites = pop[elite_idx]

    out = np.empty((pop_size, pop.shape[1]), dtype=pop.dtype)
    out[:n_elite] = elites[:n_elite] if n_elite <= pop_size else elites[:pop_size]
    if pop_size > n_elite:
        fill = rng.integers(0, n_elite, size=pop_size - n_elite)
        out[n_elite:] = elites[fill]
    return out
