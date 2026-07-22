"""
Network crossover — structure-guided recombination for EDAs

Most EDA samplers in pateda draw new solutions from the *probabilities* of the
learned model.  Network crossover instead exploits only the model's *structure*
(its linkage graph) as a recombination operator, in the spirit of Hauschild &
Pelikan (2010): tightly-linked variables are inherited together from the same
parent, so building blocks captured by the model are not disrupted.

Given two parents and an undirected linkage graph ``G`` (extracted from the
learned model, or supplied directly from the problem structure), a *connected*
crossover mask ``M`` of about ``n/2`` variables is grown on ``G``; the first
child takes the mask variables from parent 1 and the rest from parent 2, and the
second child is the complement.  Because ``M`` is connected in ``G``, the edges
it *cuts* tend to join variables that are far apart in the linkage graph -- the
weakest dependencies -- so strong linkages are preserved.

This is implemented as an ordinary :class:`~pateda.core.components.SamplingMethod`
so it plugs into an EDA exactly like probabilistic samplers: the EDA passes the
current population as ``aux_pop`` (the parent pool) and the learned model, whose
structure supplies ``G``.  It is the structural counterpart of the non-model
crossover operators in :mod:`pateda.crossover` (e.g.
:class:`~pateda.crossover.block.SampleBlockCrossover`), and complements the
partial samplers in :mod:`pateda.sampling.partial`: partial sampling *resamples*
a subset of variables from the model, whereas network crossover *swaps* a
connected subset between two parents.

Two mask-growth variants are provided (``mask_method``):

- ``"bfs"`` -- randomized breadth-first growth (Hauschild & Pelikan, 2010): the
  mask expands to all neighbors of the current mask, added in random order.
- ``"random_walk"`` -- random-walk growth (Stonedahl, Rand & Wilensky, 2008):
  the mask follows a random walk along the graph edges.

Both grow a connected cluster (restarting from a fresh random seed variable if
the current component is exhausted before the target size is reached), so on a
graph with no edges they degrade gracefully to a uniform-like crossover.

References
----------
- Hauschild, M., & Pelikan, M. (2010). "Network crossover performance on NK
  landscapes and deceptive problems." GECCO 2010 / MEDAL Report 2010003.
- Stonedahl, F., Rand, W., & Wilensky, U. (2008). "CrossNet: a framework for
  crossover with network-based chromosomal structures." GECCO 2008.
"""

from typing import Any, List, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model
from pateda.knowledge_extraction.model_structure import model_to_linkage_graph


def _adjacency_list(graph: np.ndarray) -> List[np.ndarray]:
    """Neighbor-index lists from a symmetric 0/1 adjacency matrix."""
    return [np.where(graph[i] > 0)[0] for i in range(graph.shape[0])]


def grow_mask_bfs(
    neighbors: List[np.ndarray],
    n_vars: int,
    target_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grow a connected crossover mask by randomized breadth-first search.

    Reproduces the mask construction of Hauschild & Pelikan (2010): starting
    from a random variable, repeatedly add all not-yet-included neighbors of the
    current mask (in random order) until the target size is reached; if the
    connected component is exhausted first, restart from a fresh random variable.
    """
    in_mask = np.zeros(n_vars, dtype=bool)
    mask: List[int] = []
    while len(mask) < target_size:
        # Fresh seed from the variables not yet in the mask.
        remaining = np.where(~in_mask)[0]
        if remaining.size == 0:
            break
        start = int(rng.choice(remaining))
        in_mask[start] = True
        mask.append(start)
        # Breadth-first expansion of the current mask.
        while len(mask) < target_size:
            frontier = []
            for v in mask:
                for w in neighbors[v]:
                    if not in_mask[w]:
                        frontier.append(int(w))
            if not frontier:
                break                       # component exhausted -> new seed
            frontier = np.unique(frontier)
            rng.shuffle(frontier)
            for w in frontier:
                if len(mask) >= target_size:
                    break
                if not in_mask[w]:
                    in_mask[w] = True
                    mask.append(int(w))
    return in_mask


def grow_mask_random_walk(
    neighbors: List[np.ndarray],
    n_vars: int,
    target_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grow a connected crossover mask by a random walk along the edges.

    Reproduces the network-based recombination of Stonedahl et al. (2008): from
    a random variable, step to a random neighbor and add it; when the walk gets
    stuck (all neighbors already in the mask, or an isolated variable), restart
    from a fresh random variable.
    """
    in_mask = np.zeros(n_vars, dtype=bool)
    count = 0
    current = -1
    while count < target_size:
        if current < 0:
            remaining = np.where(~in_mask)[0]
            if remaining.size == 0:
                break
            current = int(rng.choice(remaining))
            in_mask[current] = True
            count += 1
            continue
        nbrs = [int(w) for w in neighbors[current] if not in_mask[w]]
        if not nbrs:
            current = -1                    # stuck -> restart the walk
            continue
        nxt = int(rng.choice(nbrs))
        in_mask[nxt] = True
        count += 1
        current = nxt
    return in_mask


class SampleNetworkCrossover(SamplingMethod):
    """
    Network crossover sampler (structure-guided recombination).

    Grows a connected crossover mask on the model's linkage graph and produces
    two complementary offspring per parent pair.  Shares the sampling interface
    of every other pateda sampler; the parent pool is the population the EDA
    passes as ``aux_pop``.
    """

    def __init__(
        self,
        n_samples: int,
        mask_method: str = "bfs",
        linkage_graph: Optional[np.ndarray] = None,
        mask_fraction: float = 0.5,
        mutation_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_samples: Number of offspring to generate.
            mask_method: Mask-growth variant, ``"bfs"`` (Hauschild & Pelikan) or
                ``"random_walk"`` (Stonedahl et al.).
            linkage_graph: Optional fixed ``(n_vars, n_vars)`` symmetric 0/1
                interaction matrix (problem-known structure).  When ``None`` the
                graph is extracted from the learned model at sampling time.
            mask_fraction: Target mask size as a fraction of ``n_vars`` (the
                paper grows the mask up to ``n/2``, i.e. ``0.5``).
            mutation_prob: Optional per-gene random-reset mutation applied to the
                offspring (0 disables it).
            seed: Seed for this operator's private RNG (``None`` -> fresh).
        """
        if mask_method not in ("bfs", "random_walk"):
            raise ValueError(
                f"mask_method must be 'bfs' or 'random_walk', got {mask_method!r}"
            )
        if not 0.0 < mask_fraction < 1.0:
            raise ValueError(f"mask_fraction must be in (0, 1), got {mask_fraction}")
        self.n_samples = n_samples
        self.mask_method = mask_method
        self.linkage_graph = None if linkage_graph is None else np.asarray(linkage_graph)
        self.mask_fraction = mask_fraction
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)

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
        Generate offspring by network crossover.

        Args:
            n_vars: Number of variables.
            model: Learned model whose structure supplies the linkage graph
                (ignored when a fixed ``linkage_graph`` was given).
            cardinality: Variable cardinalities (used only by the optional
                mutation).
            aux_pop: Parent pool -- the current population, passed by the EDA.
                Required.
            aux_fitness: Unused.
            rng: Optional RNG (defaults to the operator's private RNG).
            **params: Per-call overrides (``n_samples``, ``mask_fraction``,
                ``mask_method``, ``mutation_prob``, ``linkage_graph``).

        Returns:
            Offspring population of shape ``(n_samples, n_vars)``.
        """
        gen = rng if rng is not None else self.rng
        if aux_pop is None:
            raise ValueError("SampleNetworkCrossover requires aux_pop (the parent pool)")

        n_samples = int(params.get("n_samples", self.n_samples))
        mask_fraction = float(params.get("mask_fraction", self.mask_fraction))
        mask_method = params.get("mask_method", self.mask_method)
        mutation_prob = float(params.get("mutation_prob", self.mutation_prob))

        graph = params.get("linkage_graph", self.linkage_graph)
        if graph is None:
            graph = model_to_linkage_graph(model, n_vars)
        graph = np.asarray(graph)
        neighbors = _adjacency_list(graph)

        parents = np.asarray(aux_pop)
        n_parents = parents.shape[0]
        target = max(1, min(n_vars - 1, int(round(mask_fraction * n_vars))))
        grow = grow_mask_bfs if mask_method == "bfs" else grow_mask_random_walk

        offspring = np.empty((n_samples, n_vars), dtype=parents.dtype)
        i = 0
        while i < n_samples:
            p1, p2 = gen.integers(0, n_parents, size=2)
            mask = grow(neighbors, n_vars, target, gen)
            # Child 1: mask from parent1, complement from parent2.
            child1 = np.where(mask, parents[p1], parents[p2])
            offspring[i] = child1
            i += 1
            if i < n_samples:
                # Child 2: the complementary inheritance.
                offspring[i] = np.where(mask, parents[p2], parents[p1])
                i += 1

        if mutation_prob > 0.0:
            card = np.asarray(cardinality).astype(int).reshape(n_vars)
            flip = gen.random((n_samples, n_vars)) < mutation_prob
            rows, cols = np.where(flip)
            if rows.size:
                col_card = card[cols]
                offspring[rows, cols] = (offspring[rows, cols]
                                         + 1 + gen.integers(0, np.maximum(col_card - 1, 1))
                                         ) % col_card

        return offspring
