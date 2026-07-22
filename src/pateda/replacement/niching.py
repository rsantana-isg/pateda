"""
Diversity-preserving (niching) replacement methods.

These replacement strategies keep the population spread over several basins of
attraction instead of collapsing onto a single optimum, which is important on
multimodal problems and reduces the spurious dependencies a converged EDA
population induces in the learned model. See ``docs/Niching/`` for background.

Three schemes are provided:

* :class:`DeterministicCrowdingReplacement` -- each offspring competes with the
  most similar current individual and wins only if it is at least as fit.
* :class:`RestrictedTournamentReplacement` -- each offspring competes with the
  most similar member of a random window (a robust, ``rng``-aware version of
  :class:`~pateda.replacement.elitist.RTRReplacement`).
* :class:`ClusteringReplacement` -- the merged population is clustered and
  elitism is applied within each cluster, preserving one niche per cluster.

All keep the population size fixed and handle both single-objective ``(N, 1)``
and multi-objective ``(N, m)`` fitness (multi-objective comparisons use the mean
over objectives as a scalar proxy).
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import ReplacementMethod


def _scalar(fitness: np.ndarray) -> np.ndarray:
    """Return a 1-D scalar fitness (mean over objectives if multi-objective)."""
    fitness = np.asarray(fitness, dtype=float)
    if fitness.ndim == 2 and fitness.shape[1] > 1:
        return np.mean(fitness, axis=1)
    return fitness.reshape(-1)


def _distances(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Distance from row vector ``x`` to every row of ``pool``.

    Hamming distance for integer representations, Euclidean otherwise.
    """
    if np.issubdtype(pool.dtype, np.integer) and np.issubdtype(x.dtype, np.integer):
        return np.sum(pool != x, axis=1)
    return np.linalg.norm(pool.astype(float) - x.astype(float), axis=1)


class DeterministicCrowdingReplacement(ReplacementMethod):
    """
    Deterministic crowding.

    Each newly sampled individual competes with the most similar individual in
    the current population and replaces it only if it is at least as fit. Because
    competition is local (nearest neighbour), distinct niches survive even when a
    globally strong solution appears elsewhere.
    """

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pop = old_pop.copy()
        fit = old_fitness.copy()
        s_old = _scalar(fit)
        s_new = _scalar(new_fitness)

        for i in range(new_pop.shape[0]):
            d = _distances(new_pop[i], pop)
            j = int(np.argmin(d))
            if s_new[i] >= s_old[j]:
                pop[j] = new_pop[i]
                fit[j] = new_fitness[i]
                s_old[j] = s_new[i]
        return pop, fit


class RestrictedTournamentReplacement(ReplacementMethod):
    """
    Restricted tournament replacement (RTR).

    For each new individual a random window of ``window_size`` current
    individuals is drawn; the new individual competes with the most similar one
    in the window and replaces it if strictly fitter. A robust, ``rng``-aware
    version of :class:`~pateda.replacement.elitist.RTRReplacement` that supports
    ``(N, 1)`` and multi-objective fitness.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = int(window_size)

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        window = int(params.get("window_size", self.window_size))
        pop = old_pop.copy()
        fit = old_fitness.copy()
        s_old = _scalar(fit)
        s_new = _scalar(new_fitness)
        n = pop.shape[0]
        w = min(window, n)

        for i in range(new_pop.shape[0]):
            idx = rng.choice(n, size=w, replace=False)
            d = _distances(new_pop[i], pop[idx])
            j = int(idx[int(np.argmin(d))])
            if s_new[i] > s_old[j]:
                pop[j] = new_pop[i]
                fit[j] = new_fitness[i]
                s_old[j] = s_new[i]
        return pop, fit


class ClusteringReplacement(ReplacementMethod):
    """
    Clustering-based replacement.

    The merged (old + new) population is partitioned into ``n_clusters`` clusters;
    within each cluster the best individuals are kept, with a per-cluster quota
    proportional to the cluster's size. This preserves at least one representative
    per cluster (niche) while remaining elitist inside each niche.

    Clustering uses k-means (``scikit-learn``) on the population cast to float;
    for discrete representations this is an approximate but effective grouping.
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = int(n_clusters)

    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target = old_pop.shape[0]
        pool = np.vstack([old_pop, new_pop])
        pool_fit = np.vstack([
            np.asarray(old_fitness).reshape(old_pop.shape[0], -1),
            np.asarray(new_fitness).reshape(new_pop.shape[0], -1),
        ])
        s_pool = _scalar(pool_fit)

        n_clusters = int(params.get("n_clusters", self.n_clusters))
        n_clusters = max(1, min(n_clusters, pool.shape[0]))

        labels = self._cluster(pool, n_clusters, rng)

        # per-cluster quota proportional to cluster size, keeping best within
        keep_idx = []
        cluster_ids = np.unique(labels)
        quotas = {}
        for c in cluster_ids:
            members = np.where(labels == c)[0]
            quotas[c] = max(1, int(round(target * len(members) / pool.shape[0])))
        # adjust quotas so they sum to exactly `target`
        self._fix_quotas(quotas, cluster_ids, labels, target)

        for c in cluster_ids:
            members = np.where(labels == c)[0]
            order = members[np.argsort(-s_pool[members])]
            keep_idx.extend(order[: quotas[c]].tolist())

        keep_idx = np.array(keep_idx[:target], dtype=int)
        # if rounding left us short, top up with the globally best remaining
        if keep_idx.shape[0] < target:
            remaining = np.setdiff1d(np.arange(pool.shape[0]), keep_idx)
            extra = remaining[np.argsort(-s_pool[remaining])][: target - keep_idx.shape[0]]
            keep_idx = np.concatenate([keep_idx, extra])

        out_pop = pool[keep_idx]
        out_fit = pool_fit[keep_idx]
        # restore original single-objective 1-column shape if applicable
        if np.asarray(old_fitness).ndim == 1:
            out_fit = out_fit.reshape(-1)
        return out_pop, out_fit

    @staticmethod
    def _cluster(pool: np.ndarray, n_clusters: int,
                 rng: Optional[np.random.Generator]) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans
            seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
            km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed)
            return km.fit_predict(pool.astype(float))
        except Exception:
            # fallback: single cluster (pure elitism)
            return np.zeros(pool.shape[0], dtype=int)

    @staticmethod
    def _fix_quotas(quotas, cluster_ids, labels, target):
        total = sum(quotas.values())
        # reduce/increase quotas (largest clusters first) until they sum to target
        order = sorted(cluster_ids, key=lambda c: -np.sum(labels == c))
        i = 0
        while total > target:
            c = order[i % len(order)]
            if quotas[c] > 1:
                quotas[c] -= 1
                total -= 1
            i += 1
            if i > 10 * len(order) and total > target:
                # hard stop safety
                break
        i = 0
        while total < target:
            c = order[i % len(order)]
            quotas[c] += 1
            total += 1
            i += 1
