"""
Visualizations for the network-theoretic analysis of EDA models.

These functions plot the evolution of the network measures computed by
:mod:`pateda.knowledge_extraction.network_measures` and the structure-mining
artefacts (edge-frequency matrices, degree distributions, motif spectra,
network snapshots) discussed in:

  * Santana et al., "Network measures for information extraction in evolutionary
    algorithms", IJCIS 6(6), 2013.
  * Santana et al., "Mining probabilistic models learned by EDAs ...",
    GECCO-2009.

All functions accept an optional ``save_path``; when given the figure is saved
(``.png``/``.pdf``/``.eps`` inferred from the extension) and closed, otherwise
the matplotlib ``Figure`` is returned for interactive use.

Author: Roberto Santana (roberto.santana@ehu.eus)
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pateda.knowledge_extraction.network_measures import (
    aggregate_degree_distribution,
    edge_frequency_matrix,
    to_networkx,
    triad_census_series,
    _CONNECTED_TRIADS,
)

try:
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False


def _finish(fig, save_path: Optional[str]):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# Evolution of network measures
# ---------------------------------------------------------------------------

def plot_measures_evolution(
    evolution: Dict[str, Any],
    measures: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot the per-generation evolution of selected scalar network measures.

    Parameters
    ----------
    evolution : dict
        Output of ``compute_measures_evolution`` (must contain ``'series'``).
    measures : sequence of str, optional
        Names of the measures to plot.  Defaults to a representative subset.
    """
    series = evolution["series"]
    if measures is None:
        measures = ["n_edges", "density", "clustering_mean",
                    "characteristic_path_length", "max_modularity",
                    "motif_number_z3"]
    measures = [m for m in measures if m in series]

    n = len(measures)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    gens = np.arange(evolution["n_generations"])

    for idx, name in enumerate(measures):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(gens, series[name], marker="o", ms=3, lw=1.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def compare_measure_evolution(
    evolutions: Dict[str, Dict[str, Any]],
    measure: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Compare one network measure across several EDA runs.

    Parameters
    ----------
    evolutions : dict
        Mapping ``label -> compute_measures_evolution(...)`` output.
    measure : str
        Scalar measure name to compare (e.g. ``'max_modularity'``).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ev in evolutions.items():
        series = ev["series"]
        if measure not in series:
            continue
        gens = np.arange(ev["n_generations"])
        ax.plot(gens, series[measure], marker="o", ms=3, lw=1.6, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel(measure)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def compare_measures_grid(
    evolutions: Dict[str, Dict[str, Any]],
    measures: Sequence[str],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Grid comparing several measures across several EDAs (one panel/measure)."""
    n = len(measures)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    for idx, measure in enumerate(measures):
        ax = axes[idx // ncols][idx % ncols]
        for label, ev in evolutions.items():
            series = ev["series"]
            if measure not in series:
                continue
            gens = np.arange(ev["n_generations"])
            ax.plot(gens, series[measure], marker="o", ms=2.5, lw=1.4, label=label)
        ax.set_xlabel("Generation")
        ax.set_ylabel(measure)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Structure-mining artefacts
# ---------------------------------------------------------------------------

def plot_edge_frequency_matrix(
    adjacencies: Sequence[np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Heat-map of the edge-frequency (coincidence) matrix over generations."""
    freq = edge_frequency_matrix(adjacencies)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(freq, cmap="hot", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("Variable j")
    ax.set_ylabel("Variable i")
    fig.colorbar(im, ax=ax, label="Arc frequency")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_degree_distribution(
    adjacencies: Sequence[np.ndarray],
    directed: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Average number of vertices for each vertex degree (Mining-paper Fig. 2)."""
    dist = aggregate_degree_distribution(adjacencies, directed=directed)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(dist)), dist, color="steelblue")
    ax.set_xlabel("Vertex degree")
    ax.set_ylabel("Average number of vertices")
    ax.grid(True, alpha=0.3, axis="y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_motif_evolution(
    adjacencies: Sequence[np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Evolution of the directed triad (Z=3 motif) frequencies across generations."""
    series = triad_census_series(adjacencies)
    gens = np.arange(len(adjacencies))
    fig, ax = plt.subplots(figsize=(9, 5))
    for triad in _CONNECTED_TRIADS:
        vals = series[triad]
        if np.any(vals > 0):
            ax.plot(gens, vals, marker="o", ms=2.5, lw=1.2, label=triad)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Motif frequency (Z=3)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, title="Triad type")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_network_snapshots(
    adjacencies: Sequence[np.ndarray],
    generations: Optional[Sequence[int]] = None,
    directed: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Draw the learned network at a few selected generations."""
    if not _HAS_NX:
        raise ImportError("networkx is required to draw network snapshots")
    n_gen = len(adjacencies)
    if generations is None:
        if n_gen <= 4:
            generations = list(range(n_gen))
        else:
            generations = [0, n_gen // 3, 2 * n_gen // 3, n_gen - 1]
    generations = [g for g in generations if 0 <= g < n_gen]

    k = len(generations)
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 4), squeeze=False)
    for ax_idx, g in enumerate(generations):
        ax = axes[0][ax_idx]
        adj = np.asarray(adjacencies[g])
        if adj.size == 0:
            ax.axis("off")
            ax.set_title(f"gen {g} (empty)")
            continue
        G = to_networkx((adj != 0).astype(int), directed=directed)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=180, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=directed,
                               arrowsize=8, edge_color="gray")
        ax.set_title(f"gen {g}  ({G.number_of_edges()} arcs)")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_betweenness_two_approaches(
    evolution: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """The two betweenness representations of the paper (Fig. 11).

    Left: *vertex approach* — average vertex betweenness condensed over all
    generations, per vertex.  Right: *generation approach* — average vertex
    betweenness at each generation.
    """
    per_gen = evolution["per_generation"]
    # Vertex approach: average each vertex's betweenness across generations,
    # recomputed from the stored adjacency matrices.
    vertex_acc: Dict[int, List[float]] = {}
    for g in per_gen:
        adj = np.asarray(g.get("adjacency", np.zeros((0, 0))))
        if adj.size == 0:
            continue
        G = to_networkx((adj != 0).astype(int), bool(g.get("directed", True)))
        bc = nx.betweenness_centrality(G)
        for node, val in bc.items():
            vertex_acc.setdefault(node, []).append(val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if vertex_acc:
        nodes = sorted(vertex_acc)
        means = [np.mean(vertex_acc[v]) for v in nodes]
        ax1.bar(nodes, means, color="seagreen")
    ax1.set_xlabel("Vertex")
    ax1.set_ylabel("Avg betweenness (over generations)")
    ax1.set_title("Vertex approach")
    ax1.grid(True, alpha=0.3, axis="y")

    gens = np.arange(evolution["n_generations"])
    ax2.plot(gens, evolution["series"]["vertex_betweenness_mean"],
             marker="o", ms=3, color="purple")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Avg betweenness (over vertices)")
    ax2.set_title("Generation approach")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)
