"""
Visualizations for the knowledge extracted from continuous-EDA models
(Gaussian networks and vine copulas).

Complements :mod:`pateda.knowledge_extraction.network_visualizations` (which is
representation-agnostic) with plots specific to the continuous case:
Gaussian-parameter / partial-correlation evolution, precision heat-maps,
partial-correlation networks, vine first-tree networks, pair-copula family
composition and Kendall's-tau-by-tree statistics, and Gaussian-vs-Bayesian
network comparisons.

References: see :mod:`pateda.knowledge_extraction.gaussian_networks` and
:mod:`pateda.knowledge_extraction.vine_analysis`.

Author: Roberto Santana (roberto.santana@ehu.eus)
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pateda.knowledge_extraction.network_measures import to_networkx
from pateda.knowledge_extraction.vine_analysis import (
    analyze_vine,
    first_tree_network,
    family_composition,
    tau_by_tree,
)
from pateda.knowledge_extraction.gaussian_networks import compare_networks

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
# Gaussian models
# ---------------------------------------------------------------------------

def plot_gaussian_parameter_evolution(
    gaussian_evolution: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Evolution of variance and dependence strength of the Gaussian model.

    Top: mean variance (search-focus / convergence).  Bottom: mean absolute
    partial correlation (strength of conditional dependencies) per generation.
    Expects the dict returned by ``gaussian_network_evolution`` (uses its
    ``covariances`` and ``partial_correlations``).
    """
    covs = gaussian_evolution["covariances"]
    pcorrs = gaussian_evolution["partial_correlations"]
    gens = np.arange(len(covs))

    mean_var = [float(np.mean(np.diag(c))) if np.asarray(c).size else np.nan for c in covs]

    def _mean_abs_offdiag(m):
        m = np.asarray(m)
        if m.size == 0 or m.shape[0] < 2:
            return np.nan
        iu = np.triu_indices_from(m, k=1)
        return float(np.mean(np.abs(m[iu])))

    mean_pcorr = [_mean_abs_offdiag(p) for p in pcorrs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(gens, mean_var, marker="o", ms=3, color="navy")
    ax1.set_ylabel("Mean variance")
    ax1.grid(True, alpha=0.3)
    ax2.plot(gens, mean_pcorr, marker="o", ms=3, color="darkred")
    ax2.set_ylabel("Mean |partial corr.|")
    ax2.set_xlabel("Generation")
    ax2.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_precision_heatmap(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    label: str = "partial correlation",
):
    """Heat-map of a precision / partial-correlation / covariance matrix."""
    m = np.asarray(matrix, dtype=float)
    vmax = float(np.nanmax(np.abs(m))) if m.size else 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Variable j")
    ax.set_ylabel("Variable i")
    fig.colorbar(im, ax=ax, label=label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_partial_correlation_network(
    network: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Draw a Gaussian interaction network with signed, weighted edges.

    Expects the dict from ``gaussian_interaction_network`` (uses ``adjacency``
    and ``weights``); blue/red edges denote positive/negative partial
    correlations, edge width is proportional to ``|weight|``.
    """
    if not _HAS_NX:
        raise ImportError("networkx is required to draw networks")
    adj = np.asarray(network["adjacency"])
    weights = np.asarray(network["weights"])
    G = to_networkx((adj != 0).astype(int), directed=False)
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightyellow",
                           edgecolors="gray", node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    for (u, v) in G.edges():
        w = weights[u, v]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=("steelblue" if w >= 0 else "firebrick"),
                lw=1 + 4 * min(abs(w), 1.0), alpha=0.7, zorder=0)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_network_comparison(
    adjacency_gaussian: np.ndarray,
    adjacency_other: np.ndarray,
    labels=("Gaussian", "Bayesian net"),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Side-by-side networks plus their agreement (Jaccard) in the title.

    Demonstrates *combining* the Gaussian-covariance network with a
    Bayesian-network (or known-interaction) network.
    """
    if not _HAS_NX:
        raise ImportError("networkx is required to draw networks")
    cmp = compare_networks(adjacency_gaussian, adjacency_other)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mats = [adjacency_gaussian, adjacency_other, cmp["intersection"]]
    names = [labels[0], labels[1], "agreement (∩)"]
    colors = ["lightyellow", "lightblue", "lightgreen"]
    for ax, m, name, col in zip(axes, mats, names, colors):
        G = to_networkx((np.asarray(m) != 0).astype(int), directed=False)
        pos = nx.circular_layout(G)
        # Draw edges manually (avoids a networkx 3.x / NumPy 2.0 incompatibility
        # in draw_networkx_edges with a string edge_color).
        for (u, v) in G.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="gray", lw=1.3, alpha=0.7, zorder=0)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col,
                               edgecolors="gray", node_size=260)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.set_title(f"{name}  ({G.number_of_edges()} edges)")
        ax.set_axis_off()
    sup = (title + "  " if title else "") + \
        f"Jaccard={cmp['jaccard']:.2f}, common={cmp['common_edges']}"
    fig.suptitle(sup)
    fig.tight_layout()
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Vine copulas
# ---------------------------------------------------------------------------

_FAMILY_COLORS = {
    "gaussian": "tab:blue", "student": "tab:cyan", "t": "tab:cyan",
    "clayton": "tab:green", "gumbel": "tab:red", "frank": "tab:orange",
    "joe": "tab:purple", "bb1": "tab:brown", "bb6": "tab:pink",
    "bb7": "tab:olive", "bb8": "tab:gray",
    "indep": "lightgray", "independence": "lightgray",
}


def plot_vine_first_tree(
    model: Any,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Draw the first vine tree ``T_1`` with tau-labelled, family-coloured edges."""
    if not _HAS_NX:
        raise ImportError("networkx is required to draw networks")
    net = first_tree_network(model)
    adj = net["adjacency"]
    G = to_networkx((adj != 0).astype(int), directed=False)
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="white",
                           edgecolors="black", node_size=320)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    for (i0, j0, tau, fam) in net["edge_list"]:
        col = _FAMILY_COLORS.get(fam.lower(), "black")
        ax.plot([pos[i0][0], pos[j0][0]], [pos[i0][1], pos[j0][1]],
                color=col, lw=1 + 4 * min(abs(tau), 1.0), alpha=0.8, zorder=0)
        mx, my = (pos[i0][0] + pos[j0][0]) / 2, (pos[i0][1] + pos[j0][1]) / 2
        ax.text(mx, my, f"{tau:.2f}", fontsize=7, color=col)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_family_composition(
    model: Any,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    include_independence: bool = False,
):
    """Bar chart of pair-copula family counts in the learned vine."""
    comp = family_composition(model, include_independence=include_independence)
    counts = comp["counts"]
    fams = sorted(counts, key=lambda f: counts[f], reverse=True)
    vals = [counts[f] for f in fams]
    colors = [_FAMILY_COLORS.get(f.lower(), "tab:blue") for f in fams]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(fams, vals, color=colors)
    ax.set_ylabel("Number of pair-copulas")
    ax.set_xlabel("Copula family")
    ax.grid(True, alpha=0.3, axis="y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_tau_by_tree(
    model: Any,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Bar chart of the mean absolute Kendall's tau per vine tree."""
    stats = tau_by_tree(model)
    trees = sorted(stats["mean_abs_tau_by_tree"])
    means = [stats["mean_abs_tau_by_tree"][t] for t in trees]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"T{t}" for t in trees], means, color="teal")
    ax.set_ylabel("Mean |Kendall's tau|")
    ax.set_xlabel("Vine tree")
    ax.grid(True, alpha=0.3, axis="y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_vine_evolution(
    vine_evolution: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Evolution of vine summary statistics and family frequencies.

    Panels: first-tree edges, effective truncation level, overall mean ``|tau|``,
    and per-family frequency, across generations.  Expects the dict from
    ``vine_analysis.vine_evolution``.
    """
    series = vine_evolution["series"]
    gens = np.arange(vine_evolution["n_generations"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(gens, series["first_tree_edges"], marker="o", ms=3, color="navy")
    axes[0, 0].set_ylabel("First-tree edges"); axes[0, 0].set_xlabel("Generation")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gens, series["effective_truncation"], marker="o", ms=3, color="darkgreen")
    axes[0, 1].set_ylabel("Effective truncation"); axes[0, 1].set_xlabel("Generation")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(gens, series["overall_mean_abs_tau"], marker="o", ms=3, color="darkred")
    axes[1, 0].set_ylabel("Overall mean |tau|"); axes[1, 0].set_xlabel("Generation")
    axes[1, 0].grid(True, alpha=0.3)

    for fam, vals in vine_evolution["family_frequency_series"].items():
        if fam.lower() in ("indep", "independence"):
            continue
        axes[1, 1].plot(gens, vals, marker="o", ms=2.5,
                        color=_FAMILY_COLORS.get(fam.lower()), label=fam)
    axes[1, 1].set_ylabel("Family frequency"); axes[1, 1].set_xlabel("Generation")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)
