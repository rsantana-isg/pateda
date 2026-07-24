"""
plot_bn_eda_extra_figures.py -- per-problem fitness-curve figures and the
algorithm x function UMDA-gain heatmap.

Inputs (from earlier extraction steps, all in bn_eda_analysis/):
    bn_eda_fitness_curves.csv   per (problem, algorithm, gen) seed-averaged fitness
    bn_eda_summary.csv          per-run best fitness (for the gain heatmap)

Outputs (bn_eda_analysis/figures/):
    fitness_curves/curve_<problem>.pdf   one per problem, 19 algorithm curves
    fitness_curves_grid.pdf              33-panel montage
    fig_umda_gain_heatmap.pdf            algorithm x function normalised gain vs UMDA

The UMDA gain of (algorithm a, problem p) is the seed-mean best fitness of a
minus that of the univariate model (UMDA baseline), normalised within the
problem by the spread across algorithms:
    gain(a,p) = (f_a - f_univ) / (max_b f_b - min_b f_b)
so gain > 0 means a beats UMDA and gain < 0 means UMDA is better.

Usage
-----
    python3 scripts/plot_bn_eda_extra_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "bn_eda_analysis")
CURVES = os.path.join(OUT, "bn_eda_fitness_curves.csv")
SUMMARY = os.path.join(OUT, "bn_eda_summary.csv")
UNIV = "univ_bn"

ALG_LABEL = {"univ_bn": "Univ (UMDA)", "k2": "K2", "k2_mi": "K2-MI", "k2_mb": "K2-MB",
             "k2_refine": "K2-Ref", "k2_ensemble": "K2-Ens", "k2_plus": "K2+",
             "fi_k2": "FI-K2", "rfe_k2": "RFE-K2", "bic": "BIC-HC", "aic": "AIC-HC",
             "stable_hc": "HC-Stable", "pc": "PC", "stable_pc": "PC-Stable",
             "dt": "DT", "dmbbn": "DMBBN", "sartre": "SARTRE", "binotears": "BINO",
             "bounded_tw": "BdTW"}
ALG_ORDER = list(ALG_LABEL)
GROUP_ORDER = {"small": 0, "medium": 1, "large": 2}
COLORS = {a: cm.tab20(i % 20) for i, a in enumerate(ALG_ORDER)}


def problem_sort_key(p):
    fam, n = p.rsplit("_", 1)
    return (fam, int(n))


def plot_one(curves, problem, ax, legend=False):
    g = curves[curves["problem"] == problem]
    for a in ALG_ORDER:
        ga = g[g["algorithm"] == a].sort_values("gen")
        if ga.empty:
            continue
        ax.plot(ga["gen"], ga["mean_fitness"], color=COLORS[a], lw=1.3,
                label=ALG_LABEL[a])
    ax.set_title(problem.replace("_", " "), fontsize=10)
    ax.set_xlabel("generation", fontsize=8)
    ax.set_ylabel("mean population fitness", fontsize=8)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=6, ncol=2, loc="best")


def per_problem_figures(curves, out_dir):
    d = os.path.join(out_dir, "figures", "fitness_curves")
    os.makedirs(d, exist_ok=True)
    problems = sorted(curves["problem"].unique(), key=problem_sort_key)
    for p in problems:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        plot_one(curves, p, ax, legend=True)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"curve_{p}.pdf"), bbox_inches="tight")
        plt.close(fig)
    # montage grid (shared legend)
    ncol = 5
    nrow = int(np.ceil(len(problems) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow))
    axes = np.atleast_2d(axes)
    for i, p in enumerate(problems):
        ax = axes[i // ncol][i % ncol]
        plot_one(curves, p, ax, legend=False)
        ax.set_xlabel(""); ax.set_ylabel("")
    for j in range(len(problems), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Line2D([0], [0], color=COLORS[a], lw=2, label=ALG_LABEL[a])
               for a in ALG_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out_dir, "figures", "fitness_curves_grid.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    return problems


def umda_gain_heatmap(summary, out_dir, families=None, suffix=""):
    df = pd.read_csv(summary)
    df["best_fitness"] = pd.to_numeric(df["best_fitness"], errors="coerce")
    if families is not None:
        df = df[df["problem"].str.rsplit("_", n=1).str[0].isin(families)]
    cell = df.groupby(["problem", "group", "algorithm"], as_index=False)["best_fitness"].mean()
    gains = {}
    for (problem, group), g in cell.groupby(["problem", "group"]):
        gg = g.set_index("algorithm")["best_fitness"]
        if UNIV not in gg.index:
            continue
        rng = gg.max() - gg.min()
        if rng <= 0:
            gains[(problem, group)] = {a: 0.0 for a in gg.index}
        else:
            gains[(problem, group)] = {a: (gg[a] - gg[UNIV]) / rng for a in gg.index}
    # build matrix: rows = algorithms (excl UNIV), cols = problems ordered by group/family
    problems = sorted(gains, key=lambda pk: (GROUP_ORDER.get(pk[1], 9),
                                             problem_sort_key(pk[0])))
    algs = [a for a in ALG_ORDER if a != UNIV]
    M = np.full((len(algs), len(problems)), np.nan)
    for j, pk in enumerate(problems):
        for i, a in enumerate(algs):
            if a in gains[pk]:
                M[i, j] = gains[pk][a]
    # order algorithms by mean gain (best first)
    order = np.argsort(-np.nanmean(M, axis=1))
    M = M[order]
    algs = [algs[i] for i in order]

    fig, ax = plt.subplots(figsize=(0.34 * len(problems) + 3, 0.36 * len(algs) + 2))
    vmax = np.nanpercentile(np.abs(M), 98)
    im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([pk[0].replace("_", " ") for pk in problems],
                       rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(algs)))
    ax.set_yticklabels([ALG_LABEL[a] for a in algs], fontsize=8)
    # group separators
    groups = [GROUP_ORDER.get(pk[1], 9) for pk in problems]
    for j in range(1, len(groups)):
        if groups[j] != groups[j - 1]:
            ax.axvline(j - 0.5, color="black", lw=1.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=4.5,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("normalised best-fitness gain vs UMDA  (blue > UMDA, red < UMDA)",
                 fontsize=8)
    fname = f"fig_umda_gain_heatmap{suffix}.pdf"
    fig.savefig(os.path.join(out_dir, "figures", fname), bbox_inches="tight")
    plt.close(fig)
    # also dump the matrix as CSV
    pd.DataFrame(M, index=[ALG_LABEL[a] for a in algs],
                 columns=[pk[0] for pk in problems]).to_csv(
        os.path.join(out_dir, f"umda_gain_matrix{suffix}.csv"))
    print(f"UMDA-gain heatmap{suffix}: {len(algs)} algorithms x {len(problems)} problems; "
          f"cells<0 (UMDA better): {int(np.nansum(M < 0))} / {int(np.sum(~np.isnan(M)))}")


def main():
    out_dir = OUT
    curves = pd.read_csv(CURVES)
    problems = per_problem_figures(curves, out_dir)
    print(f"Wrote {len(problems)} per-problem fitness-curve figures + montage.")
    umda_gain_heatmap(SUMMARY, out_dir)
    umda_gain_heatmap(SUMMARY, out_dir,
                      families={"Trap", "Ising", "Deceptive3", "Checkerboard"},
                      suffix="_selected")
    print(f"Figures in {os.path.join(out_dir, 'figures')}")


if __name__ == "__main__":
    main()
