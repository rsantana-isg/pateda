"""
stats_utils.py
--------------
Statistical comparison helpers following the conventions of Ceberio et al. (2015)
and the wider pateda experimental framework.

Public API
----------
summary_table(df)          → pd.DataFrame  (mean ± std, best, rank)
friedman_test(df)          → dict
wilcoxon_pairwise(df)      → pd.DataFrame
critical_difference_plot(df, alpha, filepath)  → None
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    group_cols: Sequence[str] = ("Problem", "Algorithm"),
) -> pd.DataFrame:
    """Return a summary table with mean ± std, best value, and rank per problem.

    Parameters
    ----------
    df:
        Raw results DataFrame with one row per (problem, algorithm, seed).
    value_col:
        Column containing the fitness / objective value.
    group_cols:
        Columns to group by (problem first, then algorithm).

    Returns
    -------
    pd.DataFrame indexed by (Problem, Algorithm) with columns:
        Mean, Std, Best, Rank
    """
    agg = (
        df.groupby(list(group_cols))[value_col]
        .agg(Mean="mean", Std="std", Best="max")
        .round(4)
    )
    # Rank within each problem (higher = better by default, flip if minimisation)
    agg["Rank"] = (
        agg.groupby(level=0)["Mean"]
        .rank(ascending=False, method="average")
    )
    agg["Mean±Std"] = agg.apply(
        lambda r: f"{r['Mean']:.2f} ± {r['Std']:.2f}", axis=1
    )
    return agg[["Mean±Std", "Best", "Rank", "Mean", "Std"]]


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def friedman_test(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    problem_col: str = "Problem",
    algorithm_col: str = "Algorithm",
    seed_col: str = "Seed",
) -> dict:
    """Perform the Friedman test across algorithms.

    For each problem the algorithms are ranked per seed; the Friedman
    statistic is computed on those ranks (over problems × seeds as blocks).

    Returns
    -------
    dict with keys: statistic, p_value, reject_H0 (at α=0.05)
    """
    algorithms = df[algorithm_col].unique()
    # Build matrix: rows = blocks (problem × seed), cols = algorithm ranks
    records = []
    for prob in df[problem_col].unique():
        for seed in df[seed_col].unique():
            row = df[(df[problem_col] == prob) & (df[seed_col] == seed)]
            if len(row) == len(algorithms):
                records.append(
                    [row[row[algorithm_col] == a][value_col].values[0] for a in algorithms]
                )
    if not records:
        return {"statistic": np.nan, "p_value": np.nan, "reject_H0": False}

    matrix = np.array(records)  # (n_blocks, n_algorithms)
    stat, p = friedmanchisquare(*matrix.T)
    return {
        "statistic": round(stat, 4),
        "p_value": round(p, 6),
        "reject_H0": p < 0.05,
        "algorithms": list(algorithms),
    }


# ---------------------------------------------------------------------------
# Wilcoxon pairwise post-hoc
# ---------------------------------------------------------------------------

def wilcoxon_pairwise(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    algorithm_col: str = "Algorithm",
    seed_col: str = "Seed",
    problem_col: str = "Problem",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test for each pair of algorithms.

    Values are pooled across seeds and problems (one observation per
    problem × seed block), so differences are evaluated on paired blocks.

    Returns
    -------
    pd.DataFrame with columns: AlgA, AlgB, statistic, p_value, significant
    """
    algorithms = sorted(df[algorithm_col].unique())
    rows = []
    for a, b in itertools.combinations(algorithms, 2):
        paired_a, paired_b = [], []
        for prob in df[problem_col].unique():
            for seed in df[seed_col].unique():
                va = df[(df[problem_col] == prob) & (df[seed_col] == seed) & (df[algorithm_col] == a)][value_col].values
                vb = df[(df[problem_col] == prob) & (df[seed_col] == seed) & (df[algorithm_col] == b)][value_col].values
                if len(va) == 1 and len(vb) == 1:
                    paired_a.append(va[0])
                    paired_b.append(vb[0])
        try:
            stat, p = wilcoxon(paired_a, paired_b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(
            {
                "AlgA": a,
                "AlgB": b,
                "statistic": round(stat, 4) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6) if not np.isnan(p) else np.nan,
                "significant": bool(p < alpha) if not np.isnan(p) else False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Critical-difference diagram (Demšar 2006)
# ---------------------------------------------------------------------------

def critical_difference_plot(
    df: pd.DataFrame,
    alpha: float = 0.05,
    filepath: str | None = None,
    value_col: str = "Best Fitness",
    algorithm_col: str = "Algorithm",
    problem_col: str = "Problem",
    seed_col: str = "Seed",
    maximize: bool = True,
) -> None:
    """Draw a critical-difference diagram (Demšar, 2006).

    Algorithms that are **not** significantly different are connected by a
    horizontal bar.

    Parameters
    ----------
    df:
        Raw results DataFrame.
    alpha:
        Significance level for the Nemenyi test approximation.
    filepath:
        If provided, save figure to this path; otherwise show interactively.
    maximize:
        If True, higher rank = better (rank 1 is best).
    """
    algorithms = sorted(df[algorithm_col].unique())
    k = len(algorithms)
    problems = df[problem_col].unique()
    seeds = df[seed_col].unique()

    # Average rank per algorithm
    avg_ranks: dict[str, float] = {a: 0.0 for a in algorithms}
    n_blocks = 0
    for prob in problems:
        for seed in seeds:
            block = {}
            for a in algorithms:
                vals = df[
                    (df[problem_col] == prob)
                    & (df[seed_col] == seed)
                    & (df[algorithm_col] == a)
                ][value_col].values
                if len(vals) == 1:
                    block[a] = vals[0]
            if len(block) == k:
                sorted_algs = sorted(block, key=block.get, reverse=maximize)
                for rank, alg in enumerate(sorted_algs, start=1):
                    avg_ranks[alg] += rank
                n_blocks += 1

    if n_blocks == 0:
        print("CD plot: no complete blocks found, skipping.")
        return

    for a in algorithms:
        avg_ranks[a] /= n_blocks

    # Nemenyi critical difference
    # CD = q_α * sqrt(k(k+1) / (6*N))
    # q_α values for α=0.05 (two-tailed) from Demšar Table 5
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                     6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_alpha_table.get(k, 2.728)
    CD = q * math.sqrt(k * (k + 1) / (6 * n_blocks))

    sorted_algs = sorted(algorithms, key=lambda a: avg_ranks[a])
    ranks = [avg_ranks[a] for a in sorted_algs]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, k * 1.5), 3))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1.8)
    ax.set_yticks([])
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel("Average Rank")
    ax.set_title(f"Critical Difference Diagram  (CD = {CD:.3f}, α = {alpha})")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Draw algorithm markers
    y_alg = 1.2
    for i, (alg, r) in enumerate(zip(sorted_algs, ranks)):
        ax.plot(r, y_alg, "o", color="#2c7bb6", markersize=8, zorder=3)
        offset = 0.3 if i % 2 == 0 else -0.2
        ax.annotate(
            alg,
            xy=(r, y_alg),
            xytext=(r, y_alg + offset),
            ha="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
        )

    # Draw CD bar in the top-left
    ax.annotate("", xy=(1 + CD, 1.7), xytext=(1, 1.7),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + CD / 2, 1.72, f"CD={CD:.2f}", ha="center", fontsize=8)

    # Connect non-significantly-different groups
    y_line = 0.9
    for i, (a1, r1) in enumerate(zip(sorted_algs, ranks)):
        for a2, r2 in zip(sorted_algs[i + 1:], ranks[i + 1:]):
            if abs(r1 - r2) < CD:
                ax.plot([r1, r2], [y_line, y_line], lw=4, color="#d7191c", alpha=0.7, zorder=2)
                y_line -= 0.12

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# needed inside this module
import math
