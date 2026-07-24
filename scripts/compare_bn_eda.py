"""
compare_bn_eda.py -- do search-distribution quality metrics predict EDA success?

Given the aggregated ``bn_eda_summary.csv`` (from ``extract_bn_eda.py``), this
script quantifies whether the offline-style model-quality metrics computed live
during the EDA -- Spearman correlation (``sp_pop``/``sp_sel``), mean
log-likelihood (``ll_pop``), KL divergence (``kl_pop``) and skeleton F1 -- are
good **surrogates** of the actual EDA performance (``best_fitness``).

Method
------
* Runs are averaged over seeds to a per-(problem, algorithm) table.
* Because the EDA objective scale differs wildly across problems, everything is
  analysed **within problem** (blocked, Demsar 2006):
    - algorithms are ranked by EDA ``best_fitness`` (the ground truth), and by
      each surrogate metric;
    - the Spearman rank-correlation between the performance ranking and each
      surrogate ranking is computed per problem, then macro-averaged.
  KL is a distance (lower is better), so its sign is flipped before ranking.
* We also report how often each surrogate's top-1 algorithm coincides with the
  performance top-1 algorithm ("hit rate").

Outputs (mirroring bn_edas_analysis/eda_cluster_analysis/)
    tables/  table_surrogate_validity.tex, table_algorithm_summary.tex
    figures/ fig_surrogate_rankcorr.pdf, fig_surrogate_scatter.pdf,
             fig_perf_vs_cost.pdf

Usage
-----
    python3 scripts/compare_bn_eda.py [summary_csv] [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(_ROOT, "results", "bn_eda_summary.csv")
DEFAULT_OUT = os.path.join(_ROOT, "results", "bn_eda_analysis")

# surrogate metric -> (higher is better?)  KL is a distance.
SURROGATES = {
    "mean_sp_pop": True,
    "mean_sp_sel": True,
    "mean_ll_pop": True,
    "mean_kl_pop": False,
    "mean_f1": True,
}
PERF = "best_fitness"           # ground-truth EDA performance (higher better)


def load(csv_path):
    df = pd.read_csv(csv_path)
    num = list(SURROGATES) + [PERF, "auc_best", "total_learn_time",
                              "total_wall_time", "generation_found"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # average replicate seeds -> one row per (problem, algorithm)
    keys = ["problem", "algorithm"]
    agg = df.groupby(keys, as_index=False)[
        [c for c in num if c in df.columns]].mean()
    return df, agg


def per_problem_rank_correlations(agg):
    """For each problem, Spearman(rank by performance, rank by surrogate)."""
    records = []
    for problem, g in agg.groupby("problem"):
        if g["algorithm"].nunique() < 3:
            continue                                  # need a few algorithms
        perf = g[PERF].to_numpy(dtype=float)
        if np.all(perf == perf[0]) or np.isnan(perf).all():
            continue
        row = {"problem": problem, "n_alg": len(g)}
        for metric, higher in SURROGATES.items():
            if metric not in g:
                continue
            vals = g[metric].to_numpy(dtype=float)
            if not higher:
                vals = -vals
            mask = ~np.isnan(vals) & ~np.isnan(perf)
            if mask.sum() < 3 or np.all(vals[mask] == vals[mask][0]):
                row[metric] = np.nan
            else:
                row[metric] = spearmanr(perf[mask], vals[mask]).correlation
        records.append(row)
    return pd.DataFrame(records)


def top1_hit_rates(agg):
    """Fraction of problems where the surrogate's best algo == performance's best."""
    hits = {m: [] for m in SURROGATES}
    for problem, g in agg.groupby("problem"):
        if g["algorithm"].nunique() < 3:
            continue
        g = g.reset_index(drop=True)
        if g[PERF].isna().all():
            continue
        best_perf = g.loc[g[PERF].idxmax(), "algorithm"]
        for metric, higher in SURROGATES.items():
            if metric not in g or g[metric].isna().all():
                continue
            idx = g[metric].idxmax() if higher else g[metric].idxmin()
            hits[metric].append(int(g.loc[idx, "algorithm"] == best_perf))
    return {m: (np.mean(v) if v else np.nan) for m, v in hits.items()}


def write_validity_table(rankcorr, hits, out_dir):
    means = rankcorr[list(SURROGATES)].mean()
    stds = rankcorr[list(SURROGATES)].std()
    path = os.path.join(out_dir, "tables", "table_surrogate_validity.tex")
    with open(path, "w") as f:
        f.write("% surrogate validity: aligned rank-correlation with EDA best_fitness\n")
        f.write("% metrics are oriented so that a higher rho = better surrogate\n")
        f.write("% (KL divergence is sign-flipped, since lower KL is better).\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Surrogate metric & mean aligned $\\rho$ & std & top-1 hit rate \\\\\n")
        f.write("\\midrule\n")
        label = {"mean_sp_pop": "Correlation (pop)",
                 "mean_sp_sel": "Correlation (selected)",
                 "mean_ll_pop": "Log-likelihood",
                 "mean_kl_pop": "KL divergence",
                 "mean_f1": "Skeleton F1"}
        best = means.idxmax()
        for m in SURROGATES:
            b = "\\textbf{%.3f}" % means[m] if m == best else "%.3f" % means[m]
            hr = hits.get(m, float("nan"))
            f.write(f"{label[m]} & {b} & {stds[m]:.3f} & "
                    f"{hr:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return means, stds


def write_algorithm_table(agg, out_dir):
    # per-problem performance rank (1 = best), averaged over problems
    agg = agg.copy()
    agg["perf_rank"] = agg.groupby("problem")[PERF].rank(
        ascending=False, method="average")
    cols = {"perf_rank": "mean", PERF: "count"}
    for m in SURROGATES:
        if m in agg:
            cols[m] = "mean"
    for c in ["total_learn_time", "auc_best"]:
        if c in agg:
            cols[c] = "mean"
    summ = agg.groupby("algorithm").agg(cols)
    summ = summ.rename(columns={PERF: "n_problems"}).sort_values("perf_rank")
    path = os.path.join(out_dir, "tables", "table_algorithm_summary.tex")
    with open(path, "w") as f:
        f.write("% per-algorithm mean performance rank and surrogate metrics\n")
        summ.to_latex(f, float_format="%.3f")
    return summ


def fig_rankcorr(rankcorr, out_dir):
    metrics = [m for m in SURROGATES if m in rankcorr]
    data = [rankcorr[m].dropna().to_numpy() for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels([m.replace("mean_", "") for m in metrics])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel(r"per-problem aligned $\rho$ with EDA best fitness")
    ax.set_ylim(-1.05, 1.05)
    fig.savefig(os.path.join(out_dir, "figures", "fig_surrogate_rankcorr.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_scatter(agg, out_dir):
    # within-problem z-scored performance vs z-scored surrogate (pooled)
    metrics = [m for m in SURROGATES if m in agg]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.2),
                             squeeze=False)
    g = agg.copy()

    def z(col):
        return g.groupby("problem")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    g["z_perf"] = z(PERF)
    for ax, m in zip(axes[0], metrics):
        gm = g.dropna(subset=[m, PERF])
        zx = z(m)[gm.index]
        ax.scatter(zx, gm["z_perf"], s=10, alpha=0.5)
        ax.set_xlabel(m.replace("mean_", "") + " (z)")
        ax.set_ylabel("best fitness (z)")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig_surrogate_scatter.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_perf_vs_cost(agg, out_dir):
    if "total_learn_time" not in agg:
        return
    a = agg.copy()
    a["perf_rank"] = a.groupby("problem")[PERF].rank(ascending=False,
                                                     method="average")
    summ = a.groupby("algorithm").agg(
        perf_rank=("perf_rank", "mean"),
        learn_time=("total_learn_time", "mean"))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(summ["learn_time"], summ["perf_rank"])
    for name, r in summ.iterrows():
        ax.annotate(name, (r["learn_time"], r["perf_rank"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean total learning time per run (s, log)")
    ax.set_ylabel("mean EDA performance rank (1 = best)")
    ax.invert_yaxis()
    fig.savefig(os.path.join(out_dir, "figures", "fig_perf_vs_cost.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    df, agg = load(csv_path)
    print(f"Loaded {len(df)} runs -> {len(agg)} (problem, algorithm) cells, "
          f"{agg['algorithm'].nunique()} algorithms, "
          f"{agg['problem'].nunique()} problems")

    rankcorr = per_problem_rank_correlations(agg)
    hits = top1_hit_rates(agg)

    if not rankcorr.empty:
        means, _ = write_validity_table(rankcorr, hits, out_dir)
        print("\nSurrogate validity (macro-avg per-problem rank-correlation "
              "with EDA best fitness):")
        for m in SURROGATES:
            print(f"  {m:14s} rho={means[m]:+.3f}   top1-hit={hits.get(m, float('nan')):.2f}")
        fig_rankcorr(rankcorr, out_dir)
        fig_scatter(agg, out_dir)
    else:
        print("Not enough problems with >=3 algorithms yet for the "
              "surrogate-validity analysis.")

    write_algorithm_table(agg, out_dir)
    fig_perf_vs_cost(agg, out_dir)
    rankcorr.to_csv(os.path.join(out_dir, "per_problem_rank_correlations.csv"),
                    index=False)
    print(f"\nWrote tables/figures to {out_dir}")


if __name__ == "__main__":
    main()
