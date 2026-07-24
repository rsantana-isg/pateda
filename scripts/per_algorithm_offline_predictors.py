"""
per_algorithm_offline_predictors.py -- case-by-case (per BN-learning algorithm)
study of whether the OFFLINE search-distribution metrics predict the ONLINE
EDA performance, and whether some metrics work better for some learners.

Earlier analyses were blocked *within problem* (across algorithms): "does an
offline metric pick the best learner for a function?".  This one is the
transpose -- blocked *within algorithm* (across problems): for a FIXED learner,
do the problems where its offline metric is high coincide with the problems
where it performs well online?

Scale-free online score: within each problem, ``best_fitness`` is min-max
normalised across the 19 learners to [0,1] (0 = worst learner on that problem,
1 = best).  Problems where every learner ties (the saturated OneMax instances)
are dropped.  For each algorithm we then take the Spearman correlation, across
its problems, between each offline metric (at the matching temperature T=1.0)
and this normalised online score.  Metrics are aligned so that a positive
correlation means "higher metric -> better relative online performance" (KL is
sign-flipped).

Outputs
    tables/table_per_algorithm_predictors.tex   algorithms x offline metrics
    figures/fig_per_algorithm_predictors.pdf     heatmap
    per_algorithm_offline_predictors.csv

Usage
-----
    python3 scripts/per_algorithm_offline_predictors.py [online_csv] [offline_csv] [out_dir]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "bn_eda_analysis")
DEF_ON = os.path.join(OUT, "bn_eda_summary.csv")
DEF_OFF = os.path.join(_ROOT, "eda_cluster_results.csv")
MATCH_T = 1.0
MIN_PROBLEMS = 8            # need enough problems for a per-algorithm correlation
SIG_RHO = 0.37             # |rho| significant at ~0.05 for n~29 (Spearman)

OFF_METRICS = {            # offline metric -> (label, higher is better?)
    "test_sp":  ("Test $\\rho$", True),
    "train_sp": ("Train $\\rho$", True),
    "f1":       ("$F_1$", True),
    "test_kl":  ("Test KL", False),
    "test_ll":  ("Test LL", True),
}
ALG_LABEL = {
    "univ_bn": "Univ", "k2": "K2", "k2_mi": "K2-MI", "k2_mb": "K2-MB",
    "k2_refine": "K2-Ref", "k2_ensemble": "K2-Ens", "k2_plus": "K2+",
    "fi_k2": "FI-K2", "rfe_k2": "RFE-K2", "bic": "BIC-HC", "aic": "AIC-HC",
    "stable_hc": "HC-Stable", "pc": "PC", "stable_pc": "PC-Stable",
    "dt": "DT", "dmbbn": "DMBBN", "sartre": "SARTRE",
    "binotears": "BINO", "bounded_tw": "BdTW",
}
ALG_ORDER = list(ALG_LABEL)


def load(online_csv, offline_csv):
    on = pd.read_csv(online_csv)
    on["best_fitness"] = pd.to_numeric(on["best_fitness"], errors="coerce")
    cell = on.groupby(["problem", "algorithm"], as_index=False)["best_fitness"].mean()
    # within-problem min-max normalisation (drop tied problems)
    def norm(g):
        v = g["best_fitness"]
        rng = v.max() - v.min()
        g = g.copy()
        g["online_norm"] = np.nan if rng <= 0 else (v - v.min()) / rng
        return g
    cell = cell.groupby("problem", group_keys=False).apply(norm)
    cell = cell.dropna(subset=["online_norm"])

    off = pd.read_csv(offline_csv)
    for m in OFF_METRICS:
        off[m] = pd.to_numeric(off[m], errors="coerce")
    offcell = (off[off["temperature"] == MATCH_T]
               .groupby(["problem", "algorithm"], as_index=False)[list(OFF_METRICS)].mean())
    return cell.merge(offcell, on=["problem", "algorithm"], how="inner")


def per_algorithm(df):
    rows = []
    for alg, g in df.groupby("algorithm"):
        y = g["online_norm"].to_numpy(float)
        rec = {"algorithm": alg, "n_problems": len(g)}
        for m, (_lab, higher) in OFF_METRICS.items():
            x = g[m].to_numpy(float) * (1 if higher else -1)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < MIN_PROBLEMS or np.all(x[mask] == x[mask][0]):
                rec[m] = np.nan
            else:
                rec[m] = spearmanr(x[mask], y[mask]).correlation
        rows.append(rec)
    res = pd.DataFrame(rows).set_index("algorithm")
    res = res.reindex([a for a in ALG_ORDER if a in res.index])
    return res


def write_table(res, out_dir):
    metrics = list(OFF_METRICS)
    path = os.path.join(out_dir, "tables", "table_per_algorithm_predictors.tex")
    with open(path, "w") as f:
        f.write("% per-algorithm offline->online predictiveness (across problems)\n")
        f.write(f"% aligned Spearman; bold = |rho|>={SIG_RHO:.2f} (approx p<0.05); best metric per row starred\n")
        f.write("\\begin{tabular}{l" + "c" * len(metrics) + "r}\n\\toprule\n")
        f.write("Algorithm & " + " & ".join(OFF_METRICS[m][0] for m in metrics)
                + " & \\#prob \\\\\n\\midrule\n")
        for alg, r in res.iterrows():
            best_m = r[metrics].astype(float).idxmax() if r[metrics].notna().any() else None
            cells = []
            for m in metrics:
                v = r[m]
                if np.isnan(v):
                    cells.append("--")
                    continue
                s = "%.2f" % v
                if abs(v) >= SIG_RHO:
                    s = "\\textbf{%s}" % s
                if m == best_m:
                    s = s + "$^\\star$"
                cells.append(s)
            f.write(f"{ALG_LABEL.get(alg, alg)} & " + " & ".join(cells)
                    + f" & {int(r['n_problems'])} \\\\\n")
        f.write("\\midrule\n")
        mean = res[metrics].mean()
        f.write("mean & " + " & ".join("%.2f" % mean[m] for m in metrics) + " &  \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def fig_heatmap(res, out_dir):
    metrics = list(OFF_METRICS)
    M = res[metrics].to_numpy(float)
    fig, ax = plt.subplots(figsize=(1.1 * len(metrics) + 3, 0.42 * len(res) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([OFF_METRICS[m][0] for m in metrics], fontsize=10)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([ALG_LABEL.get(a, a) for a in res.index], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7,
                 label="aligned Spearman (offline metric $\\to$ online score)")
    fig.savefig(os.path.join(out_dir, "figures", "fig_per_algorithm_predictors.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    online_csv = sys.argv[1] if len(sys.argv) > 1 else DEF_ON
    offline_csv = sys.argv[2] if len(sys.argv) > 2 else DEF_OFF
    out_dir = sys.argv[3] if len(sys.argv) > 3 else OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    df = load(online_csv, offline_csv)
    res = per_algorithm(df)
    res.to_csv(os.path.join(out_dir, "per_algorithm_offline_predictors.csv"))
    write_table(res, out_dir)
    fig_heatmap(res, out_dir)

    metrics = list(OFF_METRICS)
    print("=== Per-algorithm offline->online predictiveness "
          "(aligned Spearman across problems) ===")
    disp = res.copy()
    disp.index = [ALG_LABEL.get(a, a) for a in disp.index]
    print(disp[metrics + ["n_problems"]].round(2).to_string())

    print("\n=== Best offline predictor per algorithm ===")
    for alg, r in res.iterrows():
        vals = r[metrics].astype(float)
        if vals.notna().any():
            bm = vals.idxmax()
            bm_lab = OFF_METRICS[bm][0].replace("$", "").replace("\\rho", "rho")
            print(f"  {ALG_LABEL.get(alg, alg):11s} best = {bm_lab:10s} "
                  f"rho={vals[bm]:+.2f}   (F1={r['f1']:+.2f}, testrho={r['test_sp']:+.2f})")

    print("\n=== Column means (which metric is the best predictor overall) ===")
    print(res[metrics].mean().round(3).to_string())
    print(f"\n{len(df)} (problem,algorithm) cells over "
          f"{df['problem'].nunique()} non-saturated problems")
    print(f"Wrote table/figure/CSV to {out_dir}")


if __name__ == "__main__":
    main()
