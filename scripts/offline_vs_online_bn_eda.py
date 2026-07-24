"""
offline_vs_online_bn_eda.py -- can the OFFLINE search-distribution metrics
predict the ONLINE EDA behaviour, for the different functions?

Two complementary levels of prediction:

  LEVEL A -- algorithm selection *within* a function (across algorithms).
    For each function we rank the 19 learners by their OFFLINE metric and by
    their ONLINE mean best fitness, and take the aligned Spearman rank
    correlation of the two rankings.  Macro-averaged over functions and broken
    down per family.  Answers: "does a good offline metric tell me which learner
    to pick for this function?"

  LEVEL B -- function difficulty *across* functions.
    Scale-free per-function quantities:
      offline dependency gain = best dependency learner's offline metric
                                minus the univariate model's offline metric;
      online  dependency gain = (best dependency learner's online fitness
                                minus univariate fitness) / (max-min online).
    Correlated across the 33 functions.  Answers: "on the functions where BNs
    model the search distribution better than independence, do BN-EDAs actually
    optimise better than the univariate EDA?"

Offline metrics are taken at the temperature matching the online run
(Boltzmann beta=1.0 -> T=1.0); the paper's held-out (test) regime columns are
used as the primary predictors.

Usage
-----
    python3 scripts/offline_vs_online_bn_eda.py [online_summary_csv] [offline_csv] [out_dir]
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
DEFAULT_ONLINE = os.path.join(_ROOT, "bn_eda_analysis", "bn_eda_summary.csv")
DEFAULT_OFFLINE = os.path.join(_ROOT, "eda_cluster_results.csv")
DEFAULT_OUT = os.path.join(_ROOT, "bn_eda_analysis")

MATCH_T = 1.0          # online Boltzmann beta = 1.0  <->  offline temperature 1.0
UNIV = "univ_bn"

# offline metric -> (label, higher is better?)
OFF_METRICS = {
    "test_sp":  ("Test $\\rho$", True),
    "train_sp": ("Train $\\rho$", True),
    "f1":       ("$F_1$", True),
    "test_kl":  ("Test KL", False),
    "test_ll":  ("Test LL", True),
}


def offline_cells(offline_csv, temperature):
    off = pd.read_csv(offline_csv)
    for c in OFF_METRICS:
        off[c] = pd.to_numeric(off[c], errors="coerce")
    both = off[off["temperature"] == temperature]
    cell = (both.groupby(["problem", "algorithm"], as_index=False)[list(OFF_METRICS)]
            .mean())
    # also the temperature-pooled version (robustness)
    cell_all = (off.groupby(["problem", "algorithm"], as_index=False)[list(OFF_METRICS)]
                .mean())
    return cell, cell_all


def online_cells(online_csv):
    on = pd.read_csv(online_csv)
    on["best_fitness"] = pd.to_numeric(on["best_fitness"], errors="coerce")
    cell = (on.groupby(["problem", "family", "algorithm"], as_index=False)["best_fitness"]
            .mean())
    return cell


# ---------------------------------------------------------------------------
# LEVEL A: per-function algorithm-ranking agreement
# ---------------------------------------------------------------------------
def level_a(online, offcell, tag):
    merged = online.merge(offcell, on=["problem", "algorithm"], how="inner")
    recs = []
    for problem, g in merged.groupby("problem"):
        perf = g["best_fitness"].to_numpy(float)
        if g["algorithm"].nunique() < 3 or np.all(perf == perf[0]):
            continue
        row = {"problem": problem, "family": g["family"].iloc[0], "n_alg": len(g)}
        for m, (_lab, higher) in OFF_METRICS.items():
            vals = g[m].to_numpy(float) * (1 if higher else -1)
            mask = ~np.isnan(vals) & ~np.isnan(perf)
            if mask.sum() < 3 or np.all(vals[mask] == vals[mask][0]):
                row[m] = np.nan
            else:
                row[m] = spearmanr(perf[mask], vals[mask]).correlation
        recs.append(row)
    rc = pd.DataFrame(recs)
    rc.to_csv(os.path.join(DEFAULT_OUT, f"offline_online_levelA_{tag}.csv"), index=False)
    return rc


def write_levelA_table(rc_T, rc_all, out_dir):
    path = os.path.join(out_dir, "tables", "table_offline_predicts_online_A.tex")
    with open(path, "w") as f:
        f.write("% LEVEL A: offline metric ranking vs online best-fitness ranking, per function\n")
        f.write("% aligned per-function Spearman (KL flipped), macro-averaged over functions\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Offline metric & $\\rho$ (T=1.0) & $\\rho$ (all T) & \\% functions $\\rho>0$ \\\\\n")
        f.write("\\midrule\n")
        mT = rc_T[list(OFF_METRICS)].mean()
        mA = rc_all[list(OFF_METRICS)].mean()
        best = mT.idxmax()
        for m, (lab, _h) in OFF_METRICS.items():
            frac = 100 * (rc_T[m] > 0).mean()
            v = "\\textbf{%.3f}" % mT[m] if m == best else "%.3f" % mT[m]
            f.write(f"{lab} & {v} & {mA[m]:.3f} & {frac:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rc_T[list(OFF_METRICS)].mean()


def write_levelA_by_family(rc_T, out_dir):
    g = rc_T.groupby("family")[list(OFF_METRICS)].mean()
    order = g["test_sp"].sort_values(ascending=False).index
    g = g.reindex(order)
    path = os.path.join(out_dir, "tables", "table_offline_predicts_online_A_byfamily.tex")
    with open(path, "w") as f:
        f.write("% LEVEL A per problem family (offline T=1.0)\n")
        f.write("\\begin{tabular}{l" + "c" * len(OFF_METRICS) + "}\n\\toprule\n")
        f.write("Family & " + " & ".join(OFF_METRICS[m][0] for m in OFF_METRICS)
                + " \\\\\n\\midrule\n")
        for fam, r in g.iterrows():
            cells = " & ".join(("%.2f" % r[m]) if not np.isnan(r[m]) else "--"
                               for m in OFF_METRICS)
            f.write(f"{fam} & {cells} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return g


# ---------------------------------------------------------------------------
# LEVEL B: across-function dependency gain
# ---------------------------------------------------------------------------
def level_b(online, offcell, out_dir):
    merged = online.merge(offcell, on=["problem", "algorithm"], how="inner")
    recs = []
    for problem, g in merged.groupby("problem"):
        if UNIV not in set(g["algorithm"]):
            continue
        u = g[g["algorithm"] == UNIV].iloc[0]
        dep = g[g["algorithm"] != UNIV]
        perf = g["best_fitness"].to_numpy(float)
        rng = perf.max() - perf.min()
        if rng <= 0 or np.isnan(rng):
            continue
        row = {
            "problem": problem, "family": g["family"].iloc[0],
            # online: how much the best dependency EDA beats univariate (scale-free)
            "online_gain": (dep["best_fitness"].max() - u["best_fitness"]) / rng,
        }
        for m, (_lab, higher) in OFF_METRICS.items():
            best_dep = dep[m].max() if higher else -dep[m].min()
            u_off = u[m] if higher else -u[m]
            row["offgain_" + m] = best_dep - u_off
        recs.append(row)
    dfb = pd.DataFrame(recs)
    dfb.to_csv(os.path.join(out_dir, "offline_online_levelB.csv"), index=False)

    path = os.path.join(out_dir, "tables", "table_offline_predicts_online_B.tex")
    res = {}
    with open(path, "w") as f:
        f.write("% LEVEL B: across-function correlation of offline dependency gain\n")
        f.write("% with online dependency gain (best BN-EDA over univariate, scale-free)\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Offline dependency gain & Spearman $\\rho$ vs online gain & $p$ \\\\\n")
        f.write("\\midrule\n")
        for m, (lab, _h) in OFF_METRICS.items():
            x = dfb["offgain_" + m].to_numpy(float)
            y = dfb["online_gain"].to_numpy(float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            r, p = spearmanr(x[mask], y[mask])
            res[m] = (r, p)
            f.write(f"{lab} & {r:.3f} & {p:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return dfb, res


def fig_levelB(dfb, out_dir):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    x = dfb["offgain_test_sp"].to_numpy(float)
    y = dfb["online_gain"].to_numpy(float)
    ax.scatter(x, y, s=28, color="#2b6cb0")
    for _i, r in dfb.iterrows():
        ax.annotate(r["problem"].replace("_", "\n"), (r["offgain_test_sp"], r["online_gain"]),
                    fontsize=5.5, xytext=(2, 2), textcoords="offset points", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("offline dependency gain in test $\\rho$ (best BN $-$ Univ)")
    ax.set_ylabel("online dependency gain (best BN-EDA $-$ Univ, scale-free)")
    fig.savefig(os.path.join(out_dir, "figures", "fig_offline_online_levelB.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    online_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ONLINE
    offline_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OFFLINE
    out_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    offT, offAll = offline_cells(offline_csv, MATCH_T)
    online = online_cells(online_csv)

    print("=== LEVEL A: does an offline metric predict the best learner "
          "for each function? ===")
    rc_T = level_a(online, offT, "T1")
    rc_all = level_a(online, offAll, "allT")
    meansT = write_levelA_table(rc_T, rc_all, out_dir)
    byfam = write_levelA_by_family(rc_T, out_dir)
    for m, (lab, _h) in OFF_METRICS.items():
        frac = 100 * (rc_T[m] > 0).mean()
        print(f"  {lab:12s} macro-rho(T=1.0)={meansT[m]:+.3f}   "
              f"functions with rho>0: {frac:.0f}%   (n={rc_T[m].notna().sum()})")
    print("\n  per-family macro-rho for test_sp (offline T=1.0):")
    for fam, v in byfam["test_sp"].sort_values(ascending=False).items():
        print(f"    {fam:16s} {v:+.2f}")

    print("\n=== LEVEL B: across functions, does offline dependency gain predict "
          "online dependency gain? ===")
    dfb, res = level_b(online, offT, out_dir)
    fig_levelB(dfb, out_dir)
    for m, (lab, _h) in OFF_METRICS.items():
        r, p = res[m]
        star = " *" if p < 0.05 else ""
        print(f"  {lab:12s} spearman_across_functions={r:+.3f}  p={p:.3f}{star}")
    print(f"\n  ({len(dfb)} functions with a univariate baseline and non-degenerate spread)")
    print(f"\nWrote LEVEL A/B tables, figures and CSVs to {out_dir}")


if __name__ == "__main__":
    main()
