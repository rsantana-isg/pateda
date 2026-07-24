"""
analyze_trajectory_offline_online.py -- use the JSON-derived trajectory features
to explain the OFFLINE <-> ONLINE metric relationship.

Inputs (produced by extract_bn_eda_trajectory.py):
    bn_eda_trajectory_features.csv   per-run early/late-phase features + diversity
    bn_eda_avg_trajectory.csv        per-(family, gen) average trajectories
    eda_cluster_results.csv          offline per-regime metrics (sp_s0/s1/s2, ...)

Four analyses:

  1. REGIME SWEEP (figure).  The online EDA sweeps the same selection-pressure
     axis the offline benchmark samples at fixed regimes: diversity 1->0, while
     sp_pop rises to 1 and the model (edges, F1) collapses.

  2. REGIME-MATCHED AGREEMENT.  Does the offline per-regime Spearman track the
     online per-phase Spearman?  offline subset 0 (diverse) vs online early
     phase; offline subset 2 (converged) vs online late phase.  Blocked
     within-function across algorithms.  Tests whether offline and online
     measure the *same thing* once the regime is matched.

  3. sp_pop IS CONVERGENCE, NOT QUALITY.  Across algorithms within a function,
     mean sp_pop correlates with how fast the population converges (mean
     diversity), not with best fitness -- which is why it fails as a predictor.

  4. TRAJECTORY PREDICTORS + DIVERSITY-BASED PREMATURE-CONVERGENCE TEST.  Which
     trajectory feature best predicts online best fitness (blocked), and does
     low final diversity signal harmful premature convergence?

Usage
-----
    python3 scripts/analyze_trajectory_offline_online.py [features_csv] [avg_csv] [offline_csv] [out_dir]
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
DEF_FEAT = os.path.join(OUT, "bn_eda_trajectory_features.csv")
DEF_AVG = os.path.join(OUT, "bn_eda_avg_trajectory.csv")
DEF_OFF = os.path.join(_ROOT, "eda_cluster_results.csv")
MATCH_T = 1.0
PERF = "best_fitness"


def blocked_rho(agg, xcol, ycol, higher=True):
    """Macro-averaged within-problem Spearman(x, y) across algorithms."""
    rs = []
    for _p, g in agg.groupby("problem"):
        x = g[xcol].to_numpy(float) * (1 if higher else -1)
        y = g[ycol].to_numpy(float)
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 4 or np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
            continue
        rs.append(spearmanr(x[m], y[m]).correlation)
    return float(np.mean(rs)) if rs else np.nan, len(rs)


# ---------------------------------------------------------------------------
def fig_regime_sweep(avg, out_dir):
    """Overall + per-family average trajectories of the four bridge metrics."""
    overall = avg.groupby("gen")[["pop_diversity", "sp_pop", "f1", "edges"]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    panels = [("pop_diversity", "population diversity", axes[0][0]),
              ("sp_pop", "Spearman $\\rho$ (population)", axes[0][1]),
              ("f1", "skeleton $F_1$", axes[1][0]),
              ("edges", "model edges", axes[1][1])]
    for key, ylab, ax in panels:
        ax.plot(overall.index, overall[key], color="black", lw=2.2, label="all")
        for fam, g in avg.groupby("family"):
            ax.plot(g["gen"], g[key], lw=0.8, alpha=0.5)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylab)
    # annotate offline regime correspondence on the diversity panel
    ax = axes[0][0]
    ax.axvspan(0, 10, color="#cfe3f7", alpha=0.5)
    ax.axvspan(90, 100, color="#f7d0cf", alpha=0.5)
    ax.annotate("early\n~subset 0\n(diverse)", (5, 0.4), fontsize=7, ha="center")
    ax.annotate("late\n~subset 2\n(converged)", (95, 0.5), fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig_regime_sweep.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def offline_regime_cells(offline_csv):
    off = pd.read_csv(offline_csv)
    cols = ["sp_s0", "sp_s1", "sp_s2", "kl_s0", "kl_s2", "ll_s0", "ll_s2", "f1"]
    for c in cols:
        off[c] = pd.to_numeric(off[c], errors="coerce")
    cell = (off[off["temperature"] == MATCH_T]
            .groupby(["problem", "algorithm"], as_index=False)[cols].mean())
    return cell


def regime_matched_table(agg, offcell, out_dir):
    m = agg.merge(offcell, on=["problem", "algorithm"], how="inner")
    pairs = [
        ("offline subset0 (diverse) vs online early", "sp_s0", "sp_pop_early"),
        ("offline subset2 (converged) vs online late", "sp_s2", "sp_pop_late"),
        ("cross: offline subset2 vs online early",     "sp_s2", "sp_pop_early"),
        ("cross: offline subset0 vs online late",      "sp_s0", "sp_pop_late"),
    ]
    path = os.path.join(out_dir, "tables", "table_regime_matched.tex")
    res = []
    with open(path, "w") as f:
        f.write("% regime-matched agreement between offline per-subset and online per-phase Spearman\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Comparison & blocked $\\rho$ & \\# functions \\\\\n\\midrule\n")
        for lab, ox, oy in pairs:
            r, n = blocked_rho(m, ox, oy)
            res.append((lab, r, n))
            f.write(f"{lab} & {r:.3f} & {n} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return res


def sp_confound_table(agg, out_dir):
    """sp_pop tracks convergence (diversity), not fitness."""
    checks = [
        ("mean sp\\_pop vs mean diversity (convergence)", "sp_pop_late", "mean_diversity", False),
        ("mean sp\\_pop vs best fitness", "sp_pop_late", PERF, True),
        ("early sp\\_pop vs best fitness", "sp_pop_early", PERF, True),
        ("final diversity vs best fitness", "final_diversity", PERF, True),
        ("mean diversity vs best fitness", "mean_diversity", PERF, True),
    ]
    path = os.path.join(out_dir, "tables", "table_sppop_confound.tex")
    rows = []
    with open(path, "w") as f:
        f.write("% sp_pop measures convergence, not model quality\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Blocked comparison & $\\rho$ & \\# functions \\\\\n\\midrule\n")
        for lab, x, y, hi in checks:
            r, n = blocked_rho(agg, x, y, higher=hi)
            rows.append((lab, r, n))
            f.write(f"{lab} & {r:.3f} & {n} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


def predictor_table(agg, out_dir):
    """Which trajectory feature predicts online best fitness (blocked)?"""
    feats = [
        ("early sp\\_pop", "sp_pop_early", True),
        ("late sp\\_pop", "sp_pop_late", True),
        ("early F1", "f1_early", True),
        ("max F1 (over run)", "max_f1", True),
        ("early edges", "edges_early", True),
        ("early KL", "kl_pop_early", False),
        ("mean diversity (exploration)", "mean_diversity", True),
        ("final diversity", "final_diversity", True),
        ("gens to converge", "t_converge", True),
    ]
    path = os.path.join(out_dir, "tables", "table_trajectory_predictors.tex")
    rows = []
    for lab, col, hi in feats:
        r, n = blocked_rho(agg, col, PERF, higher=hi)
        rows.append((lab, col, r, n))
    rows.sort(key=lambda t: (-abs(t[2]) if not np.isnan(t[2]) else 0))
    best = max((r for _l, _c, r, _n in rows if not np.isnan(r)), default=np.nan)
    with open(path, "w") as f:
        f.write("% trajectory features as blocked predictors of online best fitness\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Trajectory feature & blocked $\\rho$ with best fitness & \\# functions \\\\\n\\midrule\n")
        for lab, _c, r, n in rows:
            v = "\\textbf{%.3f}" % r if r == best else "%.3f" % r
            f.write(f"{lab} & {v} & {n} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


def premature_diversity_table(agg, out_dir):
    """Definitive diversity-based premature-convergence test, per family."""
    fam_rows = []
    for fam, g in agg.groupby("family"):
        r1, _ = blocked_rho_within(g, "sp_pop_early", "final_diversity", True)
        r2, _ = blocked_rho_within(g, "final_diversity", PERF, True)
        r3, _ = blocked_rho_within(g, "max_f1", PERF, True)
        fam_rows.append((fam, r1, r2, r3))
    path = os.path.join(out_dir, "tables", "table_premature_diversity.tex")
    with open(path, "w") as f:
        f.write("% diversity-based premature convergence test by family\n")
        f.write("% H1: early-fit -> low final diversity (rho<0) AND low diversity -> worse fitness (rho>0)\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Family & $\\rho$(early fit, final div) & $\\rho$(final div, fitness) "
                "& $\\rho$(max F1, fitness) \\\\\n\\midrule\n")
        for fam, r1, r2, r3 in fam_rows:
            def s(x):
                return "%.2f" % x if not np.isnan(x) else "--"
            f.write(f"{fam} & {s(r1)} & {s(r2)} & {s(r3)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return fam_rows


def blocked_rho_within(g, xcol, ycol, higher=True):
    rs = []
    for _p, gg in g.groupby("problem"):
        x = gg[xcol].to_numpy(float) * (1 if higher else -1)
        y = gg[ycol].to_numpy(float)
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 4 or np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
            continue
        rs.append(spearmanr(x[m], y[m]).correlation)
    return (float(np.mean(rs)) if rs else np.nan), len(rs)


def main():
    feat_csv = sys.argv[1] if len(sys.argv) > 1 else DEF_FEAT
    avg_csv = sys.argv[2] if len(sys.argv) > 2 else DEF_AVG
    off_csv = sys.argv[3] if len(sys.argv) > 3 else DEF_OFF
    out_dir = sys.argv[4] if len(sys.argv) > 4 else OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    feat = pd.read_csv(feat_csv)
    avg = pd.read_csv(avg_csv)
    # aggregate features over seeds -> per (problem, algorithm)
    numcols = [c for c in feat.columns if c not in
               ("problem", "family", "algorithm", "n", "seed")]
    agg = feat.groupby(["problem", "family", "algorithm"], as_index=False)[numcols].mean()

    fig_regime_sweep(avg, out_dir)

    offcell = offline_regime_cells(off_csv)
    rm = regime_matched_table(agg, offcell, out_dir)
    print("=== 2. Regime-matched agreement (offline per-subset vs online per-phase) ===")
    for lab, r, n in rm:
        print(f"  {lab:44s} blocked_rho={r:+.3f}  (n={n})")

    print("\n=== 3. sp_pop is convergence, not quality ===")
    for lab, r, n in sp_confound_table(agg, out_dir):
        print(f"  {lab:48s} rho={r:+.3f} (n={n})")

    print("\n=== 4a. Trajectory features as blocked predictors of best fitness ===")
    for lab, _c, r, n in predictor_table(agg, out_dir):
        print(f"  {lab:32s} rho={r:+.3f} (n={n})")

    print("\n=== 4b. Diversity-based premature-convergence test by family ===")
    print(f"  {'family':16s} {'rho(earlyfit,finaldiv)':>22s} {'rho(finaldiv,fit)':>18s} {'rho(maxF1,fit)':>15s}")
    for fam, r1, r2, r3 in premature_diversity_table(agg, out_dir):
        def s(x):
            return f"{x:+.2f}" if not np.isnan(x) else "  --"
        print(f"  {fam:16s} {s(r1):>22s} {s(r2):>18s} {s(r3):>15s}")

    print(f"\nWrote regime/trajectory tables and fig_regime_sweep.pdf to {out_dir}")


if __name__ == "__main__":
    main()
