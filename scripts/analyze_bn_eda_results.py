"""
analyze_bn_eda_results.py -- analysis of the BN-EDA comparison grid.

Reads the per-run ``.dat`` summaries written by ``run_bn_eda.py`` (one line per
run: best_fitness + the live model-quality metrics) that live in
``results_bn_eda/`` and answers the three questions of the study:

  1. COMPLETION.  How many of the 20 seeds finished for each (objective
     function, BN-learning algorithm)?  Emitted overall and split by problem
     size (Small / Medium / Large).

  2. PREDICTORS.  Which model-quality metric -- Spearman correlation, KL
     divergence, log-likelihood or skeleton F1 -- best predicts the actual EDA
     performance (best fitness)?  Analysed *within problem* (blocked): per
     problem we rank algorithms by performance and by each metric and take the
     Spearman rank-correlation of the two rankings, then macro-average over
     problems (+ top-1 hit rate).  A second, cross-dataset check relates the
     *offline* search-distribution metrics of ``eda_cluster_results.csv`` (the
     experiments of the AIS paper) to the same EDA performance.

  3. RANKING.  Which BN-learning method works best for each objective function
     (mean best fitness over the 20 seeds), a per-problem winner table, and the
     overall ranking of the algorithms by their average rank (Friedman +
     Nemenyi critical difference).

Usage
-----
    python3 scripts/analyze_bn_eda_results.py [results_dir] [offline_csv] [out_dir]

Defaults: results_dir=results_bn_eda, offline_csv=eda_cluster_results.csv,
out_dir=bn_eda_analysis.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_ROOT, "results_bn_eda")
DEFAULT_OFFLINE = os.path.join(_ROOT, "eda_cluster_results.csv")
DEFAULT_OUT = os.path.join(_ROOT, "bn_eda_analysis")

N_SEEDS = 20

# .dat data line: best_fitness generation_found total_wall_time total_learn_time
#                 mean_sp_pop mean_ll_pop mean_kl_pop mean_f1 final_edges
DAT_COLS = ["best_fitness", "generation_found", "total_wall_time",
            "total_learn_time", "mean_sp_pop", "mean_ll_pop", "mean_kl_pop",
            "mean_f1", "final_edges"]

# live model-quality metrics -> is higher better?  (KL is a distance)
SURROGATES = {
    "mean_sp_pop": ("Spearman $\\rho$", True),
    "mean_kl_pop": ("KL divergence", False),
    "mean_ll_pop": ("Log-likelihood", True),
    "mean_f1":     ("Skeleton $F_1$", True),
}
PERF = "best_fitness"

# pretty algorithm labels (match the paper's naming)
ALG_LABEL = {
    "univ_bn": "Univ", "k2": "K2", "k2_mi": "K2-MI", "k2_mb": "K2-MB",
    "k2_refine": "K2-Ref", "k2_ensemble": "K2-Ens", "k2_plus": "K2+",
    "fi_k2": "FI-K2", "rfe_k2": "RFE-K2", "bic": "BIC-HC", "aic": "AIC-HC",
    "stable_hc": "HC-Stable", "pc": "PC", "stable_pc": "PC-Stable",
    "dt": "DT", "dmbbn": "DMBBN", "sartre": "SARTRE",
    "binotears": "BINO", "bounded_tw": "BdTW",
}
GROUP_ORDER = ["small", "medium", "large"]
GROUP_TITLE = {"small": "Small", "medium": "Medium", "large": "Large"}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_runs(results_dir):
    """Parse every .dat file into one tidy row per finished run."""
    rows = []
    hdr = re.compile(r"# problem=(\S+) algorithm=(\S+) seed=(\d+)")
    for fp in sorted(glob.glob(os.path.join(results_dir, "*.dat"))):
        with open(fp) as fh:
            lines = fh.read().splitlines()
        head = next((l for l in lines if l.startswith("# problem=")), "")
        data = next((l for l in lines if l.strip() and not l.startswith("#")), "")
        m = hdr.match(head)
        if not m or not data:
            continue
        problem, algorithm, seed = m.group(1), m.group(2), int(m.group(3))
        vals = data.split()
        if len(vals) < len(DAT_COLS):
            continue
        rec = {"problem": problem, "algorithm": algorithm, "seed": seed}
        for c, v in zip(DAT_COLS, vals):
            try:
                rec[c] = float(v)
            except ValueError:
                rec[c] = np.nan
        family, n = problem.rsplit("_", 1)
        rec["family"] = family
        rec["n"] = int(n)
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df


def size_group(n):
    if n <= 50:
        return "small" if n < 55 else "medium"
    return "medium" if n <= 66 else "large"


def add_groups(df, offline_csv):
    """Attach the size group.  Prefer the authoritative map from the offline CSV."""
    gmap = {}
    if os.path.isfile(offline_csv):
        off = pd.read_csv(offline_csv, usecols=["problem", "group"])
        gmap = dict(off.drop_duplicates().itertuples(index=False, name=None))
    df["group"] = df["problem"].map(gmap)
    # fall back to n-based bins for anything not in the map
    miss = df["group"].isna()
    if miss.any():
        df.loc[miss, "group"] = df.loc[miss, "n"].map(size_group)
    return df


# ---------------------------------------------------------------------------
# 1. Completion
# ---------------------------------------------------------------------------
# binary-only learner x non-binary family: never launched, expected = 0
INCOMPATIBLE = {("Braid", "binotears")}


def completion_matrix(df):
    """(problem x algorithm) count of finished seeds (0..20)."""
    algos = sorted(df["algorithm"].unique(),
                   key=lambda a: list(ALG_LABEL).index(a) if a in ALG_LABEL else 99)
    problems = (df[["problem", "family", "n", "group"]].drop_duplicates()
                .sort_values(["group", "family", "n"]))
    counts = (df.groupby(["problem", "algorithm"])["seed"].nunique()
              .unstack(fill_value=0).reindex(columns=algos, fill_value=0))
    return counts, algos, problems


def write_completion_tables(counts, problems, algos, out_dir):
    counts.reindex(problems["problem"]).to_csv(
        os.path.join(out_dir, "completion_matrix.csv"))

    # per-algorithm totals (out of the number of *compatible* problems x 20)
    fam_of = dict(zip(problems["problem"], problems["family"]))
    tot, exp = {}, {}
    for a in algos:
        t = e = 0
        for p in problems["problem"]:
            if (fam_of[p], a) in INCOMPATIBLE:
                continue
            e += N_SEEDS
            t += int(counts.loc[p, a]) if p in counts.index else 0
        tot[a], exp[a] = t, e
    summary = pd.DataFrame({"finished": tot, "expected": exp})
    summary["pct"] = 100 * summary["finished"] / summary["expected"]
    summary.index = [ALG_LABEL.get(a, a) for a in summary.index]
    summary.sort_values("pct", ascending=False).to_csv(
        os.path.join(out_dir, "completion_by_algorithm.csv"))

    # three LaTeX tables (problem x algorithm), one per size group
    for grp in GROUP_ORDER:
        gp = problems[problems["group"] == grp]["problem"].tolist()
        if not gp:
            continue
        sub = counts.reindex(gp)
        path = os.path.join(out_dir, "tables", f"table_completion_{grp}.tex")
        with open(path, "w") as f:
            f.write(f"% finished seeds (out of {N_SEEDS}) -- {GROUP_TITLE[grp]} problems\n")
            f.write("\\begin{tabular}{l" + "r" * len(algos) + "}\n\\toprule\n")
            f.write("Problem & " + " & ".join(ALG_LABEL.get(a, a) for a in algos)
                    + " \\\\\n\\midrule\n")
            for p in gp:
                cells = []
                for a in algos:
                    fam = fam_of[p]
                    if (fam, a) in INCOMPATIBLE:
                        cells.append("--")
                    else:
                        c = int(sub.loc[p, a])
                        cells.append(str(c) if c == N_SEEDS else "\\textbf{%d}" % c)
                f.write(p.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    return summary


# ---------------------------------------------------------------------------
# 2. Predictors (surrogate validity)
# ---------------------------------------------------------------------------
def aggregate_cells(df):
    """Average over seeds -> one row per (problem, algorithm)."""
    num = [PERF] + list(SURROGATES) + ["total_learn_time", "generation_found"]
    return (df.groupby(["problem", "group", "algorithm"], as_index=False)[num]
            .mean())


def per_problem_rankcorr(agg):
    recs = []
    for problem, g in agg.groupby("problem"):
        if g["algorithm"].nunique() < 3:
            continue
        perf = g[PERF].to_numpy(float)
        if np.all(perf == perf[0]):
            continue
        row = {"problem": problem, "group": g["group"].iloc[0], "n_alg": len(g)}
        for m, (_lab, higher) in SURROGATES.items():
            vals = g[m].to_numpy(float)
            if not higher:
                vals = -vals
            mask = ~np.isnan(vals) & ~np.isnan(perf)
            if mask.sum() < 3 or np.all(vals[mask] == vals[mask][0]):
                row[m] = np.nan
            else:
                row[m] = spearmanr(perf[mask], vals[mask]).correlation
        recs.append(row)
    return pd.DataFrame(recs)


def top1_hits(agg):
    hits = {m: [] for m in SURROGATES}
    for _p, g in agg.groupby("problem"):
        if g["algorithm"].nunique() < 3 or g[PERF].isna().all():
            continue
        g = g.reset_index(drop=True)
        best = g.loc[g[PERF].idxmax(), "algorithm"]
        for m, (_lab, higher) in SURROGATES.items():
            if g[m].isna().all():
                continue
            idx = g[m].idxmax() if higher else g[m].idxmin()
            hits[m].append(int(g.loc[idx, "algorithm"] == best))
    return {m: (float(np.mean(v)) if v else np.nan) for m, v in hits.items()}


def write_validity_table(rankcorr, hits, out_dir):
    means = rankcorr[list(SURROGATES)].mean()
    stds = rankcorr[list(SURROGATES)].std()
    best = means.idxmax()
    path = os.path.join(out_dir, "tables", "table_surrogate_validity.tex")
    with open(path, "w") as f:
        f.write("% predictive validity of each model-quality metric for EDA best fitness\n")
        f.write("% aligned per-problem Spearman rank-corr (KL sign-flipped), macro-averaged\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Metric & mean aligned $\\rho$ & std & top-1 hit \\\\\n\\midrule\n")
        for m, (lab, _h) in SURROGATES.items():
            v = "\\textbf{%.3f}" % means[m] if m == best else "%.3f" % means[m]
            f.write(f"{lab} & {v} & {stds[m]:.3f} & {hits.get(m, np.nan):.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return means, stds


def write_validity_by_group(rankcorr, out_dir):
    g = rankcorr.groupby("group")[list(SURROGATES)].mean().reindex(GROUP_ORDER)
    path = os.path.join(out_dir, "tables", "table_surrogate_validity_by_group.tex")
    with open(path, "w") as f:
        f.write("% predictive validity by problem size group\n")
        f.write("\\begin{tabular}{l" + "c" * len(SURROGATES) + "}\n\\toprule\n")
        f.write("Group & " + " & ".join(SURROGATES[m][0] for m in SURROGATES)
                + " \\\\\n\\midrule\n")
        for grp in GROUP_ORDER:
            if grp not in g.index:
                continue
            vals = " & ".join("%.3f" % g.loc[grp, m] for m in SURROGATES)
            f.write(f"{GROUP_TITLE[grp]} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def fig_predictor_box(rankcorr, means, out_dir):
    metrics = list(SURROGATES)
    data = [rankcorr[m].dropna().to_numpy() for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bp = ax.boxplot(data, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f7")
    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels([SURROGATES[m][0] for m in metrics], fontsize=11)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel(r"per-problem aligned $\rho$ with EDA best fitness", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    for i, m in enumerate(metrics, 1):
        ax.annotate(f"mean={means[m]:.2f}", (i, -0.98), ha="center", fontsize=9)
    fig.savefig(os.path.join(out_dir, "figures", "fig_predictor_rankcorr.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_predictor_scatter(agg, out_dir):
    metrics = list(SURROGATES)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.1),
                             squeeze=False)
    g = agg.copy()

    def z(col):
        return g.groupby("problem")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    g["z_perf"] = z(PERF)
    for ax, m in zip(axes[0], metrics):
        sign = 1 if SURROGATES[m][1] else -1
        gm = g.dropna(subset=[m, PERF])
        zx = sign * z(m)[gm.index]
        ax.scatter(zx, g["z_perf"][gm.index], s=8, alpha=0.4, color="#2b6cb0")
        r = spearmanr(zx, g["z_perf"][gm.index], nan_policy="omit").correlation
        ax.set_title(f"{SURROGATES[m][0]}\n(pooled $\\rho$={r:.2f})", fontsize=10)
        ax.set_xlabel("metric (within-problem z)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
    axes[0][0].set_ylabel("best fitness (within-problem z)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig_predictor_scatter.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# --- cross-dataset: offline search-distribution metrics vs EDA performance ---
def offline_predictor_analysis(agg, offline_csv, out_dir):
    if not os.path.isfile(offline_csv):
        return None
    off = pd.read_csv(offline_csv)
    for c in ["test_sp", "train_sp", "f1", "test_kl", "test_ll"]:
        if c in off:
            off[c] = pd.to_numeric(off[c], errors="coerce")
    # offline cell = mean over temperatures/splits/seeds per (problem, algorithm)
    offcell = (off.groupby(["problem", "algorithm"], as_index=False)
               [["test_sp", "train_sp", "f1", "test_kl", "test_ll"]].mean())
    merged = agg.merge(offcell, on=["problem", "algorithm"], how="inner",
                       suffixes=("", "_off"))
    off_metrics = {"test_sp": ("Offline test $\\rho$", True),
                   "train_sp": ("Offline train $\\rho$", True),
                   "f1": ("Offline $F_1$", True),
                   "test_kl": ("Offline KL", False),
                   "test_ll": ("Offline LL", True)}
    recs = []
    for _p, g in merged.groupby("problem"):
        if g["algorithm"].nunique() < 3:
            continue
        perf = g[PERF].to_numpy(float)
        if np.all(perf == perf[0]):
            continue
        row = {"problem": g["problem"].iloc[0]}
        for m, (_lab, higher) in off_metrics.items():
            vals = g[m].to_numpy(float) * (1 if higher else -1)
            mask = ~np.isnan(vals) & ~np.isnan(perf)
            if mask.sum() < 3 or np.all(vals[mask] == vals[mask][0]):
                row[m] = np.nan
            else:
                row[m] = spearmanr(perf[mask], vals[mask]).correlation
        recs.append(row)
    rc = pd.DataFrame(recs)
    if rc.empty:
        return None
    means = rc[list(off_metrics)].mean()
    best = means.idxmax()
    path = os.path.join(out_dir, "tables", "table_offline_predictor.tex")
    with open(path, "w") as f:
        f.write("% do the OFFLINE search-distribution metrics (AIS paper) predict EDA best fitness?\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Offline metric & mean aligned $\\rho$ & std \\\\\n\\midrule\n")
        for m, (lab, _h) in off_metrics.items():
            v = "\\textbf{%.3f}" % means[m] if m == best else "%.3f" % means[m]
            f.write(f"{lab} & {v} & {rc[m].std():.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    rc.to_csv(os.path.join(out_dir, "offline_predictor_rankcorr.csv"), index=False)
    return means


# ---------------------------------------------------------------------------
# 3. Ranking (best method per function + average ranks)
# ---------------------------------------------------------------------------
def best_per_problem(agg, out_dir):
    rows = []
    for problem, g in agg.groupby("problem"):
        g = g.sort_values(PERF, ascending=False).reset_index(drop=True)
        win = g.loc[0]
        run = g.loc[1] if len(g) > 1 else win
        rows.append({
            "problem": problem, "group": win["group"],
            "winner": ALG_LABEL.get(win["algorithm"], win["algorithm"]),
            "best_fitness": win[PERF],
            "runner_up": ALG_LABEL.get(run["algorithm"], run["algorithm"]),
            "runner_fitness": run[PERF],
            "n_alg": len(g),
        })
    bp = pd.DataFrame(rows).sort_values(["group", "problem"])
    bp.to_csv(os.path.join(out_dir, "best_per_problem.csv"), index=False)
    path = os.path.join(out_dir, "tables", "table_best_per_problem.tex")
    with open(path, "w") as f:
        f.write("% best BN-learning method per objective function (mean best fitness over 20 seeds)\n")
        f.write("\\begin{tabular}{llrl}\n\\toprule\n")
        f.write("Problem & Best method & Mean best fitness & Runner-up \\\\\n\\midrule\n")
        for grp in GROUP_ORDER:
            sub = bp[bp["group"] == grp]
            if sub.empty:
                continue
            f.write("\\multicolumn{4}{l}{\\emph{%s}} \\\\\n" % GROUP_TITLE[grp])
            for _i, r in sub.iterrows():
                f.write(f"\\quad {r['problem'].replace('_', chr(92)+'_')} & "
                        f"{r['winner']} & {r['best_fitness']:.4g} & {r['runner_up']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return bp


def average_ranks(agg, out_dir):
    a = agg.copy()
    # rank algorithms within each problem by mean best fitness (1 = best)
    a["rank"] = a.groupby("problem")[PERF].rank(ascending=False, method="average")
    summ = (a.groupby("algorithm")
            .agg(avg_rank=("rank", "mean"),
                 n_problems=("problem", "nunique"),
                 mean_learn_time=("total_learn_time", "mean"))
            .sort_values("avg_rank"))
    # count outright wins (per-problem argmax of performance)
    wins = (a.loc[a.groupby("problem")[PERF].idxmax(), "algorithm"]
            .value_counts())
    summ["wins"] = summ.index.map(wins).fillna(0).astype(int)
    summ["label"] = summ.index.map(lambda x: ALG_LABEL.get(x, x))

    # Friedman test on the complete-block subset (problems with all algorithms)
    piv = a.pivot_table(index="problem", columns="algorithm", values="rank")
    complete = piv.dropna(axis=0)
    fried = None
    if complete.shape[0] >= 3 and complete.shape[1] >= 3:
        stat, p = friedmanchisquare(*[complete[c].to_numpy() for c in complete.columns])
        k, N = complete.shape[1], complete.shape[0]
        q_alpha = 3.354  # Nemenyi q for infinite df, alpha=0.05 (used with k)
        # standard Nemenyi CD = q_alpha * sqrt(k(k+1)/(6N)); q_alpha depends on k.
        q_by_k = {19: 4.16, 18: 4.13, 15: 4.03, 10: 3.72}  # approx studentized range/sqrt2
        qa = q_by_k.get(k, 4.16)
        cd = qa * np.sqrt(k * (k + 1) / (6.0 * N))
        fried = dict(chi2=stat, p=p, k=k, N=N, cd=cd)

    summ_out = summ[["label", "avg_rank", "wins", "n_problems", "mean_learn_time"]]
    summ_out.to_csv(os.path.join(out_dir, "algorithm_ranking.csv"), index=False)

    path = os.path.join(out_dir, "tables", "table_algorithm_ranking.tex")
    with open(path, "w") as f:
        f.write("% algorithms ranked by average per-problem rank on EDA best fitness (1=best)\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Algorithm & Avg. rank & Wins & \\# problems & Learn time (s) \\\\\n\\midrule\n")
        best_rank = summ_out["avg_rank"].min()
        for _i, r in summ_out.iterrows():
            ar = ("\\textbf{%.2f}" % r["avg_rank"] if r["avg_rank"] == best_rank
                  else "%.2f" % r["avg_rank"])
            f.write(f"{r['label']} & {ar} & {int(r['wins'])} & "
                    f"{int(r['n_problems'])} & {r['mean_learn_time']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return summ_out, fried


def fig_avg_rank(summ, fried, out_dir):
    s = summ.sort_values("avg_rank")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(s))
    ax.barh(y, s["avg_rank"], color="#2b6cb0")
    ax.set_yticks(y)
    ax.set_yticklabels(s["label"])
    ax.invert_yaxis()
    ax.set_xlabel("average rank on EDA best fitness (1 = best)")
    if fried:
        best = s["avg_rank"].min()
        ax.axvline(best + fried["cd"], color="red", ls="--", lw=1)
        ax.annotate(f"CD={fried['cd']:.2f}", (best + fried["cd"], len(s) - 1),
                    color="red", fontsize=9, ha="left", va="bottom")
    fig.savefig(os.path.join(out_dir, "figures", "fig_avg_rank.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_family_heatmap(agg, out_dir):
    """Mean per-problem performance rank of each algorithm per problem family."""
    a = agg.copy()
    a["family"] = a["problem"].str.rsplit("_", n=1).str[0]
    a["rank"] = a.groupby("problem")[PERF].rank(ascending=False, method="average")
    piv = a.pivot_table(index="algorithm", columns="family", values="rank",
                        aggfunc="mean")
    order = a.groupby("algorithm")["rank"].mean().sort_values().index
    piv = piv.reindex(order)
    piv.index = [ALG_LABEL.get(x, x) for x in piv.index]
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2, 0.42 * piv.shape[0] + 1.5))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean rank (1=best)")
    fig.savefig(os.path.join(out_dir, "figures", "fig_family_rank_heatmap.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    offline_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OFFLINE
    out_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    df = load_runs(results_dir)
    df = add_groups(df, offline_csv)
    df.to_csv(os.path.join(out_dir, "bn_eda_summary.csv"), index=False)
    print(f"Loaded {len(df)} finished runs: "
          f"{df['problem'].nunique()} problems x {df['algorithm'].nunique()} algorithms")

    # 1. completion
    counts, algos, problems = completion_matrix(df)
    summary = write_completion_tables(counts, problems, algos, out_dir)
    print("\n=== Completion by algorithm (finished / expected) ===")
    print(summary.sort_values("pct", ascending=False).to_string(
        float_format=lambda x: f"{x:.1f}"))

    # 2. predictors
    agg = aggregate_cells(df)
    rankcorr = per_problem_rankcorr(agg)
    hits = top1_hits(agg)
    rankcorr.to_csv(os.path.join(out_dir, "per_problem_rankcorr.csv"), index=False)
    means, _ = write_validity_table(rankcorr, hits, out_dir)
    write_validity_by_group(rankcorr, out_dir)
    fig_predictor_box(rankcorr, means, out_dir)
    fig_predictor_scatter(agg, out_dir)
    off_means = offline_predictor_analysis(agg, offline_csv, out_dir)
    print("\n=== Predictor validity (live metrics vs EDA best fitness) ===")
    for m, (lab, _h) in SURROGATES.items():
        print(f"  {lab:22s} mean rho={means[m]:+.3f}  top1-hit={hits.get(m, np.nan):.2f}")
    if off_means is not None:
        print("\n=== Offline search-distribution metrics vs EDA best fitness ===")
        for m in off_means.index:
            print(f"  {m:14s} mean rho={off_means[m]:+.3f}")

    # 3. ranking
    best_per_problem(agg, out_dir)
    summ, fried = average_ranks(agg, out_dir)
    fig_avg_rank(summ, fried, out_dir)
    fig_family_heatmap(agg, out_dir)
    print("\n=== Algorithm ranking by average rank (1 = best) ===")
    print(summ.to_string(index=False,
                         float_format=lambda x: f"{x:.2f}"))
    if fried:
        print(f"\nFriedman: chi2={fried['chi2']:.1f}, p={fried['p']:.2e} "
              f"(k={fried['k']}, N={fried['N']} complete-block problems), "
              f"Nemenyi CD={fried['cd']:.2f}")

    print(f"\nWrote tables/figures/CSVs to {out_dir}")


if __name__ == "__main__":
    main()
