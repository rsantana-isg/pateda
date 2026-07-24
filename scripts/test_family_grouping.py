"""
test_family_grouping.py -- formal test of the "different offline metrics predict
different BN-EDA learners" family structure found by
per_algorithm_offline_predictors.py.

Base object: the learner x metric matrix R (19 x 5) of aligned Spearman
correlations between each offline metric (at T=1.0) and the within-problem
min-max normalised online best fitness, computed across the 29 non-saturated
problems (KL sign-flipped so + = predictive).

Four formal analyses:

  1. INTERACTION PERMUTATION TEST.  H0: the offline metrics relate to online
     performance the same way for every learner (no learner x metric
     interaction).  Statistic = summed across-learner variance of R.  Null is
     built by shuffling the online score across algorithms WITHIN each problem
     (destroys learner-specific structure, keeps every marginal).  Per-metric
     variants localise which metrics carry the heterogeneity.

  2. NUMBER OF FAMILIES.  Ward hierarchical clustering of the z-scored rows;
     silhouette for k=2..6 to choose k; dendrogram.

  3. CLUSTER STABILITY.  Bootstrap over problems (block bootstrap by problem);
     recompute R, re-cluster at the chosen k; per-cluster Jaccard stability
     (Hennig) and a 19x19 consensus co-assignment matrix.

  4. CONTRAST CI.  Bootstrap 95% CI for the difference in test-rho
     predictiveness between the highest- and lowest-scoring clusters.

Usage
-----
    python3 scripts/test_family_grouping.py [online_csv] [offline_csv] [out_dir]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "bn_eda_analysis")
DEF_ON = os.path.join(OUT, "bn_eda_summary.csv")
DEF_OFF = os.path.join(_ROOT, "eda_cluster_results.csv")
MATCH_T = 1.0
MIN_PROB = 8
B = 2000
RNG = np.random.default_rng(0)

OFF_METRICS = {"test_sp": ("Test rho", True), "train_sp": ("Train rho", True),
               "f1": ("F1", True), "test_kl": ("Test KL", False),
               "test_ll": ("Test LL", True)}
METS = list(OFF_METRICS)
ALG_LABEL = {"univ_bn": "Univ", "k2": "K2", "k2_mi": "K2-MI", "k2_mb": "K2-MB",
             "k2_refine": "K2-Ref", "k2_ensemble": "K2-Ens", "k2_plus": "K2+",
             "fi_k2": "FI-K2", "rfe_k2": "RFE-K2", "bic": "BIC-HC", "aic": "AIC-HC",
             "stable_hc": "HC-Stable", "pc": "PC", "stable_pc": "PC-Stable",
             "dt": "DT", "dmbbn": "DMBBN", "sartre": "SARTRE", "binotears": "BINO",
             "bounded_tw": "BdTW"}


def load_panel(online_csv, offline_csv):
    on = pd.read_csv(online_csv)
    on["best_fitness"] = pd.to_numeric(on["best_fitness"], errors="coerce")
    cell = on.groupby(["problem", "algorithm"], as_index=False)["best_fitness"].mean()

    def norm(g):
        v = g["best_fitness"]
        rng = v.max() - v.min()
        g = g.copy()
        g["online_norm"] = np.nan if rng <= 0 else (v - v.min()) / rng
        return g
    cell = cell.groupby("problem", group_keys=False).apply(norm).dropna(subset=["online_norm"])

    off = pd.read_csv(offline_csv)
    for m in METS:
        off[m] = pd.to_numeric(off[m], errors="coerce")
    offcell = (off[off["temperature"] == MATCH_T]
               .groupby(["problem", "algorithm"], as_index=False)[METS].mean())
    panel = cell.merge(offcell, on=["problem", "algorithm"], how="inner")
    # align metrics so + = predictive (flip KL)
    for m, (_l, hi) in OFF_METRICS.items():
        if not hi:
            panel[m] = -panel[m]
    return panel


def build_arrays(panel, algorithms):
    """Dense (n_alg x n_prob) online and (n_alg x n_prob x n_met) offline arrays."""
    problems = sorted(panel["problem"].unique())
    pidx = {p: i for i, p in enumerate(problems)}
    aidx = {a: i for i, a in enumerate(algorithms)}
    ON = np.full((len(algorithms), len(problems)), np.nan)
    OFF = np.full((len(algorithms), len(problems), len(METS)), np.nan)
    for row in panel.itertuples(index=False):
        i, j = aidx[row.algorithm], pidx[row.problem]
        ON[i, j] = row.online_norm
        for k, m in enumerate(METS):
            OFF[i, j, k] = getattr(row, m)
    return ON, OFF, problems


def _spear(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < MIN_PROB:
        return np.nan
    xm, ym = x[mask], y[mask]
    if np.all(xm == xm[0]) or np.all(ym == ym[0]):
        return np.nan
    rx, ry = rankdata(xm), rankdata(ym)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def matrix_R_arr(ON, OFF, cols):
    """learner x metric aligned Spearman over the given problem columns."""
    nA, nM = ON.shape[0], OFF.shape[2]
    R = np.full((nA, nM), np.nan)
    for i in range(nA):
        y = ON[i, cols]
        for k in range(nM):
            R[i, k] = _spear(OFF[i, cols, k], y)
    return R


def zscore_impute(R):
    Z = np.where(np.isnan(R), np.nanmean(R, axis=0, keepdims=True), R)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return Z


def cluster(Z, k):
    L = linkage(Z, method="ward")
    labels = fcluster(L, k, criterion="maxclust")
    return L, labels


# --- 1. interaction permutation test -------------------------------------
def interaction_test(ON, OFF, R_obs):
    def stat(R):
        return np.nansum(np.nanvar(R, axis=0))
    obs = stat(R_obs)
    obs_by_metric = np.nanvar(R_obs, axis=0)
    cols = np.arange(ON.shape[1])
    null = np.zeros(B)
    null_by_metric = np.zeros((B, len(METS)))
    for b in range(B):
        ONp = ON.copy()
        # shuffle online score across algorithms within each problem (keep NaN pattern)
        for j in range(ON.shape[1]):
            col = ONp[:, j]
            present = np.where(~np.isnan(col))[0]
            col[present] = col[RNG.permutation(present)]
        Rb = matrix_R_arr(ONp, OFF, cols)
        null[b] = stat(Rb)
        null_by_metric[b] = np.nanvar(Rb, axis=0)
    pval = (np.sum(null >= obs) + 1) / (B + 1)
    p_by_metric = [(np.sum(null_by_metric[:, j] >= obs_by_metric[j]) + 1) / (B + 1)
                   for j in range(len(METS))]
    return obs, pval, obs_by_metric, p_by_metric


# --- 3. bootstrap cluster stability --------------------------------------
def bootstrap_stability(ON, OFF, base_labels, k):
    n = ON.shape[0]
    nprob = ON.shape[1]
    consensus = np.zeros((n, n))
    counts = 0
    base_clusters = [set(np.where(base_labels == c)[0]) for c in np.unique(base_labels)]
    jacc = {c: [] for c in np.unique(base_labels)}
    for b in range(B):
        cols = RNG.choice(nprob, size=nprob, replace=True)
        Rb = matrix_R_arr(ON, OFF, cols)
        Zb = zscore_impute(Rb)
        try:
            _, lb = cluster(Zb, k)
        except Exception:
            continue
        for ci in np.unique(lb):
            idx = np.where(lb == ci)[0]
            consensus[np.ix_(idx, idx)] += 1
        counts += 1
        boot_clusters = [set(np.where(lb == c)[0]) for c in np.unique(lb)]
        for c, orig in zip(np.unique(base_labels), base_clusters):
            best = max((len(orig & bc) / len(orig | bc)) for bc in boot_clusters)
            jacc[c].append(best)
    consensus = consensus / max(counts, 1)
    jacc_mean = {c: float(np.mean(v)) if v else np.nan for c, v in jacc.items()}
    return consensus, jacc_mean


# --- 4. contrast bootstrap CI --------------------------------------------
def contrast_ci(ON, OFF, hi_idx, lo_idx, metric="test_sp"):
    j = METS.index(metric)
    nprob = ON.shape[1]
    deltas = []
    for b in range(B):
        cols = RNG.choice(nprob, size=nprob, replace=True)
        Rb = matrix_R_arr(ON, OFF, cols)
        deltas.append(np.nanmean(Rb[hi_idx, j]) - np.nanmean(Rb[lo_idx, j]))
    deltas = np.array(deltas)
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5]), float(np.mean(deltas <= 0))


# --- figures --------------------------------------------------------------
def fig_dendro(L, algorithms, labels, out_dir):
    """Dendrogram with links coloured to match the fcluster assignment exactly."""
    n = len(algorithms)
    # leaves under each node; palette per cluster label
    palette = ["#e6842a", "#3aa35a", "#c0392b", "#8e44ad", "#2c7fb8", "#d81b60"]
    cl_color = {c: palette[i % len(palette)] for i, c in enumerate(sorted(set(labels)))}
    leaves_under = {i: {i} for i in range(n)}
    node_cluster = {}
    for idx, (a, b, _h, _cnt) in enumerate(L):
        a, b = int(a), int(b)
        leaves = leaves_under[a] | leaves_under[b]
        leaves_under[n + idx] = leaves
    def link_color(k):
        leaves = leaves_under[k]
        cls = {labels[i] for i in leaves}
        return cl_color[next(iter(cls))] if len(cls) == 1 else "#9aa0a6"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dendrogram(L, labels=[ALG_LABEL.get(a, a) for a in algorithms], ax=ax,
               link_color_func=link_color)
    ax.set_ylabel("Ward distance")
    plt.xticks(fontsize=9, rotation=40, ha="right")
    fig.savefig(os.path.join(out_dir, "figures", "fig_family_dendrogram.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_consensus(consensus, algorithms, labels, out_dir):
    order = np.argsort(labels)
    C = consensus[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1)
    names = [ALG_LABEL.get(algorithms[i], algorithms[i]) for i in order]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="bootstrap co-assignment frequency")
    fig.savefig(os.path.join(out_dir, "figures", "fig_family_consensus.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    online_csv = sys.argv[1] if len(sys.argv) > 1 else DEF_ON
    offline_csv = sys.argv[2] if len(sys.argv) > 2 else DEF_OFF
    out_dir = sys.argv[3] if len(sys.argv) > 3 else OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    panel = load_panel(online_csv, offline_csv)
    algorithms = [a for a in ALG_LABEL if a in set(panel["algorithm"])]
    ON, OFF, problems = build_arrays(panel, algorithms)
    cols0 = np.arange(len(problems))
    R = matrix_R_arr(ON, OFF, cols0)
    Z = zscore_impute(R)
    print(f"Panel: {len(problems)} problems, {len(algorithms)} learners, "
          f"{len(METS)} offline metrics.\n")

    # 1. interaction permutation test
    obs, pval, obs_m, p_m = interaction_test(ON, OFF, R)
    print("=== 1. Learner x metric INTERACTION permutation test "
          f"(B={B}) ===")
    print(f"  summed across-learner variance = {obs:.3f}   permutation p = {pval:.4f}")
    print("  per-metric across-learner variance (p):")
    for j, m in enumerate(METS):
        print(f"    {OFF_METRICS[m][0]:9s} var={obs_m[j]:.3f}  p={p_m[j]:.4f}")

    # 2. number of families
    print("\n=== 2. Number of families (silhouette, Ward) ===")
    best_k, best_s = None, -1
    for k in range(2, 7):
        _, lab = cluster(Z, k)
        s = silhouette_score(Z, lab) if len(set(lab)) > 1 else float("nan")
        print(f"  k={k}: silhouette={s:.3f}")
        if s > best_s:
            best_s, best_k = s, k
    print(f"  -> chosen k = {best_k} (silhouette {best_s:.3f})")

    L, labels = cluster(Z, best_k)
    fig_dendro(L, algorithms, labels, out_dir)

    # cluster membership + mean response
    dfR = pd.DataFrame(R, index=[ALG_LABEL.get(a, a) for a in algorithms], columns=METS)
    dfR["cluster"] = labels
    dfR.to_csv(os.path.join(out_dir, "family_clustering.csv"))
    print("\n  cluster membership and mean metric-response:")
    for c in sorted(set(labels)):
        members = [ALG_LABEL.get(algorithms[i], algorithms[i])
                   for i in range(len(algorithms)) if labels[i] == c]
        mean_resp = np.nanmean(R[np.array(labels) == c], axis=0)
        resp = "  ".join(f"{METS[j]}={mean_resp[j]:+.2f}" for j in range(len(METS)))
        print(f"    C{c}: {', '.join(members)}")
        print(f"        {resp}")

    # 3. stability
    print(f"\n=== 3. Cluster stability (bootstrap over problems, B={B}) ===")
    consensus, jacc = bootstrap_stability(ON, OFF, labels, best_k)
    for c in sorted(jacc):
        tag = ("stable" if jacc[c] >= 0.75 else
               "pattern" if jacc[c] >= 0.6 else "unstable")
        print(f"    C{c} Jaccard stability = {jacc[c]:.2f}  ({tag})")
    fig_consensus(consensus, algorithms, labels, out_dir)

    # 4. contrast CI: highest vs lowest test_rho cluster
    j = METS.index("test_sp")
    cl_mean = {c: np.nanmean(R[np.array(labels) == c, j]) for c in set(labels)}
    hi_c = max(cl_mean, key=cl_mean.get)
    lo_c = min(cl_mean, key=cl_mean.get)
    hi_idx = [i for i in range(len(algorithms)) if labels[i] == hi_c]
    lo_idx = [i for i in range(len(algorithms)) if labels[i] == lo_c]
    hi_alg = [algorithms[i] for i in hi_idx]
    lo_alg = [algorithms[i] for i in lo_idx]
    d, ci, p0 = contrast_ci(ON, OFF, hi_idx, lo_idx)
    print(f"\n=== 4. Contrast: test-rho predictiveness, C{hi_c} vs C{lo_c} ===")
    print(f"    delta = {d:+.3f}   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]   "
          f"boot p(delta<=0) = {p0:.4f}")
    print(f"    C{hi_c} = {[ALG_LABEL.get(a,a) for a in hi_alg]}")
    print(f"    C{lo_c} = {[ALG_LABEL.get(a,a) for a in lo_alg]}")

    # write summary table
    with open(os.path.join(out_dir, "tables", "table_family_grouping_test.tex"), "w") as f:
        f.write("% formal test of the learner family grouping\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Test & statistic & result \\\\\n\\midrule\n")
        f.write(f"Learner$\\times$metric interaction (perm., B={B}) & "
                f"var$={obs:.2f}$ & $p={pval:.3f}$ \\\\\n")
        f.write(f"Number of families (silhouette) & $k={best_k}$ & "
                f"$s={best_s:.2f}$ \\\\\n")
        for c in sorted(jacc):
            f.write(f"Cluster C{c} Jaccard stability & {jacc[c]:.2f} & "
                    f"{'stable' if jacc[c]>=0.75 else 'pattern' if jacc[c]>=0.6 else 'unstable'} \\\\\n")
        f.write(f"C{hi_c}$-$C{lo_c} test-$\\rho$ contrast & "
                f"$\\Delta={d:.2f}$ & 95\\% CI $[{ci[0]:.2f},{ci[1]:.2f}]$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\nWrote family-grouping test table/figures to {out_dir}")


if __name__ == "__main__":
    main()
