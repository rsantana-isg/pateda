"""
analyze_v2.py -- recompute every table/figure of AIS_BN_EDAs_v2.tex after
excluding three learners (DT, DMBBN, K2-Ref) and two problem families
(Braid, MaxClique).

Produces (into AIS_2026/ for figures, and prints ready-to-paste LaTeX):
  * Table 2  completion (only incomplete pairs, remaining grid)
  * Table 3  offline ranking by avg rank on test-regime Spearman  (eda_cluster_results.csv)
  * Table 4  3x3 generalisation matrix                            (eda_cluster_results.csv)
  * Table 5  learner families (Ward clustering + interaction test + bootstrap)
  * Fig  avg-rank on online best fitness (fig_avg_rank_v2.pdf)
  * Fig  UMDA-gain heatmap, four decomposable families (fig_umda_gain_heatmap_selected_v2.pdf)
  * Fig  dendrogram + consensus (fig_family_dendrogram_v2.pdf, fig_family_consensus_v2.pdf)
  * console: online Friedman/CD, offline Friedman/CD, interaction p-values, contrast CI

Usage:  python3 scripts/analyze_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import test_family_grouping as T   # reuse clustering / interaction / bootstrap

ANA = os.path.join(_ROOT, "bn_eda_analysis")
PAPER = os.path.join(_ROOT, "AIS_2026")
V2 = os.path.join(_ROOT, "bn_eda_analysis_v2")
os.makedirs(os.path.join(V2, "figures"), exist_ok=True)

SUMMARY = os.path.join(ANA, "bn_eda_summary.csv")
OFFLINE = os.path.join(_ROOT, "eda_cluster_results.csv")
COMPMAT = os.path.join(ANA, "completion_matrix.csv")

EX_ALG = {"dt", "dmbbn", "k2_refine"}
EX_FAM = {"Braid", "MaxClique"}
N_SEEDS = 20

ALG_LABEL = {"univ_bn": "Univ", "k2": "K2", "k2_mi": "K2-MI", "k2_mb": "K2-MB",
             "k2_ensemble": "K2-Ens", "k2_plus": "K2+", "fi_k2": "FI-K2",
             "rfe_k2": "RFE-K2", "bic": "BIC-HC", "aic": "AIC-HC",
             "stable_hc": "HC-Stable", "pc": "PC", "stable_pc": "PC-Stable",
             "sartre": "SARTRE", "binotears": "BINO", "bounded_tw": "BdTW"}
ALG_ORDER = list(ALG_LABEL)                      # 16 remaining learners


def fam_of(problem):
    return problem.rsplit("_", 1)[0]


# ===========================================================================
# Table 2: completion
# ===========================================================================
def table_completion():
    cm = pd.read_csv(COMPMAT, index_col=0)
    cm = cm.loc[[p for p in cm.index if fam_of(p) not in EX_FAM],
                [a for a in ALG_ORDER if a in cm.columns]]
    aff_probs = [p for p in cm.index if any(cm.loc[p, a] < N_SEEDS for a in cm.columns)]
    aff_algs = [a for a in cm.columns if any(cm.loc[p, a] < N_SEEDS for p in cm.index)]
    aff_probs = sorted(aff_probs, key=lambda p: (fam_of(p), int(p.rsplit("_", 1)[1])))
    L = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Number of completed seeds (out of $20$) for the only "
         "(problem, algorithm) pairs that did \\emph{not} reach all $20$ runs, "
         "after excluding DT, DMBBN and K2-Ref and the \\textsf{Braid} and "
         "\\textsf{MaxClique} families; every pair omitted here completed all "
         "$20$. All incompletions occur at the largest sizes ($n=256/258$), where "
         "the most expensive remaining learners exceed the wall-clock budget.}",
         "\\label{tab:completion}",
         "\\begin{tabular}{l" + "c" * len(aff_algs) + "}", "\\toprule",
         "Problem & " + " & ".join(ALG_LABEL[a] for a in aff_algs) + " \\\\", "\\midrule"]
    for p in aff_probs:
        cells = []
        for a in aff_algs:
            c = int(cm.loc[p, a])
            cells.append(str(c) if c == N_SEEDS else "\\textbf{%d}" % c)
        L.append(p.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


# ===========================================================================
# Table 3 + 4: offline (search-distribution) analysis
# ===========================================================================
def _off_load():
    off = pd.read_csv(OFFLINE)
    off = off[~off["family"].isin(EX_FAM) & ~off["algorithm"].isin(EX_ALG)].copy()
    for c in ["test_sp", "train_sp", "f1", "time", "sp_s0", "sp_s1", "sp_s2"]:
        off[c] = pd.to_numeric(off[c], errors="coerce")
    return off


def _pa_matrix(off, metric, algos):
    piv = off.pivot_table(index="problem", columns="algorithm", values=metric,
                          aggfunc="mean").reindex(columns=algos)
    return piv.dropna(axis=0, how="any")


def _macro(off, metric, algos):
    piv = off.pivot_table(index="problem", columns="algorithm", values=metric,
                          aggfunc="mean")
    return piv.mean(axis=0, skipna=True).reindex(algos)


def _nemenyi_cd(k, n):
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
           15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.518, 20: 3.547}
    return q05.get(k, 3.547) * np.sqrt(k * (k + 1) / (6.0 * n))


def table_ranking_offline():
    off = _off_load()
    algos = [a for a in ALG_ORDER if a in set(off["algorithm"])]
    Mtest = _pa_matrix(off, "test_sp", algos)
    rank = Mtest.rank(axis=1, ascending=False).mean(axis=0)
    order = list(rank.sort_values().index)
    disp = pd.DataFrame(index=order)
    disp["Rank"] = rank.reindex(order)
    disp["Test $\\rho$"] = _macro(off, "test_sp", algos).reindex(order)
    disp["Train $\\rho$"] = _macro(off, "train_sp", algos).reindex(order)
    disp["$F_1$"] = _macro(off, "f1", algos).reindex(order)
    disp["Time (s)"] = _macro(off, "time", algos).reindex(order)
    higher = {"Rank": False, "Test $\\rho$": True, "Train $\\rho$": True,
              "$F_1$": True, "Time (s)": False}
    # Friedman + CD
    k, n = Mtest.shape[1], Mtest.shape[0]
    chi2, p = friedmanchisquare(*[Mtest[c].values for c in Mtest.columns])
    cd = _nemenyi_cd(k, n)
    best = order[0]
    tied = [a for a in order if (rank[a] - rank[best]) <= cd]

    cols = list(disp.columns)
    bestcell = {c: (disp[c].idxmax() if higher[c] else disp[c].idxmin()) for c in cols}
    L = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{9pt}",
         "\\caption{BN learning algorithms ranked by their average rank on the "
         "test-regime Spearman correlation $\\rho$ (blocked by problem; $1=$ best, "
         f"over {n} complete-block problems). Macro-averaged test/train $\\rho$, "
         "skeleton $F_1$ and learning time are shown for reference. Best value per "
         "column in bold.}",
         "\\label{tab:ranking}", "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Algorithm & Rank & Test $\\rho$ & Train $\\rho$ & $F_1$ & Time (s) \\\\",
         "\\midrule"]
    for a in order:
        cells = []
        for c in cols:
            v = disp.loc[a, c]
            prec = 2 if c == "Rank" else (1 if c == "Time (s)" else 3)
            s = "%.*f" % (prec, v)
            if bestcell[c] == a:
                s = "$\\mathbf{%s}$" % s
            cells.append(s)
        L.append(ALG_LABEL.get(a, a) + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    stats = dict(chi2=chi2, p=p, k=k, n=n, cd=cd, best=ALG_LABEL[best],
                 tied=[ALG_LABEL[a] for a in tied])
    return "\n".join(L), stats


def table_genmatrix_offline():
    off = _off_load()
    M = np.full((3, 3), np.nan)
    for i in range(3):
        sub = off[off["train_set"] == i]
        for j in range(3):
            M[i, j] = np.nanmean(sub[f"sp_s{j}"].values.astype(float))
    L = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{10pt}",
         "\\caption{Mean test correlation $\\rho$ by training regime (row) and "
         "evaluation regime (column), averaged over the remaining algorithms, "
         "problems and temperatures. The diagonal (bold) is the training regime; "
         "the off-diagonal entries are the held-out regimes.}",
         "\\label{tab:genmatrix}", "\\begin{tabular}{lrrr}", "\\toprule",
         "Train $\\backslash$ Eval & subset $0$ & subset $1$ & subset $2$ \\\\",
         "\\midrule"]
    for i in range(3):
        cells = []
        for j in range(3):
            s = "%.3f" % M[i, j]
            if i == j:
                s = "$\\mathbf{%s}$" % s
            cells.append(s)
        L.append(f"subset ${i}$ & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L), M


# ===========================================================================
# Online avg-rank figure  +  UMDA-gain heatmap
# ===========================================================================
def _online_cell():
    on = pd.read_csv(SUMMARY)
    on = on[~on["family"].isin(EX_FAM) & ~on["algorithm"].isin(EX_ALG)].copy()
    on["best_fitness"] = pd.to_numeric(on["best_fitness"], errors="coerce")
    return on


def online_avg_rank():
    on = _online_cell()
    algos = [a for a in ALG_ORDER if a in set(on["algorithm"])]
    cell = on.groupby(["problem", "algorithm"], as_index=False)["best_fitness"].mean()
    piv = cell.pivot_table(index="problem", columns="algorithm", values="best_fitness")
    piv = piv.reindex(columns=algos)
    ranks = piv.rank(axis=1, ascending=False)
    avg_rank = ranks.mean(axis=0).sort_values()
    order = list(avg_rank.index)
    # Friedman/CD on complete blocks
    complete = piv.dropna(axis=0, how="any")
    k, n = complete.shape[1], complete.shape[0]
    chi2, p = friedmanchisquare(*[complete[c].values for c in complete.columns])
    cd = _nemenyi_cd(k, n)
    wins = (cell.loc[cell.groupby("problem")["best_fitness"].idxmax(), "algorithm"]
            .value_counts())

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(order))
    ax.barh(y, avg_rank.values, color="#2b6cb0")
    ax.set_yticks(y); ax.set_yticklabels([ALG_LABEL[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlabel("average rank on EDA best fitness (1 = best)")
    best_r = avg_rank.values[0]
    ax.axvline(best_r + cd, color="red", ls="--", lw=1)
    ax.annotate(f"CD={cd:.2f}", (best_r + cd, len(order) - 1), color="red",
                fontsize=9, ha="left", va="bottom")
    fig.savefig(os.path.join(V2, "figures", "fig_avg_rank_v2.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig_avg_rank_v2.pdf"), bbox_inches="tight")
    plt.close(fig)
    return dict(order=[ALG_LABEL[a] for a in order],
                avg_rank=avg_rank, chi2=chi2, p=p, k=k, n=n, cd=cd,
                best=ALG_LABEL[order[0]], worst=ALG_LABEL[order[-1]],
                wins={ALG_LABEL[a]: int(wins.get(a, 0)) for a in order})


def umda_gain_selected():
    on = _online_cell()
    SEL = {"Trap", "Ising", "Deceptive3", "Checkerboard"}
    on = on[on["family"].isin(SEL)]
    cell = on.groupby(["problem", "group", "algorithm"], as_index=False)["best_fitness"].mean()
    gains = {}
    for (prob, grp), g in cell.groupby(["problem", "group"]):
        gg = g.set_index("algorithm")["best_fitness"]
        if "univ_bn" not in gg.index:
            continue
        rng = gg.max() - gg.min()
        gains[prob] = {a: (0.0 if rng <= 0 else (gg[a] - gg["univ_bn"]) / rng)
                       for a in gg.index}
    gorder = {"small": 0, "medium": 1, "large": 2}
    grp_of = dict(zip(cell["problem"], cell["group"]))
    problems = sorted(gains, key=lambda p: (gorder.get(grp_of[p], 9), fam_of(p),
                                            int(p.rsplit("_", 1)[1])))
    algs = [a for a in ALG_ORDER if a != "univ_bn"]
    Mmat = np.full((len(algs), len(problems)), np.nan)
    for j, p in enumerate(problems):
        for i, a in enumerate(algs):
            if a in gains[p]:
                Mmat[i, j] = gains[p][a]
    order = np.argsort(-np.nanmean(Mmat, axis=1))
    Mmat = Mmat[order]; algs = [algs[i] for i in order]
    fig, ax = plt.subplots(figsize=(0.4 * len(problems) + 3, 0.36 * len(algs) + 2))
    vmax = np.nanpercentile(np.abs(Mmat), 98)
    im = ax.imshow(Mmat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([p.replace("_", " ") for p in problems], rotation=90, fontsize=7)
    ax.set_yticks(range(len(algs))); ax.set_yticklabels([ALG_LABEL[a] for a in algs], fontsize=8)
    grps = [gorder.get(grp_of[p], 9) for p in problems]
    for j in range(1, len(grps)):
        if grps[j] != grps[j - 1]:
            ax.axvline(j - 0.5, color="black", lw=1.2)
    for i in range(Mmat.shape[0]):
        for j in range(Mmat.shape[1]):
            v = Mmat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("normalised best-fitness gain vs UMDA (blue > UMDA, red < UMDA)", fontsize=8)
    fig.savefig(os.path.join(V2, "figures", "fig_umda_gain_heatmap_selected_v2.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig_umda_gain_heatmap_selected_v2.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    neg = int(np.nansum(Mmat < 0)); tot = int(np.sum(~np.isnan(Mmat)))
    return dict(neg=neg, tot=tot, n_prob=len(problems), n_alg=len(algs))


# ===========================================================================
# Table 5 + Fig 2: families (clustering + interaction + bootstrap)
# ===========================================================================
def _filtered_panel():
    """Reimplements T.load_panel with exclusions applied before normalisation."""
    on = pd.read_csv(SUMMARY)
    on = on[~on["family"].isin(EX_FAM) & ~on["algorithm"].isin(EX_ALG)].copy()
    on["best_fitness"] = pd.to_numeric(on["best_fitness"], errors="coerce")
    cell = on.groupby(["problem", "algorithm"], as_index=False)["best_fitness"].mean()

    def norm(g):
        v = g["best_fitness"]; rng = v.max() - v.min(); g = g.copy()
        g["online_norm"] = np.nan if rng <= 0 else (v - v.min()) / rng
        return g
    cell = cell.groupby("problem", group_keys=False).apply(norm).dropna(subset=["online_norm"])

    off = pd.read_csv(OFFLINE)
    off = off[~off["family"].isin(EX_FAM) & ~off["algorithm"].isin(EX_ALG)]
    for m in T.METS:
        off[m] = pd.to_numeric(off[m], errors="coerce")
    offcell = (off[off["temperature"] == T.MATCH_T]
               .groupby(["problem", "algorithm"], as_index=False)[T.METS].mean())
    panel = cell.merge(offcell, on=["problem", "algorithm"], how="inner")
    for m, (_l, hi) in T.OFF_METRICS.items():
        if not hi:
            panel[m] = -panel[m]
    return panel


def families():
    panel = _filtered_panel()
    algorithms = [a for a in ALG_ORDER if a in set(panel["algorithm"])]
    ON, OFF, problems = T.build_arrays(panel, algorithms)
    R = T.matrix_R_arr(ON, OFF, np.arange(len(problems)))
    Z = T.zscore_impute(R)
    # interaction permutation test
    obs, pval, obs_m, p_m = T.interaction_test(ON, OFF, R)
    # silhouette -> k
    from sklearn.metrics import silhouette_score
    sil = {}
    for k in range(2, 7):
        _, lab = T.cluster(Z, k)
        sil[k] = silhouette_score(Z, lab) if len(set(lab)) > 1 else float("nan")
    best_k = max(sil, key=sil.get)
    L, labels = T.cluster(Z, best_k)
    # figures (write to V2 then copy into paper with _v2 names)
    T.ALG_LABEL.update(ALG_LABEL)
    consensus, jacc = T.bootstrap_stability(ON, OFF, labels, best_k)
    T.fig_dendro(L, algorithms, labels, V2)
    T.fig_consensus(consensus, algorithms, labels, V2)
    for src, dst in [("fig_family_dendrogram.pdf", "fig_family_dendrogram_v2.pdf"),
                     ("fig_family_consensus.pdf", "fig_family_consensus_v2.pdf")]:
        import shutil
        shutil.copy(os.path.join(V2, "figures", src), os.path.join(PAPER, dst))
    # contrast between highest and lowest test_sp cluster
    j = T.METS.index("test_sp")
    cl_mean = {c: np.nanmean(R[np.array(labels) == c, j]) for c in set(labels)}
    hi_c, lo_c = max(cl_mean, key=cl_mean.get), min(cl_mean, key=cl_mean.get)
    hi_idx = [i for i in range(len(algorithms)) if labels[i] == hi_c]
    lo_idx = [i for i in range(len(algorithms)) if labels[i] == lo_c]
    d, ci, p0 = T.contrast_ci(ON, OFF, hi_idx, lo_idx)

    # build family table
    rows = []
    for c in sorted(set(labels)):
        members = [ALG_LABEL[algorithms[i]] for i in range(len(algorithms)) if labels[i] == c]
        mean_resp = np.nanmean(R[np.array(labels) == c], axis=0)
        rows.append((c, members, mean_resp, jacc[c]))
    return dict(n_prob=len(problems), n_alg=len(algorithms), obs=obs, pval=pval,
                p_m=dict(zip(T.METS, p_m)), obs_m=dict(zip(T.METS, obs_m)),
                sil=sil, best_k=best_k, rows=rows, jacc=jacc,
                contrast=dict(hi=hi_c, lo=lo_c, d=d, ci=ci, p0=p0,
                              hi_members=[ALG_LABEL[algorithms[i]] for i in hi_idx],
                              lo_members=[ALG_LABEL[algorithms[i]] for i in lo_idx]))


def main():
    print("############### TABLE 2 (completion) ###############")
    print(table_completion())
    print("\n############### TABLE 3 (offline ranking) ###############")
    t3, s3 = table_ranking_offline()
    print(t3)
    print(f"\n% offline Friedman: chi2={s3['chi2']:.1f} p={s3['p']:.2e} "
          f"k={s3['k']} n={s3['n']} CD={s3['cd']:.2f}")
    print(f"% best={s3['best']} tied-within-CD={s3['tied']}")
    print("\n############### TABLE 4 (genmatrix) ###############")
    t4, M = table_genmatrix_offline()
    print(t4)
    print("\n############### ONLINE avg-rank (Fig) ###############")
    o = online_avg_rank()
    print(f"% online Friedman: chi2={o['chi2']:.1f} p={o['p']:.2e} k={o['k']} "
          f"n={o['n']} CD={o['cd']:.2f}; best={o['best']} worst={o['worst']}")
    print("% avg ranks:", {a: round(o["avg_rank"][k], 2)
                           for k, a in zip(o["avg_rank"].index, o["order"])})
    print("% wins:", o["wins"])
    print("\n############### UMDA gain (selected) ###############")
    u = umda_gain_selected()
    print(f"% selected heatmap: {u['n_alg']} algs x {u['n_prob']} problems; "
          f"cells<0 (UMDA better): {u['neg']}/{u['tot']}")
    print("\n############### TABLE 5 (families) ###############")
    f = families()
    print(f"% panel: {f['n_prob']} problems x {f['n_alg']} learners")
    print(f"% interaction permutation: summed var={f['obs']:.3f} p={f['pval']:.4f}")
    for m in T.METS:
        print(f"%   {m:9s} var={f['obs_m'][m]:.3f} p={f['p_m'][m]:.4f}")
    print(f"% silhouette: {{{', '.join(f'{k}:{v:.3f}' for k,v in f['sil'].items())}}} -> k={f['best_k']}")
    print("% families:")
    for c, members, resp, jc in f["rows"]:
        rr = "  ".join(f"{m}={resp[i]:+.2f}" for i, m in enumerate(T.METS))
        print(f"%   C{c} (Jaccard {jc:.2f}): {', '.join(members)}")
        print(f"%       {rr}")
    ct = f["contrast"]
    print(f"% contrast C{ct['hi']} vs C{ct['lo']}: d={ct['d']:+.3f} "
          f"95%CI[{ct['ci'][0]:+.3f},{ct['ci'][1]:+.3f}] p={ct['p0']:.4f}")
    print(f"%   hi={ct['hi_members']}  lo={ct['lo_members']}")


if __name__ == "__main__":
    main()
