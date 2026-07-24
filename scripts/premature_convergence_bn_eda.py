"""
premature_convergence_bn_eda.py -- mechanistic check for WHY the offline
search-distribution metrics fail to predict online EDA performance on the
deceptive / decomposable families (Deceptive3, Trap, UBQP, MaxClique).

Two competing explanations are tested, per function, across the 19 learners.
Because the ``.dat`` files hold no population-diversity trace (that lives only
in the JSON, which was not kept), the available convergence signal is
``generation_found`` -- the generation at which the run's overall best was first
found.  Small ``generation_found`` together with a low final fitness is the
signature of premature convergence.

  H1  PREMATURE CONVERGENCE.  Learners whose model fits the population better
      (high live Spearman ``mean_sp_pop``) collapse the population earlier and
      get stuck:
        rho(mean_sp_pop, generation_found) < 0   (better fit -> earlier stop)
        rho(generation_found, best_fitness) > 0   (earlier stop -> worse fitness)

  H2  METRIC / STRUCTURE MISMATCH.  The offline Spearman rewards simple,
      low-structure models that generalise the *distribution* across regimes but
      miss the variable interactions needed to *optimise*:
        rho(offline test_sp, final_edges) < 0     (offline rho favours sparse models)
        rho(mean_f1, best_fitness)        > 0     (structure, not fit, drives fitness)

For each function we compute these blocked (within-problem, across-algorithm)
Spearman correlations and aggregate them by family, contrasting the families
where the offline metric predicts online performance (Ising, Checkerboard) with
those where it anti-predicts (Deceptive3, Trap, UBQP, MaxClique).

Usage
-----
    python3 scripts/premature_convergence_bn_eda.py [online_summary_csv] [offline_csv] [out_dir]
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
MATCH_T = 1.0

# families where offline test-rho predicts online (from Level A) vs where it
# anti-predicts -- used only to order/annotate the output.
PREDICTED = ["Ising", "Checkerboard", "EqualProducts"]
ANTIPREDICTED = ["Braid", "Trap", "MaxClique", "Deceptive3", "UBQP"]

# per-function correlations we compute (label, x, y, higher-good sign note)
CHECKS = [
    ("H1a rho(fit, gen_found)",       "mean_sp_pop",   "generation_found"),
    ("H1b rho(gen_found, fitness)",   "generation_found", "best_fitness"),
    ("H2a rho(off_test_sp, edges)",   "test_sp",       "final_edges"),
    ("H2b rho(f1, fitness)",          "mean_f1",       "best_fitness"),
    ("ref rho(off_test_sp, fitness)", "test_sp",       "best_fitness"),
    ("ref rho(fit, fitness)",         "mean_sp_pop",   "best_fitness"),
]


def load(online_csv, offline_csv):
    on = pd.read_csv(online_csv)
    num = ["best_fitness", "generation_found", "mean_sp_pop", "mean_f1",
           "final_edges"]
    cell = on.groupby(["problem", "family", "algorithm"], as_index=False)[num].mean()
    off = pd.read_csv(offline_csv)
    off["test_sp"] = pd.to_numeric(off["test_sp"], errors="coerce")
    offcell = (off[off["temperature"] == MATCH_T]
               .groupby(["problem", "algorithm"], as_index=False)["test_sp"].mean())
    return cell.merge(offcell, on=["problem", "algorithm"], how="left")


def per_function_corr(df):
    recs = []
    for problem, g in df.groupby("problem"):
        if g["algorithm"].nunique() < 5:
            continue
        row = {"problem": problem, "family": g["family"].iloc[0], "n_alg": len(g)}
        for lab, xcol, ycol in CHECKS:
            x = g[xcol].to_numpy(float)
            y = g[ycol].to_numpy(float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 5 or np.all(x[mask] == x[mask][0]) or np.all(y[mask] == y[mask][0]):
                row[lab] = np.nan
            else:
                row[lab] = spearmanr(x[mask], y[mask]).correlation
        recs.append(row)
    return pd.DataFrame(recs)


def by_family(pc):
    labs = [c[0] for c in CHECKS]
    fam = pc.groupby("family")[labs].mean()
    order = [f for f in (PREDICTED + ANTIPREDICTED) if f in fam.index]
    order += [f for f in fam.index if f not in order]
    return fam.reindex(order)


def write_table(fam, out_dir):
    labs = [c[0] for c in CHECKS]
    path = os.path.join(out_dir, "tables", "table_premature_convergence.tex")
    short = {c[0]: c[0].split()[0] for c in CHECKS}
    with open(path, "w") as f:
        f.write("% mechanistic check: premature convergence (H1) vs metric/structure mismatch (H2)\n")
        f.write("% blocked within-function across-algorithm Spearman, averaged per family\n")
        f.write("\\begin{tabular}{l" + "c" * len(labs) + "}\n\\toprule\n")
        f.write("Family & " + " & ".join(short[l].replace("_", "\\_") for l in labs)
                + " \\\\\n\\midrule\n")
        for famname, r in fam.iterrows():
            cells = " & ".join(("%.2f" % r[l]) if not np.isnan(r[l]) else "--"
                               for l in labs)
            f.write(f"{famname} & {cells} \\\\\n")
        f.write("\\midrule\n")
        allmean = fam.mean()
        f.write("all & " + " & ".join("%.2f" % allmean[l] for l in labs) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def alg_convergence_profile(df, out_dir):
    """Per algorithm on the anti-predicted (deceptive) families: does the
    winner converge earlier or later, and is it the more-structured model?"""
    dec = df[df["family"].isin(ANTIPREDICTED)].copy()
    dec["rank"] = dec.groupby("problem")["best_fitness"].rank(ascending=False,
                                                              method="average")
    prof = (dec.groupby("algorithm")
            .agg(mean_perf_rank=("rank", "mean"),
                 mean_gen_found=("generation_found", "mean"),
                 mean_f1=("mean_f1", "mean"),
                 mean_edges=("final_edges", "mean"),
                 mean_off_sp=("test_sp", "mean"))
            .sort_values("mean_perf_rank"))
    prof.to_csv(os.path.join(out_dir, "deceptive_convergence_profile.csv"))
    return prof


def fig_mechanism(pc, out_dir):
    fam = by_family(pc)
    labs = [c[0] for c in CHECKS]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(fam[labs].to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labs], fontsize=8)
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels(fam.index, fontsize=9)
    for i in range(fam.shape[0]):
        for j in range(len(labs)):
            v = fam.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="within-function Spearman")
    fig.savefig(os.path.join(out_dir, "figures", "fig_premature_convergence.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    online_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ONLINE
    offline_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OFFLINE
    out_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    df = load(online_csv, offline_csv)
    pc = per_function_corr(df)
    pc.to_csv(os.path.join(out_dir, "premature_convergence_per_function.csv"),
              index=False)
    fam = by_family(pc)
    write_table(fam, out_dir)
    fig_mechanism(pc, out_dir)
    prof = alg_convergence_profile(df, out_dir)

    labs = [c[0] for c in CHECKS]
    print("=== Per-family within-function correlations (blocked across algorithms) ===")
    with pd.option_context("display.width", 160):
        print(fam[labs].round(2).to_string())
    print("\nall-family mean:")
    print(fam[labs].mean().round(3).to_string())

    print("\n=== H1 premature convergence ===")
    m = fam[labs].mean()
    print(f"  H1a rho(model fit, gen_found) = {m['H1a rho(fit, gen_found)']:+.2f}  "
          f"(H1 predicts < 0: better fit -> earlier stop)")
    print(f"  H1b rho(gen_found, fitness)   = {m['H1b rho(gen_found, fitness)']:+.2f}  "
          f"(H1 predicts > 0: earlier stop -> worse fitness)")
    print("\n=== H2 metric / structure mismatch ===")
    print(f"  H2a rho(offline test_sp, edges) = {m['H2a rho(off_test_sp, edges)']:+.2f}  "
          f"(H2 predicts < 0: offline rho favours sparse models)")
    print(f"  H2b rho(F1, fitness)            = {m['H2b rho(f1, fitness)']:+.2f}  "
          f"(H2 predicts > 0: structure drives fitness)")

    print("\n=== Convergence profile on the deceptive/anti-predicted families ===")
    print("   (learners ordered best->worst online; gen_found, F1, edges, offline rho)")
    print(prof.round(3).to_string())

    print(f"\nWrote premature-convergence table/figure/CSVs to {out_dir}")


if __name__ == "__main__":
    main()
