"""Individualized time profiles for the 11 BN-based EDAs (pbo_bn_variants).

Reads results/bn_profile/bn_profile.csv (from profile_bn_variants.py) and writes
two figures to results/bn_profile/:
  fig_bn_component_profile.pdf : per-algorithm component share (% of total, n=100)
  fig_bn_sampling_cost.pdf     : absolute sampling cost per generation, n=64 & 100
                                 (log scale) -- the per-solution vs block gap.
No titles (captions live in the LaTeX doc); large fonts for paper use.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "bn_profile"))
NGEN = 8
ORDER = ["EBNA_BIC", "EBNA_K2", "EBNA_PC", "LFDA_mp6", "BOA_mp6", "SARTRE_mp6",
         "A1_dt", "A2_mi", "A3_fast", "A4_mdl", "A5_ndg"]
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 12})


def main():
    df = pd.read_csv(os.path.join(OUT, "bn_profile.csv")).set_index(["alg", "dim"])

    # ---- Figure 1: component share (% of total) at n=100, stacked ----
    comps = ["learn", "sample", "evaluate", "select", "replace", "other"]
    colors = {"learn": "#4C72B0", "sample": "#DD8452", "evaluate": "#55A868",
              "select": "#C44E52", "replace": "#8172B3", "other": "#937860"}
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(ORDER))
    left = np.zeros(len(ORDER))
    for c in comps:
        vals = np.array([100 * df.loc[(a, 100), c] / df.loc[(a, 100), "total"]
                         for a in ORDER])
        ax.barh(y, vals, left=left, color=colors[c],
                label=c, edgecolor="white", linewidth=0.5)
        left += vals
    for i, a in enumerate(ORDER):
        s = 100 * df.loc[(a, 100), "sample"] / df.loc[(a, 100), "total"]
        ax.text(101, i, f"sample={s:.1f}%  ({1000*df.loc[(a,100),'sample']/NGEN:.0f} ms/gen)",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("share of total runtime (%), n = 100")
    ax.set_xlim(0, 145)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.42, -0.16),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bn_component_profile.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: absolute sampling cost per generation (log scale) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ORDER))
    w = 0.38
    for j, dim in enumerate((64, 100)):
        ms = [1000 * df.loc[(a, dim), "sample"] / NGEN for a in ORDER]
        cols = ["#DD8452" if df.loc[(a, dim), "sampler"] == "SampleBayesianNetwork"
                else "#55A868" for a in ORDER]
        ax.bar(x + (j - 0.5) * w, ms, w, color=cols,
               alpha=1.0 if dim == 100 else 0.55,
               edgecolor="black", linewidth=0.4,
               label=f"n={dim}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=35, ha="right")
    ax.set_ylabel("sampling time / generation (ms), pop=200", labelpad=8)
    ax.set_ylim(1.5, 900)
    ax.axvline(5.5, color="grey", ls="--", lw=1)
    ax.text(2.5, 620, "per-solution  (SampleBayesianNetwork)",
            ha="center", fontsize=11, color="#C0631F")
    ax.text(8.5, 620, "block / vectorised  (SampleLocalStructureBN)",
            ha="center", fontsize=11, color="#2E7D32")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#DD8452", label="per-solution sampler"),
                       Patch(color="#55A868", label="block (vectorised) sampler")],
              frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bn_sampling_cost.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote:", os.path.join(OUT, "fig_bn_component_profile.pdf"))
    print("wrote:", os.path.join(OUT, "fig_bn_sampling_cost.pdf"))


if __name__ == "__main__":
    main()
