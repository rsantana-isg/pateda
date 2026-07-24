"""
Analyze PBO results produced by ``compare_discrete_edas_pbo.py``.

Loads the IOHexperimenter data folders (one per algorithm) with the
``iohinspector`` package and produces paper-ready figures and tables in
``results/pbo_analysis/``:

Figures (saved as .pdf and .eps, no titles, large fonts):
  - ``fixed_budget_f{fid}_dim{dim}``  mean best-so-far fitness vs
    evaluations, one line per algorithm (fixed-budget convergence plot).
  - ``ecdf_dim{dim}``                 ECDF of target achievement,
    aggregated over all functions of one dimension.

Tables:
  - ``final_fitness.csv``             per-run final best fitness (raw data).
  - ``table_final_dim{dim}.tex``      mean +- std of the final best fitness,
                                      functions x algorithms, best in bold.
  - ``aocc_dim{dim}.csv`` and ``table_aocc_dim{dim}.tex``
                                      Area Over the Convergence Curve
                                      (anytime performance in [0, 1],
                                      higher is better).
  - ``kruskal_wallis.csv``            Kruskal-Wallis test across algorithms
                                      per (function, dimension).
  - ``dunn_f{fid}_dim{dim}.csv``      post hoc Dunn tests (only if the
                                      ``scikit_posthocs`` package is
                                      installed).

Usage (all arguments optional, positional):
    python scripts/analyze_pbo_results.py [data_root] [output_dir]

    data_root   folder with one IOH data folder per algorithm
                (default: results/pbo_data)
    output_dir  where figures and tables are written
                (default: results/pbo_analysis)
"""

import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from iohinspector import (
    DataManager,
    plot_single_function_fixed_budget,
    get_data_ecdf,
    get_sequence,
    transform_fval,
    get_aocc,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_data")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_analysis")
)

# Paper-ready fonts (figures carry no titles; captions live in the LaTeX doc).
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})


def save_figure(fig, output_dir, stem):
    """Save one figure as both .pdf and .eps for LaTeX inclusion."""
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  figure: {stem}.pdf / .eps")


def load_manager(data_root):
    """Create a DataManager over every algorithm folder under data_root."""
    folders = [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
    ]
    if not folders:
        raise SystemExit(f"No data folders found in {data_root}")
    manager = DataManager()
    manager.add_folders(folders)
    return manager


def function_label(overview, fid):
    """Return a 'f{fid} Name' label for tables."""
    name = (
        overview.filter(pl.col("function_id") == fid)["function_name"][0]
    )
    return f"f{fid} {name}"


def write_final_fitness_tables(overview, output_dir, dims, algorithms):
    """Per-run CSV plus one LaTeX table (mean +- std, best bold) per dim."""
    df = overview.to_pandas()
    df = df.rename(columns={"best_y": "final_best"})
    csv_path = os.path.join(output_dir, "final_fitness.csv")
    # data_id is the globally unique run identifier assigned by iohinspector
    # (run_id is only unique within one data folder).
    df[["data_id", "algorithm_name", "function_id", "function_name",
        "dimension", "instance", "run_id", "evals",
        "final_best"]].to_csv(csv_path, index=False)
    print(f"  table:  final_fitness.csv ({len(df)} runs)")

    for dim in dims:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            continue
        grouped = sub.groupby(["function_id", "algorithm_name"])["final_best"]
        mean = grouped.mean().unstack()[algorithms]
        std = grouped.std().unstack()[algorithms]

        lines = [
            "% Final best fitness (mean +- std over runs), "
            f"PBO suite, n={dim}.  Best mean per function in bold.",
            "\\setlength{\\tabcolsep}{10pt}",
            "\\begin{tabular}{l" + "r" * len(algorithms) + "}",
            "\\hline",
            "Function & " + " & ".join(algorithms) + " \\\\",
            "\\hline",
        ]
        for fid in mean.index:
            label = function_label(overview, int(fid))
            best_mean = mean.loc[fid].max()
            cells = []
            for alg in algorithms:
                m, s = mean.loc[fid, alg], std.loc[fid, alg]
                if np.isnan(m):
                    cells.append("--")
                    continue
                body = f"{m:.2f} \\pm {0.0 if np.isnan(s) else s:.2f}"
                if np.isclose(m, best_mean):    # bold all tied bests
                    body = f"\\mathbf{{{body}}}"
                cells.append(f"${body}$")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]

        tex_path = os.path.join(output_dir, f"table_final_dim{dim}.tex")
        with open(tex_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  table:  table_final_dim{dim}.tex")


def write_statistical_tests(overview, output_dir, dims):
    """Kruskal-Wallis across algorithms per (function, dim) + Dunn post hoc."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
        print("  note:   scikit_posthocs not installed; skipping Dunn tests")

    df = overview.to_pandas()
    rows = []
    for (fid, dim), sub in df.groupby(["function_id", "dimension"]):
        if dim not in dims:
            continue
        groups = [g["best_y"].values
                  for _, g in sub.groupby("algorithm_name") if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, pval = scipy_stats.kruskal(*groups)
        except ValueError:
            stat, pval = np.nan, np.nan
        if np.isnan(pval):      # all values identical -> no difference
            pval = 1.0
        rows.append({"function_id": fid, "dimension": dim,
                     "H_statistic": stat, "p_value": pval})

        if sp is not None and pval < 0.05:
            dunn = sp.posthoc_dunn(sub, val_col="best_y",
                                   group_col="algorithm_name",
                                   p_adjust="holm")
            dunn.to_csv(os.path.join(output_dir,
                                     f"dunn_f{int(fid)}_dim{int(dim)}.csv"))

    kw = pd.DataFrame(rows)
    kw.to_csv(os.path.join(output_dir, "kruskal_wallis.csv"), index=False)
    print(f"  table:  kruskal_wallis.csv ({len(kw)} tests)")


def write_aocc_tables(manager, overview, output_dir, dims, algorithms):
    """Area Over the Convergence Curve per (function, algorithm, dim)."""
    for dim in dims:
        parts = []
        fids = sorted(
            overview.filter(pl.col("dimension") == dim)["function_id"]
            .unique().to_list()
        )
        eval_max = int(overview.filter(pl.col("dimension") == dim)["evals"].max())
        for fid in fids:
            data = (
                manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                .load(monotonic=True, include_meta_data=True)
            )
            if len(data) == 0:
                continue
            # Normalize fitness to [0, 1] per function (linear scale: PBO
            # values can be <= 0, so no log transform).
            lb = float(data["raw_y"].min())
            ub = float(data["raw_y"].max())
            if ub <= lb:
                continue
            data = transform_fval(data, lb=lb, ub=ub, scale_log=False,
                                  maximization=True)
            parts.append(
                get_aocc(data, eval_max=eval_max,
                         free_vars=["function_id", "function_name",
                                    "algorithm_name"],
                         return_as_pandas=True)
            )
        if not parts:
            continue
        aocc = pd.concat(parts, ignore_index=True)
        aocc.to_csv(os.path.join(output_dir, f"aocc_dim{dim}.csv"),
                    index=False)

        pivot = aocc.pivot_table(index="function_id",
                                 columns="algorithm_name", values="AOCC")
        pivot = pivot[[a for a in algorithms if a in pivot.columns]]
        lines = [
            f"% AOCC (area over the convergence curve), PBO suite, n={dim}.",
            "% Anytime performance in [0,1]; higher is better; best in bold.",
            "\\setlength{\\tabcolsep}{10pt}",
            "\\begin{tabular}{l" + "r" * len(pivot.columns) + "}",
            "\\hline",
            "Function & " + " & ".join(pivot.columns) + " \\\\",
            "\\hline",
        ]
        for fid in pivot.index:
            label = function_label(overview, int(fid))
            best_aocc = pivot.loc[fid].max()
            cells = []
            for alg in pivot.columns:
                v = pivot.loc[fid, alg]
                if np.isnan(v):
                    cells.append("--")
                elif np.isclose(v, best_aocc):  # bold all tied bests
                    cells.append(f"$\\mathbf{{{v:.3f}}}$")
                else:
                    cells.append(f"${v:.3f}$")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(os.path.join(output_dir, f"table_aocc_dim{dim}.tex"),
                  "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  table:  aocc_dim{dim}.csv / table_aocc_dim{dim}.tex")


def make_fixed_budget_figures(manager, overview, output_dir, dims):
    """One convergence plot (mean best-so-far vs evaluations) per (f, dim)."""
    for dim in dims:
        fids = sorted(
            overview.filter(pl.col("dimension") == dim)["function_id"]
            .unique().to_list()
        )
        for fid in fids:
            data = (
                manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                .load(monotonic=True, include_meta_data=True)
            )
            if len(data) == 0:
                continue
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            # Arithmetic mean (PBO values can be <= 0, so no geometric mean).
            plot_single_function_fixed_budget(
                data, maximization=True, measures=["mean"], ax=ax,
            )
            ax.set_title("")
            ax.set_yscale("linear")     # PBO fitness scales are linear
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best fitness")
            # Clean the seaborn legend: drop pseudo-entries, unpack tuples.
            handles, labels = ax.get_legend_handles_labels()
            keep = [(h, l.strip("(),'\" ")) for h, l in zip(handles, labels)
                    if l not in ("None", "variable", "mean")]
            ax.legend([h for h, _ in keep], [l for _, l in keep],
                      ncol=2, frameon=False)
            save_figure(fig, output_dir, f"fixed_budget_f{int(fid)}_dim{dim}")


def make_ecdf_figures(manager, overview, output_dir, dims):
    """ECDF aggregated over all functions of one dimension.

    Targets are normalized per function (linear scale, since PBO values can
    be <= 0) before averaging, following the aggregated ECDF of IOHanalyzer.
    """
    for dim in dims:
        sub = overview.filter(pl.col("dimension") == dim)
        if len(sub) == 0:
            continue
        fids = sorted(sub["function_id"].unique().to_list())
        eval_max = int(sub["evals"].max())
        eval_values = get_sequence(1, eval_max, 50, scale_log=True,
                                   cast_to_int=True)
        parts = []
        for fid in fids:
            data = (
                manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                .load(monotonic=True, include_meta_data=True)
            )
            if len(data) == 0:
                continue
            f_min = float(data["raw_y"].min())
            f_max = float(data["raw_y"].max())
            if f_max <= f_min:
                continue
            parts.append(
                get_data_ecdf(data, maximization=True, scale_f_log=False,
                              f_min=f_min, f_max=f_max,
                              eval_values=eval_values,
                              return_as_pandas=True)
            )
        if not parts:
            continue
        ecdf = (
            pd.concat(parts, ignore_index=True)
            .groupby(["evaluations", "algorithm_name"])["eaf"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for alg, grp in ecdf.groupby("algorithm_name"):
            ax.plot(grp["evaluations"], grp["eaf"], label=alg)
        ax.set_xscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Proportion of (target, run) pairs")
        ax.set_ylim(0, 1.02)
        ax.legend(ncol=2, frameon=False)
        save_figure(fig, output_dir, f"ecdf_dim{dim}")


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")

    manager = load_manager(data_root)
    overview = manager.overview
    dims = sorted(overview["dimension"].unique().to_list())
    algorithms = sorted(overview["algorithm_name"].unique().to_list())
    n_funcs = overview["function_id"].n_unique()
    print(f"Loaded: {len(algorithms)} algorithms, {n_funcs} functions, "
          f"dims={dims}, {len(overview)} runs\n")

    print("Tables:")
    write_final_fitness_tables(overview, output_dir, dims, algorithms)
    write_aocc_tables(manager, overview, output_dir, dims, algorithms)
    write_statistical_tests(overview, output_dir, dims)

    print("Figures:")
    make_fixed_budget_figures(manager, overview, output_dir, dims)
    make_ecdf_figures(manager, overview, output_dir, dims)

    print(f"\nDone.  Output in {output_dir}")


if __name__ == "__main__":
    main()
