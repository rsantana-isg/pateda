"""
Analyze the weighted-probability PBO results produced by
``compare_weighted_edas_pbo.py`` / ``run_weighted_pbo_eda.py``.

The IOH data folders are named ``{ALG}__{SEL}`` (``SEL`` in ``FP``, ``BZ``,
``RTS``), so the ``algorithm_name`` stored by IOH encodes both the algorithm and
the selection method.  This script loads them with ``iohinspector``'s
``DataManager`` and produces, under ``results/pbo_weighted_analysis/``:

Per selection method (subdir ``FP/``, ``BZ/``, ``RTS/`` -- algorithms compared
WITHIN one weighting/diversity scheme):
  - ``fixed_budget_f{fid}_dim{dim}.pdf/.eps``  mean best-so-far vs evaluations,
    one line per algorithm (fixed-budget convergence plot).
  - ``ecdf_dim{dim}.pdf/.eps``                 ECDF of target achievement,
    aggregated over all functions of one dimension.
  - ``final_fitness.csv``                      per-run final best fitness.
  - ``table_final_dim{dim}.tex``               mean +- std, functions x algorithms,
                                               tied bests bold, ``\\tabcolsep{10pt}``.
  - ``aocc_dim{dim}.csv`` / ``table_aocc_dim{dim}.tex``  Area Over the Convergence
                                               Curve (anytime performance in [0,1]).
  - ``kruskal_wallis.csv``                     KW across algorithms per (function,dim).
  - ``dunn_f{fid}_dim{dim}.csv``               post hoc Dunn (only if
                                               ``scikit_posthocs`` is installed).

Cross-method summary (subdir ``summary/`` -- the three selection methods
compared directly):
  - ``ecdf_methods_dim{dim}.pdf/.eps``         aggregated ECDF, one line per
                                               selection method (pooled over all
                                               functions and algorithms).
  - ``aocc_by_method_dim{dim}.csv`` / ``table_aocc_by_method_dim{dim}.tex``
                                               mean AOCC, algorithms x selection
                                               methods, best method per algorithm bold.
  - ``kruskal_methods.csv``                    KW across the three selection methods
                                               per (function, dim).

Figures carry no titles (captions live in the LaTeX document) and use large,
paper-ready fonts; each is saved as both ``.pdf`` and ``.eps``.

Usage (all arguments optional, positional):
    python scripts/analyze_weighted_pbo_results.py [data_root] [output_dir] [selection]

    data_root   folder with one IOH data folder per (algorithm, selection method)
                (default: results/pbo_weighted_data)
    output_dir  where figures and tables are written
                (default: results/pbo_weighted_analysis)
    selection   optional: FP, BZ or RTS -- analyse ONLY that selection method
                (loads only its folders, writes only its outputs, no cross-method
                summary).  Omit to analyse all three together (per-method subdirs
                FP/, BZ/, RTS/ plus a cross-method summary/).

Examples:
    # all three selection methods in one run (subdirs FP/ BZ/ RTS/ + summary/)
    python scripts/analyze_weighted_pbo_results.py pbo_weighted_data_cluster \\
        results/pbo_weighted_analysis

    # three independent, lighter analyses (one selection each, no clutter)
    python scripts/analyze_weighted_pbo_results.py pbo_weighted_data_cluster \\
        results/pbo_analysis_FP  FP
    python scripts/analyze_weighted_pbo_results.py pbo_weighted_data_cluster \\
        results/pbo_analysis_BZ  BZ
    python scripts/analyze_weighted_pbo_results.py pbo_weighted_data_cluster \\
        results/pbo_analysis_RTS RTS
"""

import os
import sys
import glob

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
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_weighted_data")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_weighted_analysis")
)

SELECTION_ORDER = ["FP", "BZ", "RTS"]

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_figure(fig, output_dir, stem):
    """Save one figure as both .pdf and .eps for LaTeX inclusion."""
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  figure: {os.path.relpath(os.path.join(output_dir, stem))}.pdf / .eps")


def _folder_selection(basename):
    """Selection method encoded in a data-folder name.

    Folders are named ``{ALG}__{SEL}`` (local) or ``{ALG}__{SEL}_f..dim..s..``
    (cluster); the selection is the token right after ``__``.
    """
    if "__" not in basename:
        return None
    return basename.split("__", 1)[1].split("_", 1)[0]


def load_manager(data_root, sel_filter=None):
    """Create a DataManager over the (algorithm, selection) folders.

    When ``sel_filter`` is given (``FP``/``BZ``/``RTS``) only that selection
    method's folders are loaded, so each selection can be analysed
    independently and far more cheaply than loading all of them.
    """
    folders = [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
        and (sel_filter is None or _folder_selection(d) == sel_filter)
    ]
    if not folders:
        raise SystemExit(
            f"No data folders found in {data_root}"
            + (f" for selection method {sel_filter!r}" if sel_filter else ""))

    # Skip incomplete/crashed runs: a valid IOH ``Analyzer`` folder has a
    # finalized ``IOHprofiler_f*.json`` at its top level.  Jobs that were killed
    # mid-run (e.g. the prohibitive n=625 model-building EDAs) leave only a
    # ``data_f*/`` subdir with no json, which would otherwise abort the load.
    manager = DataManager()
    skipped = []
    added = 0
    for folder in folders:
        if not glob.glob(os.path.join(folder, "*.json")):
            skipped.append(os.path.basename(folder))
            continue
        try:
            manager.add_folder(folder)
            added += 1
        except Exception as exc:                      # malformed/partial json
            skipped.append(f"{os.path.basename(folder)} ({type(exc).__name__})")
    if added == 0:
        raise SystemExit(
            f"No valid IOH data folders in {data_root}"
            + (f" for selection {sel_filter!r}" if sel_filter else "")
            + f" ({len(skipped)} folders had no json).")
    if skipped:
        print(f"  note: skipped {len(skipped)} incomplete/invalid run folders "
              f"(no finalized json), e.g. {skipped[:3]}")
    return manager


def _base_alg(name):
    if not isinstance(name, str):
        return "NA"
    return name.split("__")[0]


def _sel_method(name):
    if not isinstance(name, str):
        return "NA"
    parts = name.split("__")
    return parts[1] if len(parts) > 1 else "NA"


def add_split_columns(df):
    """Add polars 'algorithm' and 'selection' columns from 'algorithm_name'."""
    return df.with_columns([
        pl.col("algorithm_name").map_elements(_base_alg, return_dtype=pl.Utf8)
          .alias("algorithm"),
        pl.col("algorithm_name").map_elements(_sel_method, return_dtype=pl.Utf8)
          .alias("selection"),
    ])


def subset_for_method(data, sel):
    """Rows of `data` for one selection method, with algorithm_name set to the
    base algorithm name (so plots/legends show 'UMDA' rather than 'UMDA__FP')."""
    sub = data.filter(pl.col("selection") == sel)
    if len(sub) == 0:
        return sub
    return sub.with_columns(pl.col("algorithm").alias("algorithm_name"))


def function_label(overview_pd, fid):
    name = overview_pd.loc[overview_pd["function_id"] == fid, "function_name"].iloc[0]
    return f"f{fid} {name}"


def _bold_table(index_labels, columns, value_of, is_best, caption, fmt="{:.2f}"):
    """Assemble a LaTeX tabular; value_of(row, col)->str cell body, is_best->bool."""
    lines = [
        caption,
        "\\setlength{\\tabcolsep}{10pt}",
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\hline",
        "Function & " + " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row_key, label in index_labels:
        cells = []
        for col in columns:
            body = value_of(row_key, col)
            if body is None:
                cells.append("--")
            elif is_best(row_key, col):
                cells.append(f"$\\mathbf{{{body}}}$")
            else:
                cells.append(f"${body}$")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-selection-method tables
# ---------------------------------------------------------------------------
def write_final_fitness_tables(overview_pd, out_dir, dims, sel, algorithms):
    df = overview_pd[overview_pd["selection"] == sel].copy()
    df = df.rename(columns={"best_y": "final_best"})
    df[["data_id", "algorithm", "selection", "function_id", "function_name",
        "dimension", "instance", "run_id", "evals", "final_best"]].to_csv(
        os.path.join(out_dir, "final_fitness.csv"), index=False)
    print(f"  [{sel}] final_fitness.csv ({len(df)} runs)")

    for dim in dims:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            continue
        grouped = sub.groupby(["function_id", "algorithm"])["final_best"]
        mean = grouped.mean().unstack()
        std = grouped.std().unstack()
        cols = [a for a in algorithms if a in mean.columns]
        mean, std = mean[cols], std[cols]

        best = {int(fid): mean.loc[fid].max() for fid in mean.index}

        def value_of(fid, alg):
            m, s = mean.loc[fid, alg], std.loc[fid, alg]
            if np.isnan(m):
                return None
            return f"{m:.2f} \\pm {0.0 if np.isnan(s) else s:.2f}"

        def is_best(fid, alg):
            m = mean.loc[fid, alg]
            return (not np.isnan(m)) and np.isclose(m, best[int(fid)])

        index_labels = [(fid, function_label(overview_pd, int(fid)))
                        for fid in mean.index]
        caption = ("% Final best fitness (mean +- std), PBO suite, "
                   f"n={dim}, selection method {sel}.  Best mean per function bold.")
        with open(os.path.join(out_dir, f"table_final_dim{dim}.tex"), "w") as fh:
            fh.write(_bold_table(index_labels, cols, value_of, is_best, caption))
        print(f"  [{sel}] table_final_dim{dim}.tex")


def write_statistical_tests(overview_pd, out_dir, dims, sel):
    """KW across algorithms per (function,dim) + Dunn post hoc (within method)."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
        print(f"  [{sel}] note: scikit_posthocs not installed; skipping Dunn tests")

    df = overview_pd[overview_pd["selection"] == sel]
    rows = []
    for (fid, dim), sub in df.groupby(["function_id", "dimension"]):
        if dim not in dims:
            continue
        groups = [g["best_y"].values for _, g in sub.groupby("algorithm")
                  if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, pval = scipy_stats.kruskal(*groups)
        except ValueError:
            stat, pval = np.nan, np.nan
        if np.isnan(pval):
            pval = 1.0
        rows.append({"function_id": fid, "dimension": dim,
                     "H_statistic": stat, "p_value": pval})
        if sp is not None and pval < 0.05:
            dunn = sp.posthoc_dunn(sub, val_col="best_y", group_col="algorithm",
                                   p_adjust="holm")
            dunn.to_csv(os.path.join(out_dir, f"dunn_f{int(fid)}_dim{int(dim)}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "kruskal_wallis.csv"),
                              index=False)
    print(f"  [{sel}] kruskal_wallis.csv ({len(rows)} tests)")


# ---------------------------------------------------------------------------
# AOCC (loaded once per (fid,dim), split per method + cross-method table)
# ---------------------------------------------------------------------------
def write_aocc_tables(manager, overview_pd, out_root, dims, sel_methods,
                      algorithms, write_summary=True):
    for dim in dims:
        eval_max = int(overview_pd[overview_pd["dimension"] == dim]["evals"].max())
        fids = sorted(overview_pd[overview_pd["dimension"] == dim]
                      ["function_id"].unique().tolist())
        parts = []
        for fid in fids:
            data = (manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                    .load(monotonic=True, include_meta_data=True))
            if len(data) == 0:
                continue
            lb, ub = float(data["raw_y"].min()), float(data["raw_y"].max())
            if ub <= lb:
                continue
            data = transform_fval(data, lb=lb, ub=ub, scale_log=False,
                                  maximization=True)
            aocc = get_aocc(data, eval_max=eval_max,
                            free_vars=["function_id", "function_name",
                                       "algorithm_name"],
                            return_as_pandas=True)
            parts.append(aocc)
        if not parts:
            continue
        aocc = pd.concat(parts, ignore_index=True)
        aocc["algorithm"] = aocc["algorithm_name"].map(_base_alg)
        aocc["selection"] = aocc["algorithm_name"].map(_sel_method)

        # ---- per selection-method AOCC tables (functions x algorithms) ----
        for sel in sel_methods:
            sub = aocc[aocc["selection"] == sel]
            if sub.empty:
                continue
            out_dir = os.path.join(out_root, sel)
            sub.to_csv(os.path.join(out_dir, f"aocc_dim{dim}.csv"), index=False)
            pivot = sub.pivot_table(index="function_id", columns="algorithm",
                                    values="AOCC")
            cols = [a for a in algorithms if a in pivot.columns]
            pivot = pivot[cols]
            best = {int(fid): pivot.loc[fid].max() for fid in pivot.index}
            index_labels = [(fid, function_label(overview_pd, int(fid)))
                            for fid in pivot.index]

            def value_of(fid, alg, pivot=pivot):
                v = pivot.loc[fid, alg]
                return None if np.isnan(v) else f"{v:.3f}"

            def is_best(fid, alg, pivot=pivot, best=best):
                v = pivot.loc[fid, alg]
                return (not np.isnan(v)) and np.isclose(v, best[int(fid)])

            caption = ("% AOCC (area over convergence curve, higher better), "
                       f"PBO n={dim}, selection method {sel}.  Best per function bold.")
            with open(os.path.join(out_dir, f"table_aocc_dim{dim}.tex"), "w") as fh:
                fh.write(_bold_table(index_labels, cols, value_of, is_best, caption))
            print(f"  [{sel}] aocc_dim{dim}.csv / table_aocc_dim{dim}.tex")

        # ---- cross-method AOCC table (algorithms x selection methods) ----
        if not write_summary:
            continue
        summary_dir = os.path.join(out_root, "summary")
        by_method = aocc.pivot_table(index="algorithm", columns="selection",
                                     values="AOCC", aggfunc="mean")
        method_cols = [s for s in sel_methods if s in by_method.columns]
        by_method = by_method[method_cols]
        by_method.to_csv(os.path.join(summary_dir,
                                      f"aocc_by_method_dim{dim}.csv"))
        alg_order = [a for a in algorithms if a in by_method.index]
        best = {a: by_method.loc[a].max() for a in alg_order}
        lines = [
            f"% Mean AOCC per algorithm and selection method, PBO n={dim}.",
            "% Averaged over all functions; best selection method per algorithm bold.",
            "\\setlength{\\tabcolsep}{10pt}",
            "\\begin{tabular}{l" + "r" * len(method_cols) + "}",
            "\\hline",
            "Algorithm & " + " & ".join(method_cols) + " \\\\",
            "\\hline",
        ]
        for a in alg_order:
            cells = []
            for s in method_cols:
                v = by_method.loc[a, s]
                if np.isnan(v):
                    cells.append("--")
                elif np.isclose(v, best[a]):
                    cells.append(f"$\\mathbf{{{v:.3f}}}$")
                else:
                    cells.append(f"${v:.3f}$")
            lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(os.path.join(summary_dir,
                               f"table_aocc_by_method_dim{dim}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  [summary] aocc_by_method_dim{dim}.csv / .tex")


# ---------------------------------------------------------------------------
# Figures (loaded once per (fid,dim), plotted per method)
# ---------------------------------------------------------------------------
def make_fixed_budget_figures(manager, overview_pd, out_root, dims, sel_methods):
    for dim in dims:
        fids = sorted(overview_pd[overview_pd["dimension"] == dim]
                      ["function_id"].unique().tolist())
        for fid in fids:
            data = (manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                    .load(monotonic=True, include_meta_data=True))
            if len(data) == 0:
                continue
            data = add_split_columns(data)
            for sel in sel_methods:
                sub = subset_for_method(data, sel)
                if len(sub) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(7.0, 5.0))
                plot_single_function_fixed_budget(
                    sub, maximization=True, measures=["mean"], ax=ax)
                ax.set_title("")
                ax.set_yscale("linear")
                ax.set_xlabel("Evaluations")
                ax.set_ylabel("Best fitness")
                handles, labels = ax.get_legend_handles_labels()
                keep = [(h, l.strip("(),'\" ")) for h, l in zip(handles, labels)
                        if l not in ("None", "variable", "mean")]
                if keep:
                    ax.legend([h for h, _ in keep], [l for _, l in keep],
                              ncol=2, frameon=False)
                save_figure(fig, os.path.join(out_root, sel),
                            f"fixed_budget_f{int(fid)}_dim{dim}")


def _ecdf_parts(manager, overview_pd, dim):
    """ECDF rows for every algorithm_name at one dimension, pooled per function
    with a consistent per-function normalization across all methods."""
    sub = overview_pd[overview_pd["dimension"] == dim]
    fids = sorted(sub["function_id"].unique().tolist())
    eval_max = int(sub["evals"].max())
    eval_values = get_sequence(1, eval_max, 50, scale_log=True, cast_to_int=True)
    parts = []
    for fid in fids:
        data = (manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                .load(monotonic=True, include_meta_data=True))
        if len(data) == 0:
            continue
        f_min, f_max = float(data["raw_y"].min()), float(data["raw_y"].max())
        if f_max <= f_min:
            continue
        parts.append(get_data_ecdf(data, maximization=True, scale_f_log=False,
                                   f_min=f_min, f_max=f_max,
                                   eval_values=eval_values,
                                   return_as_pandas=True))
    return parts


def make_ecdf_figures(manager, overview_pd, out_root, dims, sel_methods,
                      cross_method=True):
    for dim in dims:
        parts = _ecdf_parts(manager, overview_pd, dim)
        if not parts:
            continue
        ecdf = pd.concat(parts, ignore_index=True)
        # get_data_ecdf can emit rows with a null algorithm_name (aggregate/pad
        # rows, seen on dims with partial algorithm coverage such as n=625 where
        # some jobs were skipped).  They carry no algorithm identity, so drop
        # them before mapping instead of crashing.
        n_before = len(ecdf)
        ecdf = ecdf[ecdf["algorithm_name"].notna()].copy()
        if len(ecdf) < n_before:
            print(f"  note: dropped {n_before - len(ecdf)} ECDF rows with null "
                  f"algorithm_name (dim={dim})")
        if ecdf.empty:
            continue
        ecdf["algorithm"] = ecdf["algorithm_name"].map(_base_alg)
        ecdf["selection"] = ecdf["algorithm_name"].map(_sel_method)

        # ---- per-method ECDF (one line per algorithm) ----
        for sel in sel_methods:
            sub = ecdf[ecdf["selection"] == sel]
            if sub.empty:
                continue
            agg = (sub.groupby(["evaluations", "algorithm"])["eaf"]
                   .mean().reset_index())
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            for alg, grp in agg.groupby("algorithm"):
                ax.plot(grp["evaluations"], grp["eaf"], label=alg)
            ax.set_xscale("log")
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Proportion of (target, run) pairs")
            ax.set_ylim(0, 1.02)
            ax.legend(ncol=2, frameon=False)
            save_figure(fig, os.path.join(out_root, sel), f"ecdf_dim{dim}")

        if not cross_method:
            continue
        # ---- cross-method ECDF (one line per selection method) ----
        agg = (ecdf.groupby(["evaluations", "selection"])["eaf"]
               .mean().reset_index())
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for sel in sel_methods:
            grp = agg[agg["selection"] == sel]
            if grp.empty:
                continue
            ax.plot(grp["evaluations"], grp["eaf"], label=sel, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Proportion of (target, run) pairs")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
        save_figure(fig, os.path.join(out_root, "summary"),
                    f"ecdf_methods_dim{dim}")


def write_method_kruskal(overview_pd, out_root, dims, sel_methods):
    """KW across the three selection methods per (function, dim), pooling the
    final fitness of all algorithms within each method."""
    rows = []
    for (fid, dim), sub in overview_pd.groupby(["function_id", "dimension"]):
        if dim not in dims:
            continue
        groups = [g["best_y"].values
                  for s in sel_methods
                  for _, g in [(s, sub[sub["selection"] == s])] if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, pval = scipy_stats.kruskal(*groups)
        except ValueError:
            stat, pval = np.nan, np.nan
        if np.isnan(pval):
            pval = 1.0
        rows.append({"function_id": fid, "dimension": dim,
                     "H_statistic": stat, "p_value": pval})
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "summary",
                                           "kruskal_methods.csv"), index=False)
    print(f"  [summary] kruskal_methods.csv ({len(rows)} tests)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    sel_filter = sys.argv[3].upper() if len(sys.argv) > 3 else None
    if sel_filter is not None and sel_filter not in SELECTION_ORDER:
        raise SystemExit(f"selection must be one of {SELECTION_ORDER} (got "
                         f"{sel_filter!r}); omit it to analyse all together.")

    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Selection:   {sel_filter or 'all (per-method subdirs + summary)'}")

    manager = load_manager(data_root, sel_filter)
    overview = add_split_columns(manager.overview)
    overview_pd = overview.to_pandas()

    dims = sorted(overview_pd["dimension"].unique().tolist())
    sel_methods = [s for s in SELECTION_ORDER
                   if s in overview_pd["selection"].unique().tolist()]
    algorithms = sorted(overview_pd["algorithm"].unique().tolist())
    n_funcs = overview_pd["function_id"].nunique()
    print(f"Loaded: {len(algorithms)} algorithms x {len(sel_methods)} selection "
          f"methods, {n_funcs} functions, dims={dims}, {len(overview_pd)} runs")
    print(f"  algorithms: {algorithms}")
    print(f"  selection : {sel_methods}\n")

    cross_method = len(sel_methods) > 1

    # Output directory tree.
    for sel in sel_methods:
        os.makedirs(os.path.join(output_dir, sel), exist_ok=True)
    if cross_method:
        os.makedirs(os.path.join(output_dir, "summary"), exist_ok=True)

    print("Per-selection-method tables:")
    for sel in sel_methods:
        write_final_fitness_tables(overview_pd, os.path.join(output_dir, sel),
                                   dims, sel, algorithms)
        write_statistical_tests(overview_pd, os.path.join(output_dir, sel),
                                dims, sel)

    print("AOCC tables" + (" (per method + cross-method summary):" if cross_method
                           else " (per method):"))
    write_aocc_tables(manager, overview_pd, output_dir, dims, sel_methods,
                      algorithms, write_summary=cross_method)

    if cross_method:
        print("Cross-method statistics:")
        write_method_kruskal(overview_pd, output_dir, dims, sel_methods)

    print("Figures:")
    make_fixed_budget_figures(manager, overview_pd, output_dir, dims, sel_methods)
    make_ecdf_figures(manager, overview_pd, output_dir, dims, sel_methods,
                      cross_method=cross_method)

    print(f"\nDone.  Output in {output_dir}")


if __name__ == "__main__":
    main()
