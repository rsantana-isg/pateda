"""
Analyze the SELECTED weighted-probability PBO results produced by
``compare_selected_edas_pbo.py`` / ``run_selected_pbo_eda.py``.

The IOH data folders are named ``{ALG}__{SEL}`` (``SEL`` in ``FP``, ``BZ``,
``RTS``), so the ``algorithm_name`` stored by IOH encodes both the algorithm and
the selection method.  Because every algorithm is run under three selection
schemes, the analysis is produced **per selection method**: one subdirectory
``FP/``, ``BZ/``, ``RTS/`` under ``results/pbo_analysis/``, each comparing the
algorithms *within* that scheme (exactly the outputs of
``analyze_pbo_results.py``, applied once per method — no cross-method summary).

For each selection method the following are written to ``<output_dir>/<SEL>/``:

Figures (``.pdf`` and ``.eps``, no titles, large paper fonts):
  - ``fixed_budget_f{fid}_dim{dim}``  mean best-so-far fitness vs evaluations,
    one line per algorithm (fixed-budget convergence plot).
  - ``ecdf_dim{dim}``                 ECDF of target achievement, aggregated over
    all functions of one dimension.

Tables:
  - ``final_fitness.csv``             per-run final best fitness (raw data).
  - ``table_final_dim{dim}.tex``      mean +- std of the final best fitness,
                                      functions x algorithms, all tied bests
                                      bold, ``\\tabcolsep{10pt}``.
  - ``aocc_dim{dim}.csv`` / ``table_aocc_dim{dim}.tex``  Area Over the
                                      Convergence Curve (anytime performance in
                                      [0, 1], higher is better).
  - ``kruskal_wallis.csv``            Kruskal-Wallis across algorithms per
                                      (function, dimension).
  - ``dunn_f{fid}_dim{dim}.csv``      post hoc Dunn tests (only if
                                      ``scikit_posthocs`` is installed).

Usage (all arguments optional, positional):
    python scripts/analyze_selected_pbo_results.py [data_root] [output_dir] [selection]

    data_root   folder with one IOH data folder per (algorithm, selection method)
                (default: results/pbo_selected_data)
    output_dir  where the per-method subdirectories are written
                (default: results/pbo_analysis)
    selection   optional: FP, BZ or RTS -- analyse ONLY that method.  Omit to
                analyse all present methods (one subdirectory each).
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
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_selected_data")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_analysis")
)

SELECTION_ORDER = ["FP", "BZ", "RTS"]

# Paper-ready fonts (figures carry no titles; captions live in the LaTeX doc).
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10,
})


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _folder_selection(basename):
    """Selection method encoded in a data-folder name ``{ALG}__{SEL}...``."""
    if "__" not in basename:
        return None
    return basename.split("__", 1)[1].split("_", 1)[0]


def _base_alg(name):
    """Base algorithm from an IOH ``algorithm_name`` ``{ALG}__{SEL}``."""
    if not isinstance(name, str):
        return "NA"
    return name.split("__")[0]


def load_manager(data_root, sel_filter):
    """DataManager over one selection method's folders.

    Skips incomplete/crashed runs: a valid IOH ``Analyzer`` folder has a
    finalized ``IOHprofiler_f*.json`` at its top level.  Jobs killed mid-run
    (e.g. the prohibitive n=625 model-building EDAs) leave only a ``data_f*/``
    subdir with no json, which would otherwise abort the load.
    """
    folders = [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
        and _folder_selection(d) == sel_filter
    ]
    manager = DataManager()
    skipped, added = [], 0
    for folder in folders:
        if not glob.glob(os.path.join(folder, "*.json")):
            skipped.append(os.path.basename(folder))
            continue
        try:
            manager.add_folder(folder)
            added += 1
        except Exception as exc:                       # malformed/partial json
            skipped.append(f"{os.path.basename(folder)} ({type(exc).__name__})")
    if added == 0:
        return None
    if skipped:
        print(f"  [{sel_filter}] note: skipped {len(skipped)} incomplete run "
              f"folders (no finalized json), e.g. {skipped[:3]}")
    return manager


def available_selections(data_root):
    """Selection methods that have at least one data folder."""
    sels = set()
    for d in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, d)):
            s = _folder_selection(d)
            if s in SELECTION_ORDER:
                sels.add(s)
    return [s for s in SELECTION_ORDER if s in sels]


def _to_base(data):
    """Replace the ``algorithm_name`` column of loaded data with the base
    algorithm (strip ``__{SEL}``) so legends / columns show clean names."""
    return data.with_columns(
        pl.col("algorithm_name")
        .map_elements(_base_alg, return_dtype=pl.Utf8)
        .alias("algorithm_name")
    )


def function_label(overview_pd, fid):
    name = overview_pd.loc[overview_pd["function_id"] == fid,
                           "function_name"].iloc[0]
    return f"f{fid} {name}"


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  figure: {os.path.join(os.path.basename(output_dir), stem)}.pdf / .eps")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_final_fitness_tables(overview_pd, out_dir, dims, algorithms):
    df = overview_pd.rename(columns={"best_y": "final_best"}).copy()
    df[["data_id", "algorithm", "function_id", "function_name", "dimension",
        "instance", "run_id", "evals", "final_best"]].to_csv(
        os.path.join(out_dir, "final_fitness.csv"), index=False)
    print(f"  table:  final_fitness.csv ({len(df)} runs)")

    for dim in dims:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            continue
        grouped = sub.groupby(["function_id", "algorithm"])["final_best"]
        mean = grouped.mean().unstack()
        std = grouped.std().unstack()
        cols = [a for a in algorithms if a in mean.columns]
        mean, std = mean[cols], std[cols]

        lines = [
            f"% Final best fitness (mean +- std over runs), PBO suite, n={dim}."
            "  Best mean per function in bold.",
            "\\setlength{\\tabcolsep}{10pt}",
            "\\begin{tabular}{l" + "r" * len(cols) + "}",
            "\\hline",
            "Function & " + " & ".join(cols) + " \\\\",
            "\\hline",
        ]
        for fid in mean.index:
            best_mean = mean.loc[fid].max()
            cells = []
            for alg in cols:
                m, s = mean.loc[fid, alg], std.loc[fid, alg]
                if np.isnan(m):
                    cells.append("--")
                    continue
                body = f"{m:.2f} \\pm {0.0 if np.isnan(s) else s:.2f}"
                if np.isclose(m, best_mean):     # bold all tied bests
                    body = f"\\mathbf{{{body}}}"
                cells.append(f"${body}$")
            lines.append(f"{function_label(overview_pd, int(fid))} & "
                         + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(os.path.join(out_dir, f"table_final_dim{dim}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  table:  table_final_dim{dim}.tex")


def write_statistical_tests(overview_pd, out_dir, dims):
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
        print("  note:   scikit_posthocs not installed; skipping Dunn tests")

    rows = []
    for (fid, dim), sub in overview_pd.groupby(["function_id", "dimension"]):
        if dim not in dims:
            continue
        groups = [g["best_y"].values
                  for _, g in sub.groupby("algorithm") if len(g) > 0]
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
            dunn.to_csv(os.path.join(out_dir,
                                     f"dunn_f{int(fid)}_dim{int(dim)}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "kruskal_wallis.csv"),
                              index=False)
    print(f"  table:  kruskal_wallis.csv ({len(rows)} tests)")


def write_aocc_tables(manager, overview_pd, out_dir, dims, algorithms):
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
            data = _to_base(data)
            lb, ub = float(data["raw_y"].min()), float(data["raw_y"].max())
            if ub <= lb:
                continue
            data = transform_fval(data, lb=lb, ub=ub, scale_log=False,
                                  maximization=True)
            parts.append(get_aocc(data, eval_max=eval_max,
                                  free_vars=["function_id", "function_name",
                                             "algorithm_name"],
                                  return_as_pandas=True))
        if not parts:
            continue
        aocc = pd.concat(parts, ignore_index=True)
        aocc = aocc.rename(columns={"algorithm_name": "algorithm"})
        aocc.to_csv(os.path.join(out_dir, f"aocc_dim{dim}.csv"), index=False)

        pivot = aocc.pivot_table(index="function_id", columns="algorithm",
                                 values="AOCC")
        cols = [a for a in algorithms if a in pivot.columns]
        pivot = pivot[cols]
        lines = [
            f"% AOCC (area over the convergence curve), PBO suite, n={dim}.",
            "% Anytime performance in [0,1]; higher is better; best in bold.",
            "\\setlength{\\tabcolsep}{10pt}",
            "\\begin{tabular}{l" + "r" * len(cols) + "}",
            "\\hline",
            "Function & " + " & ".join(cols) + " \\\\",
            "\\hline",
        ]
        for fid in pivot.index:
            best_aocc = pivot.loc[fid].max()
            cells = []
            for alg in cols:
                v = pivot.loc[fid, alg]
                if np.isnan(v):
                    cells.append("--")
                elif np.isclose(v, best_aocc):
                    cells.append(f"$\\mathbf{{{v:.3f}}}$")
                else:
                    cells.append(f"${v:.3f}$")
            lines.append(f"{function_label(overview_pd, int(fid))} & "
                         + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(os.path.join(out_dir, f"table_aocc_dim{dim}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  table:  aocc_dim{dim}.csv / table_aocc_dim{dim}.tex")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_fixed_budget_figures(manager, overview_pd, out_dir, dims):
    for dim in dims:
        fids = sorted(overview_pd[overview_pd["dimension"] == dim]
                      ["function_id"].unique().tolist())
        for fid in fids:
            data = (manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                    .load(monotonic=True, include_meta_data=True))
            if len(data) == 0:
                continue
            data = _to_base(data)
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            plot_single_function_fixed_budget(
                data, maximization=True, measures=["mean"], ax=ax)
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
            save_figure(fig, out_dir, f"fixed_budget_f{int(fid)}_dim{dim}")


def make_ecdf_figures(manager, overview_pd, out_dir, dims):
    for dim in dims:
        sub = overview_pd[overview_pd["dimension"] == dim]
        if sub.empty:
            continue
        fids = sorted(sub["function_id"].unique().tolist())
        eval_max = int(sub["evals"].max())
        eval_values = get_sequence(1, eval_max, 50, scale_log=True,
                                   cast_to_int=True)
        parts = []
        for fid in fids:
            data = (manager.select(function_ids=[int(fid)], dimensions=[int(dim)])
                    .load(monotonic=True, include_meta_data=True))
            if len(data) == 0:
                continue
            data = _to_base(data)
            f_min, f_max = float(data["raw_y"].min()), float(data["raw_y"].max())
            if f_max <= f_min:
                continue
            parts.append(get_data_ecdf(data, maximization=True, scale_f_log=False,
                                       f_min=f_min, f_max=f_max,
                                       eval_values=eval_values,
                                       return_as_pandas=True))
        if not parts:
            continue
        ecdf = pd.concat(parts, ignore_index=True)
        ecdf = ecdf[ecdf["algorithm_name"].notna()]
        if ecdf.empty:
            continue
        agg = (ecdf.groupby(["evaluations", "algorithm_name"])["eaf"]
               .mean().reset_index())
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for alg, grp in agg.groupby("algorithm_name"):
            ax.plot(grp["evaluations"], grp["eaf"], label=alg)
        ax.set_xscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Proportion of (target, run) pairs")
        ax.set_ylim(0, 1.02)
        ax.legend(ncol=2, frameon=False)
        save_figure(fig, out_dir, f"ecdf_dim{dim}")


# ---------------------------------------------------------------------------
# Per-selection-method driver
# ---------------------------------------------------------------------------
def analyze_selection(data_root, output_dir, sel):
    manager = load_manager(data_root, sel)
    if manager is None:
        print(f"[{sel}] no valid data folders; skipping.")
        return
    overview = manager.overview.with_columns(
        pl.col("algorithm_name").map_elements(_base_alg, return_dtype=pl.Utf8)
        .alias("algorithm"))
    overview_pd = overview.to_pandas()

    dims = sorted(overview_pd["dimension"].unique().tolist())
    algorithms = sorted(overview_pd["algorithm"].unique().tolist())
    n_funcs = overview_pd["function_id"].nunique()
    out_dir = os.path.join(output_dir, sel)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{sel}] {len(algorithms)} algorithms, {n_funcs} functions, "
          f"dims={dims}, {len(overview_pd)} runs -> {out_dir}")

    write_final_fitness_tables(overview_pd, out_dir, dims, algorithms)
    write_aocc_tables(manager, overview_pd, out_dir, dims, algorithms)
    write_statistical_tests(overview_pd, out_dir, dims)
    make_fixed_budget_figures(manager, overview_pd, out_dir, dims)
    make_ecdf_figures(manager, overview_pd, out_dir, dims)


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    sel_filter = sys.argv[3].upper() if len(sys.argv) > 3 else None
    if sel_filter is not None and sel_filter not in SELECTION_ORDER:
        raise SystemExit(f"selection must be one of {SELECTION_ORDER} "
                         f"(got {sel_filter!r}); omit it to analyse all.")

    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")

    sels = [sel_filter] if sel_filter else available_selections(data_root)
    if not sels:
        raise SystemExit(f"No {SELECTION_ORDER} data folders found in {data_root}")
    print(f"Selections:  {sels}\n")

    os.makedirs(output_dir, exist_ok=True)
    for sel in sels:
        analyze_selection(data_root, output_dir, sel)

    print(f"\nDone.  Output in {output_dir}")


if __name__ == "__main__":
    main()
