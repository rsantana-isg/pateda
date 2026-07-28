"""
Assemble a single LaTeX document with ALL the comparison *tables* (no figures)
produced by ``analyze_weighted_pbo_results.py``, organised into three sections
(one per selection method: FP, BZ, RTS), plus a computed **overall summary of
the best algorithms across all functions and dimensions**.

For every selection method and every dimension the per-function tables
(final best fitness and AOCC, functions x algorithms, best in bold) are the
existing ``table_final_dim*.tex`` / ``table_aocc_dim*.tex`` fragments; they are
``\\input`` inside rotated, width-fitted tables so the 22-algorithm columns fit.

The summary section is computed from the raw CSVs:
  * mean AOCC over all (function, dimension) per algorithm, per selection and
    overall (AOCC is normalised per function, so it is comparable across
    functions; higher is better);
  * average rank of the mean final fitness within each (function, dimension)
    (scale-free; lower is better);
  * number of (function, dimension) "wins" (tied best final fitness).

Usage:
    python3 scripts/make_pbo_latex_report.py [analysis_dir] [output_tex]

    analysis_dir  directory with FP/ BZ/ RTS/ subdirs
                  (default: results/pbo_weighted_analysis)
    output_tex    output .tex path
                  (default: <analysis_dir>/pbo_report.tex)

Then compile from inside analysis_dir (so the \\input paths resolve):
    cd <analysis_dir> && pdflatex pbo_report.tex && pdflatex pbo_report.tex
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd

SEL_ORDER = ["FP", "BZ", "RTS"]
SEL_TITLE = {"FP": "FP", "BZ": "BZ", "RTS": "RTS"}
SEL_DESC = {
    "FP":  r"truncation ($\tau=0.5$) $+$ fitness-proportional weighting $+$ "
           r"elitist replacement",
    "BZ":  r"truncation ($\tau=0.5$) $+$ Boltzmann weighting ($\beta=1$) $+$ "
           r"elitist replacement",
    "RTS": r"truncation ($\tau=0.5$) $+$ Boltzmann weighting ($\beta=1$) $+$ "
           r"restricted-tournament replacement",
}


def _dims_in(subdir):
    dims = set()
    for f in glob.glob(os.path.join(subdir, "table_final_dim*.tex")):
        m = re.search(r"dim(\d+)\.tex$", f)
        if m:
            dims.add(int(m.group(1)))
    return sorted(dims)


def load_final(analysis_dir, sels):
    frames = []
    for s in sels:
        p = os.path.join(analysis_dir, s, "final_fitness.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_aocc(analysis_dir, sels):
    frames = []
    for s in sels:
        for f in glob.glob(os.path.join(analysis_dir, s, "aocc_dim*.csv")):
            d = pd.read_csv(f)
            d["dimension"] = int(re.search(r"dim(\d+)", f).group(1))
            if "selection" not in d.columns:
                d["selection"] = s
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Summary computations
# ---------------------------------------------------------------------------
def compute_summary(final_df, aocc_df, sels):
    """Return (aocc_tbl, rank_tbl, wins_tbl, coverage) DataFrames indexed by
    algorithm with columns = sels + ['Overall']."""
    # --- mean AOCC ---
    aocc_sel = (aocc_df.groupby(["algorithm", "selection"])["AOCC"].mean()
                .unstack().reindex(columns=sels))
    aocc_sel["Overall"] = aocc_df.groupby("algorithm")["AOCC"].mean()

    # --- average rank by mean final fitness within each (function, dim, sel) ---
    g = (final_df.groupby(["function_id", "dimension", "selection", "algorithm"])
         ["final_best"].mean().reset_index())
    g["rank"] = (g.groupby(["function_id", "dimension", "selection"])["final_best"]
                 .rank(ascending=False, method="min"))
    rank_sel = (g.groupby(["algorithm", "selection"])["rank"].mean()
                .unstack().reindex(columns=sels))
    rank_sel["Overall"] = g.groupby("algorithm")["rank"].mean()

    # --- wins (tied best final fitness) ---
    g["win"] = (g["rank"] == 1.0).astype(int)
    wins_sel = (g.groupby(["algorithm", "selection"])["win"].sum()
                .unstack().reindex(columns=sels))
    wins_sel["Overall"] = g.groupby("algorithm")["win"].sum()

    # --- coverage: #(function,dim) cells the algorithm participated in ---
    coverage = g.groupby("algorithm").size().rename("cells")
    return aocc_sel, rank_sel, wins_sel, coverage


def _tex(s):
    """Escape LaTeX-special characters in a label (algorithm names may contain
    underscores, e.g. ``A1_dt``, ``BOA_mp6``, ``EBNA_BIC``)."""
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _fmt_table(df, cols, fmt, best="max", coverage=None, intcols=False):
    """LaTeX rows for a summary table (best per column in bold)."""
    best_val = {c: (df[c].max() if best == "max" else df[c].min()) for c in cols}
    lines = []
    for alg in df.index:
        cells = []
        for c in cols:
            v = df.loc[alg, c]
            if pd.isna(v):
                cells.append("--")
                continue
            body = (f"{int(round(v))}" if intcols else fmt.format(v))
            if np.isclose(v, best_val[c]):
                body = f"\\mathbf{{{body}}}"
            cells.append(f"${body}$")
        cov = f" & {int(coverage.get(alg, 0))}" if coverage is not None else ""
        lines.append(f"{_tex(alg)} & " + " & ".join(cells) + cov + " \\\\")
    return "\n".join(lines)


def summary_section(aocc_sel, rank_sel, wins_sel, coverage, sels):
    order = aocc_sel["Overall"].sort_values(ascending=False).index.tolist()
    aocc_sel, rank_sel, wins_sel = (aocc_sel.loc[order], rank_sel.loc[order],
                                    wins_sel.loc[order])
    cols = sels + ["Overall"]
    colspec = "l" + "r" * len(cols)
    header = "Algorithm & " + " & ".join(cols)

    top_aocc = ", ".join(_tex(a) for a in order[:3])
    top_rank = rank_sel["Overall"].sort_values().index.tolist()[:3]
    top_rank = ", ".join(_tex(a) for a in top_rank)

    out = []
    out.append(r"\section{Overall summary: best algorithms across all functions "
               r"and dimensions}")
    out.append(
        r"The tables below aggregate every (function, dimension) cell.  "
        r"\emph{Mean AOCC} is comparable across functions because AOCC is "
        r"normalised to $[0,1]$ per function (higher is better).  "
        r"\emph{Average rank} ranks the algorithms by mean final fitness within "
        r"each (function, dimension) and averages those ranks (lower is better); "
        r"it is scale-free.  \emph{Wins} counts the (function, dimension) cells "
        r"where the algorithm attains the (tied) best mean final fitness.  "
        r"The \emph{cells} column is how many (function, dimension) cases the "
        r"algorithm participated in (less than the maximum only for the "
        r"expensive learners whose $n=625$ jobs did not finish).")
    out.append("")
    out.append(rf"\noindent\textbf{{Best overall by mean AOCC:}} {top_aocc}.\\")
    out.append(rf"\textbf{{Best overall by average rank:}} {top_rank}.")
    out.append("")

    # Mean AOCC table
    out.append(r"\begin{table}[h!]\centering")
    out.append(r"\caption{Mean AOCC per algorithm (higher is better); best per "
               r"column in bold.  Rows sorted by the Overall column.}")
    out.append(r"\setlength{\tabcolsep}{8pt}")
    out.append(r"\begin{tabular}{" + colspec + " r}")
    out.append(r"\hline")
    out.append(header + r" & cells \\")
    out.append(r"\hline")
    out.append(_fmt_table(aocc_sel, cols, "{:.3f}", best="max", coverage=coverage))
    out.append(r"\hline\end{tabular}\end{table}")
    out.append("")

    # Average rank table
    out.append(r"\begin{table}[h!]\centering")
    out.append(r"\caption{Average rank of the mean final fitness (lower is "
               r"better); best per column in bold.}")
    out.append(r"\setlength{\tabcolsep}{8pt}")
    out.append(r"\begin{tabular}{" + colspec + "}")
    out.append(r"\hline")
    out.append(header + r" \\")
    out.append(r"\hline")
    out.append(_fmt_table(rank_sel, cols, "{:.2f}", best="min"))
    out.append(r"\hline\end{tabular}\end{table}")
    out.append("")

    # Wins table
    out.append(r"\begin{table}[h!]\centering")
    out.append(r"\caption{Number of (function, dimension) wins (tied best mean "
               r"final fitness); best per column in bold.}")
    out.append(r"\setlength{\tabcolsep}{8pt}")
    out.append(r"\begin{tabular}{" + colspec + "}")
    out.append(r"\hline")
    out.append(header + r" \\")
    out.append(r"\hline")
    out.append(_fmt_table(wins_sel, cols, "{:.0f}", best="max", intcols=True))
    out.append(r"\hline\end{tabular}\end{table}")
    return "\n".join(out)


def landscape_table(fragment_relpath, caption):
    """A rotated, width-fitted table wrapping an existing tabular fragment."""
    return "\n".join([
        r"\begin{landscape}",
        r"\begin{table}[p]\centering",
        rf"\caption{{{caption}}}",
        rf"\resizebox{{\linewidth}}{{!}}{{\input{{{fragment_relpath}}}}}",
        r"\end{table}",
        r"\end{landscape}",
        "",
    ])


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{pdflscape}
\usepackage{longtable}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\title{Weighted-probability EDAs on the PBO suite:\\ per-function comparison tables}
\author{Generated by scripts/make\_pbo\_latex\_report.py}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def main():
    analysis_dir = (sys.argv[1] if len(sys.argv) > 1
                    else "results/pbo_weighted_analysis")
    out_tex = (sys.argv[2] if len(sys.argv) > 2
               else os.path.join(analysis_dir, "pbo_report.tex"))

    sels = [s for s in SEL_ORDER if os.path.isdir(os.path.join(analysis_dir, s))]
    if not sels:
        raise SystemExit(f"No FP/BZ/RTS subdirs found in {analysis_dir}")
    print(f"Analysis dir: {analysis_dir}")
    print(f"Selections:   {sels}")

    parts = [PREAMBLE]
    for sel in sels:
        subdir = os.path.join(analysis_dir, sel)
        dims = _dims_in(subdir)
        parts.append(rf"\section{{Selection method {SEL_TITLE[sel]}: {SEL_DESC[sel]}}}")
        parts.append(rf"Per-function comparison of all algorithms under the "
                     rf"{SEL_TITLE[sel]} selection scheme, for each dimension "
                     rf"$n \in \{{{', '.join(str(d) for d in dims)}\}}$.  In every "
                     rf"table rows are PBO functions, columns are algorithms, and "
                     rf"the best mean per function is in bold.")
        for dim in dims:
            ff = os.path.join(sel, f"table_final_dim{dim}.tex")
            ao = os.path.join(sel, f"table_aocc_dim{dim}.tex")
            if os.path.exists(os.path.join(analysis_dir, ff)):
                parts.append(landscape_table(
                    ff, rf"Final best fitness (mean $\pm$ std), "
                        rf"{SEL_TITLE[sel]}, $n={dim}$."))
            if os.path.exists(os.path.join(analysis_dir, ao)):
                parts.append(landscape_table(
                    ao, rf"AOCC (area over the convergence curve, higher is "
                        rf"better), {SEL_TITLE[sel]}, $n={dim}$."))
        parts.append(r"\clearpage")

    # Summary from the raw CSVs.
    final_df = load_final(analysis_dir, sels)
    aocc_df = load_aocc(analysis_dir, sels)
    if not final_df.empty and not aocc_df.empty:
        aocc_sel, rank_sel, wins_sel, coverage = compute_summary(
            final_df, aocc_df, sels)
        parts.append(summary_section(aocc_sel, rank_sel, wins_sel, coverage, sels))
    else:
        parts.append(r"\section{Overall summary}"
                     "\nSummary CSVs not found; skipped.")

    parts.append(r"\end{document}")

    with open(out_tex, "w") as fh:
        fh.write("\n".join(parts) + "\n")
    n_tables = sum(1 for p in parts if r"\begin{landscape}" in p)
    print(f"Wrote {out_tex}  ({n_tables} per-function tables + summary)")
    print(f"Compile with:\n  cd {analysis_dir} && "
          f"pdflatex {os.path.basename(out_tex)} && pdflatex {os.path.basename(out_tex)}")


if __name__ == "__main__":
    main()
