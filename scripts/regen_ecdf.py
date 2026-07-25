"""
Regenerate ONLY the ECDF figures of the weighted-PBO analysis.

Use this after a full ``analyze_weighted_pbo_results.py`` run whose ECDF step
failed (or was fixed) but whose fixed-budget plots and LaTeX tables are already
correct -- it reuses the existing output subdirectories and does a single
data-load pass instead of re-running the whole (3-pass) analysis.

Usage (all optional, positional):
    python3 scripts/regen_ecdf.py [data_root] [output_dir] [selection]

    data_root   IOH data folders   (default: pbo_weighted_data_cluster)
    output_dir  analysis output     (default: results/pbo_weighted_analysis)
    selection   FP, BZ or RTS -- only that method (matches the 3rd argument of
                analyze_weighted_pbo_results.py); omit for all + cross-method.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_weighted_pbo_results as A


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "pbo_weighted_data_cluster"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/pbo_weighted_analysis"
    sel_filter = sys.argv[3].upper() if len(sys.argv) > 3 else None
    if sel_filter is not None and sel_filter not in A.SELECTION_ORDER:
        raise SystemExit(f"selection must be one of {A.SELECTION_ORDER} "
                         f"(got {sel_filter!r}); omit it for all methods.")

    print(f"Data root:  {data_root}")
    print(f"Output dir: {out}")
    print(f"Selection:  {sel_filter or 'all'}")

    manager = A.load_manager(data_root, sel_filter)
    overview_pd = A.add_split_columns(manager.overview).to_pandas()
    dims = sorted(overview_pd["dimension"].unique().tolist())
    sels = [s for s in A.SELECTION_ORDER
            if s in overview_pd["selection"].unique().tolist()]
    cross_method = len(sels) > 1
    print(f"dims={dims}  selections={sels}")

    for s in sels + (["summary"] if cross_method else []):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    A.make_ecdf_figures(manager, overview_pd, out, dims, sels,
                        cross_method=cross_method)
    print(f"\nDone.  ECDF figures regenerated in {out}")


if __name__ == "__main__":
    main()
