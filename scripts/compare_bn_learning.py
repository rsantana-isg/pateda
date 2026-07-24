"""
compare_bn_learning.py — Compare BN structure learning methods on benchmark datasets.

Usage (positional args, no flags):
    python3 scripts/compare_bn_learning.py [data_dir [n_samples [seed]]]

Defaults:
    data_dir  = data/bn_benchmarks
    n_samples = 1000
    seed      = 0

For each of the 10 benchmark BNs × each learning method:
  - Load true adjacency from <data_dir>/<name>_meta.json
  - Load data from <data_dir>/<name>_data_N*.csv  (first n_samples rows)
  - Fit BN structure; record wall-clock time
  - Compute directed and skeleton metrics + SHD

Prints a per-BN table and an aggregated summary.
Saves all results to <data_dir>/comparison_results.csv.

Adding a new method: add one dict to LEARNING_METHODS only.

Methods implemented:
  Score-based   : K2, BIC-HC, AIC-HC, HC-Stable, Tabu
  Local-structure: DT (decision-tree MDL), DG (decision-graph Bayesian)
  Constraint-based: GS, RCD, RPCD, PC, Stable-PC
  Polytree        : ChowLiu, RebanePearl, LPA-PADA, LPA-marg, Sheaf-CP
"""

import sys
import os
import time
import json
import csv
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bayes_nets import BayesianNetwork  # noqa: E402

# ---------------------------------------------------------------------------
# Learning methods registry
# Adding a new method: append one dict here — nothing else to change.
# ---------------------------------------------------------------------------
LEARNING_METHODS = [
    # ── Score-based (tabular CPDs) ─────────────────────────────────────
    dict(name="K2",         method="k2",         kwargs={}),
    dict(name="BIC-HC",     method="bic",         kwargs={}),
    dict(name="AIC-HC",     method="aic",         kwargs={}),
    dict(name="HC-Stable",  method="stable_hc",   kwargs={}),
    dict(name="Tabu",       method="tabu",        kwargs={}),
    # ── Local-structure CPD scoring ────────────────────────────────────
    dict(name="DT-MDL",     method="dt",          kwargs={}),
    dict(name="DG-Bayes",   method="dg",          kwargs={}),
    # ── Constraint-based ───────────────────────────────────────────────
    dict(name="GS",         method="gs",          kwargs={}),
    dict(name="RCD",        method="rcd",         kwargs={}),
    dict(name="RPCD",       method="rpcd",        kwargs={}),
    dict(name="PC",         method="pc",          kwargs={}),
    dict(name="Stable-PC",  method="stable_pc",   kwargs={}),
    # ── Polytree / singly connected ────────────────────────────────────
    dict(name="ChowLiu",    method="chow_liu",     kwargs={}),
    dict(name="RebanePearl",method="rebane_pearl", kwargs={}),
    dict(name="LPA-PADA",   method="lpa",          kwargs={}),
    dict(name="LPA-marg",   method="lpa_marginal", kwargs={}),
    dict(name="Sheaf-CP",   method="causal_polytree", kwargs={}),
    # ── Markov-blanket / decomposition-based ───────────────────────────
    dict(name="DMBBN",      method="dmbbn",       kwargs={}),
    dict(name="iter-DSLA",  method="iterdsla",    kwargs={}),
    # ── Differentiable (binary benchmarks only; errors are caught) ──────
    dict(name="BINOTEARS",  method="binotears",   kwargs={}),
    # NOTE: 'levelwise'/'exact' (LevelWiseDPLearner) is optimal but
    # exponential — enable it only on small benchmarks (n_vars <= ~18).
    # 'sartre' (SARTREPruner) needs a good topological order; chain it
    # after an order-estimating learner rather than running it stand-alone.
]

# The 10 benchmarks (same list as generate_bn_datasets.py)
BENCHMARK_NAMES = [
    "asia", "moodstate", "electrolysis", "accidents", "corrosion",
    "flood", "BOPfailure3", "yangtze", "BOPfailure2", "pneumonia",
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(true_adj: np.ndarray, learned_adj: np.ndarray) -> Dict[str, float]:
    """
    Compute structure learning metrics.

    Parameters
    ----------
    true_adj    : (n, n) int adjacency matrix of the ground-truth DAG.
    learned_adj : (n, n) int adjacency matrix of the learned DAG.

    Returns
    -------
    dict with keys: directed_precision, directed_recall, directed_f1,
                    skeleton_precision, skeleton_recall, skeleton_f1, shd
    """
    # --- Directed metrics ---
    true_edges = set(zip(*np.where(true_adj > 0))) if true_adj.sum() > 0 else set()
    learned_edges = set(zip(*np.where(learned_adj > 0))) if learned_adj.sum() > 0 else set()

    tp_dir = len(true_edges & learned_edges)
    fp_dir = len(learned_edges - true_edges)
    fn_dir = len(true_edges - learned_edges)

    if learned_edges:
        dir_prec = tp_dir / len(learned_edges)
    else:
        dir_prec = 1.0  # vacuously true: no false positives

    if true_edges:
        dir_rec = tp_dir / len(true_edges)
    else:
        dir_rec = 1.0 if not learned_edges else 0.0

    if dir_prec + dir_rec > 0:
        dir_f1 = 2 * dir_prec * dir_rec / (dir_prec + dir_rec)
    else:
        dir_f1 = 0.0

    # --- Skeleton metrics (undirected) ---
    def to_skeleton(adj: np.ndarray) -> set:
        rows, cols = np.where(adj > 0)
        return {(min(r, c), max(r, c)) for r, c in zip(rows, cols)}

    true_skel = to_skeleton(true_adj)
    learned_skel = to_skeleton(learned_adj)

    tp_skel = len(true_skel & learned_skel)
    fp_skel = len(learned_skel - true_skel)
    fn_skel = len(true_skel - learned_skel)

    if learned_skel:
        skel_prec = tp_skel / len(learned_skel)
    else:
        skel_prec = 1.0

    if true_skel:
        skel_rec = tp_skel / len(true_skel)
    else:
        skel_rec = 1.0 if not learned_skel else 0.0

    if skel_prec + skel_rec > 0:
        skel_f1 = 2 * skel_prec * skel_rec / (skel_prec + skel_rec)
    else:
        skel_f1 = 0.0

    # --- Structural Hamming Distance (SHD) ---
    # SHD = FP_directed + FN_directed + reversals
    # Reversals: edge in both but wrong direction
    reversals = 0
    for (r, c) in learned_edges:
        if (c, r) in true_edges:  # reversed
            reversals += 1

    # A reversed edge is a FP in learned and FN in true, so:
    # SHD = (fp_dir - reversals) + (fn_dir - reversals) + reversals
    #      = fp_dir + fn_dir - reversals
    shd = fp_dir + fn_dir - reversals

    return {
        "directed_precision": dir_prec,
        "directed_recall":    dir_rec,
        "directed_f1":        dir_f1,
        "skeleton_precision": skel_prec,
        "skeleton_recall":    skel_rec,
        "skeleton_f1":        skel_f1,
        "shd":                float(shd),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_meta(data_dir: Path, name: str) -> Optional[Dict]:
    meta_path = data_dir / f"{name}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as fh:
        return json.load(fh)


def _find_data_file(data_dir: Path, name: str, n_samples: int) -> Optional[Path]:
    """Find <name>_data_N<any>.csv, preferring exact n_samples match."""
    exact = data_dir / f"{name}_data_N{n_samples}.csv"
    if exact.exists():
        return exact
    # Accept any file matching the pattern
    matches = sorted(data_dir.glob(f"{name}_data_N*.csv"))
    if matches:
        return matches[-1]  # use the largest available
    return None


def _load_data(csv_path: Path, n_samples: int, n_vars: int) -> Optional[np.ndarray]:
    """Load up to n_samples rows from a headerless integer CSV."""
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            try:
                rows.append([int(x) for x in row])
            except ValueError:
                continue
    if not rows:
        return None
    data = np.array(rows, dtype=int)
    if data.shape[1] != n_vars:
        return None
    return data


def _fit_bn(data: np.ndarray, cardinality: np.ndarray, method: str, kwargs: Dict) -> np.ndarray:
    """Fit a BN and return the learned adjacency matrix."""
    n_vars = data.shape[1]
    bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
    bn.learn_structure(
        data,
        method=method,
        max_parents=None,
        alpha=1.0,
        **kwargs
    )
    return bn.adjacency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    data_dir  = Path(args[0]) if len(args) > 0 else _PROJECT_ROOT / "data" / "bn_benchmarks"
    n_samples = int(args[1])  if len(args) > 1 else 1000
    seed      = int(args[2])  if len(args) > 2 else 0

    np.random.seed(seed)

    print(f"data_dir:  {data_dir}")
    print(f"n_samples: {n_samples}")
    print(f"seed:      {seed}")
    print()
    print("Metrics:")
    print("  Dir-P/R/F1 — directed edge precision / recall / F1 (exact direction match)")
    print("  Sk-P/R/F1  — skeleton precision / recall / F1 (undirected, Markov-equivalent friendly)")
    print("  SHD        — Structural Hamming Distance (FP + FN - reversals)")
    print()

    method_names = [m["name"] for m in LEARNING_METHODS]
    metric_keys  = ["directed_precision", "directed_recall", "directed_f1",
                    "skeleton_precision", "skeleton_recall", "skeleton_f1",
                    "shd", "time_s"]

    all_rows: List[Dict] = []          # for CSV output
    # aggregation: method_name -> list of metric dicts
    agg: Dict[str, List[Dict]] = {m["name"]: [] for m in LEARNING_METHODS}

    for bn_name in BENCHMARK_NAMES:
        meta = _load_meta(data_dir, bn_name)
        if meta is None:
            print(f"[SKIP] {bn_name}: metadata not found (run generate_bn_datasets.py first)")
            continue

        data_file = _find_data_file(data_dir, bn_name, n_samples)
        if data_file is None:
            print(f"[SKIP] {bn_name}: data file not found")
            continue

        n_vars     = meta["n_vars"]
        cardinality = np.array(meta["cardinality"], dtype=int)
        true_adj   = np.array(meta["adjacency"], dtype=int)
        n_edges    = int(true_adj.sum())

        data = _load_data(data_file, n_samples, n_vars)
        if data is None:
            print(f"[SKIP] {bn_name}: could not load data")
            continue

        actual_n = data.shape[0]

        print(f"=== {bn_name}  (n_vars={n_vars}, n_edges={n_edges}, n_samples={actual_n}) ===")
        header = f"  {'Method':<12} {'Dir-P':>6} {'Dir-R':>6} {'Dir-F1':>7} {'Sk-P':>6} {'Sk-R':>6} {'Sk-F1':>7} {'SHD':>6} {'Time':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for method_info in LEARNING_METHODS:
            mname  = method_info["name"]
            method = method_info["method"]
            kwargs = method_info["kwargs"]

            t0 = time.time()
            try:
                learned_adj = _fit_bn(data, cardinality, method, kwargs)
            except Exception as exc:
                print(f"  {mname:<12} ERROR: {exc}")
                continue
            elapsed = time.time() - t0

            metrics = compute_metrics(true_adj, learned_adj)
            metrics["time_s"] = elapsed

            row_csv = {"bn_name": bn_name, "method": mname}
            row_csv.update(metrics)
            all_rows.append(row_csv)
            agg[mname].append(metrics)

            print(
                f"  {mname:<12}"
                f" {metrics['directed_precision']:>6.3f}"
                f" {metrics['directed_recall']:>6.3f}"
                f" {metrics['directed_f1']:>7.3f}"
                f" {metrics['skeleton_precision']:>6.3f}"
                f" {metrics['skeleton_recall']:>6.3f}"
                f" {metrics['skeleton_f1']:>7.3f}"
                f" {metrics['shd']:>6.0f}"
                f" {elapsed:>6.2f}s"
            )

        print()

    # ------------------------------------------------------------------
    # Aggregated summary (mean over all BNs)
    # ------------------------------------------------------------------
    print("=== AGGREGATED SUMMARY (mean over all BNs) ===")
    header = f"  {'Method':<12} {'Dir-P':>6} {'Dir-R':>6} {'Dir-F1':>7} {'Sk-P':>6} {'Sk-R':>6} {'Sk-F1':>7} {'SHD':>6} {'Time':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method_info in LEARNING_METHODS:
        mname = method_info["name"]
        records = agg[mname]
        if not records:
            print(f"  {mname:<12} no results")
            continue
        avgs = {}
        for k in metric_keys:
            vals = [r[k] for r in records if k in r]
            avgs[k] = float(np.mean(vals)) if vals else float('nan')

        print(
            f"  {mname:<12}"
            f" {avgs['directed_precision']:>6.3f}"
            f" {avgs['directed_recall']:>6.3f}"
            f" {avgs['directed_f1']:>7.3f}"
            f" {avgs['skeleton_precision']:>6.3f}"
            f" {avgs['skeleton_recall']:>6.3f}"
            f" {avgs['skeleton_f1']:>7.3f}"
            f" {avgs['shd']:>6.1f}"
            f" {avgs['time_s']:>6.2f}s"
        )

    print()

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if all_rows:
        csv_path = data_dir / "comparison_results.csv"
        fieldnames = ["bn_name", "method"] + metric_keys
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to: {csv_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
