"""
extract_eda_cluster.py — parse the cluster result files of the extended EDA
benchmark into a single tidy CSV.

Input
-----
A directory of result files produced by ``scripts/gen_eval_eda_benchmark.py``,
one per (problem, train_set, algorithm, seed, temperature) combination, each
named ``eval_<problem>_tr<train>_<algo>_T<temp>_s<seed>.dat`` and containing an
11-value vector:

    f1 time  ll_s0 kl_s0 sp_s0  ll_s1 kl_s1 sp_s1  ll_s2 kl_s2 sp_s2

Only the *finished* runs are present (the launcher grid is idempotent and some
of the largest jobs may not have completed in the wall-clock limit).

Output
------
A tidy CSV (one row per result file) with the parsed metadata, the raw
per-split metrics, and — crucially — the **train vs. test** split of every
prediction metric.  For a run whose training pool is split ``t``:

  * the ``train_*`` columns hold the metric on split ``t`` (the split the BN was
    learned on, evaluated on 100% of its rows), and
  * the ``test_*`` columns hold the mean over the **two other splits** — the
    two held-out "test cases" whose Boltzmann probability ranking was never
    used for learning.

``sp`` (Spearman correlation between the BN-predicted probability and the
original Boltzmann probability) is the priority metric downstream.

Usage
-----
    python3 scripts/extract_eda_cluster.py [results_dir] [out_csv]

    results_dir  default: eda_eval_cluster/
    out_csv      default: results/eda_cluster_results.csv
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FNAME_RE = re.compile(
    r"^eval_(?P<problem>.+)_tr(?P<train>\d+)_(?P<algo>.+)_T(?P<temp>[0-9.]+)_s(?P<seed>\d+)\.dat$"
)
VALUE_NAMES = ["f1", "time",
               "ll_s0", "kl_s0", "sp_s0",
               "ll_s1", "kl_s1", "sp_s1",
               "ll_s2", "kl_s2", "sp_s2"]


def problem_group(n: int) -> str:
    """Small / Medium / Large by number of variables (covers every dataset)."""
    if n < 50:
        return "small"          # 30, 36, 39
    if n < 100:
        return "medium"         # 50, 60, 64, 66
    return "large"              # 100 .. 258


def parse_header_extras(path: str) -> dict:
    """Read n / edges / n_selected from the first comment line, if present."""
    extras = {}
    try:
        with open(path) as fh:
            first = fh.readline()
    except OSError:
        return extras
    if first.startswith("#"):
        for tok in first[1:].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                extras[k] = v
    return extras


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


def load_one(path: str) -> dict | None:
    """Return a tidy record for one result file, or None if unparsable."""
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    try:
        v = np.loadtxt(path)          # skips '#' comment lines automatically
    except Exception:                 # noqa: BLE001
        return None
    v = np.atleast_1d(v).astype(float)
    if v.shape[0] != 11:
        return None

    problem = m["problem"]
    train_set = int(m["train"])
    algo = m["algo"]
    temp = m["temp"]
    seed = int(m["seed"])
    family = problem.rsplit("_", 1)[0]
    try:
        n = int(problem.rsplit("_", 1)[1])
    except ValueError:
        n = -1

    vals = {name: v[i] for i, name in enumerate(VALUE_NAMES)}

    # split the per-split metrics into (train split == train_set) vs (other two)
    other = [s for s in (0, 1, 2) if s != train_set]
    rec = {
        "problem": problem, "family": family, "n": n, "group": problem_group(n),
        "algorithm": algo, "temperature": float(temp), "temp_str": temp,
        "train_set": train_set, "seed": seed,
        "f1": vals["f1"], "time": vals["time"],
    }
    extras = parse_header_extras(path)
    rec["edges"] = _fnum(extras.get("edges", ""))
    rec["n_selected"] = _fnum(extras.get("n_selected", ""))

    for metric in ("ll", "kl", "sp"):
        rec[f"train_{metric}"] = vals[f"{metric}_s{train_set}"]
        test_vals = [vals[f"{metric}_s{s}"] for s in other]
        # nanmean over the two held-out test splits (a split may be NaN, e.g.
        # constant predictions on a separable problem such as OneMax)
        with np.errstate(invalid="ignore"):
            rec[f"test_{metric}"] = float(np.nanmean(test_vals))
        rec[f"test1_{metric}"] = test_vals[0]
        rec[f"test2_{metric}"] = test_vals[1]
    # keep raw per-split values too (useful for ad-hoc analysis)
    for name in VALUE_NAMES:
        rec[name] = vals[name]
    return rec


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "eda_eval_cluster")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "results", "eda_cluster_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "*.dat")))
    print(f"Found {len(files)} result files in {results_dir}")

    records, skipped = [], 0
    for f in files:
        rec = load_one(f)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    if not records:
        raise SystemExit("No parsable result files found.")

    # stable, readable column order
    lead = ["problem", "family", "n", "group", "algorithm", "temperature",
            "temp_str", "train_set", "seed", "edges", "n_selected", "f1", "time"]
    derived = [f"{r}_{m}" for m in ("sp", "ll", "kl")
               for r in ("train", "test", "test1", "test2")]
    fields = lead + derived + VALUE_NAMES
    fields = list(dict.fromkeys(fields))   # dedupe, preserve order

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    n_prob = len({r["problem"] for r in records})
    n_algo = len({r["algorithm"] for r in records})
    print(f"Wrote {len(records)} rows ({skipped} skipped) to {out_csv}")
    print(f"  problems={n_prob}  algorithms={n_algo}  "
          f"temps={sorted({r['temp_str'] for r in records})}  "
          f"train_sets={sorted({r['train_set'] for r in records})}")
    groups = {}
    for r in records:
        groups.setdefault(r["group"], set()).add(r["problem"])
    for g in ("small", "medium", "large"):
        if g in groups:
            print(f"  group {g}: {len(groups[g])} problems")


if __name__ == "__main__":
    main()
