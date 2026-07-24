"""
extract_bn_eda.py -- aggregate the per-run JSON files written by run_bn_eda.py
into two tidy CSVs for analysis.

    results/bn_eda_cluster/*.json
        -> bn_eda_summary.csv       one row per (problem, algorithm, seed) run
        -> bn_eda_trajectory.csv    one row per (run, generation) -- long form

Usage
-----
    python3 scripts/extract_bn_eda.py [results_dir] [out_dir]

Defaults: results_dir = results/bn_eda_cluster, out_dir = results/.
Both CSVs are safe to regenerate; partial result sets are handled (missing or
unfinished runs are simply absent).
"""
import csv
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_ROOT, "results", "bn_eda_cluster")
DEFAULT_OUT = os.path.join(_ROOT, "results")

# run-level config/summary columns (flattened)
SUMMARY_COLUMNS = [
    "problem", "family", "n_vars", "max_cardinality", "algorithm", "seed",
    "pop_size", "n_gen", "max_parents", "weighting_beta",
    "best_fitness", "final_best", "generation_found", "auc_best",
    "total_wall_time", "total_learn_time", "mean_learn_time",
    "mean_edges", "final_edges", "mean_f1",
    "mean_sp_sel", "final_sp_sel", "mean_sp_pop", "final_sp_pop",
    "mean_ll_pop", "mean_kl_pop", "final_diversity",
]

TRAJ_METRICS = [
    "gen", "best_fitness", "mean_fitness", "std_fitness",
    "learn_time", "edges", "f1", "n_selected", "pop_diversity",
    "sp_sel", "ll_sel", "kl_sel", "sp_pop", "ll_pop", "kl_pop",
]


def _row_summary(rec):
    s = rec.get("summary", {})
    problem = rec.get("problem", "")
    # family = alphabetic prefix (e.g. Deceptive3_39 -> Deceptive3)
    family = rec.get("family") or (problem.rsplit("_", 1)[0] if "_" in problem else problem)
    row = {
        "problem": problem,
        "family": family,
        "n_vars": rec.get("n_vars"),
        "max_cardinality": rec.get("max_cardinality"),
        "algorithm": rec.get("algorithm"),
        "seed": rec.get("seed"),
        "pop_size": rec.get("pop_size"),
        "n_gen": rec.get("n_gen"),
        "max_parents": rec.get("max_parents"),
        "weighting_beta": rec.get("weighting_beta"),
    }
    for k in SUMMARY_COLUMNS:
        if k in row:
            continue
        row[k] = s.get(k)
    return row


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "bneda_*.json")))
    if not files:
        raise SystemExit(f"No bneda_*.json files found in {results_dir}")

    summary_path = os.path.join(out_dir, "bn_eda_summary.csv")
    traj_path = os.path.join(out_dir, "bn_eda_trajectory.csv")

    n_ok = 0
    with open(summary_path, "w", newline="") as sf, \
         open(traj_path, "w", newline="") as tf:
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
        sw.writeheader()
        tw = csv.DictWriter(
            tf, fieldnames=["problem", "algorithm", "seed"] + TRAJ_METRICS)
        tw.writeheader()

        for path in files:
            try:
                rec = json.load(open(path))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"skip unreadable {os.path.basename(path)}: {exc}",
                      file=sys.stderr)
                continue
            srow = _row_summary(rec)
            sw.writerow({k: srow.get(k) for k in SUMMARY_COLUMNS})
            for g in rec.get("trajectory", []):
                trow = {"problem": srow["problem"],
                        "algorithm": srow["algorithm"], "seed": srow["seed"]}
                for m in TRAJ_METRICS:
                    trow[m] = g.get(m)
                tw.writerow(trow)
            n_ok += 1

    print(f"Aggregated {n_ok} runs from {results_dir}")
    print(f"  wrote {summary_path}")
    print(f"  wrote {traj_path}")


if __name__ == "__main__":
    main()
