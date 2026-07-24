"""
extract_bn_eda_trajectory.py -- mine the per-generation JSON trajectories written
by run_bn_eda.py to explain the OFFLINE<->ONLINE metric relationship.

The online BN-EDA sweeps through the same selection-pressure axis the offline
benchmark samples at three fixed regimes: at gen 0 the population is diverse
(pop_diversity ~ 1, like offline subset 0) and by the last generation it has
converged (pop_diversity ~ 0, like offline subset 2).  Along the way sp_pop
rises toward 1 (an artefact of convergence, not model quality) while the model
edges and F1 collapse toward 0.  This script turns each trajectory into:

  * a per-run FEATURE row (early / late phase means, convergence dynamics,
    real population-diversity signals) -> bn_eda_trajectory_features.csv
  * per-(family, generation) AVERAGE trajectories of the key metrics
    -> bn_eda_avg_trajectory.csv

Phases:  early = generations 0..EARLY-1 (exploratory / diverse regime);
         late  = last LATE generations (converged regime).

Usage
-----
    python3 scripts/extract_bn_eda_trajectory.py [results_dir] [out_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_ROOT, "results_bn_eda")
DEFAULT_OUT = os.path.join(_ROOT, "bn_eda_analysis")

EARLY = 10          # first 10 generations = exploratory / diverse regime
LATE = 10           # last 10 generations = converged regime
CONV_THRESH = 0.1   # pop_diversity below this = "converged"
TRAJ_KEYS = ["pop_diversity", "sp_pop", "ll_pop", "kl_pop", "f1", "edges",
             "best_fitness", "mean_fitness", "std_fitness"]


def _nan_mean(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _slice_mean(traj, key, idx):
    return _nan_mean([traj[i].get(key, np.nan) for i in idx if i < len(traj)])


def run_features(d):
    traj = d["trajectory"]
    G = len(traj)
    early = range(0, min(EARLY, G))
    late = range(max(0, G - LATE), G)
    div = np.array([t.get("pop_diversity", np.nan) for t in traj], float)
    edges = np.array([t.get("edges", np.nan) for t in traj], float)

    # convergence generation: first gen with diversity < threshold
    below = np.where(div < CONV_THRESH)[0]
    t_conv = int(below[0]) if below.size else G
    # model-collapse generation: first gen where edges hit 0 after being > 0
    collapse = G
    seen_edge = False
    for i, e in enumerate(edges):
        if e > 0:
            seen_edge = True
        elif seen_edge and e == 0:
            collapse = i
            break

    s = d["summary"]
    feat = {
        "problem": d["problem"], "family": d.get("family", d["problem"].rsplit("_", 1)[0]),
        "n": int(d["n_vars"]), "algorithm": d["algorithm"], "seed": int(d["seed"]),
        "best_fitness": float(s.get("best_fitness", np.nan)),
        "auc_best": float(s.get("auc_best", np.nan)),
        "final_diversity": float(s.get("final_diversity", div[-1] if len(div) else np.nan)),
        "mean_diversity": _nan_mean(div),          # exploration budget over the run
        "t_converge": t_conv,                        # gens until population collapses
        "t_collapse_model": collapse,                # gens until BN loses all edges
        "max_f1": float(np.nanmax([t.get("f1", np.nan) for t in traj])),
    }
    for key in ["sp_pop", "f1", "edges", "kl_pop", "ll_pop", "pop_diversity"]:
        feat[f"{key}_early"] = _slice_mean(traj, key, early)
        feat[f"{key}_late"] = _slice_mean(traj, key, late)
    return feat


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    feats = []
    # accumulate per-(family, gen) sums for average trajectories
    acc_sum = defaultdict(lambda: defaultdict(float))
    acc_cnt = defaultdict(lambda: defaultdict(int))
    bad = 0
    for k, fp in enumerate(files):
        try:
            d = json.load(open(fp))
        except Exception:
            bad += 1
            continue
        feats.append(run_features(d))
        fam = d.get("family", d["problem"].rsplit("_", 1)[0])
        for t in d["trajectory"]:
            g = t.get("gen")
            for key in TRAJ_KEYS:
                v = t.get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    acc_sum[(fam, g)][key] += v
                    acc_cnt[(fam, g)][key] += 1
        if (k + 1) % 2000 == 0:
            print(f"  processed {k + 1}/{len(files)} json files", file=sys.stderr)

    df = pd.DataFrame(feats)
    df.to_csv(os.path.join(out_dir, "bn_eda_trajectory_features.csv"), index=False)

    rows = []
    for (fam, g), sums in acc_sum.items():
        row = {"family": fam, "gen": g}
        for key in TRAJ_KEYS:
            c = acc_cnt[(fam, g)][key]
            row[key] = sums[key] / c if c else np.nan
        rows.append(row)
    avg = pd.DataFrame(rows).sort_values(["family", "gen"])
    avg.to_csv(os.path.join(out_dir, "bn_eda_avg_trajectory.csv"), index=False)

    print(f"Parsed {len(df)} json runs ({bad} unreadable) -> "
          f"bn_eda_trajectory_features.csv, bn_eda_avg_trajectory.csv")
    # quick sanity: mean early vs late sp_pop / diversity / f1 over all runs
    print("\nOver all runs (mean):")
    for c in ["pop_diversity", "sp_pop", "f1", "edges"]:
        print(f"  {c:14s} early={df[c+'_early'].mean():.3f}  late={df[c+'_late'].mean():.3f}")


if __name__ == "__main__":
    main()
