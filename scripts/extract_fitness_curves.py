"""
extract_fitness_curves.py -- per-(problem, algorithm, generation) average fitness
curves from the JSON trajectories, for the per-problem convergence figures.

Output: bn_eda_fitness_curves.csv with, for each (problem, algorithm, gen), the
seed-averaged population mean fitness and best-so-far fitness, plus the number
of seeds contributing to that cell.

Usage
-----
    python3 scripts/extract_fitness_curves.py [results_dir] [out_csv]
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
DEFAULT_OUT = os.path.join(_ROOT, "bn_eda_analysis", "bn_eda_fitness_curves.csv")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    out_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    mean_sum = defaultdict(float)   # (problem, algorithm, gen) -> sum mean_fitness
    best_sum = defaultdict(float)   # -> sum best-so-far fitness
    cnt = defaultdict(int)
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    for k, fp in enumerate(files):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        prob, alg = d["problem"], d["algorithm"]
        for t in d["trajectory"]:
            g = t.get("gen")
            mf = t.get("mean_fitness")
            bf = t.get("best_fitness")
            if mf is None or (isinstance(mf, float) and np.isnan(mf)):
                continue
            key = (prob, alg, g)
            mean_sum[key] += mf
            best_sum[key] += bf if bf is not None else np.nan
            cnt[key] += 1
        if (k + 1) % 3000 == 0:
            print(f"  {k + 1}/{len(files)}", file=sys.stderr)

    rows = []
    for key, c in cnt.items():
        prob, alg, g = key
        rows.append({"problem": prob, "algorithm": alg, "gen": g,
                     "mean_fitness": mean_sum[key] / c,
                     "best_fitness": best_sum[key] / c, "n_seeds": c})
    df = pd.DataFrame(rows).sort_values(["problem", "algorithm", "gen"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} (problem,algorithm,gen) rows to {out_csv}")


if __name__ == "__main__":
    main()
