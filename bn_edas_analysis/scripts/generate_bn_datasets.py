"""
generate_bn_datasets.py — Generate benchmark datasets from bnRep Bayesian networks.

Usage (positional args, no flags):
    python3 scripts/generate_bn_datasets.py [n_samples [seed [r_data_dir [output_dir]]]]

Defaults:
    n_samples  = 2000
    seed       = 42
    r_data_dir = docs/BN_data_generation/bnRep-master/data-raw
    output_dir = data/bn_benchmarks

For each of 10 selected BNs:
  - Parse the corresponding R file
  - Sample n_samples rows via ancestral sampling
  - Save data as  <output_dir>/<name>_data_N<n>.csv  (no header, integer values)
  - Save metadata as <output_dir>/<name>_meta.json
  - Print one summary line per BN
"""

import sys
import os
import time
import json
import csv
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.bn_io import parse_bnrep_r_file, sample_bn  # noqa: E402

# ---------------------------------------------------------------------------
# Selected benchmarks
# ---------------------------------------------------------------------------
SELECTED_BNS = [
    dict(name="asia",         file="asia.R",         n_vars=8,  description="Small binary BN, lung disease"),
    dict(name="moodstate",    file="moodstate.R",    n_vars=7,  description="Small, mixed cardinality (up to 7 states)"),
    dict(name="electrolysis", file="electrolysis.R", n_vars=16, description="Medium-small, binary, sparse"),
    dict(name="accidents",    file="accidents.R",    n_vars=17, description="Medium, mixed cardinality (up to 10 states)"),
    dict(name="corrosion",    file="corrosion.R",    n_vars=22, description="Medium, mixed cardinality (2-4 states)"),
    dict(name="flood",        file="flood.R",        n_vars=22, description="Medium, high cardinality (2-7 states)"),
    dict(name="BOPfailure3",  file="BOPfailure3.R",  n_vars=28, description="Medium-large, binary"),
    dict(name="yangtze",      file="yangtze.R",      n_vars=31, description="Larger, mixed (2-3 states)"),
    dict(name="BOPfailure2",  file="BOPfailure2.R",  n_vars=46, description="Large, binary"),
    dict(name="pneumonia",    file="pneumonia.R",    n_vars=62, description="Largest available, mostly binary"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parse arguments ---
    args = sys.argv[1:]
    n_samples  = int(args[0])  if len(args) > 0 else 2000
    seed       = int(args[1])  if len(args) > 1 else 42
    r_data_dir = Path(args[2]) if len(args) > 2 else _PROJECT_ROOT / "docs" / "BN_data_generation" / "bnRep-master" / "data-raw"
    output_dir = Path(args[3]) if len(args) > 3 else _PROJECT_ROOT / "data" / "bn_benchmarks"

    r_data_dir = Path(r_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"n_samples:  {n_samples}")
    print(f"seed:       {seed}")
    print(f"r_data_dir: {r_data_dir}")
    print(f"output_dir: {output_dir}")
    print()
    print(f"{'Name':<16} {'n_vars':>6} {'n_edges':>7} {'cardinalities':<30} {'time(s)':>8}")
    print("-" * 74)

    rng = np.random.default_rng(seed)

    for bn_info in SELECTED_BNS:
        name     = bn_info["name"]
        r_file   = r_data_dir / bn_info["file"]

        t0 = time.time()
        try:
            bn_def = parse_bnrep_r_file(str(r_file), name=name)
        except Exception as exc:
            print(f"  ERROR parsing {name}: {exc}", file=sys.stderr)
            continue

        try:
            data = sample_bn(bn_def, n_samples, rng=rng)
        except Exception as exc:
            print(f"  ERROR sampling {name}: {exc}", file=sys.stderr)
            continue

        elapsed = time.time() - t0

        # Save data CSV (no header, integer values)
        data_path = output_dir / f"{name}_data_N{n_samples}.csv"
        with open(data_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            for row in data:
                writer.writerow(row.tolist())

        # Save metadata JSON
        n_edges = int(bn_def.adjacency.sum())
        meta = {
            "name":        bn_def.name,
            "n_vars":      bn_def.n_vars,
            "n_edges":     n_edges,
            "n_samples":   n_samples,
            "description": bn_info.get("description", ""),
            "var_names":   bn_def.var_names,
            "cardinality": bn_def.cardinality.tolist(),
            "adjacency":   bn_def.adjacency.tolist(),
            "var_states":  bn_def.var_states,
        }
        meta_path = output_dir / f"{name}_meta.json"
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)

        # Summary line
        card_str = str(sorted(set(bn_def.cardinality.tolist())))
        print(f"{name:<16} {bn_def.n_vars:>6} {n_edges:>7} {card_str:<30} {elapsed:>8.2f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
