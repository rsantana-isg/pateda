"""
statistical_comparison_benchmark.py
----------------------------------
Extract 20-run best fitness values for benchmark combinatorial instances and compare:
  Diff-EDA, DbD-EDA, UMDA, TreeEDA, EBNA, MN-FDAG, C-VAE-EDA

For each instance (SAT, Ising, UBQP), the script:
  1) Selects the best configuration per algorithm from benchmark CSV summaries.
  2) Reads best-fitness values from raw .dat files (up to 20 seeds).
  3) Runs Kruskal-Wallis.
  4) Runs post-hoc pairwise Mann-Whitney U with Bonferroni correction.
  5) Exports results to CSV.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kruskal, mannwhitneyu


N_SEEDS = 20
N_GEN = 250
TRUNC = 0.1
N = 100
P_SIZE = N * 5

PROBLEMS = {
    "sat": {
        "name_in_files": "SAT",
        "instances": ["uf100-01", "uf100-02", "uf100-03", "uf100-04", "uf100-05"],
    },
    "ising": {
        "name_in_files": "Ising",
        "instances": ["SG_100_1", "SG_100_2", "SG_100_3", "SG_100_4"],
    },
    "ubqp": {
        "name_in_files": "UBQP",
        "instances": ["bqp100"],
    },
}

ALGORITHMS = ["Diff-EDA", "DbD-EDA", "UMDA", "TreeEDA", "EBNA", "MN-FDAG", "C-VAE-EDA"]

RAW_FOLDERS = {
    "dendiff": "results_benchmark_Dendiff",
    "dbd": "results_benchmark_DbD",
    "eda": "results_benchmark_EDA_RW",
    "vae": "results_benchmark_VAE",
}

SUMMARY_FILES = {
    "dendiff": "dendiff_benchmark_{problem}_results.csv",
    "dbd": "dbd_benchmark_{problem}_results.csv",
    "eda": "discrete_EDA_RW_benchmark_{problem}_results.csv",
    "vae": "vae_benchmark_{problem}_results.csv",
}

DENDIFF_SAMPLING = {
    "dendiff_gumbel": "gumbel",
    "dendiff_corruption": "corruption",
    "dendiff_ste": "ste",
    "dendiff_hard_concrete": "hard_concrete",
    "dendiff_deterministic": "deterministic",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def parse_best_fitness(path: str) -> float | None:
    try:
        with open(path, "r") as fh:
            for line in fh:
                if "Best fitness found:" in line:
                    return float(line.split(":", 1)[1].strip())
    except FileNotFoundError:
        return None
    return None


def to_float(value: str, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trim_num(value: float) -> str:
    return format(value, ".12g")


def alpha_str_for_eda_and_diff(value: str) -> str:
    x = to_float(value, 0.0)
    if abs(x) < 1e-12:
        return "0"
    return trim_num(x)


def alpha_str_for_vae(value: str) -> str:
    x = to_float(value, 0.0)
    if abs(x) < 1e-12:
        return "0.0"
    return trim_num(x)


def int_str(value: str) -> str:
    return str(int(round(to_float(value, 0.0))))


def rank_key(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    return (
        to_float(row.get("success", ""), float("-inf")),
        to_float(row.get("best_fitness", ""), float("-inf")),
        -to_float(row.get("generation", ""), float("inf")),
        -to_float(row.get("elapsed_time", ""), float("inf")),
    )


def select_best(rows: List[Dict[str, str]]) -> Dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=rank_key)


def extract_dendiff(problem_name: str, instance: str, cfg: Dict[str, str], folder: str) -> List[float]:
    variant = cfg["variant"]
    sampling = DENDIFF_SAMPLING.get(variant)
    if sampling is None:
        return []

    activation = cfg["activation"]
    loss = cfg["loss"]
    fg = int_str(cfg.get("fitness_guided", "0"))
    alpha = alpha_str_for_eda_and_diff(cfg.get("alpha", "0"))

    values: List[float] = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_benchmark_dendiff_{problem_name}_{instance}_{P_SIZE}_{N_GEN}_{TRUNC}_"
            f"{variant}_{sampling}_{activation}_{loss}_"
            f"400_20_{fg}_1.0_0.01_1_{alpha}_{seed}.dat"
        )
        v = parse_best_fitness(os.path.join(folder, fname))
        if v is not None:
            values.append(v)
    return values


def extract_dbd(problem_name: str, instance: str, cfg: Dict[str, str], folder: str) -> List[float]:
    variant = cfg["variant"]
    activation = cfg["activation"]
    loss = cfg["loss"]
    k = int_str(cfg.get("k", "0"))
    fg = int_str(cfg.get("fitness_guided", "0"))
    alpha = alpha_str_for_eda_and_diff(cfg.get("alpha", "0"))

    values: List[float] = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_benchmark_dbd_{problem_name}_{instance}_{P_SIZE}_{N_GEN}_{TRUNC}_"
            f"{variant}_{activation}_{loss}_100_20_{k}_0.1_{fg}_0_{alpha}_{seed}.dat"
        )
        v = parse_best_fitness(os.path.join(folder, fname))
        if v is not None:
            values.append(v)
    return values


def extract_eda(problem_name: str, instance: str, cfg: Dict[str, str], folder: str) -> List[float]:
    alg = cfg["alg"]
    alpha = alpha_str_for_eda_and_diff(cfg.get("alpha", "0"))

    values: List[float] = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_benchmark_EDA_RW_{problem_name}_{instance}_{P_SIZE}_{N_GEN}_{alg}_{alpha}_{TRUNC}_{seed}.dat"
        )
        v = parse_best_fitness(os.path.join(folder, fname))
        if v is not None:
            values.append(v)
    return values


def extract_vae(problem_name: str, instance: str, cfg: Dict[str, str], folder: str) -> List[float]:
    variant = cfg["variant"]
    act_enc = cfg["act_enc"]
    act_dec = cfg["act_dec"]
    alpha = alpha_str_for_vae(cfg.get("alpha", "0.0"))

    values: List[float] = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_benchmark_vae_{problem_name}_{instance}_{P_SIZE}_{N_GEN}_{TRUNC}_"
            f"{variant}_{act_enc}_{act_dec}_0.0_1.0_0_50_0_{alpha}_{seed}.dat"
        )
        v = parse_best_fitness(os.path.join(folder, fname))
        if v is not None:
            values.append(v)
    return values


def run_kruskal(groups: Dict[str, List[float]]) -> Tuple[float, float]:
    arrays = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(arrays) < 2:
        return float("nan"), float("nan")
    stat, p = kruskal(*arrays)
    return float(stat), float(p)


def posthoc_mannwhitney_bonferroni(groups: Dict[str, List[float]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    names = [k for k, v in groups.items() if len(v) > 0]
    pairs = list(itertools.combinations(names, 2))
    n_comp = len(pairs)

    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    for a, b in pairs:
        x = np.array(groups[a])
        y = np.array(groups[b])
        try:
            u_stat, p_raw = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            u_stat, p_raw = float("nan"), float("nan")

        p_adj = min(p_raw * n_comp, 1.0) if not np.isnan(p_raw) else float("nan")
        results[(a, b)] = {"U": float(u_stat), "p_raw": float(p_raw), "p_adj": float(p_adj)}

    return results


def sig_marker(p: float) -> str:
    if np.isnan(p):
        return "na"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def choose_configs(summary_root: str, problem: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    base = os.path.join(summary_root, "combinatorial", problem)

    dendiff_rows = read_csv_rows(os.path.join(base, SUMMARY_FILES["dendiff"].format(problem=problem)))
    dbd_rows = read_csv_rows(os.path.join(base, SUMMARY_FILES["dbd"].format(problem=problem)))
    eda_rows = read_csv_rows(os.path.join(base, SUMMARY_FILES["eda"].format(problem=problem)))
    vae_rows = read_csv_rows(os.path.join(base, SUMMARY_FILES["vae"].format(problem=problem)))

    configs: Dict[str, Dict[str, Dict[str, str]]] = {
        "Diff-EDA": {},
        "DbD-EDA": {},
        "UMDA": {},
        "TreeEDA": {},
        "EBNA": {},
        "MN-FDAG": {},
        "C-VAE-EDA": {},
    }

    instances = PROBLEMS[problem]["instances"]
    for instance in instances:
        best_diff = select_best([r for r in dendiff_rows if r.get("instance") == instance])
        if best_diff:
            configs["Diff-EDA"][instance] = best_diff

        best_dbd = select_best([r for r in dbd_rows if r.get("instance") == instance])
        if best_dbd:
            configs["DbD-EDA"][instance] = best_dbd

        best_vae = select_best([
            r for r in vae_rows if r.get("instance") == instance and r.get("variant") == "C-VAE"
        ])
        if best_vae:
            configs["C-VAE-EDA"][instance] = best_vae

        for alg in ["UMDA", "TreeEDA", "EBNA", "MN-FDAG"]:
            best_eda = select_best([
                r for r in eda_rows if r.get("instance") == instance and r.get("alg") == alg
            ])
            if best_eda:
                configs[alg][instance] = best_eda

    return configs


def analyze_problem(
    summary_root: str,
    raw_root: str,
    problem_key: str,
) -> List[Dict[str, object]]:
    problem_name = PROBLEMS[problem_key]["name_in_files"]
    instances = PROBLEMS[problem_key]["instances"]
    configs = choose_configs(summary_root, problem_key)

    rows: List[Dict[str, object]] = []

    dendiff_folder = os.path.join(raw_root, RAW_FOLDERS["dendiff"])
    dbd_folder = os.path.join(raw_root, RAW_FOLDERS["dbd"])
    eda_folder = os.path.join(raw_root, RAW_FOLDERS["eda"])
    vae_folder = os.path.join(raw_root, RAW_FOLDERS["vae"])

    for instance in instances:
        groups: Dict[str, List[float]] = {k: [] for k in ALGORITHMS}

        if instance in configs["Diff-EDA"]:
            groups["Diff-EDA"] = extract_dendiff(problem_name, instance, configs["Diff-EDA"][instance], dendiff_folder)
        if instance in configs["DbD-EDA"]:
            groups["DbD-EDA"] = extract_dbd(problem_name, instance, configs["DbD-EDA"][instance], dbd_folder)
        if instance in configs["UMDA"]:
            groups["UMDA"] = extract_eda(problem_name, instance, configs["UMDA"][instance], eda_folder)
        if instance in configs["TreeEDA"]:
            groups["TreeEDA"] = extract_eda(problem_name, instance, configs["TreeEDA"][instance], eda_folder)
        if instance in configs["EBNA"]:
            groups["EBNA"] = extract_eda(problem_name, instance, configs["EBNA"][instance], eda_folder)
        if instance in configs["MN-FDAG"]:
            groups["MN-FDAG"] = extract_eda(problem_name, instance, configs["MN-FDAG"][instance], eda_folder)
        if instance in configs["C-VAE-EDA"]:
            groups["C-VAE-EDA"] = extract_vae(problem_name, instance, configs["C-VAE-EDA"][instance], vae_folder)

        groups_with_data = {k: v for k, v in groups.items() if len(v) > 0}
        if len(groups_with_data) < 2:
            print(f"[SKIP] {problem_key}/{instance}: fewer than 2 algorithms with data.")
            continue

        kw_stat, kw_p = run_kruskal(groups_with_data)
        posthoc = posthoc_mannwhitney_bonferroni(groups_with_data)

        print("=" * 76)
        print(f"Problem={problem_key}  Instance={instance}")
        print(f"Kruskal-Wallis: H={kw_stat:.4f}  p={kw_p:.6f}  ({sig_marker(kw_p)})")
        for alg in ALGORITHMS:
            vals = groups[alg]
            if vals:
                print(f"  {alg:<10} N={len(vals):>2} mean={np.mean(vals):>10.4f} std={np.std(vals):>10.4f}")
            else:
                print(f"  {alg:<10} N= 0")

        for (a, b), res in posthoc.items():
            rows.append(
                {
                    "problem": problem_key,
                    "instance": instance,
                    "alg_a": a,
                    "alg_b": b,
                    "n_a": len(groups[a]),
                    "n_b": len(groups[b]),
                    "U": res["U"],
                    "p_raw": res["p_raw"],
                    "p_adj_bonferroni": res["p_adj"],
                    "significance": sig_marker(res["p_adj"]),
                    "kruskal_H": kw_stat,
                    "kruskal_p": kw_p,
                }
            )

    return rows


def save_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        print("No result rows to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {out_path}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_root = script_dir
    raw_root = script_dir

    if len(sys.argv) > 1:
        raw_root = sys.argv[1]
    if len(sys.argv) > 2:
        summary_root = sys.argv[2]

    all_rows: List[Dict[str, object]] = []
    for problem in ["sat", "ising", "ubqp"]:
        all_rows.extend(analyze_problem(summary_root=summary_root, raw_root=raw_root, problem_key=problem))

    save_csv(all_rows, os.path.join(script_dir, "benchmark_statistical_comparison_results.csv"))


if __name__ == "__main__":
    main()
