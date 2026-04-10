"""
statistical_comparison.py
--------------------------
Reads experiment result files (20 runs per algorithm) and performs:
  1. Kruskal-Wallis test across all algorithms for each objective function.
  2. Post-hoc pairwise Mann-Whitney U tests with Bonferroni correction.

Algorithms compared:
  Diff-EDA, DbD-EDA, UMDA, TreeEDA, EBNA, MN-FDAG, C-VAE-EDA

Result folders (relative to the working directory when the script is run):
  Diff-EDA  : results_Dendiff_n30_N15n_new/
  DbD-EDA   : results_DbD_n30_N15n_new/
  UMDA, TreeEDA, EBNA, MN-FDAG : results_discrete_eda_v2_450_n30/
  C-VAE-EDA : results_VAE_450_n30_0.95/

The best-known hyper-parameter configuration for each deep-learning EDA is
hard-coded below (matching the selection used in Extract_Succ_Fitness_Gen_Additive.py).
"""

import os
import sys
import itertools
import numpy as np
from scipy.stats import kruskal, mannwhitneyu

# ---------------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------------
N_SEEDS = 20
N_GEN = 250
TRUNC = 0.5
ALPHA = 0.95

OBJ_FUNCTIONS = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5']

N_SMALL = 30   # problem size for all objectives except HIFF
N_LARGE = 64   # problem size for HIFF

ALGORITHMS = ['Diff-EDA', 'DbD-EDA', 'UMDA', 'TreeEDA', 'EBNA', 'MN-FDAG', 'C-VAE-EDA']

# ---------------------------------------------------------------------------
# Folder paths (relative to the working directory)
# ---------------------------------------------------------------------------
FOLDER_DENDIFF = 'results_Dendiff_n30_N15n_new'
FOLDER_DBD     = 'results_DbD_n30_N15n_new'
FOLDER_EDA     = 'results_discrete_eda_v2_450_n30'
FOLDER_VAE     = 'results_VAE_450_n30_0.95'

# ---------------------------------------------------------------------------
# Best hyper-parameter configurations (from Extract_Succ_Fitness_Gen_Additive.py)
# ---------------------------------------------------------------------------
# Diff-EDA: dendiff_gumbel variant, relu activation, ranking loss, fitness_guided=0
DENDIFF_CONFIG = {
    'variant': 'dendiff_gumbel',
    'sampling_strategy': 'gumbel',
    'activation': 'relu',
    'loss': 'ranking',
    'fitness_guided': 0,
    'n_timesteps': 400,
    'n_sampling_steps': 20,
    'temperature': 1.0,
    'beta_start': 0.01,
    'beta_end': 1,
    'alpha_smooth': 0.1,
}

# DbD-EDA: dbd_cs variant, relu activation, huber loss, fitness_guided=1
DBD_CONFIG = {
    'variant': 'dbd_cs',
    'activation': 'relu',
    'loss': 'huber',
    'num_alpha_samples': 100,
    'n_steps': 20,
    'k': 0,           # dbd_cs has no topology training, k=0
    'alpha_smooth': 0.1,
    'fitness_guided': 0,
    'markov_init': 0,
    'alpha': ALPHA,
}

# C-VAE-EDA: C-VAE variant, relu encoder, relu decoder
VAE_CONFIG = {
    'variant': 'C-VAE',
    'enc': 'relu',
    'dec': 'relu',
    'beta_start': 0.0,
    'beta_end': 1.0,
    'latent_dim': 0,
    'epochs': 50,
    'mi_layer': 0,
}

# ---------------------------------------------------------------------------
# Helper: parse a single .dat file and return the best fitness value
# ---------------------------------------------------------------------------

def parse_dat_file(path: str):
    """Return the best fitness value found in a .dat result file, or None."""
    best_fitness = None
    try:
        with open(path, 'r') as fh:
            for line in fh:
                if 'Best fitness found:' in line:
                    best_fitness = float(line.split(': ')[1].strip())
    except FileNotFoundError:
        pass
    return best_fitness


# ---------------------------------------------------------------------------
# Per-algorithm extraction functions
# ---------------------------------------------------------------------------

def extract_dendiff(obj_func: str, folder: str = FOLDER_DENDIFF) -> list:
    """Extract 20 best-fitness values for Diff-EDA."""
    n = N_LARGE if obj_func == 'HIFF' else N_SMALL
    p_size = n * 15
    cfg = DENDIFF_CONFIG
    values = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_dendiff_{obj_func}_{n}_{p_size}_{N_GEN}_{TRUNC}_"
            f"{cfg['variant']}_{cfg['sampling_strategy']}_{cfg['activation']}_{cfg['loss']}_"
            f"{cfg['n_timesteps']}_{cfg['n_sampling_steps']}_{cfg['fitness_guided']}_"
            f"{cfg['temperature']}_{cfg['beta_start']}_{cfg['beta_end']}_{ALPHA}_{seed}.dat"
        )
        val = parse_dat_file(os.path.join(folder, fname))
        if val is not None:
            values.append(val)
    return values


def extract_dbd(obj_func: str, folder: str = FOLDER_DBD) -> list:
    """Extract 20 best-fitness values for DbD-EDA."""
    n = N_LARGE if obj_func == 'HIFF' else N_SMALL
    p_size = n * 15
    cfg = DBD_CONFIG
    values = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_dbd_{obj_func}_{n}_{p_size}_{N_GEN}_{TRUNC}_"
            f"{cfg['variant']}_{cfg['activation']}_{cfg['loss']}_"
            f"{cfg['num_alpha_samples']}_{cfg['n_steps']}_"
            f"{cfg['k']}_{cfg['alpha_smooth']}_{cfg['fitness_guided']}_"
            f"{cfg['markov_init']}_{cfg['alpha']}_{seed}.dat"
        )
        val = parse_dat_file(os.path.join(folder, fname))
        if val is not None:
            values.append(val)
    return values


def extract_eda(obj_func: str, alg_name: str, folder: str = FOLDER_EDA) -> list:
    """Extract 20 best-fitness values for classical EDA variants (UMDA, TreeEDA, EBNA, MN-FDAG)."""
    n = N_LARGE if obj_func == 'HIFF' else N_SMALL
    p_size = n * 15
    values = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_discrete_eda_v2_{obj_func}_{n}_{p_size}_{N_GEN}_{alg_name}_{seed}.dat"
        )
        val = parse_dat_file(os.path.join(folder, fname))
        if val is not None:
            values.append(val)
    return values


def extract_vae(obj_func: str, folder: str = FOLDER_VAE) -> list:
    """Extract 20 best-fitness values for C-VAE-EDA."""
    n = N_LARGE if obj_func == 'HIFF' else N_SMALL
    p_size = n * 15
    cfg = VAE_CONFIG
    values = []
    for seed in range(1, N_SEEDS + 1):
        fname = (
            f"results_discrete_vae_{obj_func}_{n}_{p_size}_{N_GEN}_"
            f"{TRUNC}_{seed}_{cfg['variant']}_{cfg['enc']}_{cfg['dec']}_"
            f"{cfg['beta_start']}_{cfg['beta_end']}_{cfg['latent_dim']}_"
            f"{cfg['epochs']}_{cfg['mi_layer']}_{ALPHA}.dat"
        )
        val = parse_dat_file(os.path.join(folder, fname))
        if val is not None:
            values.append(val)
    return values


# ---------------------------------------------------------------------------
# Gather all algorithm results for a given objective function
# ---------------------------------------------------------------------------

def gather_results(obj_func: str) -> dict:
    """Return a dict {algorithm_name: [best_fitness_values...]} for one objective."""
    eda_map = {
        'UMDA':    'UMDA',
        'TreeEDA': 'TreeEDA',
        'EBNA':    'EBNA',
        'MN-FDAG': 'MN-FDAG',
    }
    data = {
        'Diff-EDA':  extract_dendiff(obj_func),
        'DbD-EDA':   extract_dbd(obj_func),
        'UMDA':      extract_eda(obj_func, eda_map['UMDA']),
        'TreeEDA':   extract_eda(obj_func, eda_map['TreeEDA']),
        'EBNA':      extract_eda(obj_func, eda_map['EBNA']),
        'MN-FDAG':   extract_eda(obj_func, eda_map['MN-FDAG']),
        'C-VAE-EDA': extract_vae(obj_func),
    }
    return data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_kruskal_wallis(groups: dict) -> tuple:
    """
    Run Kruskal-Wallis H-test on the provided groups.

    Parameters
    ----------
    groups : dict  {name: [values]}

    Returns
    -------
    stat : float   H statistic
    p    : float   p-value
    """
    arrays = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(arrays) < 2:
        return float('nan'), float('nan')
    stat, p = kruskal(*arrays)
    return stat, p


def posthoc_mannwhitney_bonferroni(groups: dict) -> dict:
    """
    Perform all pairwise Mann-Whitney U tests between groups and apply
    Bonferroni correction to the resulting p-values.

    Parameters
    ----------
    groups : dict  {name: [values]}

    Returns
    -------
    results : dict  {(alg_a, alg_b): {'U': float, 'p_raw': float, 'p_adj': float}}
    """
    names = [k for k, v in groups.items() if len(v) > 0]
    pairs = list(itertools.combinations(names, 2))
    n_comparisons = len(pairs)

    raw_results = {}
    for a, b in pairs:
        x = np.array(groups[a])
        y = np.array(groups[b])
        try:
            u_stat, p_raw = mannwhitneyu(x, y, alternative='two-sided')
        except ValueError:
            u_stat, p_raw = float('nan'), float('nan')
        raw_results[(a, b)] = {'U': u_stat, 'p_raw': p_raw}

    # Bonferroni correction: multiply each raw p-value by the number of comparisons
    adjusted = {}
    for pair, res in raw_results.items():
        p_adj = min(res['p_raw'] * n_comparisons, 1.0)
        adjusted[pair] = {**res, 'p_adj': p_adj}

    return adjusted


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
SIG_LEVELS = [(0.001, '***'), (0.01, '**'), (0.05, '*'), (1.0, 'ns')]


def sig_marker(p: float) -> str:
    """Return a significance marker string for a given p-value."""
    for threshold, marker in SIG_LEVELS:
        if p <= threshold:
            return marker
    return 'ns'


def print_summary(obj_func: str, data: dict, kw_stat: float, kw_p: float, posthoc: dict):
    """Print a human-readable summary of the statistical tests."""
    sep = '=' * 72
    print(f"\n{sep}")
    print(f"Objective function: {obj_func}")
    print(sep)

    # Descriptive statistics
    print(f"\n{'Algorithm':<14} {'N':>4} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print('-' * 54)
    for alg in ALGORITHMS:
        vals = np.array(data.get(alg, []))
        if len(vals) == 0:
            print(f"  {alg:<12} {'N/A':>4}")
        else:
            print(
                f"  {alg:<12} {len(vals):>4} {np.mean(vals):>10.4f} "
                f"{np.std(vals):>10.4f} {np.median(vals):>10.4f}"
            )

    # Kruskal-Wallis result
    print(f"\nKruskal-Wallis H = {kw_stat:.4f},  p = {kw_p:.6f}  {sig_marker(kw_p)}")

    if kw_p >= 0.05:
        print("  → No statistically significant differences found (α = 0.05).")
        return

    # Post-hoc pairwise comparisons
    print("\nPost-hoc pairwise Mann-Whitney U (Bonferroni-corrected):")
    print(f"  {'Comparison':<30} {'U':>10} {'p_raw':>10} {'p_adj':>10} {'sig':>5}")
    print('  ' + '-' * 60)
    for (a, b), res in posthoc.items():
        label = f"{a} vs {b}"
        marker = sig_marker(res['p_adj'])
        print(
            f"  {label:<30} {res['U']:>10.1f} {res['p_raw']:>10.6f} "
            f"{res['p_adj']:>10.6f} {marker:>5}"
        )
    print(f"\n  Significance: *** p≤0.001  ** p≤0.01  * p≤0.05  ns p>0.05")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results_rows: list, path: str = 'statistical_comparison_results.csv'):
    """Save post-hoc results to a CSV file."""
    import csv
    if not results_rows:
        return
    fieldnames = list(results_rows[0].keys())
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nResults saved to '{path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []

    # Track which objectives are missing data per algorithm for a consolidated warning
    missing_objectives: dict = {alg: [] for alg in ALGORITHMS}

    for obj_func in OBJ_FUNCTIONS:
        data = gather_results(obj_func)

        # Collect missing-data information for each algorithm
        for alg, vals in data.items():
            if len(vals) == 0:
                missing_objectives[alg].append(obj_func)

        # Run tests only if at least 2 algorithms have data
        groups_with_data = {k: v for k, v in data.items() if len(v) > 0}
        if len(groups_with_data) < 2:
            print(f"\n[SKIP] {obj_func}: fewer than 2 algorithms have data.")
            continue

        kw_stat, kw_p = run_kruskal_wallis(groups_with_data)
        posthoc = posthoc_mannwhitney_bonferroni(groups_with_data)

        print_summary(obj_func, data, kw_stat, kw_p, posthoc)

        # Collect rows for CSV
        for (a, b), res in posthoc.items():
            all_rows.append({
                'objective': obj_func,
                'alg_a': a,
                'alg_b': b,
                'U': res['U'],
                'p_raw': res['p_raw'],
                'p_adj_bonferroni': res['p_adj'],
                'significant_0.05': 'yes' if res['p_adj'] <= 0.05 else 'no',
                'kruskal_H': kw_stat,
                'kruskal_p': kw_p,
            })

    # Emit a single consolidated warning per algorithm listing all affected objectives
    for alg, objs in missing_objectives.items():
        if objs:
            print(
                f"[WARNING] No data found for algorithm '{alg}' "
                f"on objective(s): {', '.join(objs)}. Check folder paths.",
                file=sys.stderr,
            )

    save_csv(all_rows)


if __name__ == '__main__':
    main()
