"""
Compare all permutation EDAs on proxy benchmark functions.

Problems (all MAXIMIZED):
  1. QAP proxy   maximize -kendall_distance(perm, identity)
  2. LOP proxy   maximize sum(perm[i] * i)
  3. TSP proxy   maximize -(sum of |perm[i] - perm[i+1]| for consecutive pairs)

Usage:
    python3.11 scripts/compare_permutation_edas.py
"""

import time
import traceback
import numpy as np

from pateda import (
    EHMEDA, NHMEDA,
    MallowsKendallEDA, MallowsCayleyEDA,
    GMallowsKendallEDA, GMallowsCayleyEDA,
)

N_VARS = 12

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def _kendall_distance(perm1, perm2):
    """Number of pairwise inversions between perm1 and perm2."""
    n = len(perm1)
    p2inv = np.argsort(perm2)
    rel = p2inv[perm1]
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rel[i] > rel[j]:
                dist += 1
    return dist


_identity = np.arange(N_VARS)


def qap_proxy(perm):
    """Maximize -kendall_distance(perm, identity).  Optimum=0 at identity."""
    return -float(_kendall_distance(np.asarray(perm, dtype=int), _identity))


def lop_proxy(perm):
    """Maximize sum(perm[i]*i).  Favours identity ordering."""
    perm = np.asarray(perm, dtype=float)
    weights = np.arange(len(perm), dtype=float)
    return float(np.dot(perm, weights))


def tsp_proxy(perm):
    """Maximize -(sum of |perm[i] - perm[i+1]|).  Favours smooth sequences."""
    perm = np.asarray(perm, dtype=float)
    return -float(np.sum(np.abs(np.diff(perm))))


# ---------------------------------------------------------------------------
# Algorithm list
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("EHM-EDA",             EHMEDA),
    ("NHM-EDA",             NHMEDA),
    ("Mallows-Kendall",     MallowsKendallEDA),
    ("Mallows-Cayley",      MallowsCayleyEDA),
    ("GMallows-Kendall",    GMallowsKendallEDA),
    ("GMallows-Cayley",     GMallowsCayleyEDA),
]

PROBLEMS = [
    ("QAP-proxy", qap_proxy),
    ("LOP-proxy", lop_proxy),
    ("TSP-proxy", tsp_proxy),
]

POP_SIZE = 100
N_GEN = 50
SEL_RATIO = 0.5
SEED = 42


def run_one(alg_cls, n_vars, fitness_func):
    alg = alg_cls(
        n_vars=n_vars,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=SEED,
    )
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    best = stats.best_fitness_overall
    mean_last = stats.mean_fitness[-1] if stats.mean_fitness else float('nan')
    return best, mean_last, elapsed


def main():
    header_width = 18
    col_w = 14

    for prob_name, fitness_func in PROBLEMS:
        print(f"\n{'=' * 68}")
        print(f"Problem: {prob_name}  (n_vars={N_VARS})")
        print(f"{'=' * 68}")
        print(f"{'Algorithm':<{header_width}} {'Best':>{col_w}} {'MeanLast':>{col_w}} {'Time(s)':>{col_w}}")
        print("-" * (header_width + 3 * col_w + 4))

        for alg_name, alg_cls in ALGORITHMS:
            try:
                best, mean_last, elapsed = run_one(alg_cls, N_VARS, fitness_func)
                print(f"{alg_name:<{header_width}} {best:>{col_w}.4f} {mean_last:>{col_w}.4f} {elapsed:>{col_w}.2f}")
            except Exception as e:
                print(f"{alg_name:<{header_width}} {'ERROR':>{col_w}} {str(e)[:30]:>{col_w}} {'':>{col_w}}")
                traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
