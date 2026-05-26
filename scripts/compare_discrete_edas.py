"""
Compare all discrete EDAs on standard benchmark functions.

Problems:
  1. OneMax     n_vars=20, binary
  2. Trap-3     n_vars=21, binary (7 non-overlapping 3-bit traps)
  3. HIFF       n_vars=16, binary (hierarchical if-and-only-if)

Usage:
    python3.11 scripts/compare_discrete_edas.py
"""

import time
import traceback
import numpy as np

from pateda import (
    UMDA, BMDA, TreeEDA, TreeEDAR, MIMIC, PBIL,
    EBNA, BOA, AffEDA, MKEDA, MTED,
    MNFDA, MNFDAR, MNFDAG, MNFDAGR, MOA,
    FDA, BSC,
)

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def onemax(x):
    """Maximize sum of bits."""
    return float(np.sum(x))


def trap3(x):
    """7 non-overlapping 3-bit deceptive traps (n_vars=21)."""
    x = np.asarray(x, dtype=int)
    total = 0.0
    k = 3
    for start in range(0, len(x), k):
        block = x[start:start + k]
        s = int(np.sum(block))
        if s == k:
            total += k
        else:
            total += k - 1 - s
    return total


def hiff(x):
    """
    Hierarchical If-and-only-If (HIFF), n_vars must be a power of 2.
    H(x) = sum of block contributions across all hierarchical levels.
    """
    x = np.asarray(x, dtype=int)
    n = len(x)
    vals = x.copy().astype(float)
    score = 0.0
    block_size = 2
    while block_size <= n:
        new_vals = np.zeros(n // block_size)
        for i in range(n // block_size):
            block = vals[i * block_size: (i + 1) * block_size]
            if np.all(block == 0) or np.all(block == 1):
                score += block_size
                new_vals[i] = block[0]
            else:
                new_vals[i] = -1  # sentinel: not uniform
        vals = new_vals
        block_size *= 2
    return score


# ---------------------------------------------------------------------------
# Algorithm configurations
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("UMDA",      UMDA),
    ("BMDA",      BMDA),
    ("TreeEDA",   TreeEDA),
    ("TreeEDA-r", TreeEDAR),
    ("MIMIC",     MIMIC),
    ("PBIL",      PBIL),
    ("EBNA",      EBNA),
    ("BOA",       BOA),
    ("AffEDA",    AffEDA),
    ("MKEDA",     MKEDA),
    ("MTED",      MTED),
    ("MNFDA",     MNFDA),
    ("MNFDAR",    MNFDAR),
    ("MNFDAG",    MNFDAG),
    ("MNFDAGR",   MNFDAGR),
    ("MOA",       MOA),
    ("FDA",       FDA),
    ("BSC",       BSC),
]

PROBLEMS = [
    ("OneMax",  onemax,  20, 2),
    ("Trap-3",  trap3,   21, 2),
    ("HIFF",    hiff,    16, 2),
]

POP_SIZE = 200
N_GEN = 50
SEL_RATIO = 0.5
SEED = 42


def run_one(alg_cls, n_vars, cardinality, fitness_func, **kwargs):
    alg = alg_cls(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=SEED,
        **kwargs,
    )
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    best = stats.best_fitness_overall
    mean_last = stats.mean_fitness[-1] if stats.mean_fitness else float('nan')
    return best, mean_last, elapsed


def main():
    header_width = 14
    col_w = 14

    for prob_name, fitness_func, n_vars, card in PROBLEMS:
        print(f"\n{'=' * 70}")
        print(f"Problem: {prob_name}  (n_vars={n_vars}, cardinality={card})")
        print(f"{'=' * 70}")
        print(f"{'Algorithm':<{header_width}} {'Best':>{col_w}} {'MeanLast':>{col_w}} {'Time(s)':>{col_w}}")
        print("-" * (header_width + 3 * col_w + 4))

        for alg_name, alg_cls in ALGORITHMS:
            try:
                best, mean_last, elapsed = run_one(alg_cls, n_vars, card, fitness_func)
                print(f"{alg_name:<{header_width}} {best:>{col_w}.4f} {mean_last:>{col_w}.4f} {elapsed:>{col_w}.2f}")
            except Exception as e:
                print(f"{alg_name:<{header_width}} {'ERROR':>{col_w}} {str(e)[:30]:>{col_w}} {'':>{col_w}}")
                traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
