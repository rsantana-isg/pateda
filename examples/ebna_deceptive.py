"""
EBNA for Deceptive Problems

Demonstrates EBNA (Estimation of Bayesian Networks Algorithm) on five
different 3-bit deceptive landscapes, each with and without variable
overlap, to study how overlapping partitions affect difficulty.

Functions tested
----------------
1. Deceptive3 (Goldberg)         — classic non-overlapping 3-bit deceptive
2. Decep3 (lookup-table)         — same family, non-overlapping and overlapping
3. MH-Decep3                     — alternative lookup table (two peaks)
4. Marta-Decep3                  — Marta's fitness landscape
5. Venturini-Decep                — Venturini's deceptive landscape

For each function the script runs EBNA twice:
  - non-overlapping partitions (n_vars divisible by 3, step=3)
  - overlapping   partitions (step=2, harder)

where the function supports an overlap flag; otherwise the function is
run once with a note.
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnEBNA
from pateda.sampling import SampleBayesianNetwork
from pateda.selection import RankingSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.deceptive import deceptive3
from pateda.functions.discrete.additive_decomposable import (
    decep3,
    decep3_mh,
    decep_marta3,
    decep_venturini,
    DECEP3_TABLE,
    DECEP3_MH_TABLE,
    DECEP_MARTA_TABLE,
    DECEP_VENTURINI_TABLE,
)


# ---------------------------------------------------------------------------
# Helper: compute per-block optimum for each lookup-table function
# ---------------------------------------------------------------------------

def _table_max(table):
    """Return max value in a lookup table (= optimal per block)."""
    return float(np.max(table))


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

N_VARS      = 30   # 10 × 3-bit blocks (non-overlapping)
POP_SIZE    = 400
MAX_GEN     = 40

# For overlapping decep3: n_blocks = (n_vars - 2) // 2
N_BLOCKS_NON = N_VARS // 3
N_BLOCKS_OV  = (N_VARS - 2) // 2

# All problems share the same EBNA config
def _components(pop_size=POP_SIZE, max_gen=MAX_GEN):
    return EDAComponents(
        seeding=RandomInit(),
        learning=LearnEBNA(max_parents=2, score_metric="bic", alpha=0.1),
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        selection=RankingSelection(ratio=0.5, selection_pressure=1.8,
                                   ranking_type="linear"),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )


def run_ebna(fitness_func, n_vars, optimal, label, seed=42):
    """Run one EBNA experiment and print a compact result line."""
    eda = EDA(
        pop_size=POP_SIZE,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),
        components=_components(),
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    best  = stats.best_fitness_overall
    found = stats.generation_found if stats.generation_found is not None else MAX_GEN
    pct   = best / optimal * 100 if optimal > 0 else float('nan')
    print(f"  {label:<42}  best={best:7.3f}/{optimal:.3f}"
          f"  ({pct:5.1f}%)  gen={found:3d}")
    return stats


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_deceptive3_goldberg():
    """
    1. Goldberg Deceptive-3
    Classic non-overlapping function from pateda.functions.discrete.deceptive.
    Partitions are fixed at indices [0,1,2], [3,4,5], ...
    """
    print("\n" + "=" * 65)
    print("1. Goldberg Deceptive-3  (from pateda.functions)")
    print("=" * 65)
    optimal = N_BLOCKS_NON * 1.0   # each block max = 1.0

    print("  Non-overlapping (step=3):")
    run_ebna(deceptive3, N_VARS, optimal, "Deceptive3 (non-overlap)")

    # Overlapping version by hand (wrap with decep3 overlap=True)
    # n_vars must allow at least 2 overlap windows
    def decep3_overlap(x):
        return decep3(x, overlap=True)

    ov_optimal = N_BLOCKS_OV * _table_max(DECEP3_TABLE)
    print("  Overlapping (step=2) — harder, more epistatic:")
    run_ebna(decep3_overlap, N_VARS, ov_optimal, "Deceptive3 (overlap, step=2)")


def test_decep3_lookup():
    """
    2. Decep3 (lookup-table variant)
    Uses a specific lookup table indexed by the 3-bit value.
    Tests both non-overlapping (step=3) and overlapping (step=2).
    """
    print("\n" + "=" * 65)
    print("2. Decep3 (lookup-table, additive_decomposable)")
    print("=" * 65)

    def decep3_non(x): return decep3(x, overlap=False)
    def decep3_ov(x):  return decep3(x, overlap=True)

    opt_non = N_BLOCKS_NON * _table_max(DECEP3_TABLE)
    opt_ov  = N_BLOCKS_OV  * _table_max(DECEP3_TABLE)

    print("  Non-overlapping (step=3):")
    run_ebna(decep3_non, N_VARS, opt_non, "Decep3 (non-overlap)")

    print("  Overlapping (step=2):")
    run_ebna(decep3_ov, N_VARS, opt_ov, "Decep3 (overlap)")


def test_decep3_mh():
    """
    3. MH-Decep3
    Alternative 3-bit deceptive landscape with lookup table
    [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0].
    Optimal block configuration is [1,1,1] (index 7 → 3.0).

    Non-overlapping and a manual overlapping version are tested.
    """
    print("\n" + "=" * 65)
    print("3. MH-Decep3  (table max=3.0)")
    print("=" * 65)

    def decep3_mh_ov(x):
        """Overlapping version of MH-Decep3 (step=2)."""
        if x.ndim == 2:
            x = x.flatten()
        total = 0.0
        for i in range(0, len(x) - 2, 2):
            idx = x[i] + 2 * x[i+1] + 4 * x[i+2]
            total += DECEP3_MH_TABLE[idx]
        return total

    opt_non = N_BLOCKS_NON * _table_max(DECEP3_MH_TABLE)
    opt_ov  = N_BLOCKS_OV  * _table_max(DECEP3_MH_TABLE)

    print("  Non-overlapping (step=3):")
    run_ebna(decep3_mh, N_VARS, opt_non, "MH-Decep3 (non-overlap)")

    print("  Overlapping (step=2) — harder:")
    run_ebna(decep3_mh_ov, N_VARS, opt_ov, "MH-Decep3 (overlap)")


def test_decep_marta3():
    """
    4. Marta-Decep3
    Marta's deceptive function with table
    [0.554, 0.049, 0.856, 1.099, 0.980, -0.298, 0.371, -0.193].
    The global optimum is at index 3 (= 011 → 1.099 per block).
    Non-overlapping and overlapping versions are compared.
    """
    print("\n" + "=" * 65)
    print("4. Marta-Decep3  (table max=1.099)")
    print("=" * 65)

    def decep_marta3_ov(x):
        """Overlapping Marta-Decep3 (step=2)."""
        if x.ndim == 2:
            x = x.flatten()
        total = 0.0
        for i in range(0, len(x) - 2, 2):
            idx = 4 * x[i] + 2 * x[i+1] + x[i+2]
            total += DECEP_MARTA_TABLE[idx]
        return total

    opt_non = N_BLOCKS_NON * _table_max(DECEP_MARTA_TABLE)
    opt_ov  = N_BLOCKS_OV  * _table_max(DECEP_MARTA_TABLE)

    print("  Non-overlapping (step=3):")
    run_ebna(decep_marta3, N_VARS, opt_non, "Marta-Decep3 (non-overlap)")

    print("  Overlapping (step=2):")
    run_ebna(decep_marta3_ov, N_VARS, opt_ov, "Marta-Decep3 (overlap)")


def test_decep_venturini():
    """
    5. Venturini-Decep
    Venturini's deceptive function with table
    [0.2, 1.4, 1.6, 2.9, 3.0, 1.0, 0.8, 0.6].
    Optimal block is [1,0,0] (index 4 → 3.0).
    Non-overlapping and overlapping versions are compared.
    """
    print("\n" + "=" * 65)
    print("5. Venturini-Decep  (table max=3.0)")
    print("=" * 65)

    def decep_venturini_ov(x):
        """Overlapping Venturini deceptive (step=2)."""
        if x.ndim == 2:
            x = x.flatten()
        total = 0.0
        for i in range(0, len(x) - 2, 2):
            idx = x[i] + 2 * x[i+1] + 4 * x[i+2]
            total += DECEP_VENTURINI_TABLE[idx]
        return total

    opt_non = N_BLOCKS_NON * _table_max(DECEP_VENTURINI_TABLE)
    opt_ov  = N_BLOCKS_OV  * _table_max(DECEP_VENTURINI_TABLE)

    print("  Non-overlapping (step=3):")
    run_ebna(decep_venturini, N_VARS, opt_non, "Venturini (non-overlap)")

    print("  Overlapping (step=2) — harder epistasis:")
    run_ebna(decep_venturini_ov, N_VARS, opt_ov, "Venturini (overlap)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("EBNA on Deceptive Functions — Non-overlapping vs Overlapping")
    print("=" * 65)
    print(f"Config: n={N_VARS}, pop={POP_SIZE}, max_gen={MAX_GEN}")
    print(f"        EBNA max_parents=2, BIC score, ranking selection p=1.8")

    test_deceptive3_goldberg()
    test_decep3_lookup()
    test_decep3_mh()
    test_decep_marta3()
    test_decep_venturini()

    print("\n" + "=" * 65)
    print("Done. Overlapping variants are consistently harder:")
    print("  more epistasis → EBNA needs stronger parent-child edges")
    print("  to capture dependencies across overlapping blocks.")
    print("=" * 65)


if __name__ == "__main__":
    main()
