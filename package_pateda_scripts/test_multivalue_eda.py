"""
Test Multi-value (Non-binary) EDA Implementations

This script tests Estimation of Distribution Algorithms (EDAs) on benchmark
functions with non-binary (cardinality > 2) discrete representations.

Algorithms tested:
  - UMDA  (univariate, independence model)
  - Tree-EDA (tree-structured pairwise dependencies)
  - MN-FDA  (Markov network factorized distribution)

Benchmark functions used (all support arbitrary cardinality):
  - Integer OneMax           – simple maximization, tests basic convergence
  - Generalized k-deceptive  – deceptive trap, tests escape from local optima
  - Integer Max Blocks       – block-level dependencies
  - Integer Multi-level Trap – multi-level deceptive function
  - Integer Dependency Chain – sequential pairwise dependencies

Usage
-----
Full benchmark (multiple cardinalities, multiple runs):
    python scripts/test_multivalue_eda.py

Quick smoke test (single run, small problem):
    python scripts/test_multivalue_eda.py --quick
"""

import sys
import argparse
import time
import warnings
import numpy as np

from pateda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.learning.mnfda import LearnMNFDA
from pateda.sampling.fda import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.mutation import FrequencyBalanceMultivalueMutation

from pateda.functions.discrete.integer_functions import (
    integer_onemax,
    integer_max_blocks,
    integer_dependency_chain,
    integer_multi_level_trap,
)
from pateda.functions.discrete.additive_decomposable import gen_k_decep


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "integer_onemax": {
        "fn": integer_onemax,
        "n_vars": 30,
        "optimal": lambda n, c: n * (c - 1),
        "description": "Integer OneMax: sum of all variable values",
    },
    "gen_k_decep": {
        "fn": lambda x, c: gen_k_decep(x, k=3, cardinality=c),
        "n_vars": 30,
        "optimal": lambda n, c: (n // 3) * (3 * (c - 1)),
        "description": "Generalized k-deceptive (k=3): deceptive trap function",
    },
    "integer_max_blocks": {
        "fn": lambda x, c: integer_max_blocks(x, k=3, cardinality=c),
        "n_vars": 30,
        "optimal": lambda n, c: (n // 3) * (3 * (c - 1) + 3),
        "description": "Integer Max Blocks (k=3): block-level dependencies",
    },
    "integer_multi_level_trap": {
        "fn": lambda x, c: integer_multi_level_trap(x, k=3, cardinality=c),
        "n_vars": 30,
        "optimal": lambda n, c: (n // 3) * (3 * (c - 1) + 3),
        "description": "Integer Multi-level Trap (k=3): multi-level deceptive",
    },
    "integer_dependency_chain": {
        "fn": integer_dependency_chain,
        "n_vars": 30,
        "optimal": None,   # complex formula; skip success check
        "description": "Integer Dependency Chain: sequential pairwise dependencies",
    },
}

QUICK_BENCHMARKS = {
    "integer_onemax": BENCHMARKS["integer_onemax"],
    "gen_k_decep": BENCHMARKS["gen_k_decep"],
}

EDA_CONFIGS = {
    "umda": {
        "learning": lambda: LearnUMDA(alpha=0.01),
        "description": "UMDA (independence model)",
    },
    "tree_eda": {
        "learning": lambda: LearnTreeModel(alpha=0.01, mi_threshold=0.01),
        "description": "Tree-EDA (pairwise tree dependencies)",
    },
    "mnfda": {
        "learning": lambda: LearnMNFDA(max_clique_size=3, threshold=0.05,
                                       prior=True, return_factorized=True),
        "description": "MN-FDA (Markov network factorization)",
    },
}


# ---------------------------------------------------------------------------
# Helper: wrap a benchmark function for the EDA framework
# ---------------------------------------------------------------------------

def make_fitness(fn, cardinality):
    """Return a population-level fitness function compatible with EDA."""
    def fitness_func(x):
        if x.ndim == 1:
            return np.array([fn(x, cardinality)])
        return np.array([fn(x[i], cardinality) for i in range(x.shape[0])],
                        dtype=float)
    return fitness_func


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(eda_name, bench_name, cardinality, pop_size, max_gen,
                   seed=None, verbose=True, alpha=0.0):
    """
    Run one EDA on one benchmark function.

    Returns a dict with keys: eda, benchmark, cardinality, n_vars, best_fitness,
    optimal, success, generation_found, runtime_s.
    """
    bench = BENCHMARKS[bench_name]
    fn = bench["fn"]
    n_vars = bench["n_vars"]

    optimal = (bench["optimal"](n_vars, cardinality)
               if bench["optimal"] is not None else None)

    fitness_func = make_fitness(fn, cardinality)
    card_array = np.full(n_vars, cardinality)

    eda_cfg = EDA_CONFIGS[eda_name]
    mutation = FrequencyBalanceMultivalueMutation(alpha=alpha) if alpha > 0 else None
    components = EDAComponents(
        seeding=RandomInit(),
        learning=eda_cfg["learning"](),
        sampling=SampleFDA(n_samples=pop_size),
        selection=TruncationSelection(ratio=0.5),
        stop_condition=MaxGenerations(max_gen=max_gen),
        mutation=mutation,
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=card_array,
        components=components,
        random_seed=seed,
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("error")   # fail loudly on any RuntimeWarning
        stats, _ = eda.run(verbose=False)
    runtime = time.time() - t0

    best = float(stats.best_fitness_overall)
    success = (optimal is not None and
               np.isclose(best, optimal, rtol=1e-5))

    if verbose:
        opt_str = f"{optimal:.1f}" if optimal is not None else "N/A"
        succ_str = "✓" if success else ("✗" if optimal is not None else "-")
        print(f"  [{succ_str}] best={best:.2f}  optimal={opt_str}  "
              f"gen={stats.generation_found}  t={runtime:.2f}s")

    return {
        "eda": eda_name,
        "benchmark": bench_name,
        "cardinality": cardinality,
        "n_vars": n_vars,
        "best_fitness": best,
        "optimal": optimal,
        "success": success,
        "generation_found": int(stats.generation_found),
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def run_quick_test():
    """
    Fast smoke test: one run per algorithm × benchmark × cardinality.
    Uses small problem sizes for speed. Verifies that no RuntimeWarning
    (divide-by-zero, etc.) is raised.
    Also tests multi-value frequency balance mutation (alpha=0.95).
    """
    print("=" * 70)
    print("Quick smoke test – non-binary EDA implementations")
    print("=" * 70)

    # Use small problems for speed (n_vars=9, divisible by 3 for block functions)
    QUICK_N_VARS = 9
    cardinalities = [3, 4]
    pop_size = 50
    max_gen = 20
    seed = 42
    n_errors = 0

    quick_benchmarks_list = list(QUICK_BENCHMARKS.keys())
    quick_eda_list = ["umda", "tree_eda"]  # Skip slow MNFDA for quick test

    # Test without mutation and with mutation (alpha=0.95)
    mutation_configs = [(0.0, "no mutation"), (0.95, "alpha=0.95")]

    for card in cardinalities:
        print(f"\nCardinality = {card}")
        for eda_name in quick_eda_list:
            for bench_name in quick_benchmarks_list:
                for alpha, mut_label in mutation_configs:
                    label = (f"  {eda_name:12s}  {bench_name:30s}  "
                             f"c={card}  mut={mut_label}")
                    try:
                        bench = BENCHMARKS[bench_name]
                        fn = bench["fn"]
                        optimal = (bench["optimal"](QUICK_N_VARS, card)
                                   if bench["optimal"] is not None else None)
                        fitness_func = make_fitness(fn, card)
                        card_array = np.full(QUICK_N_VARS, card)

                        mutation = (FrequencyBalanceMultivalueMutation(alpha=alpha)
                                    if alpha > 0 else None)
                        eda_cfg = EDA_CONFIGS[eda_name]
                        components = EDAComponents(
                            seeding=RandomInit(),
                            learning=eda_cfg["learning"](),
                            sampling=SampleFDA(n_samples=pop_size),
                            selection=TruncationSelection(ratio=0.5),
                            stop_condition=MaxGenerations(max_gen=max_gen),
                            mutation=mutation,
                        )
                        eda = EDA(
                            pop_size=pop_size,
                            n_vars=QUICK_N_VARS,
                            fitness_func=fitness_func,
                            cardinality=card_array,
                            components=components,
                            random_seed=seed,
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("error")
                            stats, _ = eda.run(verbose=False)

                        best = float(stats.best_fitness_overall)
                        opt_str = f"{optimal:.1f}" if optimal is not None else "N/A"
                        print(f"{label}  best={best:.2f}  optimal={opt_str}  "
                              f"gen={stats.generation_found}")
                    except RuntimeWarning as e:
                        print(f"{label}  *** RuntimeWarning: {e}")
                        n_errors += 1
                    except Exception as e:
                        print(f"{label}  *** ERROR: {e}")
                        n_errors += 1

    print()
    if n_errors == 0:
        print("All smoke tests passed – no RuntimeWarnings or exceptions.")
    else:
        print(f"FAILURES: {n_errors} error(s) detected.")
    return n_errors == 0


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark(n_runs=5, cardinalities=None, pop_size=200,
                       max_gen=100, verbose=True, alpha=0.0):
    """
    Run a comprehensive multi-value EDA benchmark.

    Parameters
    ----------
    n_runs : int
        Number of independent runs per configuration.
    cardinalities : list of int
        List of cardinality values to test.
    pop_size : int
        Population size.
    max_gen : int
        Maximum generations.
    verbose : bool
        Print progress.
    alpha : float
        Maximum allowed frequency threshold for multi-value mutation
        (default 0.0, no mutation). If alpha > 0, applies
        FrequencyBalanceMultivalueMutation after each sampling step.
    """
    if cardinalities is None:
        cardinalities = [3, 4, 5]

    print("=" * 70)
    print("Full Multi-value EDA Benchmark")
    print("=" * 70)
    print(f"Cardinalities : {cardinalities}")
    print(f"EDAs          : {list(EDA_CONFIGS.keys())}")
    print(f"Benchmarks    : {list(BENCHMARKS.keys())}")
    print(f"Runs          : {n_runs}")
    print(f"Pop size      : {pop_size}")
    print(f"Max gen       : {max_gen}")
    print(f"Alpha (mut)   : {alpha}")
    print()

    all_results = []
    total = (len(cardinalities) * len(EDA_CONFIGS) * len(BENCHMARKS) * n_runs)
    done = 0

    for card in cardinalities:
        for eda_name in EDA_CONFIGS:
            for bench_name in BENCHMARKS:
                successes = []
                best_list = []
                for run in range(n_runs):
                    done += 1
                    seed = hash((card, eda_name, bench_name, run)) % (2 ** 31)
                    if verbose:
                        print(f"[{done}/{total}] "
                              f"{eda_name:10s} | {bench_name:30s} | "
                              f"c={card} | run={run+1}", end="  ")
                    try:
                        r = run_experiment(eda_name, bench_name, card,
                                           pop_size, max_gen, seed=seed,
                                           verbose=verbose, alpha=alpha)
                        all_results.append(r)
                        successes.append(r["success"])
                        best_list.append(r["best_fitness"])
                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {e}")
                        all_results.append({
                            "eda": eda_name, "benchmark": bench_name,
                            "cardinality": card, "error": str(e),
                        })

    # Print summary
    print("\n" + "=" * 70)
    print("Summary (mean best fitness ± std, success rate)")
    print("=" * 70)

    from itertools import product as iproduct
    for card, eda_name, bench_name in iproduct(cardinalities, EDA_CONFIGS,
                                               BENCHMARKS):
        runs = [r for r in all_results
                if r.get("eda") == eda_name
                and r.get("benchmark") == bench_name
                and r.get("cardinality") == card
                and "error" not in r]
        if not runs:
            continue
        best_vals = [r["best_fitness"] for r in runs]
        opt = runs[0]["optimal"]
        succ_rate = (sum(r["success"] for r in runs) / len(runs) * 100
                     if opt is not None else float("nan"))
        opt_str = f"{opt:.1f}" if opt is not None else "N/A"
        print(f"  c={card}  {eda_name:10s}  {bench_name:30s}  "
              f"mean={np.mean(best_vals):.2f}±{np.std(best_vals):.2f}  "
              f"optimal={opt_str}  "
              f"success={succ_rate:.0f}%")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test non-binary EDA implementations on multi-value benchmarks."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick smoke test instead of the full benchmark."
    )
    parser.add_argument(
        "--cardinalities", nargs="+", type=int, default=[3, 4, 5],
        help="List of cardinalities to test (default: 3 4 5)."
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of independent runs per configuration (default: 5)."
    )
    parser.add_argument(
        "--pop-size", type=int, default=200,
        help="Population size (default: 200)."
    )
    parser.add_argument(
        "--max-gen", type=int, default=100,
        help="Maximum generations (default: 100)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Max frequency threshold for multi-value mutation (default: 0.0, "
             "no mutation). If > 0, applies FrequencyBalanceMultivalueMutation.",
    )
    args = parser.parse_args()

    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        run_full_benchmark(
            n_runs=args.n_runs,
            cardinalities=args.cardinalities,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
