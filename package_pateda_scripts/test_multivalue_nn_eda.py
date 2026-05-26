"""
Test Neural Network EDAs on Multi-value (Non-binary) Benchmarks

This script compares neural network-based Estimation of Distribution Algorithms
(EDAs) that have full support for non-binary (multi-value) discrete problems,
using UMDA as a classical baseline.

Neural network EDAs tested (all have ✅ Full multi-value support per
EDA_Multivalue_Representations.md):
  - Categorical VAE  (Gumbel-Softmax; one-hot over each variable)
  - Categorical GAN  (Gumbel-Softmax over full cardinality)
  - Softmax RBM      (one-hot inputs; softmax visible layer)
  - Discrete Backdrive (embedding layers activated for cardinality > 2)

Baseline (classical probabilistic model):
  - UMDA             (univariate, independence model)

Benchmark functions (cardinality c=3):
  - integer_onemax  – simple convergence test
  - gen_k_decep     – generalized k-deceptive trap (k=3)

Usage
-----
Full benchmark (multiple runs):
    python scripts/test_multivalue_nn_eda.py

Quick smoke test (single run, small problem):
    python scripts/test_multivalue_nn_eda.py --quick
"""

import sys
import argparse
import time
import warnings
import numpy as np
from pathlib import Path

# Allow running as a standalone script from any directory.
# Layout: <project_root>/pateda/scripts/test_multivalue_nn_eda.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Standard EDA framework (used for the UMDA baseline) ──────────────────────
from pateda import EDA, EDAComponents
from pateda.learning import LearnUMDA
from pateda.sampling.fda import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.mutation import (
    FrequencyBalanceMultivalueMutation,
    frequency_balance_multivalue_mutation,
)

# ── Neural network learning / sampling functions ──────────────────────────────
from pateda.learning.discrete_vae import learn_categorical_vae
from pateda.learning.discrete_gan import learn_categorical_gan
from pateda.learning.rbm import learn_softmax_rbm
from pateda.learning.discrete_backdrive import learn_discrete_backdrive

from pateda.sampling.discrete_neural import (
    sample_categorical_vae,
    sample_categorical_gan,
    sample_discrete_backdrive,
)
from pateda.sampling.rbm import sample_softmax_rbm

# ── Benchmark functions ───────────────────────────────────────────────────────
from pateda.functions.discrete.integer_functions import integer_onemax
from pateda.functions.discrete.additive_decomposable import gen_k_decep


# ---------------------------------------------------------------------------
# Benchmark definitions  (fixed to c=3 as required)
# ---------------------------------------------------------------------------

CARDINALITY = 3
N_VARS = 30

BENCHMARKS = {
    "integer_onemax": {
        "fn": integer_onemax,
        "description": "Integer OneMax: sum of all variable values",
        "optimal": lambda n, c: n * (c - 1),
    },
    "gen_k_decep": {
        "fn": lambda x, c: gen_k_decep(x, k=3, cardinality=c),
        "description": "Generalized k-deceptive (k=3): deceptive trap function",
        "optimal": lambda n, c: (n // 3) * (3 * (c - 1)),
    },
}


# ---------------------------------------------------------------------------
# Fitness wrapper
# ---------------------------------------------------------------------------

def make_fitness(fn, cardinality):
    """Return a population-level fitness callable compatible with EDA."""
    def fitness_func(x):
        if x.ndim == 1:
            return np.array([fn(x, cardinality)])
        return np.array([fn(x[i], cardinality) for i in range(x.shape[0])],
                        dtype=float)
    return fitness_func


# ---------------------------------------------------------------------------
# UMDA baseline – uses the standard EDA framework
# ---------------------------------------------------------------------------

def run_umda(bench_name, n_vars, cardinality, pop_size, max_gen, seed=None,
             alpha=0.0):
    """Run one UMDA experiment; returns (best_fitness, n_generations, runtime)."""
    bench = BENCHMARKS[bench_name]
    fitness_func = make_fitness(bench["fn"], cardinality)
    card_array = np.full(n_vars, cardinality)

    mutation = FrequencyBalanceMultivalueMutation(alpha=alpha) if alpha > 0 else None
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnUMDA(alpha=0.01),
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
    stats, _ = eda.run(verbose=False)
    runtime = time.time() - t0

    return float(stats.best_fitness_overall), int(stats.generation_found), runtime


# ---------------------------------------------------------------------------
# Neural network EDA  –  custom generation loop
# ---------------------------------------------------------------------------

# Configuration for each neural EDA.
# Each entry has:
#   learn_fn  – learning function
#   sample_fn – sampling function
#   needs_card_in_sample – True if sample_fn requires cardinality argument
#   learn_params  – default training hyper-parameters
#   description   – human-readable label
NN_EDA_CONFIGS = {
    "vae": {
        "learn_fn": learn_categorical_vae,
        "sample_fn": sample_categorical_vae,
        "needs_card_in_sample": False,
        "learn_params": {
            "epochs": 50,
            "latent_dim": 8,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "description": "Categorical VAE (Gumbel-Softmax)",
    },
    "gan": {
        "learn_fn": learn_categorical_gan,
        "sample_fn": sample_categorical_gan,
        "needs_card_in_sample": False,
        "learn_params": {
            "epochs": 80,
            "latent_dim": 15,
            "batch_size": 32,
            "learning_rate_g": 0.0002,
            "learning_rate_d": 0.0002,
        },
        "description": "Categorical GAN (Gumbel-Softmax)",
    },
    "rbm": {
        "learn_fn": learn_softmax_rbm,
        "sample_fn": sample_softmax_rbm,
        "needs_card_in_sample": True,
        "learn_params": {
            "epochs": 20,
            "n_hidden": 60,
            "k_cd": 1,
            "learning_rate": 0.01,
        },
        "description": "Softmax RBM (one-hot / softmax visible)",
    },
    "backdrive": {
        "learn_fn": learn_discrete_backdrive,
        "sample_fn": sample_discrete_backdrive,
        "needs_card_in_sample": False,
        "learn_params": {
            "epochs": 50,
            "hidden_layers": [64, 32],
            "learning_rate": 0.001,
        },
        "description": "Discrete Backdrive (embedding layers)",
    },
}


def run_nn_eda(eda_name, bench_name, n_vars, cardinality, pop_size,
               max_gen, seed=None, alpha=0.0):
    """
    Run one neural-network EDA experiment.

    Returns (best_fitness, n_generations, runtime).
    """
    if seed is not None:
        np.random.seed(seed)

    bench = BENCHMARKS[bench_name]
    fn = bench["fn"]
    card_array = np.full(n_vars, cardinality)
    cfg = NN_EDA_CONFIGS[eda_name]

    learn_fn = cfg["learn_fn"]
    sample_fn = cfg["sample_fn"]
    needs_card = cfg["needs_card_in_sample"]
    learn_params = dict(cfg["learn_params"])

    # ── Initialise population ────────────────────────────────────────────────
    population = np.stack(
        [np.random.randint(0, cardinality, n_vars) for _ in range(pop_size)]
    )
    fitness = np.array([fn(population[i], cardinality) for i in range(pop_size)],
                       dtype=float)

    best_fitness = float(np.max(fitness))
    generation_found = 0
    n_selected = max(2, pop_size // 2)

    t0 = time.time()
    for gen in range(max_gen):
        # Selection (truncation)
        idx = np.argsort(fitness)[-n_selected:]
        selected = population[idx]
        sel_fitness = fitness[idx]

        # Learn model
        model = learn_fn(selected, sel_fitness, card_array, learn_params)

        # Sample new population
        if needs_card:
            new_pop = sample_fn(model, pop_size, card_array)
        else:
            new_pop = sample_fn(model, pop_size)

        new_pop = np.asarray(new_pop, dtype=int)

        # Apply multi-value frequency balance mutation if alpha > 0
        if alpha > 0:
            mutation_params = {'alpha': alpha}
            new_pop = frequency_balance_multivalue_mutation(
                n_vars, card_array, new_pop, mutation_params
            )

        # Evaluate
        new_fitness = np.array([fn(new_pop[i], cardinality) for i in range(pop_size)],
                                dtype=float)

        population = new_pop
        fitness = new_fitness

        gen_best = float(np.max(fitness))
        if gen_best > best_fitness:
            best_fitness = gen_best
            generation_found = gen + 1

    runtime = time.time() - t0
    return best_fitness, generation_found, runtime


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(eda_name, bench_name, cardinality, n_vars, pop_size,
                   max_gen, seed=None, verbose=True, alpha=0.0):
    """
    Run one EDA on one benchmark.  Returns a result dict.
    """
    bench = BENCHMARKS[bench_name]
    optimal = bench["optimal"](n_vars, cardinality)

    try:
        if eda_name == "umda":
            best, gen_found, runtime = run_umda(
                bench_name, n_vars, cardinality, pop_size, max_gen, seed,
                alpha=alpha)
        else:
            best, gen_found, runtime = run_nn_eda(
                eda_name, bench_name, n_vars, cardinality, pop_size, max_gen,
                seed, alpha=alpha)
    except Exception as exc:
        if verbose:
            print(f"  ERROR: {exc}")
        return {
            "eda": eda_name, "benchmark": bench_name,
            "cardinality": cardinality, "error": str(exc),
        }

    success = np.isclose(best, optimal, rtol=1e-5)

    if verbose:
        succ_str = "✓" if success else "✗"
        print(f"  [{succ_str}] best={best:.2f}  optimal={optimal:.1f}  "
              f"gen={gen_found}  t={runtime:.2f}s")

    return {
        "eda": eda_name,
        "benchmark": bench_name,
        "cardinality": cardinality,
        "n_vars": n_vars,
        "best_fitness": best,
        "optimal": optimal,
        "success": success,
        "generation_found": gen_found,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def run_quick_test():
    """
    Fast smoke test: one run per algorithm × benchmark with small problem size.
    Also tests multi-value frequency balance mutation (alpha=0.95).
    """
    print("=" * 70)
    print("Quick smoke test – neural network multi-value EDAs  (c=3)")
    print("=" * 70)

    quick_n_vars = 9   # divisible by 3 for gen_k_decep
    pop_size = 40
    max_gen = 10
    seed = 42
    n_errors = 0

    eda_names = ["umda"] + list(NN_EDA_CONFIGS.keys())
    bench_names = list(BENCHMARKS.keys())

    # Test without mutation and with mutation (alpha=0.95)
    mutation_configs = [(0.0, "no mutation"), (0.95, "alpha=0.95")]

    for alpha, mut_label in mutation_configs:
        print(f"\n--- Mutation: {mut_label} ---")
        for eda_name in eda_names:
            for bench_name in bench_names:
                label = f"  {eda_name:12s}  {bench_name:20s}  c={CARDINALITY}"
                try:
                    result = run_experiment(
                        eda_name, bench_name, CARDINALITY, quick_n_vars,
                        pop_size, max_gen, seed=seed, verbose=False,
                        alpha=alpha,
                    )
                    if "error" in result:
                        print(f"{label}  *** ERROR: {result['error']}")
                        n_errors += 1
                    else:
                        opt_str = f"{result['optimal']:.1f}"
                        print(f"{label}  best={result['best_fitness']:.2f}  "
                              f"optimal={opt_str}  gen={result['generation_found']}")
                except Exception as exc:
                    print(f"{label}  *** ERROR: {exc}")
                    n_errors += 1

    print()
    if n_errors == 0:
        print("All smoke tests passed.")
    else:
        print(f"FAILURES: {n_errors} error(s) detected.")
    return n_errors == 0


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark(n_runs=5, pop_size=200, max_gen=100, verbose=True,
                       alpha=0.0):
    """
    Run a full multi-run benchmark comparing neural-network EDAs with UMDA.

    Results are printed as mean best fitness ± std and success rate.

    Parameters
    ----------
    n_runs : int
        Number of independent runs per configuration.
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
    cardinality = CARDINALITY
    n_vars = N_VARS

    eda_names = ["umda"] + list(NN_EDA_CONFIGS.keys())
    bench_names = list(BENCHMARKS.keys())

    print("=" * 70)
    print("Neural Network Multi-value EDA Benchmark  (c=3)")
    print("=" * 70)
    print(f"EDAs       : {eda_names}")
    print(f"Benchmarks : {bench_names}")
    print(f"Cardinality: {cardinality}")
    print(f"n_vars     : {n_vars}")
    print(f"Runs       : {n_runs}")
    print(f"Pop size   : {pop_size}")
    print(f"Max gen    : {max_gen}")
    print(f"Alpha (mut): {alpha}")
    print()

    all_results = []
    total = len(eda_names) * len(bench_names) * n_runs
    done = 0

    for eda_name in eda_names:
        for bench_name in bench_names:
            for run in range(n_runs):
                done += 1
                seed = hash((cardinality, eda_name, bench_name, run)) % (2 ** 31)
                if verbose:
                    eda_desc = (NN_EDA_CONFIGS[eda_name]["description"]
                                if eda_name != "umda"
                                else "UMDA (independence model)")
                    print(f"[{done}/{total}] {eda_name:12s} | "
                          f"{bench_name:20s} | run={run + 1}", end="  ")

                result = run_experiment(
                    eda_name, bench_name, cardinality, n_vars,
                    pop_size, max_gen, seed=seed, verbose=verbose,
                    alpha=alpha,
                )
                all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary  (mean best fitness ± std | success rate)")
    print("=" * 70)

    for bench_name in bench_names:
        print(f"\nBenchmark: {bench_name}  "
              f"[{BENCHMARKS[bench_name]['description']}]")
        print(f"  {'EDA':12s}  {'mean':>8s}  {'std':>6s}  "
              f"{'optimal':>8s}  {'success':>8s}")
        print("  " + "-" * 52)
        for eda_name in eda_names:
            runs = [r for r in all_results
                    if r.get("eda") == eda_name
                    and r.get("benchmark") == bench_name
                    and "error" not in r]
            if not runs:
                print(f"  {eda_name:12s}  (all runs failed)")
                continue
            best_vals = [r["best_fitness"] for r in runs]
            opt = runs[0]["optimal"]
            succ = sum(r["success"] for r in runs) / len(runs) * 100
            print(f"  {eda_name:12s}  {np.mean(best_vals):8.2f}  "
                  f"{np.std(best_vals):6.2f}  "
                  f"{opt:8.1f}  {succ:7.0f}%")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare neural network EDAs on multi-value benchmarks "
                    "(cardinality=3, integer_onemax and gen_k_decep)."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick smoke test instead of the full benchmark.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of independent runs per configuration (default: 5).",
    )
    parser.add_argument(
        "--pop-size", type=int, default=200,
        help="Population size (default: 200).",
    )
    parser.add_argument(
        "--max-gen", type=int, default=100,
        help="Maximum generations (default: 100).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Max frequency threshold for multi-value mutation (default: 0.0, "
             "no mutation). If > 0, applies FrequencyBalanceMultivalueMutation.",
    )
    args = parser.parse_args()

    # Suppress PyTorch / numpy noise that is not relevant to correctness
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        run_full_benchmark(
            n_runs=args.n_runs,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
