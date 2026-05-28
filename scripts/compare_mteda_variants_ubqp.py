"""
Compare MTEDA variants against UMDA and TreeEDA on UBQP benchmark instances.

Variants tested:
  - UMDA (baseline, no dependencies)
  - TreeEDA (single tree, uniform weights)
  - MTED-k2   : mixture of 2 trees, uniform weights
  - MTED-k3   : mixture of 3 trees, uniform weights
  - MTED-k5   : mixture of 5 trees, uniform weights
  - MTED-k3-EM: mixture of 3 trees, EM weight learning
  - MTED-k3-FP: mixture of 3 trees, fitness-proportional weights
  - MTED-k3-P : mixture of 3 trees, uniform weights + priors
  - MTED-k3-A : mixture of 3 trees, uniform weights + adaptive learning
  - MTED-k3-PA: mixture of 3 trees, uniform weights + priors + adaptive

Instances:
  - bqp50  (50-variable UBQP)
  - bqp100 (100-variable UBQP)

Usage:
    python scripts/compare_mteda_variants_ubqp.py
"""

import time
import traceback
import functools
import numpy as np

from pateda import UMDA, TreeEDA, MTED
from pateda.functions.discrete.ubqp import (
    evaluate_ubqp,
    load_ubqp_benchmark_instance,
    build_ubqp_interaction_matrix,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

POP_SIZE         = 500
N_GEN            = 50
SEL_RATIO        = 0.15
N_RUNS           = 5
SEEDS            = list(range(1, N_RUNS + 1))
UBQP_THRESHOLD   = 0.5    # threshold_ratio for build_ubqp_interaction_matrix

INSTANCES = ["bqp50", "bqp100"]


# ---------------------------------------------------------------------------
# MTED variant factory
# ---------------------------------------------------------------------------

def _mted(n_trees, weight_learning="uniform", use_priors=False,
          use_adaptive=False, adaptive_mu=0.9):
    """Return an MTED-like constructor with preset variant parameters."""

    @functools.wraps(MTED.__init__)
    def _constructor(n_vars, cardinality, fitness_func,
                     pop_size=100, n_gen=50, selection_ratio=0.5,
                     elitism=True, random_seed=None, **_ignored):
        return MTED(
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            n_trees=n_trees,
            weight_learning=weight_learning,
            use_priors=use_priors,
            use_adaptive=use_adaptive,
            adaptive_mu=adaptive_mu,
            random_seed=random_seed,
        )

    return _constructor


ALGORITHMS = [
    ("UMDA",        UMDA),
    ("TreeEDA",     TreeEDA),
    ("MTED-k2",     _mted(n_trees=2)),
    ("MTED-k3",     _mted(n_trees=3)),
    ("MTED-k5",     _mted(n_trees=5)),
    ("MTED-k3-EM",  _mted(n_trees=3, weight_learning="em")),
    ("MTED-k3-FP",  _mted(n_trees=3, weight_learning="fitness_proportional")),
    ("MTED-k3-P",   _mted(n_trees=3, use_priors=True)),
    ("MTED-k3-A",   _mted(n_trees=3, use_adaptive=True)),
    ("MTED-k3-PA",  _mted(n_trees=3, use_priors=True, use_adaptive=True)),
]


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_ubqp(instance_name: str):
    """Load a UBQP benchmark instance and return (fitness_func, n_vars, optimal)."""
    ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
    interaction_matrix = build_ubqp_interaction_matrix(
        ubqp_instance, threshold_ratio=UBQP_THRESHOLD
    )

    def fitness_func(solution):
        arr = np.asarray(solution)
        val = evaluate_ubqp(arr, ubqp_instance)
        val = np.asarray(val)
        if val.ndim == 0:
            return float(val)
        return float(val[0]) if val.size == 1 else val

    return fitness_func, ubqp_instance.n_vars, optimal


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(alg_name, alg_factory, n_vars, fitness_func, seed):
    alg = alg_factory(
        n_vars=n_vars,
        cardinality=2,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=seed,
    )
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    return float(stats.best_fitness_overall), time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    name_w = 14

    for instance_name in INSTANCES:
        fitness_func, n_vars, optimal = load_ubqp(instance_name)

        print(f"\n{'=' * 88}")
        print(
            f"Problem: UBQP  Instance: {instance_name}  "
            f"(n_vars={n_vars}, optimal={optimal})"
        )
        print(
            f"pop_size={POP_SIZE}  n_gen={N_GEN}  sel_ratio={SEL_RATIO}  "
            f"n_runs={N_RUNS}  seeds={SEEDS}"
        )
        print(f"{'=' * 88}")

        problem_tag = f"UBQP {instance_name}"

        for alg_name, alg_factory in ALGORITHMS:
            bests = []
            times = []
            for seed in SEEDS:
                try:
                    best, elapsed = run_once(alg_name, alg_factory, n_vars, fitness_func, seed)
                    bests.append(best)
                    times.append(elapsed)
                except Exception as exc:
                    print(
                        f"{alg_name:<{name_w}} {problem_tag:<16}: "
                        f"ERROR (seed={seed}) -- {exc}"
                    )
                    traceback.print_exc()
                    bests.append(float("nan"))
                    times.append(float("nan"))

            valid = [b for b in bests if not np.isnan(b)]
            if valid:
                mean_best = float(np.mean(valid))
                mean_time = float(np.nanmean(times))
                bests_str = "[" + ", ".join(f"{b:.1f}" for b in bests) + "]"
                print(
                    f"{alg_name:<{name_w}} {problem_tag:<16}: "
                    f"{bests_str}  mean={mean_best:.2f}  time={mean_time:.2f}s"
                )
            else:
                print(f"{alg_name:<{name_w}} {problem_tag:<16}: ALL RUNS FAILED")

    print()


if __name__ == "__main__":
    main()
