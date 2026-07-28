"""
Computational-time profile of the MN-FDA algorithm.

Reproduces the MN-FDA configuration used in
``slurm/launch_weighted_pbo_experiments.py`` (the weighted-PBO study, where
MN-FDA was one of the best performers):

    MNFDA(max_clique_size=3, return_factorized=True)   # chi-square, PLS sampling
    pop_size=200, n_gen=50, selection_ratio=0.5 (truncation), binary vars
    customized-selection weighting FP (proportional, beta=1) by default

and times the three top-level components (initialization, learning, sampling)
plus a breakdown of the learning step into its sub-operations
(MI matrix, dependency graph / chi-square test, clique finding, clique
ordering, structure build, probability tables).

The goal is to see which components drive the growth of the per-generation cost
as the problem dimension increases, using Deceptive3 at n in {30,60,90,120}.

Usage (all positional, all optional):
    python3.11 scripts/profile_mnfda.py [seed] [sizes] [pop_size] [n_gen] [weighting]

    seed       RNG seed                                  (default 1)
    sizes      comma-separated dimensions                (default 30,60,90,120)
    pop_size   population size                           (default 200)
    n_gen      number of generations                    (default 50)
    weighting  uniform|proportional|boltzmann           (default proportional = FP)

Writes a CSV next to the analysis document and prints a summary table.
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, os.pardir, "src"))

from pateda.algorithms.discrete import MNFDA
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
import pateda.learning.mnfda as MNFDA_MOD


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
class Timers:
    """Accumulate wall-clock time and call counts per named key."""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def wrap(self, key, fn):
        def wrapped(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[key] += time.perf_counter() - t0
                self.n[key] += 1
        return wrapped


def profile_one(n_vars, pop_size, n_gen, seed, weighting):
    """Run one MN-FDA instance with instrumentation; return a timings dict."""
    timers = Timers()

    # --- patch the learning sub-operations (module-level names in mnfda) ---
    # MN-FDA now uses the vectorized kernels (proposals A/B/C/E); the profiler
    # wraps those so the breakdown still reflects what the learner actually runs.
    orig = {
        "mi": MNFDA_MOD.compute_mi_matrix_fast,
        "chi2": MNFDA_MOD.chi2_adjacency,
        "cliques": MNFDA_MOD.find_maximal_cliques_greedy,
        "order": MNFDA_MOD.order_cliques_for_sampling_fast,
        "structure": MNFDA_MOD.convert_cliques_to_factorized_structure,
        "tables": MNFDA_MOD.compute_clique_tables_fast,
    }
    MNFDA_MOD.compute_mi_matrix_fast = timers.wrap(
        "learn.mi_matrix", orig["mi"])
    MNFDA_MOD.chi2_adjacency = timers.wrap(
        "learn.graph_chi2", orig["chi2"])
    MNFDA_MOD.find_maximal_cliques_greedy = timers.wrap(
        "learn.clique_find", orig["cliques"])
    MNFDA_MOD.order_cliques_for_sampling_fast = timers.wrap(
        "learn.clique_order", orig["order"])
    MNFDA_MOD.convert_cliques_to_factorized_structure = timers.wrap(
        "learn.structure", orig["structure"])
    MNFDA_MOD.compute_clique_tables_fast = timers.wrap(
        "learn.prob_tables", orig["tables"])

    try:
        # Build the exact wrapper used by the weighted-PBO launcher.
        alg = MNFDA(
            n_vars=n_vars, cardinality=2, fitness_func=deceptive3,
            pop_size=pop_size, n_gen=n_gen, selection_ratio=0.5,
            random_seed=seed,
        )
        alg.set_weighting(weighting, beta=1.0)
        eda = alg._eda

        # Wrap top-level components.
        eda.components.learning.learn = timers.wrap(
            "learn.TOTAL", eda.components.learning.learn)
        eda.components.sampling.sample = timers.wrap(
            "sample.TOTAL", eda.components.sampling.sample)
        eda.components.seeding.seed = timers.wrap(
            "init.seed", eda.components.seeding.seed)

        # Split fitness evaluation into the gen-0 (initialization) call and the
        # per-generation calls.
        eval_state = {"first": True}
        _orig_eval = eda.evaluate_fitness

        def timed_eval(pop):
            key = "init.eval" if eval_state["first"] else "eval.offspring"
            eval_state["first"] = False
            t0 = time.perf_counter()
            try:
                return _orig_eval(pop)
            finally:
                timers.t[key] += time.perf_counter() - t0
                timers.n[key] += 1

        eda.evaluate_fitness = timed_eval

        t0 = time.perf_counter()
        eda.run(verbose=False)
        total = time.perf_counter() - t0

    finally:
        # restore module globals so repeated sizes don't stack wrappers
        MNFDA_MOD.compute_mi_matrix_fast = orig["mi"]
        MNFDA_MOD.chi2_adjacency = orig["chi2"]
        MNFDA_MOD.find_maximal_cliques_greedy = orig["cliques"]
        MNFDA_MOD.order_cliques_for_sampling_fast = orig["order"]
        MNFDA_MOD.convert_cliques_to_factorized_structure = orig["structure"]
        MNFDA_MOD.compute_clique_tables_fast = orig["tables"]

    return timers, total


LEARN_SUBKEYS = [
    "learn.mi_matrix", "learn.graph_chi2", "learn.clique_find",
    "learn.clique_order", "learn.structure", "learn.prob_tables",
]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sizes = ([int(s) for s in sys.argv[2].split(",")]
             if len(sys.argv) > 2 else [30, 60, 90, 120])
    pop_size = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    weighting = sys.argv[5] if len(sys.argv) > 5 else "proportional"

    print(f"Seed:             {seed}")
    print(f"Algorithm:        MN-FDA (max_clique_size=3, chi2, PLS sampling)")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    print(f"Selection ratio:  0.5 (truncation)")
    print(f"Weighting:        {weighting}")
    print(f"Sizes:            {sizes}")
    print(f"Problem:          Deceptive3\n")

    rows = []
    for n in sizes:
        timers, total = profile_one(n, pop_size, n_gen, seed, weighting)
        init = timers.t["init.seed"] + timers.t["init.eval"]
        learn = timers.t["learn.TOTAL"]
        sample = timers.t["sample.TOTAL"]
        eval_off = timers.t["eval.offspring"]
        overhead = total - init - learn - sample - eval_off
        row = {
            "n_vars": n,
            "total_s": total,
            "init_s": init,
            "learn_s": learn,
            "sample_s": sample,
            "eval_offspring_s": eval_off,
            "overhead_s": overhead,
        }
        for k in LEARN_SUBKEYS:
            row[k.replace("learn.", "learn_") + "_s"] = timers.t[k]
        rows.append(row)

        print(f"n={n:4d}  total={total:8.2f}s | "
              f"init={init:6.2f} learn={learn:7.2f} sample={sample:6.2f} "
              f"evalOff={eval_off:5.2f} other={overhead:5.2f}")
        learn_parts = "  ".join(
            f"{k.split('.')[1]}={timers.t[k]:.2f}" for k in LEARN_SUBKEYS)
        print(f"        learn breakdown: {learn_parts}")
        pct = {k: 100 * timers.t[k] / learn if learn > 0 else 0
               for k in LEARN_SUBKEYS}
        pct_parts = "  ".join(
            f"{k.split('.')[1]}={pct[k]:4.1f}%" for k in LEARN_SUBKEYS)
        print(f"        learn %:         {pct_parts}\n")

    # write CSV
    import csv
    out_dir = os.path.join(SCRIPT_DIR, os.pardir, "results", "mnfda_profile")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "mnfda_time_profile.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {os.path.relpath(out_csv)}")


if __name__ == "__main__":
    main()
