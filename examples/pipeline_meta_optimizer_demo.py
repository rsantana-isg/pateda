"""
Meta-optimizing EDA pipelines: the quality vs. time Pareto set

This drives the grammar-guided, multi-objective meta-optimizer
(:class:`~pateda.pipelines.meta_optimizer.PipelineMetaOptimizer`) that searches
the space of EDA pipelines to simultaneously

    * maximize the objective value reached (quality), and
    * minimize the running time,

returning the **Pareto set** of pipelines -- from the cheap-but-weaker to the
strong-but-slower -- rather than a single winner.

Difficult test problem.  A *concatenated trap-5* is used: the string is split
into blocks of 5, each fully deceptive (all-ones scores 5, otherwise ``4 - u``
where ``u`` is the number of ones).  Univariate models (UMDA, PBIL) are driven
into the deceptive attractor and stay cheap but weak; linkage-learning models
(Tree-EDA, EBNA, MN-FDA, ...) recover the building blocks and reach far higher
quality at a higher time cost.  This makes the quality/time trade-off -- and the
Pareto front over pipelines -- genuinely non-trivial.

Usage
-----
    python3 pipeline_meta_optimizer_demo.py [seed]
"""

import sys
import os
import time
import functools
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda.pipelines import PipelineMetaOptimizer, MetaProblem


# A *module-level* fitness (picklable) so pipelines can be evaluated in parallel
# worker processes.  Concatenated trap-k: each block of k is fully deceptive.
def concatenated_trap(x, k=5):
    x = np.asarray(x)
    total = 0.0
    for b in range(0, len(x), k):
        u = int(x[b:b + k].sum())
        total += float(k) if u == k else float(k - 1 - u)
    return total


def main(seed=42, n_jobs=None):
    print("#" * 80)
    print("# Meta-optimizing EDA pipelines - quality vs. running-time Pareto set")
    print(f"# seed = {seed}")
    print("#" * 80 + "\n")

    # --- difficult problem: concatenated trap-5, n = 25 (5 deceptive blocks) ---
    n_blocks, k = 5, 5
    n = n_blocks * k
    fitness = functools.partial(concatenated_trap, k=k)     # picklable fitness
    optimum = float(n_blocks * k)
    problem = MetaProblem(fitness=fitness, n_vars=n, cardinality=np.full(n, 2),
                          optimum=optimum, name=f"trap-{k} x{n_blocks} (n={n})")
    print(f"Problem: {problem.name}, optimum={optimum:.0f} "
          f"(fully deceptive -> needs linkage learning)\n")

    # --- parallel evaluation: up to 15 CPUs (one pipeline per CPU) ---
    meta_pop = 16
    if n_jobs is None:
        n_jobs = min(15, meta_pop, os.cpu_count() or 1)

    # --- meta-optimizer configuration ---
    mo = PipelineMetaOptimizer(
        problem,
        inner_pop=150, inner_gen=20,        # fixed inner budget every pipeline runs at
        meta_pop=meta_pop, meta_gens=6,     # meta-GA size
        n_eval_seeds=1,
        crossover_prob=0.8, mutation_prob=0.5,
        n_jobs=n_jobs,                      # one pipeline per CPU
        eval_timeout=10.0,                 # cap slow/hanging pipelines
        seed=seed,
    )
    print(f"Meta-GA: pop={meta_pop}, generations=6, inner budget = 150 x 20;")
    print(f"objectives = (maximize quality, minimize time), NSGA-II selection;")
    print(f"parallel evaluation on {n_jobs} CPUs (per-pipeline timeout "
          f"{mo.eval_timeout}s).\n")

    t0 = time.time()
    result = mo.optimize(verbose=True)
    wall = time.time() - t0
    # Effective parallelism: total pipeline CPU-time vs. wall-clock time.
    cpu_seconds = sum(ind.runtime for ind in result.evaluated
                      if np.isfinite(ind.runtime))
    print(f"\nSearch finished in {wall:.1f}s wall-clock; "
          f"{len(result.evaluated)} distinct pipelines evaluated.")
    print(f"Pipelines consumed {cpu_seconds:.1f} CPU-seconds of work -> "
          f"effective parallelism ~ {cpu_seconds / max(wall, 1e-9):.1f}x "
          f"(on {n_jobs} CPUs).\n")

    # --- the Pareto set ---
    print("=" * 80)
    print("Pareto set of pipelines (non-dominated on quality vs. time)")
    print("=" * 80)
    print(f"  {'quality':>7} | {'time(s)':>7} | pipeline")
    print("  " + "-" * 72)
    for ind in result.pareto_front:            # sorted by time (cheapest first)
        print(f"  {ind.quality:>7.3f} | {ind.runtime:>7.3f} | {ind.spec}")

    print("\n  Two useful extremes of the front:")
    bq, fa = result.best_quality, result.fastest
    print(f"    highest quality : q={bq.quality:.3f}, t={bq.runtime:.3f}s")
    print(f"      -> {bq.spec}")
    print(f"    fastest         : q={fa.quality:.3f}, t={fa.runtime:.3f}s")
    print(f"      -> {fa.spec}")

    print("\n" + "=" * 80)
    print("Reading the result")
    print("=" * 80)
    print("  The Pareto front trades quality for time: at the cheap end sit fast,")
    print("  simpler pipelines that only partially solve the deceptive trap; at the")
    print("  costly end sit linkage-learning pipelines that reach higher quality.")
    print("  A practitioner picks the knee of the front for their time budget; the")
    print("  meta-optimizer discovered the whole trade-off automatically.")
    print("=" * 80)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(s, jobs)
