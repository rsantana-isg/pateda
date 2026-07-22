"""
Pipeline search on a UBQP n=250 instance (multi-objective, parallel)

Same as ``pipeline_search_ubqp100.py`` but for the larger packaged ``bqp250``
unconstrained binary quadratic (UBQP) benchmark instance (n = 250).  Because the
instance is larger, individual pipelines take longer, so the per-pipeline
wall-time cap is raised to 600 s; every other parameter is kept:

  * inner budget per pipeline: 200 x 50 = 10,000 evaluations (>= 10,000),
  * meta-population of 32 pipelines evolved for 30 generations,
  * parallel evaluation across up to 15 CPUs (one pipeline per CPU).

Objectives (Pareto): maximize the UBQP objective value reached, minimize the
pipeline running time.  The script reports the progress of the search after every
generation and writes, after every generation and at the end, both the **Pareto
front** (objective values: value vs. time) and the **Pareto set** (the actual
pipelines) to files, so a long run can be inspected or resumed from its last
checkpoint.

Output files (in ``--outdir``, default ``ubqp250_pipeline_search/``):

    pareto_front.csv     -- objective_value, time_seconds     (the front)
    pareto_set.json      -- the pipelines + their objective/time (the set)
    all_evaluated.csv    -- every distinct pipeline evaluated
    progress.csv         -- per-generation search progress

Usage
-----
    python3 pipeline_search_ubqp250.py [seed] [n_jobs] [outdir]

Note: this is a heavy run (32 x 30 pipelines, >=10k evaluations each, n=250);
expect it to take a long time.  The intermediate checkpoints let you stop early
and still keep results.
"""

import sys
import os
import csv
import json
import time
import functools
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda.pipelines import PipelineMetaOptimizer, MetaProblem
from pateda.functions.discrete_binary.problems.ubqp import load_ubqp_benchmark_instance


# --- configuration (same as the n=100 script, larger per-pipeline timeout) ---
INSTANCE_NAME = "bqp250"
INNER_POP = 200            # population per pipeline
INNER_GEN = 50             # generations per pipeline  -> 200 x 50 = 10,000 evals
META_POP = 32              # number of pipelines in the meta-population
META_GENS = 30             # meta-generations
N_JOBS_DEFAULT = 15        # CPUs (one pipeline per CPU)
EVAL_TIMEOUT = 600.0       # per-pipeline wall-time cap (s); slower pipelines are
                           # marked infeasible so they never stall a generation


# A *module-level* fitness (picklable) so pipelines run in parallel processes.
def ubqp_objective(x, instance):
    return float(np.ravel(instance.evaluate(np.asarray(x, dtype=int)))[0])


# ---------------------------------------------------------------------------
# Saving Pareto front / set (used as the per-generation checkpoint too)
# ---------------------------------------------------------------------------

def _spec_dict(spec):
    return {
        "selection": spec.selection,
        "learner": spec.learner,
        "operators": list(spec.operators),
        "sampler": spec.sampler,
        "replacement": spec.replacement,
        "local_opt": spec.local_opt,
        "mutation": spec.mutation,
        "text": str(spec),
    }


def save_pareto(pareto, outdir, meta):
    os.makedirs(outdir, exist_ok=True)

    # Pareto FRONT (objective space): objective_value, time_seconds.
    with open(os.path.join(outdir, "pareto_front.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objective_value", "time_seconds"])
        for ind in pareto:
            w.writerow([f"{ind.quality:.6f}", f"{ind.runtime:.6f}"])

    # Pareto SET (decision space): the pipelines with their objective/time.
    payload = {
        "meta": meta,
        "pareto_set": [
            {"objective_value": ind.quality, "time_seconds": ind.runtime,
             "pipeline": _spec_dict(ind.spec)}
            for ind in pareto
        ],
    }
    with open(os.path.join(outdir, "pareto_set.json"), "w") as f:
        json.dump(payload, f, indent=2)


def save_all_evaluated(evaluated, outdir):
    with open(os.path.join(outdir, "all_evaluated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feasible", "objective_value", "time_seconds", "pipeline"])
        for ind in evaluated:
            w.writerow([int(ind.feasible),
                        f"{ind.quality:.6f}" if np.isfinite(ind.quality) else "",
                        f"{ind.runtime:.6f}" if np.isfinite(ind.runtime) else "",
                        str(ind.spec)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42, n_jobs=N_JOBS_DEFAULT, outdir="ubqp250_pipeline_search"):
    print("#" * 80)
    print("# Pipeline search on UBQP bqp250 (n=250) - quality vs. time Pareto")
    print(f"# seed={seed}  n_jobs={n_jobs}  outdir={outdir}")
    print("#" * 80 + "\n")

    instance, known_opt = load_ubqp_benchmark_instance(INSTANCE_NAME)
    n = instance.n_vars
    fitness = functools.partial(ubqp_objective, instance=instance)
    problem = MetaProblem(fitness=fitness, n_vars=n, cardinality=np.full(n, 2),
                          optimum=None, name=INSTANCE_NAME)      # optimum=None -> raw value
    print(f"Instance: {INSTANCE_NAME}, n={n}, best-known objective={known_opt}")
    print(f"Inner budget per pipeline: {INNER_POP} x {INNER_GEN} = "
          f"{INNER_POP * INNER_GEN} evaluations (>= 10,000)")
    print(f"Meta-GA: population={META_POP}, generations={META_GENS}, "
          f"CPUs={n_jobs}, per-pipeline timeout={EVAL_TIMEOUT}s\n")

    os.makedirs(outdir, exist_ok=True)
    progress_path = os.path.join(outdir, "progress.csv")
    with open(progress_path, "w", newline="") as f:
        csv.writer(f).writerow(["generation", "elapsed_s", "n_evaluated",
                                "n_feasible", "best_objective", "pareto_size"])

    meta = {"instance": INSTANCE_NAME, "n_vars": n, "known_optimum": known_opt,
            "inner_pop": INNER_POP, "inner_gen": INNER_GEN,
            "meta_pop": META_POP, "meta_gens": META_GENS,
            "eval_timeout": EVAL_TIMEOUT, "seed": seed}

    # Per-generation reporting + checkpointing.
    def on_generation(stats):
        with open(progress_path, "a", newline="") as f:
            csv.writer(f).writerow([
                stats["generation"], f"{stats['elapsed']:.1f}",
                stats["n_evaluated"], stats["n_feasible"],
                f"{stats['best_objective']:.3f}", stats["pareto_size"]])
        # Checkpoint the current Pareto front/set so an interrupted run keeps data.
        save_pareto(stats["pareto_front"], outdir,
                    {**meta, "checkpoint_generation": stats["generation"],
                     "elapsed_s": round(stats["elapsed"], 1)})

    mo = PipelineMetaOptimizer(
        problem,
        inner_pop=INNER_POP, inner_gen=INNER_GEN,
        meta_pop=META_POP, meta_gens=META_GENS,
        n_eval_seeds=1,
        crossover_prob=0.8, mutation_prob=0.5,
        n_jobs=n_jobs, eval_timeout=EVAL_TIMEOUT,
        seed=seed,
    )

    t0 = time.time()
    result = mo.optimize(verbose=True, callback=on_generation)
    wall = time.time() - t0

    # Final save.
    save_pareto(result.pareto_front, outdir, {**meta, "final": True,
                                              "wall_seconds": round(wall, 1)})
    save_all_evaluated(result.evaluated, outdir)

    # --- report ---
    print(f"\nSearch finished in {wall:.1f}s; {len(result.evaluated)} distinct "
          f"pipelines evaluated.\n")
    print("=" * 80)
    print("PARETO FRONT (objective value vs. time)  +  PARETO SET (pipelines)")
    print("=" * 80)
    print(f"  {'objective':>10} | {'time(s)':>8} | pipeline")
    print("  " + "-" * 74)
    for ind in result.pareto_front:
        print(f"  {ind.quality:>10.1f} | {ind.runtime:>8.2f} | {ind.spec}")
    bq, fa = result.best_quality, result.fastest
    print(f"\n  best objective : {bq.quality:.1f} in {bq.runtime:.2f}s  ->  {bq.spec}")
    print(f"  fastest        : {fa.quality:.1f} in {fa.runtime:.2f}s  ->  {fa.spec}")
    print(f"\n  Saved to '{outdir}/': pareto_front.csv, pareto_set.json, "
          f"all_evaluated.csv, progress.csv")
    print("=" * 80)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else N_JOBS_DEFAULT
    od = sys.argv[3] if len(sys.argv) > 3 else "ubqp250_pipeline_search"
    main(s, jobs, od)
