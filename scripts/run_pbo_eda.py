"""
Run ONE discrete pateda EDA on ONE PBO problem (single cluster process).

Designed for SLURM: each process handles one (algorithm, function,
dimension) combination and executes ``n_runs`` consecutive seeds
(``seed, seed+1, ..., seed+n_runs-1``) so that runs of the same
combination land in the same IOH data folder.

Every process writes to its OWN folder (the IOH ``Analyzer`` logger
cannot share a folder between processes)::

    results/pbo_data_cluster/{ALG}_f{FID}_dim{DIM}_s{SEED}/

The algorithm name stored inside the IOH metadata is just ``ALG``, so
``iohinspector`` (see ``analyze_pbo_results.py``) merges all folders of
one algorithm automatically when analyzing jointly::

    python3 scripts/analyze_pbo_results.py results/pbo_data_cluster

If the output folder already exists the process exits immediately
(idempotent, safe to re-launch).  Delete the folder of a crashed job
before re-running it.

Usage (positional arguments, seed first):
    python3 scripts/run_pbo_eda.py seed n_runs algorithm fid dim pop_size n_gen [sel_ratio]

Example:
    python3 scripts/run_pbo_eda.py 1 5 UMDA 19 100 200 50 0.5
"""

import os
import sys
import time

import numpy as np
import ioh

# Reuse the algorithm registry of the local comparison script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_discrete_edas_pbo import ALGORITHMS, INSTANCE, make_fitness

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_data_cluster")
)


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        sys.exit(1)

    myseed = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    alg_name = sys.argv[3]
    fid = int(sys.argv[4])
    dim = int(sys.argv[5])
    pop_size = int(sys.argv[6])
    n_gen = int(sys.argv[7])
    sel_ratio = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5

    registry = dict(ALGORITHMS)
    if alg_name not in registry:
        raise ValueError(
            f"Unknown algorithm '{alg_name}'.  Known: {sorted(registry)}"
        )
    alg_cls = registry[alg_name]
    seeds = list(range(myseed, myseed + n_runs))

    print(f"Seed:             {myseed}  (runs {seeds})")
    print(f"Algorithm:        {alg_name}")
    print(f"Function:         f{fid}")
    print(f"Dimension:        {dim}")
    print(f"Instance:         {INSTANCE}")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    print(f"Selection ratio:  {sel_ratio}")
    print(f"Budget per run:   {pop_size * (n_gen + 1)} evaluations")

    folder_name = f"{alg_name}_f{fid}_dim{dim}_s{myseed}"
    out_folder = os.path.join(OUTPUT_ROOT, folder_name)
    if os.path.exists(out_folder):
        print(f"Output folder already exists, skipping: {out_folder}")
        return
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Output folder:    {out_folder}")

    logger = ioh.logger.Analyzer(
        triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
        root=OUTPUT_ROOT,
        folder_name=folder_name,
        algorithm_name=alg_name,   # merging key for joint iohinspector analysis
        algorithm_info=f"pateda pop={pop_size} gen={n_gen} sel={sel_ratio}",
    )
    problem = ioh.get_problem(
        fid, instance=INSTANCE, dimension=dim,
        problem_class=ioh.ProblemClass.PBO,
    )
    problem.attach_logger(logger)

    for seed in seeds:
        alg = alg_cls(
            n_vars=problem.meta_data.n_variables,
            cardinality=2,
            fitness_func=make_fitness(problem),
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=sel_ratio,
            random_seed=seed,
        )
        t0 = time.time()
        alg.run(verbose=False)
        elapsed = time.time() - t0
        print(
            f"seed={seed}  best={problem.state.current_best.y:.4f}  "
            f"evals={problem.state.evaluations}  time={elapsed:.2f}s"
        )
        problem.reset()

    problem.detach_logger()
    logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
