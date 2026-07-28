"""
Run ONE selected weighted-probability pateda EDA on ONE PBO problem under ONE
selection method (single cluster process).

Companion of ``compare_selected_edas_pbo.py`` (the analogue of
``run_weighted_pbo_eda.py``).  Each process handles one
(algorithm, selection-method, function, dimension) tuple and executes ``n_runs``
consecutive seeds (``seed, ..., seed+n_runs-1``) so that all runs of that tuple
land in the same IOH data folder.

Every process writes to its OWN folder (the IOH ``Analyzer`` logger cannot share
a folder between processes)::

    results/pbo_selected_data_cluster/{ALG}__{SEL}_f{FID}_dim{DIM}_s{SEED}/

The IOH ``algorithm_name`` stored inside the metadata is ``{ALG}__{SEL}`` (no
per-run suffix), so ``iohinspector`` merges every folder of one (algorithm,
selection-method) pair automatically, and the folder / algorithm names reflect
both the algorithm and the selection method, as required::

    python3 scripts/analyze_selected_pbo_results.py results/pbo_selected_data_cluster

If the output folder already exists the process exits immediately (idempotent,
safe to re-launch).  Delete the folder of a crashed job before re-running it.

Usage (positional arguments, seed first):
    python3 scripts/run_selected_pbo_eda.py seed n_runs alg sel fid dim pop_size n_gen [sel_ratio]

Example:
    python3 scripts/run_selected_pbo_eda.py 1 5 MNFDAS3 RTS 19 100 200 50 0.5
"""

import os
import sys
import time

import ioh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_selected_edas_pbo import (
    ALGORITHM_BUILDERS, SELECTION_METHODS, INSTANCE,
    make_fitness, apply_selection_method, folder_name,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_selected_data_cluster")
)


def main():
    if len(sys.argv) < 9:
        print(__doc__)
        sys.exit(1)

    myseed = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    alg = sys.argv[3]
    sel = sys.argv[4]
    fid = int(sys.argv[5])
    dim = int(sys.argv[6])
    pop_size = int(sys.argv[7])
    n_gen = int(sys.argv[8])
    sel_ratio = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5

    if alg not in ALGORITHM_BUILDERS:
        raise ValueError(f"Unknown algorithm '{alg}'. Known: {list(ALGORITHM_BUILDERS)}")
    if sel not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection method '{sel}'. Known: {list(SELECTION_METHODS)}")

    seeds = list(range(myseed, myseed + n_runs))
    tag = folder_name(alg, sel)

    print(f"Seed:             {myseed}  (runs {seeds})")
    print(f"Algorithm:        {alg}")
    print(f"Selection method: {sel}  ({SELECTION_METHODS[sel]})")
    print(f"Function:         f{fid}")
    print(f"Dimension:        {dim}")
    print(f"Instance:         {INSTANCE}")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    print(f"Selection ratio:  {sel_ratio}")

    proc_folder = f"{tag}_f{fid}_dim{dim}_s{myseed}"
    out_folder = os.path.join(OUTPUT_ROOT, proc_folder)
    if os.path.exists(out_folder):
        print(f"Output folder already exists, skipping: {out_folder}")
        return
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Output folder:    {out_folder}")

    logger = ioh.logger.Analyzer(
        triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
        root=OUTPUT_ROOT,
        folder_name=proc_folder,
        algorithm_name=tag,            # merge key: {ALG}__{SEL}
        algorithm_info=(f"pateda pop={pop_size} gen={n_gen} sel={sel_ratio} "
                        f"method={sel}"),
    )
    problem = ioh.get_problem(fid, instance=INSTANCE, dimension=dim,
                              problem_class=ioh.ProblemClass.PBO)
    problem.attach_logger(logger)

    for seed in seeds:
        eda = ALGORITHM_BUILDERS[alg](
            problem.meta_data.n_variables, 2, make_fitness(problem),
            pop_size, n_gen, sel_ratio, seed,
        )
        apply_selection_method(eda, sel, pop_size)
        t0 = time.time()
        eda.run(verbose=False)
        elapsed = time.time() - t0
        print(f"seed={seed}  best={problem.state.current_best.y:.4f}  "
              f"evals={problem.state.evaluations}  time={elapsed:.2f}s")
        problem.reset()

    problem.detach_logger()
    logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
