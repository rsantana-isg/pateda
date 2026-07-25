"""
Run ONE recently-developed BN variant on ONE PBO problem under ONE selection
method (single cluster process).

Companion of ``run_weighted_pbo_eda.py`` for ``compare_bn_variants_pbo.py``.
Each process handles one (algorithm, selection-method, function, dimension)
tuple and runs ``n_runs`` consecutive seeds into a single IOH data folder::

    results/pbo_bnvariants_data_cluster/{ALG}__{SEL}_f{FID}_dim{DIM}_s{SEED}/

The IOH ``algorithm_name`` is ``{ALG}__{SEL}`` (the merging key), so
``analyze_weighted_pbo_results.py`` groups every folder of one (algorithm,
selection-method) pair automatically::

    python3 scripts/analyze_weighted_pbo_results.py results/pbo_bnvariants_data_cluster \\
        results/pbo_bnvariants_analysis

Idempotent: if the output folder already exists the process exits immediately.

Usage (positional, seed first):
    python3 scripts/run_bn_variant_pbo.py seed n_runs alg sel fid dim pop_size n_gen [sel_ratio]

Example:
    python3 scripts/run_bn_variant_pbo.py 1 5 A5_ndg RTS 19 100 200 50 0.5
"""

import os
import sys
import time

import ioh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_bn_variants_pbo import (
    ALGORITHM_BUILDERS, SELECTION_METHODS, INSTANCE, MAX_PARENTS,
    make_fitness, apply_selection_method, folder_name, build_configured_eda,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_bnvariants_data_cluster")
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
    print(f"Algorithm:        {alg}  (max_parents={MAX_PARENTS})")
    print(f"Selection method: {sel}  ({SELECTION_METHODS[sel]})")
    print(f"Function:         f{fid}   Dimension: {dim}   Instance: {INSTANCE}")
    print(f"Population Size:  {pop_size}   Generations: {n_gen}   Sel ratio: {sel_ratio}")

    proc_folder = f"{tag}_f{fid}_dim{dim}_s{myseed}"
    out_folder = os.path.join(OUTPUT_ROOT, proc_folder)
    if os.path.exists(out_folder):
        print(f"Output folder already exists, skipping: {out_folder}")
        return
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Output folder:    {out_folder}")

    logger = ioh.logger.Analyzer(
        triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
        root=OUTPUT_ROOT, folder_name=proc_folder, algorithm_name=tag,
        algorithm_info=(f"pateda pop={pop_size} gen={n_gen} sel={sel_ratio} "
                        f"mp={MAX_PARENTS} method={sel}"),
    )
    problem = ioh.get_problem(fid, instance=INSTANCE, dimension=dim,
                              problem_class=ioh.ProblemClass.PBO)
    problem.attach_logger(logger)

    for seed in seeds:
        eda = build_configured_eda(alg, sel, problem, pop_size, n_gen,
                                   sel_ratio, seed)
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
