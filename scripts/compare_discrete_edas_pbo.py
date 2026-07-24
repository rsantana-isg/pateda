"""
Compare all discrete pateda EDAs on the PBO suite of IOHexperimenter.

The PBO (Pseudo-Boolean Optimization) suite [Doerr et al., GECCO'19]
contains 25 binary maximization problems (OneMax, LeadingOnes and their
W-model transformations, LABS, Ising models, MIVS, N-Queens, Concatenated
Trap, NK landscapes).  Problems are instantiated through the ``ioh``
package and every fitness evaluation performed by a pateda EDA is
recorded by the IOHexperimenter ``Analyzer`` logger, so the results can
be post-processed with ``iohinspector`` (see ``analyze_pbo_results.py``)
or uploaded to the IOHanalyzer web service.

pateda EDAs maximize by default, which matches the PBO convention, so
raw fitness values are passed through unchanged.

For every (algorithm, function, dimension) triple the script runs
``N_RUNS`` independent seeds and prints a summary line::

    UMDA       f1  OneMax           dim=16 : [16.0, 16.0, 16.0]  mean=16.0000  time=0.52s

Logged data is written to one folder per algorithm::

    results/pbo_data/<ALGORITHM>/IOHprofiler_f*.json + data_f*/ ...

Note: the ``Analyzer`` logger never overwrites existing data; if an
algorithm folder already exists a suffixed folder (e.g. ``UMDA-1``) is
created.  Delete ``results/pbo_data`` for a fresh start.

Restricted variants that require a prior interaction structure
(TreeEDA-r, MNFDAR, MNFDAGR) are excluded because PBO is a black-box
setting.  Follow-up scripts will link the suite to the BN-based EDAs
(``compare_bn_learning_edas_rw.py``) and the Mixture-of-Trees EDAs
(``compare_mteda_variants_ubqp.py``).

Usage (all arguments optional, positional):
    python scripts/compare_discrete_edas_pbo.py [dims] [fids] [n_runs] [algs]

    dims    comma-separated dimensions, e.g. "16,64,100"   (default: 16,64,100)
    fids    comma-separated PBO function ids or "all"      (default: all = 1..25)
    n_runs  number of independent runs per triple          (default: 5)
    algs    comma-separated algorithm names or "all"       (default: all)

Examples:
    python scripts/compare_discrete_edas_pbo.py
    python scripts/compare_discrete_edas_pbo.py 16 1,2,18,19 3 UMDA,TreeEDA
"""

import os
import sys
import time
import traceback

import numpy as np
import ioh

from pateda import (
    UMDA, BMDA, TreeEDA, MIMIC, PBIL,
    EBNA, BOA, AffEDA, MKEDA, MTED,
    MNFDA, MNFDAG, MOA,
    FDA, BSC,
)


ALGORITHMS = [
    ("UMDA", UMDA),
    ("BMDA", BMDA),
    ("TreeEDA", TreeEDA),
    ("MIMIC", MIMIC),
    ("PBIL", PBIL),
    ("EBNA", EBNA),
    ("BOA", BOA),
    ("AffEDA", AffEDA),
    ("MKEDA", MKEDA),
    ("MTED", MTED),
    ("MNFDA", MNFDA),
    #("MNFDAG", MNFDAG),
    #("MOA", MOA),
    ("FDA", FDA),
    ("BSC", BSC),
]

# PBO experimental setup following Doerr et al. (GECCO'19): instance 1,
# dimensions in {16, 64, 100} (625 is also standard but too expensive for
# the model-building EDAs; add it explicitly if needed).
FUNCTION_IDS = list(range(1, 26))
DIMENSIONS = [16, 64, 100]
INSTANCE = 1

POP_SIZE = 200
N_GEN = 50                 # budget per run = POP_SIZE * (N_GEN + 1) evaluations
SEL_RATIO = 0.5
N_RUNS = 5                 # repetitions per (algorithm, function, dimension)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_data")
)


def make_fitness(problem):
    """Wrap an ioh problem as a pateda fitness function (one call = one eval)."""
    def fitness(solution):
        return float(problem(np.asarray(solution, dtype=int)))
    return fitness


def run_single_seed(alg_cls, problem, seed):
    """Run one EDA on one attached PBO problem for one seed."""
    alg = alg_cls(
        n_vars=problem.meta_data.n_variables,
        cardinality=2,
        fitness_func=make_fitness(problem),
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=seed,
    )
    t0 = time.time()
    alg.run(verbose=False)
    elapsed = time.time() - t0
    # Best-so-far as tracked by IOHexperimenter (identical to the EDA's own).
    best = float(problem.state.current_best.y)
    evals = int(problem.state.evaluations)
    return best, evals, elapsed


def parse_args(argv):
    """Parse the optional positional arguments (dims, fids, n_runs, algs)."""
    dims = DIMENSIONS
    fids = FUNCTION_IDS
    n_runs = N_RUNS
    algorithms = ALGORITHMS

    if len(argv) > 1:
        dims = [int(d) for d in argv[1].split(",")]
    if len(argv) > 2 and argv[2].lower() != "all":
        fids = [int(f) for f in argv[2].split(",")]
    if len(argv) > 3:
        n_runs = int(argv[3])
    if len(argv) > 4 and argv[4].lower() != "all":
        wanted = set(argv[4].split(","))
        algorithms = [(n, c) for n, c in ALGORITHMS if n in wanted]
        missing = wanted - {n for n, _ in algorithms}
        if missing:
            raise ValueError(f"Unknown algorithm names: {sorted(missing)}")
    return dims, fids, n_runs, algorithms


def main():
    dims, fids, n_runs, algorithms = parse_args(sys.argv)
    seeds = list(range(1, n_runs + 1))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"PBO suite comparison of pateda discrete EDAs")
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Dimensions:       {dims}")
    print(f"Functions:        {fids}")
    print(f"Instance:         {INSTANCE}")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {N_GEN}")
    print(f"Selection ratio:  {SEL_RATIO}")
    print(f"Runs (seeds):     {seeds}")
    print(f"Budget per run:   {POP_SIZE * (N_GEN + 1)} evaluations")

    name_w = 10

    for alg_name, alg_cls in algorithms:
        print(f"\n{'=' * 84}")
        print(f"Algorithm: {alg_name}")
        print(f"{'=' * 84}")

        logger = ioh.logger.Analyzer(
            triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
            root=OUTPUT_ROOT,
            folder_name=alg_name,
            algorithm_name=alg_name,
            algorithm_info=f"pateda pop={POP_SIZE} gen={N_GEN} sel={SEL_RATIO}",
        )

        for dim in dims:
            for fid in fids:
                try:
                    problem = ioh.get_problem(
                        fid,
                        instance=INSTANCE,
                        dimension=dim,
                        problem_class=ioh.ProblemClass.PBO,
                    )
                except Exception as exc:
                    print(f"{alg_name:<{name_w}} f{fid} dim={dim}: "
                          f"UNAVAILABLE -- {exc}")
                    continue

                func_tag = f"f{fid:<3} {problem.meta_data.name:<22} dim={dim}"
                problem.attach_logger(logger)
                try:
                    bests, times = [], []
                    for seed in seeds:
                        best, _, elapsed = run_single_seed(alg_cls, problem, seed)
                        bests.append(best)
                        times.append(elapsed)
                        problem.reset()   # finalizes the run in the log files
                    bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                    print(
                        f"{alg_name:<{name_w}} {func_tag}: {bests_str}  "
                        f"mean={np.mean(bests):.4f}  time={np.mean(times):.2f}s"
                    )
                except Exception as exc:
                    print(f"{alg_name:<{name_w}} {func_tag}: ERROR -- {exc}")
                    traceback.print_exc()
                finally:
                    problem.detach_logger()

        logger.close()

    print(f"\nDone.  Analyze with: python scripts/analyze_pbo_results.py")


if __name__ == "__main__":
    main()
