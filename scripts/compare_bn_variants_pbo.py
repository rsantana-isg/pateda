"""
PBO-suite registry for the recently-developed BN-based EDA variants, run under
the three customized-selection schemes (FP, BZ, RTS).

This mirrors ``compare_weighted_edas_pbo.py`` but with a different algorithm set
-- the 11 recently-developed BN variants -- all sharing a **common
``max_parents = 6``** so the comparison isolates the learning method:

    EBNA_BIC   penalized-ML BIC, warm-started add/delete/reverse search
    EBNA_K2    penalized-K2 (EBNA_K2+pen), fixed max_parents=6
    EBNA_PC    constraint-based (PC-Stable)
    LFDA_mp6   add-only greedy BIC, from scratch each generation
    BOA_mp6    K2/BDe greedy Bayesian network
    SARTRE_mp6 order-based sparse-additive BN (SARTRE-EDA)
    A1_dt      HBOA-Light: decision-tree CPDs

``LFDA_mp6`` / ``BOA_mp6`` / ``SARTRE_mp6`` carry the ``_mp6`` suffix (common
``max_parents=6``) so their IOH folders stay distinct from the ``LFDA`` / ``BOA``
/ ``SARTRE`` folders produced by ``launch_weighted_pbo_experiments.py`` (which
use each method's natural parent budget), letting the two studies' folders be
combined and analysed together without collision.  The remaining variants have
names that do not occur in that study.
    A2_mi      HBOA-Light: decision graphs + top-k MI candidate pruning
    A3_fast    HBOA-Light: decision graphs + cached-statistics split scoring
    A4_mdl     HBOA-Light: bounded decision graphs, MDL split score
    A5_ndg     HBOA-Light: non-search decision-graph construction (Tree-in-Tree/NDG)

The three selection schemes (weighting + replacement) are reused verbatim from
``compare_weighted_edas_pbo.py``:

    FP   truncation-0.5 + fitness-proportional weighting + elitist replacement
    BZ   truncation-0.5 + Boltzmann weighting (beta=1)    + elitist replacement
    RTS  truncation-0.5 + Boltzmann weighting (beta=1)    + restricted-tournament

``apply_selection_method`` overrides each algorithm's built-in replacement with
the scheme's replacement (so, e.g., the HBOA-Light variants run with elitism
under FP/BZ and with niching under RTS, uniformly with every other method).

Every fitness evaluation is logged by an ``ioh.logger.Analyzer`` to one folder
per (algorithm, selection) pair, named ``<ALG>__<SEL>`` -- the same convention
``analyze_weighted_pbo_results.py`` consumes, so the existing analysis / report
scripts work unchanged on this data.

Usage (all arguments optional, positional):
    python scripts/compare_bn_variants_pbo.py [dims] [fids] [n_runs] [algs] [sels]
"""

import os
import sys
import time
import traceback

import numpy as np
import ioh

from pateda import (
    EBNA_BIC, EBNA_K2, EBNA_PC, LFDA, BOA, SARTRE_EDA,
    HBOA_Light_A1, HBOA_Light_A2, HBOA_Light_A3, HBOA_Light_A4, HBOA_Light_A5,
)

# Reuse the selection-scheme machinery and PBO constants.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compare_weighted_edas_pbo import (   # noqa: E402  (re-exported for run/launch)
    SELECTION_METHODS, SELECTION_ORDER, apply_selection_method,
    make_fitness, folder_name, INSTANCE, FUNCTION_IDS,
    POP_SIZE, N_GEN, SEL_RATIO,
)

# Common parent budget for every method (the controlled variable).
MAX_PARENTS = 6

# n=625 included (per the study setup); the prohibitive large-dim jobs that do
# not finish are simply skipped by the analysis.
DIMENSIONS = [16, 64, 100, 625]

OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_bnvariants_data")
)


def _wrapped(cls, **extra):
    """Builder from a plug-and-play wrapper class (returns its inner EDA)."""
    def build(n_vars, card, fitness, pop_size, n_gen, sel_ratio, seed):
        alg = cls(
            n_vars=n_vars, cardinality=card, fitness_func=fitness,
            pop_size=pop_size, n_gen=n_gen, selection_ratio=sel_ratio,
            random_seed=seed, **extra,
        )
        return alg._eda
    return build


# All 11 variants share max_parents=6.
ALGORITHM_BUILDERS = {
    "EBNA_BIC": _wrapped(EBNA_BIC,      max_parents=MAX_PARENTS),
    "EBNA_K2":  _wrapped(EBNA_K2,       max_parents=MAX_PARENTS),
    "EBNA_PC":  _wrapped(EBNA_PC,       max_parents=MAX_PARENTS),
    # LFDA / BOA / SARTRE also appear (with their *natural* max_parents) in
    # compare_weighted_edas_pbo.py / launch_weighted_pbo_experiments.py.  These
    # here use the common max_parents=6, so they are renamed with an ``_mp6``
    # suffix to keep their IOH folders (``LFDA_mp6__FP`` etc.) distinct from the
    # existing ``LFDA__FP`` folders when the two studies are analysed together.
    "LFDA_mp6":   _wrapped(LFDA,       max_parents=MAX_PARENTS),
    "BOA_mp6":    _wrapped(BOA,        max_parents=MAX_PARENTS),
    "SARTRE_mp6": _wrapped(SARTRE_EDA, max_parents=MAX_PARENTS),
    "A1_dt":    _wrapped(HBOA_Light_A1, max_parents=MAX_PARENTS),
    "A2_mi":    _wrapped(HBOA_Light_A2, max_parents=MAX_PARENTS),
    "A3_fast":  _wrapped(HBOA_Light_A3, max_parents=MAX_PARENTS),
    "A4_mdl":   _wrapped(HBOA_Light_A4, max_parents=MAX_PARENTS),
    "A5_ndg":   _wrapped(HBOA_Light_A5, max_parents=MAX_PARENTS),
}
ALGORITHM_NAMES = list(ALGORITHM_BUILDERS.keys())


def build_configured_eda(alg, sel, problem, pop_size, n_gen, sel_ratio, seed):
    """Build the EDA for one (algorithm, selection-method) and configure it."""
    eda = ALGORITHM_BUILDERS[alg](
        problem.meta_data.n_variables, 2, make_fitness(problem),
        pop_size, n_gen, sel_ratio, seed,
    )
    return apply_selection_method(eda, sel, pop_size)


# ---------------------------------------------------------------------------
# Local comparison main (small grids; the cluster uses run_bn_variant_pbo.py)
# ---------------------------------------------------------------------------
def parse_args(argv):
    dims, fids, n_runs = DIMENSIONS, FUNCTION_IDS, 5
    algs, sels = ALGORITHM_NAMES, list(SELECTION_ORDER)
    if len(argv) > 1:
        dims = [int(d) for d in argv[1].split(",")]
    if len(argv) > 2 and argv[2].lower() != "all":
        fids = [int(f) for f in argv[2].split(",")]
    if len(argv) > 3:
        n_runs = int(argv[3])
    if len(argv) > 4 and argv[4].lower() != "all":
        algs = argv[4].split(",")
        unknown = [a for a in algs if a not in ALGORITHM_BUILDERS]
        if unknown:
            raise ValueError(f"Unknown algorithms: {unknown}")
    if len(argv) > 5 and argv[5].lower() != "all":
        sels = argv[5].split(",")
        unknown = [s for s in sels if s not in SELECTION_METHODS]
        if unknown:
            raise ValueError(f"Unknown selection methods: {unknown}")
    return dims, fids, n_runs, algs, sels


def main():
    dims, fids, n_runs, algs, sels = parse_args(sys.argv)
    seeds = list(range(1, n_runs + 1))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("PBO suite comparison of the recently-developed BN variants")
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Dimensions:       {dims}")
    print(f"Functions:        {fids}")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {N_GEN}")
    print(f"max_parents:      {MAX_PARENTS} (common)")
    print(f"Selection methods:{sels}")
    print(f"Algorithms:       {algs}")
    print(f"Runs (seeds):     {seeds}")

    name_w = 16
    for sel in sels:
        for alg in algs:
            tag = folder_name(alg, sel)
            print(f"\n{'=' * 84}\n{tag}\n{'=' * 84}")
            logger = ioh.logger.Analyzer(
                triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
                root=OUTPUT_ROOT, folder_name=tag, algorithm_name=tag,
                algorithm_info=(f"pateda pop={POP_SIZE} gen={N_GEN} "
                                f"sel={SEL_RATIO} mp={MAX_PARENTS} method={sel}"),
            )
            for dim in dims:
                for fid in fids:
                    try:
                        problem = ioh.get_problem(
                            fid, instance=INSTANCE, dimension=dim,
                            problem_class=ioh.ProblemClass.PBO)
                    except Exception as exc:
                        print(f"{tag:<{name_w}} f{fid} dim={dim}: UNAVAILABLE -- {exc}")
                        continue
                    func_tag = f"f{fid:<3} {problem.meta_data.name:<22} dim={dim}"
                    problem.attach_logger(logger)
                    try:
                        bests, times = [], []
                        for seed in seeds:
                            eda = build_configured_eda(
                                alg, sel, problem, POP_SIZE, N_GEN, SEL_RATIO, seed)
                            t0 = time.time()
                            eda.run(verbose=False)
                            times.append(time.time() - t0)
                            bests.append(float(problem.state.current_best.y))
                            problem.reset()
                        bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                        print(f"{tag:<{name_w}} {func_tag}: {bests_str}  "
                              f"mean={np.mean(bests):.4f}  time={np.mean(times):.2f}s")
                    except Exception as exc:
                        print(f"{tag:<{name_w}} {func_tag}: ERROR -- {exc}")
                        traceback.print_exc()
                    finally:
                        problem.detach_logger()
            logger.close()

    print("\nDone.  Analyze with: "
          "python scripts/analyze_weighted_pbo_results.py "
          f"{os.path.relpath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()
