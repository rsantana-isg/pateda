"""
Compare a SELECTED set of affinity- and Markov-network-factorization EDAs on the
PBO suite of IOHexperimenter, under weighted-probability (customized) selection.

This mirrors ``compare_weighted_edas_pbo.py`` (all algorithms use weighted
probabilities and the three selection schemes FP / BZ / RTS) but restricts the
algorithm set to the affinity and MN-FDA-S/-sparse/-P family and sweeps the
relevant hyper-parameters (Markov order ``k`` for MK-EDA, maximum clique size for
the MN-FDA variants):

    AffEDA, AffEDA-sparse,
    MK-EDA (k = 2, 3, 4),
    MN-FDA-S       (max_clique = 3, 4, 5),
    MN-FDA-sparse  (max_clique = 3, 4, 5),
    MN-FDA-S-sparse(max_clique = 3, 4, 5),
    MN-FDA-P       (max_clique = 3, 4, 5).

Every fitness evaluation is recorded by the IOHexperimenter ``Analyzer`` logger,
so the results can be post-processed with ``iohinspector`` (see
``analyze_selected_pbo_results.py``) or uploaded to IOHanalyzer.  Logged data is
written to one folder per (algorithm, selection-method) pair, whose name encodes
both, under ``results/pbo_selected_data/``::

    results/pbo_selected_data/<ALG>__<SEL>/IOHprofiler_f*.json + data_f*/ ...

The three selection methods (weighted-probability schemes, identical to
``compare_weighted_edas_pbo.py``):

    FP  : truncation (tau=0.5) + fitness-proportional weighting + elitist repl.
    BZ  : truncation (tau=0.5) + Boltzmann weighting (beta=1)   + elitist repl.
    RTS : truncation (tau=0.5) + Boltzmann weighting (beta=1)   + restricted
          tournament replacement.

Usage (all arguments optional, positional):
    python scripts/compare_selected_edas_pbo.py [dims] [fids] [n_runs] [algs] [sels]

    dims    comma-separated dimensions            (default: 16,64,100,625)
    fids    comma-separated PBO function ids/"all" (default: all = 1..25)
    n_runs  runs per (alg, sel, function, dim)     (default: 5)
    algs    comma-separated algorithm names/"all"  (default: all)
    sels    comma-separated selection methods/"all"(default: FP,BZ,RTS)

Examples:
    python scripts/compare_selected_edas_pbo.py
    python scripts/compare_selected_edas_pbo.py 16 1,2,18,19 3 MNFDAS3,MKEDA2 FP
"""

import os
import sys
import time
import traceback

import numpy as np
import ioh

from pateda import (
    AffEDA, AffEDASparse, MKEDA,
    MNFDAS, MNFDASparse, MNFDASSparse, MNFDAP,
)
from pateda.replacement.elitist import ElitistReplacement
from pateda.replacement.niching import RestrictedTournamentReplacement


# ---------------------------------------------------------------------------
# Fixed experimental parameters (kept identical to the weighted-PBO study for
# comparability).
# ---------------------------------------------------------------------------
FUNCTION_IDS = list(range(1, 26))
DIMENSIONS = [16, 64, 100, 625]      # all perfect squares (required by f23)
INSTANCE = 1

POP_SIZE = 200
N_GEN = 50                           # budget/run = POP_SIZE * (N_GEN + 1) evals
SEL_RATIO = 0.5
N_RUNS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_selected_data")
)


# ---------------------------------------------------------------------------
# Selection methods (weighted-probability schemes) -- identical to
# compare_weighted_edas_pbo.py.
# ---------------------------------------------------------------------------
SELECTION_METHODS = {
    "FP":  {"weighting": "proportional", "beta": 1.0, "replacement": "elitist"},
    "BZ":  {"weighting": "boltzmann",    "beta": 1.0, "replacement": "elitist"},
    "RTS": {"weighting": "boltzmann",    "beta": 1.0, "replacement": "rts"},
}
SELECTION_ORDER = ["FP", "BZ", "RTS"]


def rts_window_size(pop_size):
    """hBOA-style restricted-tournament window (~ pop/20, at least 2)."""
    return max(2, pop_size // 20)


def apply_selection_method(eda, sel, pop_size):
    """Configure an already-built EDA for one selection method: set the
    customized-selection weighting and swap in the replacement mechanism, so the
    same model-learner is exercised under every scheme."""
    cfg = SELECTION_METHODS[sel]
    eda.selection_weighting = cfg["weighting"]
    eda.weighting_beta = cfg["beta"]
    if cfg["replacement"] == "elitist":
        eda.components.replacement = ElitistReplacement(n_elite=1)
    elif cfg["replacement"] == "rts":
        eda.components.replacement = RestrictedTournamentReplacement(
            window_size=rts_window_size(pop_size))
    else:  # pragma: no cover - guarded by SELECTION_METHODS
        raise ValueError(f"Unknown replacement for selection method {sel!r}")
    return eda


# ---------------------------------------------------------------------------
# Algorithm builders.  Each returns a *core* EDA; the selection method's
# replacement is applied afterwards by apply_selection_method().
# ---------------------------------------------------------------------------
def _wrapped(cls, **extra):
    """Builder from a plug-and-play wrapper class (returns its inner EDA).

    Extra keyword arguments (e.g. ``k`` for MK-EDA, ``max_clique_size`` for the
    MN-FDA variants) are forwarded to the wrapper constructor.
    """
    def build(n_vars, card, fitness, pop_size, n_gen, sel_ratio, seed):
        alg = cls(
            n_vars=n_vars, cardinality=card, fitness_func=fitness,
            pop_size=pop_size, n_gen=n_gen, selection_ratio=sel_ratio,
            random_seed=seed, **extra,
        )
        return alg._eda
    return build


# Ordered registry: name -> builder.  Names encode the hyper-parameter so the
# IOH folder ``<name>__<sel>`` is self-describing.
ALGORITHM_BUILDERS = {
    "AffEDA":        _wrapped(AffEDA),
    "AffEDASparse":  _wrapped(AffEDASparse),
    "MKEDA2":        _wrapped(MKEDA, k=2),
    "MKEDA3":        _wrapped(MKEDA, k=3),
    "MKEDA4":        _wrapped(MKEDA, k=4),
    "MNFDAS3":       _wrapped(MNFDAS,        max_clique_size=3),
    "MNFDAS4":       _wrapped(MNFDAS,        max_clique_size=4),
    "MNFDAS5":       _wrapped(MNFDAS,        max_clique_size=5),
    "MNFDASparse3":  _wrapped(MNFDASparse,   max_clique_size=3),
    "MNFDASparse4":  _wrapped(MNFDASparse,   max_clique_size=4),
    "MNFDASparse5":  _wrapped(MNFDASparse,   max_clique_size=5),
    "MNFDASSparse3": _wrapped(MNFDASSparse,  max_clique_size=3),
    "MNFDASSparse4": _wrapped(MNFDASSparse,  max_clique_size=4),
    "MNFDASSparse5": _wrapped(MNFDASSparse,  max_clique_size=5),
    "MNFDAP3":       _wrapped(MNFDAP,        max_clique_size=3),
    "MNFDAP4":       _wrapped(MNFDAP,        max_clique_size=4),
    "MNFDAP5":       _wrapped(MNFDAP,        max_clique_size=5),
}
ALGORITHM_NAMES = list(ALGORITHM_BUILDERS.keys())


def folder_name(alg, sel):
    """IOH data-folder / algorithm_name reflecting algorithm and selection."""
    return f"{alg}__{sel}"


def make_fitness(problem):
    """Wrap an ioh problem as a pateda fitness function (one call = one eval)."""
    def fitness(solution):
        return float(problem(np.asarray(solution, dtype=int)))
    return fitness


def build_configured_eda(alg, sel, problem, pop_size, n_gen, sel_ratio, seed):
    """Build the EDA for one (algorithm, selection-method) and configure it."""
    builder = ALGORITHM_BUILDERS[alg]
    eda = builder(
        problem.meta_data.n_variables, 2, make_fitness(problem),
        pop_size, n_gen, sel_ratio, seed,
    )
    return apply_selection_method(eda, sel, pop_size)


# ---------------------------------------------------------------------------
# Argument parsing / local comparison main
# ---------------------------------------------------------------------------
def parse_args(argv):
    dims, fids, n_runs = DIMENSIONS, FUNCTION_IDS, N_RUNS
    algs, sels = ALGORITHM_NAMES, list(SELECTION_ORDER)

    if len(argv) > 1:
        dims = [int(d) for d in argv[1].split(",")]
    if len(argv) > 2 and argv[2].lower() != "all":
        fids = [int(f) for f in argv[2].split(",")]
    if len(argv) > 3:
        n_runs = int(argv[3])
    if len(argv) > 4 and argv[4].lower() != "all":
        wanted = [a for a in argv[4].split(",")]
        unknown = [a for a in wanted if a not in ALGORITHM_BUILDERS]
        if unknown:
            raise ValueError(f"Unknown algorithm names: {unknown}")
        algs = wanted
    if len(argv) > 5 and argv[5].lower() != "all":
        sels = [s for s in argv[5].split(",")]
        unknown = [s for s in sels if s not in SELECTION_METHODS]
        if unknown:
            raise ValueError(f"Unknown selection methods: {unknown}")
    return dims, fids, n_runs, algs, sels


def main():
    dims, fids, n_runs, algs, sels = parse_args(sys.argv)
    seeds = list(range(1, n_runs + 1))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("PBO suite comparison of SELECTED weighted-probability EDAs")
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Dimensions:       {dims}")
    print(f"Functions:        {fids}")
    print(f"Instance:         {INSTANCE}")
    print(f"Algorithms:       {algs}")
    print(f"Selections:       {sels}")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {N_GEN}")
    print(f"Selection ratio:  {SEL_RATIO}")
    print(f"Runs (seeds):     {seeds}")
    print(f"Budget per run:   {POP_SIZE * (N_GEN + 1)} evaluations")

    name_w = 14
    for alg in algs:
        for sel in sels:
            tag = folder_name(alg, sel)
            print(f"\n{'=' * 84}\nAlgorithm: {tag}\n{'=' * 84}")
            logger = ioh.logger.Analyzer(
                triggers=[ioh.logger.trigger.ON_IMPROVEMENT],
                root=OUTPUT_ROOT,
                folder_name=tag,
                algorithm_name=tag,       # merge key: {ALG}__{SEL}
                algorithm_info=(f"pateda pop={POP_SIZE} gen={N_GEN} "
                                f"sel={SEL_RATIO} method={sel}"),
            )
            for dim in dims:
                for fid in fids:
                    try:
                        problem = ioh.get_problem(
                            fid, instance=INSTANCE, dimension=dim,
                            problem_class=ioh.ProblemClass.PBO)
                    except Exception as exc:
                        print(f"{tag:<{name_w}} f{fid} dim={dim}: "
                              f"UNAVAILABLE -- {exc}")
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
          "python scripts/analyze_selected_pbo_results.py")


if __name__ == "__main__":
    main()
