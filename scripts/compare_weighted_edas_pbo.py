"""
Compare weighted-probability pateda EDAs on the PBO suite, under three
customized-selection schemes.

This is the *weighted-probability* companion of
``compare_discrete_edas_pbo.py``.  Every algorithm here learns its model from
the selected population using **customized selection** (Santana, Mendiburu &
Lozano, 2014): instead of weighting every selected solution equally (1/N), each
selected individual ``i`` carries a probability ``p_i`` derived from its fitness,
and the learning method weights its counts / probability tables by ``p``.  The
weighting is applied by the EDA core (``EDA.selection_weighting`` /
``weighting_beta``); truncation selection (ratio 0.5) is still applied first, so
``p`` re-weights the *selected* set.

Three *selection methods* (weighting + diversity mechanism) are compared:

    FP   truncation-0.5  +  fitness-proportional weighting  +  elitist replacement
    BZ   truncation-0.5  +  Boltzmann weighting (beta=1)     +  elitist replacement
    RTS  truncation-0.5  +  Boltzmann weighting (beta=1)     +  restricted-tournament
                                                                (hBOA niching) replacement

Algorithms (all run under each of FP, BZ, RTS):

  * Base discrete EDAs (from ``compare_discrete_edas_pbo.py`` minus ``BSC``):
    UMDA, BMDA, TreeEDA, MIMIC, PBIL, EBNA, BOA, AffEDA, MKEDA, MTED, MNFDA, FDA.
  * All plug-and-play BN-based EDAs (in addition to EBNA/BOA):
    LFDA, HBOA, SARTRE_EDA, BINOTEARS_EDA, PCBN_EDA, HSARTRE_EDA, HBINOTEARS_EDA.
  * Network-crossover EDAs (structure-guided recombination on the learned Tree
    linkage graph) with **substructural local search** instead of mutation:
    NetXBFS (BFS mask) and NetXRW (random-walk mask).
  * Int_FDA: the tree-based FDA for integer/high-cardinality variables.

Every fitness evaluation is recorded by an ``ioh.logger.Analyzer`` so the runs
can be post-processed with ``iohinspector`` (see
``analyze_weighted_pbo_results.py``).  Data is written to one folder per
(algorithm, selection-method) pair, the folder name reflecting both::

    results/pbo_weighted_data/<ALG>__<SEL>/     e.g.  UMDA__FP, EBNA__BZ, NetXBFS__RTS

Notes
-----
* Dimension 625 is *omitted* (prohibitively expensive for the model-building
  EDAs); the standard grid is {16, 64, 100}.  Run a timing probe first with
  ``scripts/time_weighted_edas.py``.
* ``MTED`` (mixture-of-trees) does not consume the per-sample weights ``p`` in
  the current ``bayes_nets``; under the weighting schemes it behaves as its
  uniform self.  It is kept for parity with ``compare_discrete_edas_pbo.py``.
* ``BINOTEARS_EDA`` / ``HBINOTEARS_EDA`` are binary-only; PBO is binary, so they
  apply throughout.

Usage (all arguments optional, positional):
    python scripts/compare_weighted_edas_pbo.py [dims] [fids] [n_runs] [algs] [sels]

    dims    comma-separated dimensions, e.g. "16,64,100"   (default: 16,64,100)
    fids    comma-separated PBO function ids or "all"       (default: all = 1..25)
    n_runs  number of independent runs per triple           (default: 5)
    algs    comma-separated algorithm names or "all"        (default: all)
    sels    comma-separated selection methods or "all"      (default: FP,BZ,RTS)

Examples:
    python scripts/compare_weighted_edas_pbo.py
    python scripts/compare_weighted_edas_pbo.py 16 1,2,18 3 UMDA,EBNA,NetXBFS FP,RTS
"""

import os
import sys
import time
import traceback

import numpy as np
import ioh

from pateda import (
    EDA, EDAComponents,
    UMDA, BMDA, TreeEDA, MIMIC, PBIL,
    EBNA, BOA, LFDA, HBOA,
    SARTRE_EDA, BINOTEARS_EDA, PCBN_EDA, HSARTRE_EDA, HBINOTEARS_EDA,
    AffEDA, MKEDA, MTED, MNFDA, FDA,
)
from pateda.seeding.random_init import RandomInit
from pateda.selection.truncation import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.replacement.niching import RestrictedTournamentReplacement
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.learning.tree import LearnTreeModel
from pateda.learning.int_fda import LearnIntFDA
from pateda.sampling.int_fda import SampleIntFDA
from pateda.sampling.network_crossover import SampleNetworkCrossover
from pateda.local_optimization.substructural_search import SubstructuralLocalSearch


# ---------------------------------------------------------------------------
# Experimental setup (kept consistent with compare_discrete_edas_pbo.py)
# ---------------------------------------------------------------------------
FUNCTION_IDS = list(range(1, 26))
DIMENSIONS = [16, 64, 100]          # 625 omitted (too expensive for BN EDAs)
INSTANCE = 1

POP_SIZE = 200
N_GEN = 50                          # base budget/run = POP_SIZE * (N_GEN + 1) evals
SEL_RATIO = 0.5
N_RUNS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_weighted_data")
)


# ---------------------------------------------------------------------------
# Selection methods (weighting + diversity mechanism)
# ---------------------------------------------------------------------------
# Each entry: weighting scheme, Boltzmann inverse-temperature (beta = 1/T),
# and the replacement/diversity mechanism.
SELECTION_METHODS = {
    "FP":  {"weighting": "proportional", "beta": 1.0, "replacement": "elitist"},
    "BZ":  {"weighting": "boltzmann",    "beta": 1.0, "replacement": "elitist"},
    "RTS": {"weighting": "boltzmann",    "beta": 1.0, "replacement": "rts"},
}
SELECTION_ORDER = ["FP", "BZ", "RTS"]


def rts_window_size(pop_size: int) -> int:
    """hBOA-style restricted-tournament window (~ pop/20, at least 2)."""
    return max(2, pop_size // 20)


def apply_selection_method(eda: EDA, sel: str, pop_size: int) -> EDA:
    """Configure an already-built EDA for one selection method.

    Sets the customized-selection weighting (``selection_weighting`` /
    ``weighting_beta``) and swaps in the replacement/diversity mechanism, so the
    same model-learner is exercised under every scheme.
    """
    cfg = SELECTION_METHODS[sel]
    eda.selection_weighting = cfg["weighting"]
    eda.weighting_beta = cfg["beta"]
    if cfg["replacement"] == "elitist":
        eda.components.replacement = ElitistReplacement(n_elite=1)
    elif cfg["replacement"] == "rts":
        eda.components.replacement = RestrictedTournamentReplacement(
            window_size=rts_window_size(pop_size)
        )
    else:  # pragma: no cover - guarded by SELECTION_METHODS
        raise ValueError(f"Unknown replacement for selection method {sel!r}")
    return eda


# ---------------------------------------------------------------------------
# Algorithm builders — each returns a *core* EDA (replacement is a placeholder
# that apply_selection_method() overrides).
# ---------------------------------------------------------------------------
def _as_card(card, n_vars):
    """Scalar/array cardinality -> 1-D int array (as the wrappers do internally)."""
    if np.ndim(card) == 0:
        return np.full(n_vars, int(card), dtype=int)
    return np.asarray(card, dtype=int)


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


def _build_netx(mask_method):
    """Network-crossover EDA: Tree linkage model + network crossover sampler +
    substructural local search (no mutation)."""
    def build(n_vars, card, fitness, pop_size, n_gen, sel_ratio, seed):
        card = _as_card(card, n_vars)
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=sel_ratio),
            learning=LearnTreeModel(alpha=1.0),
            sampling=SampleNetworkCrossover(
                n_samples=pop_size, mask_method=mask_method, seed=seed,
            ),
            local_opt=SubstructuralLocalSearch(
                neighborhood="both",
                max_substructure_size=6,
                subset_fraction=0.25,
                evaluation_budget=pop_size,   # ~1x sampling budget of extra evals/gen
                subset_selection="best",
                seed=seed,
            ),
            stop_condition=MaxGenerations(n_gen),
            replacement=ElitistReplacement(n_elite=1),   # placeholder
        )
        return EDA(pop_size, n_vars, fitness, card, components, random_seed=seed)
    return build


def _build_int_fda():
    """Int_FDA: tree-based FDA for integers (works on binary PBO too)."""
    def build(n_vars, card, fitness, pop_size, n_gen, sel_ratio, seed):
        card = _as_card(card, n_vars)
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=sel_ratio),
            learning=LearnIntFDA(),
            sampling=SampleIntFDA(n_samples=pop_size),
            stop_condition=MaxGenerations(n_gen),
            replacement=ElitistReplacement(n_elite=1),   # placeholder
        )
        return EDA(pop_size, n_vars, fitness, card, components, random_seed=seed)
    return build


# Ordered registry: name -> builder.  Order groups base / BN / NetX / Int_FDA.
ALGORITHM_BUILDERS = {
    # --- base discrete EDAs (compare_discrete_edas_pbo.py minus BSC) ---
    "UMDA":       _wrapped(UMDA),
    "BMDA":       _wrapped(BMDA),
    "TreeEDA":    _wrapped(TreeEDA),
    "MIMIC":      _wrapped(MIMIC),
    "PBIL":       _wrapped(PBIL),
    "EBNA":       _wrapped(EBNA),
    "BOA":        _wrapped(BOA),
    "AffEDA":     _wrapped(AffEDA),
    "MKEDA":      _wrapped(MKEDA),
    "MTED":       _wrapped(MTED),
    "MNFDA":      _wrapped(MNFDA),
    "FDA":        _wrapped(FDA),
    # --- all plug-and-play BN-based EDAs (beyond EBNA/BOA) ---
    "LFDA":       _wrapped(LFDA),
    "HBOA":       _wrapped(HBOA),
    "SARTRE":     _wrapped(SARTRE_EDA),
    "BINOTEARS":  _wrapped(BINOTEARS_EDA),
    "PCBN":       _wrapped(PCBN_EDA),
    "HSARTRE":    _wrapped(HSARTRE_EDA),
    "HBINOTEARS": _wrapped(HBINOTEARS_EDA),
    # --- network-crossover EDAs + substructural local search ---
    "NetXBFS":    _build_netx("bfs"),
    "NetXRW":     _build_netx("random_walk"),
    # --- Int_FDA ---
    "IntFDA":     _build_int_fda(),
}
ALGORITHM_NAMES = list(ALGORITHM_BUILDERS.keys())


def folder_name(alg: str, sel: str) -> str:
    """IOH data-folder / algorithm_name reflecting algorithm and selection method."""
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

    print("PBO suite comparison of weighted-probability pateda EDAs")
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Dimensions:       {dims}")
    print(f"Functions:        {fids}")
    print(f"Instance:         {INSTANCE}")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {N_GEN}")
    print(f"Selection ratio:  {SEL_RATIO}")
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
                root=OUTPUT_ROOT,
                folder_name=tag,
                algorithm_name=tag,
                algorithm_info=(f"pateda pop={POP_SIZE} gen={N_GEN} sel={SEL_RATIO} "
                                f"method={sel}"),
            )

            for dim in dims:
                for fid in fids:
                    try:
                        problem = ioh.get_problem(
                            fid, instance=INSTANCE, dimension=dim,
                            problem_class=ioh.ProblemClass.PBO,
                        )
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

    print("\nDone.  Analyze with: python scripts/analyze_weighted_pbo_results.py")


if __name__ == "__main__":
    main()
