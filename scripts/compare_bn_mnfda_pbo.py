"""Unified PBO registry: all BN-based EDAs (unbounded + treewidth-bounded Phase-2
variants) plus MN-FDA, for the cluster comparison at n in {16, 64, 625}.

Three algorithm groups share one ``build_configured_eda`` and the FP/BZ/RTS
selection machinery of ``compare_weighted_edas_pbo.py`` (so the IOH folder naming
``{ALG}__{SEL}`` and the existing ``analyze_weighted_pbo_results.py`` /
``make_pbo_latex_report.py`` pipeline work unchanged):

1. **Unbounded BN variants** -- the 11 report EDAs (``compare_bn_variants_pbo``):
   EBNA_BIC, EBNA_K2, EBNA_PC, LFDA_mp6, BOA_mp6, SARTRE_mp6, A1_dt..A5_ndg.

2. **Bounded (Phase-2) variants** -- ``BND{k}_{alg}``: the same learner restricted
   to an MI-guided k-tree scaffold so the learned BN's moral-triangulated
   treewidth is <= k *by construction* (``scripts/hboa_bounded_tw.LearnBoundedBN``;
   see ``docs/BN_bounded_clique_learning.md``).  Built by taking the base EDA
   (which fixes the correct sampler -- ``SampleLocalStructureBN`` for the
   decision-graph variants, ``SampleBayesianNetwork`` for the tabular ones) and
   swapping in the bounded learner.  These are the variants expected to remain
   feasible at n = 625, where the unbounded learners do not finish.

3. **MN-FDA** -- ``MNFDA`` (junction/clique factorization, ``max_clique_size=3``)
   and ``MNFDAG`` (its graph variant), the incumbent bounded-clique EDA to beat.

Usage mirrors compare_bn_variants_pbo (positional; the cluster uses
``run_bn_mnfda_pbo.py``)::

    python scripts/compare_bn_mnfda_pbo.py [dims] [fids] [n_runs] [algs] [sels]
"""

import os
import sys

import numpy as np
import ioh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Reuse the 11 unbounded BN builders + selection machinery + PBO constants.
import compare_bn_variants_pbo as BNV
from compare_bn_variants_pbo import (            # noqa: E402  (re-exported)
    SELECTION_METHODS, SELECTION_ORDER, apply_selection_method,
    make_fitness, folder_name, INSTANCE, FUNCTION_IDS,
    POP_SIZE, N_GEN, SEL_RATIO, MAX_PARENTS,
)
from hboa_bounded_tw import LearnBoundedBN, SampleBNWithMPC   # noqa: E402
from pateda import MNFDA, MNFDAP                    # noqa: E402

# n in {16, 64, 625} for this comparison (64 kept as the mid anchor; 100 dropped).
DIMENSIONS = [16, 64, 625]
OUTPUT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, "results", "pbo_bn_mnfda_data")
)

# The unbounded BN variants are the *bases* the bounded learners are built from,
# but they are NOT launched in this comparison (removed per the treewidth study).

# --- group 1: bounded (Phase-2) BN variants, PLS/BN sampling ----------------
# report alg -> bayes_nets bounded-learner kwargs (method + local structure).
BOUNDED_SPEC = {
    "EBNA_BIC":   dict(method="bic"),
    "EBNA_K2":    dict(method="k2_pen"),
    "EBNA_PC":    dict(method="stable_pc"),
    "LFDA_mp6":   dict(method="bic"),                       # weighted BIC via p
    "BOA_mp6":    dict(method="k2"),
    "SARTRE_mp6": dict(method="sartre"),
    "A1_dt":      dict(method="dt",     local_structure="dt"),
    "A2_mi":      dict(method="dg",     local_structure="dg"),
    "A3_fast":    dict(method="dg",     local_structure="dg", fast_local_scoring=True),
    "A4_mdl":     dict(method="dg",     local_structure="dg", max_leaves=32,
                       split_score="mdl"),
    "A5_ndg":     dict(method="dg_ndg", local_structure="dg"),
}
# Treewidth budget swept for the whole comparison.  ``k`` is the treewidth
# bound: the bounded BN-EDAs guarantee treewidth <= k, and MN-FDA is aligned to
# the *same* treewidth by setting its ``max_clique_size = k + 1`` (a k-tree /
# forest of max clique k+1 has treewidth k).  So at each k every algorithm shares
# the same treewidth budget.
BOUNDED_KS = [4, 8]
BOUNDED = {f"BND{k}_{base}": (base, k) for k in BOUNDED_KS for base in BOUNDED_SPEC}

# --- group 2: bounded BN variants + most-probable configuration -------------
# Analogue of MN-FDA-P: each generation the exact MPC of the bounded BN is
# computed by junction-tree max-product (tractable since treewidth <= k) and
# inserted as one individual; the rest are sampled by the base sampler.
BOUNDED_MPC = {f"{name}_MPC": spec for name, spec in BOUNDED.items()}

# --- group 3: MN-FDA (newest unified learner), treewidth-aligned per k -------
# MNFDA_k{k}  : sparse graph -> per-variable cliques -> running-intersection
#               forest (max_clique_size = k+1 -> treewidth k), PLS (SampleFDA).
# MNFDAP_k{k} : same learner + native MPC via SampleFDAWithMPC.
MNFDA_BUILDERS = {}
for _k in BOUNDED_KS:
    MNFDA_BUILDERS[f"MNFDA_k{_k}"] = (MNFDA, dict(max_clique_size=_k + 1))
    MNFDA_BUILDERS[f"MNFDAP_k{_k}"] = (MNFDAP, dict(max_clique_size=_k + 1))

ALGORITHM_NAMES = list(BOUNDED) + list(BOUNDED_MPC) + list(MNFDA_BUILDERS)


def _build_bounded(base, k, sel, problem, pop_size, n_gen, sel_ratio, seed):
    """Bounded (treewidth <= k) BN EDA: the base EDA (which fixes the correct
    sampler) with its learner swapped for the k-tree-scaffold learner."""
    eda = BNV.build_configured_eda(base, sel, problem, pop_size, n_gen,
                                   sel_ratio, seed)
    eda.components.learning = LearnBoundedBN(
        k=k, max_parents=MAX_PARENTS, alpha=1.0, **BOUNDED_SPEC[base])
    return eda


def build_configured_eda(alg, sel, problem, pop_size, n_gen, sel_ratio, seed):
    """Build one (algorithm, selection-method) EDA, dispatching by group."""
    if alg in BOUNDED:                                       # group 1
        base, k = BOUNDED[alg]
        return _build_bounded(base, k, sel, problem, pop_size, n_gen,
                              sel_ratio, seed)
    if alg in BOUNDED_MPC:                                   # group 2
        base, k = BOUNDED_MPC[alg]
        eda = _build_bounded(base, k, sel, problem, pop_size, n_gen,
                             sel_ratio, seed)
        eda.components.sampling = SampleBNWithMPC(
            base_sampler=eda.components.sampling, n_samples=pop_size)
        return eda
    if alg in MNFDA_BUILDERS:                                # group 3
        cls, extra = MNFDA_BUILDERS[alg]
        a = cls(n_vars=problem.meta_data.n_variables, cardinality=2,
                fitness_func=make_fitness(problem), pop_size=pop_size,
                n_gen=n_gen, selection_ratio=sel_ratio, random_seed=seed, **extra)
        return apply_selection_method(a._eda, sel, pop_size)
    raise ValueError(f"Unknown algorithm '{alg}'. Known: {ALGORITHM_NAMES}")


# ---------------------------------------------------------------------------
# Local smoke comparison (small grids); the cluster uses run_bn_mnfda_pbo.py.
# ---------------------------------------------------------------------------
def main():
    import time
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16]
    fids = ([int(f) for f in sys.argv[2].split(",")]
            if len(sys.argv) > 2 and sys.argv[2].lower() != "all" else [1, 18])
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    algs = (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4].lower() != "all"
            else ALGORITHM_NAMES)
    sels = (sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5].lower() != "all"
            else ["BZ"])
    print(f"Algorithms ({len(algs)}): {algs}")
    print(f"Dims={dims}  fids={fids}  sels={sels}  runs={n_runs}  pop={POP_SIZE}")
    for sel in sels:
        for dim in dims:
            for fid in fids:
                for alg in algs:
                    prob = ioh.get_problem(fid, instance=INSTANCE, dimension=dim,
                                           problem_class=ioh.ProblemClass.PBO)
                    bests = []
                    for seed in range(1, n_runs + 1):
                        eda = build_configured_eda(alg, sel, prob, POP_SIZE,
                                                   N_GEN, SEL_RATIO, seed)
                        t0 = time.time(); eda.run(verbose=False); dt = time.time() - t0
                        bests.append(float(prob.state.current_best.y)); prob.reset()
                    print(f"  {alg:16s} {sel} f{fid:<2} n={dim:<3} "
                          f"best={np.mean(bests):.4f}  time={dt:.2f}s")


if __name__ == "__main__":
    main()
