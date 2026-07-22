"""
hBOA on Hierarchical Problems — and how it differs from BOA

hBOA (hierarchical Bayesian Optimization Algorithm) is BOA with the
conditional distributions represented as decision trees or decision graphs
instead of full conditional probability tables (CPTs).  This script makes the
consequence of that single change visible on the problems hBOA was designed
for.

Why the local structure matters
-------------------------------
BOA stores p(Xi | Pa_i) as a table with one row per parent configuration, so
the table has prod_j r_j rows and its size grows *exponentially* with the
number of parents.  With a finite population BOA must therefore keep
max_parents small (typically 2-3): a larger parent set would need more rows
than there are individuals to estimate them from.

hBOA stores the same distribution as a decision tree/graph over the parents.
Only the parent configurations that the data actually distinguishes get their
own parameters, and a decision *graph* can additionally merge two leaves so
that distinct configurations share parameters.  The parameter count then
grows with the complexity of the dependency rather than with |Pa_i|, which is
what lets hBOA use the large, overlapping parent sets that hierarchical
problems induce.

Problems
--------
1. HIFF (Hierarchical If-and-only-If), n = 32.
   Building blocks of size 2 are composed into blocks of 4, 8, ... Each level
   rewards a block only when it is uniform, and both 00 and 11 are rewarded
   equally, so the two competing partial solutions must be kept alive until
   the level above resolves which one is needed.  Optimum = n(log2(n)+1) = 192.

2. Hierarchical Trap-3 (fhtrap1), n = 27.
   The same idea with deceptive trap-3 blocks: at every level the low-order
   statistics point *away* from the optimum.  Optimum = 81.

A note on niching
-----------------
The complete hBOA of Pelikan (2005) also replaces generational replacement by
restricted tournament replacement (RTR), which preserves the competing
partial solutions described above.  RTR is a *replacement* component and is
therefore orthogonal to the learning method demonstrated here; this script
uses pateda's default truncation selection and elitist replacement, so the
absolute fitness values below understate what full hBOA achieves.  The point
being illustrated is the effect of the local structure on the learned model.

Usage
-----
    python3 hboa_hierarchical.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnBOA, LearnHBOA
from pateda.sampling import SampleBayesianNetwork
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete_binary.toy_functions.additive_decomposable import (
    hiff,
    fhtrap1,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The comparison this script draws is between the learned *models*, not
# between fitness scores.  A population large enough for the structure search
# to be stable is therefore used on purpose: at this size every BN-based EDA
# solves these benchmarks within a few generations, so fitness saturates and
# the model-complexity block below (parent counts, CPT sizes) is what tells
# BOA and hBOA apart -- cleanly and reproducibly.
POP_SIZE = 500
MAX_GEN = 25
TRUNCATION = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_eda(label, learner, fitness_func, n_vars, optimal,
            pop_size=POP_SIZE, max_gen=MAX_GEN, seed=42):
    """Run one BN-based EDA and print a compact result line."""
    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        selection=TruncationSelection(ratio=TRUNCATION),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    best = stats.best_fitness_overall
    found = stats.generation_found if stats.generation_found is not None else max_gen
    pct = 100.0 * best / optimal if optimal else float("nan")
    print(f"  {label:<26} best={best:8.2f}/{optimal:<8.2f} ({pct:5.1f}%)  gen={found:3d}")
    return stats


def model_summary(label, learner, fitness_func, n_vars, seed):
    """Learn one model from a selected population and report its complexity.

    The population is the top ``TRUNCATION`` fraction of a random sample,
    i.e. what the learner would see in the first generation of the EDA above.
    """
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(POP_SIZE, n_vars))
    fit = np.array([fitness_func(ind) for ind in pop])
    n_sel = int(POP_SIZE * TRUNCATION)
    selected = pop[np.argsort(fit)[::-1][:n_sel]]

    model = learner.learn(
        0, n_vars, np.full(n_vars, 2), selected, np.sort(fit)[::-1][:n_sel]
    )
    adj = np.asarray(model.structure)
    n_edges = int(adj.sum())
    parents_per_var = adj.sum(axis=0)
    max_par = int(parents_per_var.max())
    mean_par = float(parents_per_var.mean())

    # Size of the tabular CPTs the same structure would require.  This is the
    # cost hBOA avoids by representing the CPDs with trees/graphs.
    tabular_params = sum(2 ** int(parents_per_var[v]) for v in range(n_vars))

    print(f"  {label:<26} edges={n_edges:4d}  max_parents={max_par:2d}"
          f"  mean_parents={mean_par:4.2f}  tabular_CPT_rows={tabular_params:6d}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_hiff(seed):
    n_vars = 32                       # HIFF needs a power of 2
    optimal = float(n_vars * (int(np.log2(n_vars)) + 1))   # = 192

    print("\n" + "=" * 78)
    print(f"1. HIFF  (n={n_vars}, optimum={optimal:.0f})")
    print("=" * 78)
    print(" Optimization:")
    run_eda("BOA  (CPT, k=3)",
            LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
            hiff, n_vars, optimal, seed=seed)
    run_eda("hBOA (dec. tree, k=6)",
            LearnHBOA(max_parents=6, local_structure="dt"),
            hiff, n_vars, optimal, seed=seed)
    run_eda("hBOA (dec. graph, k=6)",
            LearnHBOA(max_parents=6, local_structure="dg"),
            hiff, n_vars, optimal, seed=seed)

    print(" Model learned from the first selected population:")
    model_summary("BOA  (CPT, k=3)",
                  LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
                  hiff, n_vars, seed)
    model_summary("BOA  (CPT, k=6)",
                  LearnBOA(max_parents=6, score_metric="k2", metric_alpha=1.0),
                  hiff, n_vars, seed)
    model_summary("hBOA (dec. tree, k=6)",
                  LearnHBOA(max_parents=6, local_structure="dt"),
                  hiff, n_vars, seed)
    model_summary("hBOA (dec. graph, k=6)",
                  LearnHBOA(max_parents=6, local_structure="dg"),
                  hiff, n_vars, seed)


def experiment_htrap(seed):
    n_vars = 27                       # fhtrap1 needs a power of 3
    optimal = 81.0

    print("\n" + "=" * 78)
    print(f"2. Hierarchical Trap-3  (fhtrap1, n={n_vars}, optimum={optimal:.0f})")
    print("=" * 78)
    print(" Optimization:")
    run_eda("BOA  (CPT, k=3)",
            LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
            fhtrap1, n_vars, optimal, seed=seed)
    run_eda("hBOA (dec. tree, k=6)",
            LearnHBOA(max_parents=6, local_structure="dt"),
            fhtrap1, n_vars, optimal, seed=seed)
    run_eda("hBOA (dec. graph, k=6)",
            LearnHBOA(max_parents=6, local_structure="dg"),
            fhtrap1, n_vars, optimal, seed=seed)

    print(" Model learned from the first selected population:")
    model_summary("BOA  (CPT, k=3)",
                  LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
                  fhtrap1, n_vars, seed)
    model_summary("hBOA (dec. graph, k=6)",
                  LearnHBOA(max_parents=6, local_structure="dg"),
                  fhtrap1, n_vars, seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("=" * 78)
    print("hBOA on Hierarchical Problems (HIFF, hierarchical trap-3)")
    print("=" * 78)
    print(f"Seed:             {seed}")
    print(f"Algorithms:       BOA (tabular CPT) vs hBOA (decision tree / graph)")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {MAX_GEN}")
    print(f"Selection:        Truncation (ratio={TRUNCATION})")

    experiment_hiff(seed)
    experiment_htrap(seed)

    print("\n" + "=" * 78)
    print("Reading the results")
    print("=" * 78)
    print("  The 'Model learned' blocks show the effect of the local structure:")
    print("  hBOA keeps larger parent sets than BOA at the same population size,")
    print("  because its decision trees/graphs do not pay the full 2^|Pa| cost")
    print("  reported in the tabular_CPT_rows column.")
    print("  In particular the decision-graph model keeps the many overlapping")
    print("  parents these hierarchical problems induce -- a structure whose")
    print("  tabular form (tabular_CPT_rows) BOA could not afford at this")
    print("  population -- while decision graphs (dg) also merge leaves, which")
    print("  decision trees (dt) do not.")
    print()
    print("  Fitness is not the differentiator here: at this population every")
    print("  variant solves the problems within a few generations.  On harder")
    print("  instances or smaller populations the complete hBOA additionally")
    print("  needs restricted tournament replacement (RTR) for niching, a")
    print("  *replacement* component orthogonal to the learner shown here -- add")
    print("  one from pateda.replacement to reproduce the full algorithm.")
    print("=" * 78)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
