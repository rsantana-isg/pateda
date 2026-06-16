"""
Compare bayes_nets Bayesian-network learning algorithms *inside* EDAs on
real-world benchmarks with mixed-cardinality super-variable encoding.

This is the EDA counterpart of ``edas_bayes_nets/scripts/compare_bn_learning.py``:
that script measures how well each structure-learning method recovers a known
ground-truth network from a fixed dataset; here we instead plug each method
into the generative loop of an Estimation of Distribution Algorithm and measure
the *optimisation* quality it produces.

Each generation the EDA:
  1. selects the best ``selection_ratio`` fraction of the population,
  2. learns a Bayesian network from the selected set with the chosen
     ``bayes_nets`` method (k2, bic, aic, stable_hc, tabu, dt, dg, gs, rcd,
     rpcd, pc, stable_pc) — this is delegated to :class:`pateda.learning.ebna.LearnEBNA`,
     which wraps :class:`bayes_nets.BayesianNetwork`,
  3. samples a new population from the learned network
     (:class:`pateda.sampling.bayesian_network.SampleBayesianNetwork`).

Mixed-cardinality encoding
--------------------------
Exactly as in ``compare_mixed_cardinality_edas_rw.py``: the 100 binary
variables of each benchmark are partitioned into non-overlapping groups of size
1..6 and each group is one super-variable of cardinality 2^|group|.  The
super-variable partition, problem loaders and interaction-matrix lifting are
imported from that script so both comparisons use an identical encoding.

Customized selection
--------------------
Every (method) is run under three per-individual probability weightings that
``bayes_nets`` consumes through its ``sample_weights`` argument:

  - "uniform"      : p_i = 1/N            (classic EDA)
  - "proportional" : p_i ∝ (shifted) fitness
  - "boltzmann"    : p_i ∝ exp(beta * standardised fitness)

Usage:
    python scripts/compare_bn_learning_edas_rw.py
"""

import os
import sys
import time
import traceback

import numpy as np

# Reuse the super-variable encoding + problem loaders from the sibling script so
# the two comparisons share an identical experimental setup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_mixed_cardinality_edas_rw import (  # noqa: E402
    make_grouping,
    load_problem,
    PROBLEMS,
    N_BINARY_VARS,
    MAX_GROUP_SIZE,
    GROUPING_SEED,
)

from pateda.core.eda import EDA  # noqa: E402
from pateda.core.components import EDAComponents  # noqa: E402
from pateda.seeding.random_init import RandomInit  # noqa: E402
from pateda.selection.truncation import TruncationSelection  # noqa: E402
from pateda.replacement.elitist import ElitistReplacement  # noqa: E402
from pateda.stop_conditions.max_generations import MaxGenerations  # noqa: E402
from pateda.learning.ebna import LearnEBNA  # noqa: E402
from pateda.sampling.bayesian_network import SampleBayesianNetwork  # noqa: E402


# ---------------------------------------------------------------------------
# bayes_nets learning methods (same registry as compare_bn_learning.py)
# ---------------------------------------------------------------------------
# Each entry is (display name, bayes_nets method string).  LearnEBNA forwards
# the method to BayesianNetwork.fit(method=...).
BN_METHODS = [
    # Constraint-based — independence-test driven; substantially slower per
    # generation than the score-based methods (each fit runs many CI tests).
    # Comment these out for a quick score-based-only comparison.
 #  ("GS",        "gs"),
    ("RCD",       "rcd"),
    ("RPCD",      "rpcd"),
    ("PC",        "pc"),
    ("Stable-PC", "stable_pc"),    
    # Score-based (tabular CPDs)
    ("K2",        "k2"),
    ("BIC-HC",    "bic"),
    ("AIC-HC",    "aic"),
    ("HC-Stable", "stable_hc"),
 #  ("Tabu",      "tabu"),
    # Local-structure CPD scoring
    ("DT-MDL",    "dt"),
    ("DG-Bayes",  "dg"),
 
]

# Customized-selection weightings (see module docstring).
WEIGHTINGS = [
    ("uniform",      1.0),
    ("proportional", 1.0),
    ("boltzmann",    5.0),
]

# EDA hyper-parameters.  Kept smaller than the classic-EDA comparison because
# Bayesian-network structure learning (especially the constraint-based methods)
# is markedly more expensive per generation.
POP_SIZE    = 1000
N_GEN       = 50
SEL_RATIO   = 0.30
MAX_PARENTS = 2
ALPHA       = 1.0
N_RUNS      = 3
SEEDS       = list(range(1, N_RUNS + 1))


# ---------------------------------------------------------------------------
# EDA construction
# ---------------------------------------------------------------------------

def build_bn_eda(method, n_vars, cardinality, fitness_func, seed, weighting, beta):
    """
    Assemble an EDA whose model is a Bayesian network learned with the given
    ``bayes_nets`` ``method`` and whose selected individuals are weighted by the
    requested customized-selection scheme.
    """
    learner = LearnEBNA(
        max_parents=MAX_PARENTS,
        score_metric=method,
        alpha=ALPHA,
        limit_joint_table_size=True,
    )
    sampler = SampleBayesianNetwork(n_samples=POP_SIZE)

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=SEL_RATIO),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(N_GEN),
        replacement=ElitistReplacement(n_elite=1),
    )

    return EDA(
        POP_SIZE, n_vars, fitness_func, cardinality, components,
        random_seed=seed,
        selection_weighting=weighting,
        weighting_beta=beta,
    )


def run_all_seeds(method, n_vars, cardinality, fitness_func, weighting, beta):
    bests, times = [], []
    for seed in SEEDS:
        eda = build_bn_eda(
            method, n_vars, cardinality, fitness_func, seed, weighting, beta
        )
        t0 = time.time()
        stats, _ = eda.run(verbose=False)
        times.append(time.time() - t0)
        bests.append(float(stats.best_fitness_overall))
    return bests, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # One fixed super-variable grouping shared across every problem and seed —
    # identical to compare_mixed_cardinality_edas_rw.py.
    groups = make_grouping(N_BINARY_VARS, MAX_GROUP_SIZE, GROUPING_SEED)
    cardinality_vec = np.array([2 ** len(g) for g in groups], dtype=int)
    group_sizes = sorted([len(g) for g in groups], reverse=True)
    card_counts: dict = {}
    for c in cardinality_vec:
        card_counts[c] = card_counts.get(c, 0) + 1

    print(f"\n{'=' * 90}")
    print("BN-learning EDAs on mixed-cardinality real-world benchmarks")
    print(f"  {N_BINARY_VARS} binary variables  →  {len(groups)} super-variables")
    print(f"  Group sizes (desc):  {group_sizes}")
    print(
        "  Cardinality counts:  "
        + "  ".join(f"card={c}×{cnt}" for c, cnt in sorted(card_counts.items()))
    )
    print(
        f"  pop_size={POP_SIZE}  n_gen={N_GEN}  selection_ratio={SEL_RATIO}  "
        f"max_parents={MAX_PARENTS}  n_runs={N_RUNS}"
    )
    print(f"{'=' * 90}")

    name_w = 10

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_sv, card_vec, _super_interaction, optimal = load_problem(
            problem_type, instance_name, groups
        )

        print(f"\n{'=' * 90}")
        print(
            f"Problem: {problem_type}  Instance: {instance_name}  "
            f"(n_supervars={n_sv}, optimal={optimal})"
        )
        print(f"{'=' * 90}")

        problem_tag = f"{problem_type} {instance_name}"

        for weighting, beta in WEIGHTINGS:
            wlabel = weighting if weighting != "boltzmann" else f"boltzmann(beta={beta:g})"
            print(f"\n-- weighting: {wlabel} " + "-" * (88 - len(wlabel)))

            for disp_name, method in BN_METHODS:
                try:
                    bests, times = run_all_seeds(
                        method, n_sv, card_vec, fitness_func, weighting, beta
                    )
                    mean_best = float(np.mean(bests))
                    mean_time = float(np.mean(times))
                    bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                    print(
                        f"{disp_name:<{name_w}} {problem_tag:<16}: "
                        f"{bests_str}  mean={mean_best:.4f}  time={mean_time:.2f}s"
                    )
                except Exception as exc:
                    print(f"{disp_name:<{name_w}} {problem_tag:<16}: ERROR -- {exc}")
                    traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
