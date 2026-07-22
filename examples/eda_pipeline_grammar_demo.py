"""
EDA pipeline grammar: feasibility and coverage

Demonstrates the context-free grammar of :mod:`pateda.pipelines.grammar`, which
generates *feasible EDA pipelines* -- consistent assemblies of seeding,
selection, learning, sampling, replacement, optional local search / mutation and
a stopping condition.  The grammar is made rich by the MODMOD / MODCONV model
operators (:mod:`pateda.pipelines.model_operators`), which let a learning method
of one model type feed a sampler of another (e.g. a Bayesian-network learner
reaching the factorized samplers via ``bn_to_factorized``, or a tree pruned into
a forest before sampling).

The script:

  1. samples random pipelines from the grammar and shows a few;
  2. builds and runs each on a small discrete problem, reporting the feasibility
     rate (it need not be 100%: some run-time edge cases are expected);
  3. reports the grammar's coverage of pateda's discrete-EDA component
     implementations, per category.

A later phase (not implemented here) will use this grammar to seed a GA-like
meta-optimizer over the space of pipelines.

Usage
-----
    python3 eda_pipeline_grammar_demo.py [seed] [n_pipelines]
"""

import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda.pipelines import (
    sample_derivation, parse_derivation, build_pipeline,
    grammar_terminals_by_category,
)


# ---------------------------------------------------------------------------
# Reference sets: discrete-EDA component implementations available in pateda
# (the "universe" against which grammar coverage is measured).
# ---------------------------------------------------------------------------

DISCRETE_UNIVERSE = {
    "seeding": ["RandomInit", "BiasInit", "SeedThisPop", "SeedingUnitationConstraint"],
    "selection": ["TruncationSelection", "TournamentSelection", "BoltzmannSelection",
                  "ProportionalSelection", "RankingSelection",
                  "StochasticUniversalSampling"],
    "learner": [
        "LearnUMDA", "LearnPBIL", "LearnFDA", "LearnCFDA", "LearnCUMDA", "LearnBMDA",
        "LearnBSC", "LearnMIMIC", "LearnMNFDA", "LearnMNFDAG", "LearnMNFDAR",
        "LearnMNFDAGR", "LearnTreeModel", "LearnTreeModelR", "LearnTreeModelM",
        "LearnMixtureTrees", "LearnAffinityFactorization",
        "LearnAffinityFactorizationElim", "LearnMarkovChain", "LearnMOA",
        "LearnEBNA", "LearnBOA", "LearnHBOA", "LearnLFDA", "LearnPADA",
        "LearnIntFDA", "LearnRegularizedMarkov",
    ],
    "sampler": ["SampleFDA", "SampleBayesianNetwork", "SampleGibbs", "SampleIntFDA",
                "SampleRegularizedMarkov", "SampleMarkovChain", "SampleMixtureTrees",
                "SamplePartialFDA", "SampleCFDA", "SampleCUMDA"],
    "local_opt": ["DeterministicHillClimber", "FirstImprovementHillClimber",
                  "StochasticHillClimber", "SimulatedAnnealing",
                  "VariableNeighborhoodSearch", "ReducedVariableNeighborhoodSearch",
                  "SubstructuralLocalSearch", "DiscreteGreedySearch",
                  "DiscreteSimulatedAnnealing", "DiscreteSimulatedAnnealingLinear"],
    "mutation": ["RandomResetMutation", "FrequencyBalanceMutation",
                 "FrequencyBalanceMultivalueMutation", "bit_flip_mutation"],
    "replacement": ["ElitistReplacement", "GenerationalReplacement", "RTRReplacement"],
    "stop": ["MaxGenerations", "MaxGenerationsOrOptimum"],
    "modop": ["bn_to_factorized", "prune_factorized", "tree_to_forest", "tree_to_malign"],
}


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def feasibility(base_seed, n_pipelines, n_vars=14, pop_size=40, n_gen=3):
    print("=" * 78)
    print(f"1. Feasibility of {n_pipelines} random pipelines "
          f"(n={n_vars}, pop={pop_size}, {n_gen} generations)")
    print("=" * 78)
    rng = np.random.default_rng(base_seed)
    card = np.full(n_vars, 2)

    def onemax(x):
        return float(np.sum(x))

    ok, fails = 0, {}
    used = {}                      # count how often each implementation is used
    modconv = 0
    t0 = time.time()
    for _ in range(n_pipelines):
        terms = sample_derivation(rng)
        spec = parse_derivation(terms)
        for name in (spec.selection, spec.learner, spec.sampler, spec.replacement,
                     spec.local_opt, spec.mutation, *spec.operators):
            if name:
                used[name] = used.get(name, 0) + 1
        if "bn_to_factorized" in spec.operators:
            modconv += 1
        try:
            eda, _ = build_pipeline(terms, n_vars, onemax, card,
                                    pop_size=pop_size, n_gen=n_gen, random_seed=0)
            eda.run(verbose=False)
            ok += 1
        except Exception as e:
            key = f"{type(e).__name__}"
            fails[key] = fails.get(key, 0) + 1

    print(f"\n  feasible pipelines : {ok}/{n_pipelines} = {100 * ok / n_pipelines:.1f}%")
    print(f"  MODCONV used (bn_to_factorized): {modconv} pipelines")
    print(f"  distinct implementations exercised: {len(used)}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    if fails:
        print("  failure modes:")
        for k, v in sorted(fails.items(), key=lambda x: -x[1]):
            print(f"    {v:3d}  {k}")
    print()
    return used


# ---------------------------------------------------------------------------
# Example pipelines
# ---------------------------------------------------------------------------

def show_examples(base_seed, k=6):
    print("=" * 78)
    print(f"2. Example generated pipelines")
    print("=" * 78)
    rng = np.random.default_rng(base_seed + 1)
    for i in range(k):
        spec = parse_derivation(sample_derivation(rng))
        print(f"  {i + 1}. {spec}")
    print()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage():
    print("=" * 78)
    print("3. Grammar coverage of pateda discrete-EDA component implementations")
    print("=" * 78)
    covered = grammar_terminals_by_category()
    print(f"\n  {'category':<12} | {'covered':>7} / {'total':<5} | {'%':>5}  | not covered")
    print("  " + "-" * 74)
    total_cov, total_all = 0, 0
    for cat, universe in DISCRETE_UNIVERSE.items():
        cov = set(covered.get(cat, [])) & set(universe)
        missing = sorted(set(universe) - cov)
        total_cov += len(cov)
        total_all += len(universe)
        pct = 100 * len(cov) / len(universe)
        miss = ", ".join(m.replace("Learn", "").replace("Selection", "")
                         for m in missing[:4])
        if len(missing) > 4:
            miss += ", ..."
        print(f"  {cat:<12} | {len(cov):>7} / {len(universe):<5} | {pct:>4.0f}% | {miss}")
    print("  " + "-" * 74)
    print(f"  {'OVERALL':<12} | {total_cov:>7} / {total_all:<5} | "
          f"{100 * total_cov / total_all:>4.0f}%")
    print("\n  Not covered are mostly components that need a problem-specific")
    print("  argument (an interaction matrix: Tree-EDA_r, MN-FDA_r; a template:")
    print("  partial samplers) or a special seeding/stop; these can be added to a")
    print("  problem-aware grammar.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42, n_pipelines=60):
    print("#" * 78)
    print("# EDA pipeline grammar: feasibility and coverage")
    print(f"# seed = {seed}")
    print("#" * 78 + "\n")
    show_examples(seed)
    feasibility(seed, n_pipelines)
    coverage()


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    npl = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(s, npl)
