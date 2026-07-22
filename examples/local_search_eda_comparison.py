"""
Comparing EDA pipelines with different local-search components

This script hybridizes a global EDA (UMDA as the probabilistic global searcher)
with each of the budget/subset-aware local optimizers in pateda, and compares
the resulting *memetic* EDA pipelines on a combinatorial problem: the NK
landscape, a tunably rugged black-box function with many local optima, on which
local search is known to help a global searcher (Radetic, Pelikan & Goldberg,
2009).

All local optimizers share the same interface, so each pipeline differs by a
single component.  The optimizers compared are:

    - DeterministicHillClimber        (steepest-ascent single-flip, the DHC)
    - FirstImprovementHillClimber     (first-improvement descent)
    - StochasticHillClimber           (random-mutation hill climbing, RMHC)
    - SimulatedAnnealing              (Metropolis, auto initial temperature)
    - VariableNeighborhoodSearch      (basic VNS: shake + descent)
    - ReducedVariableNeighborhoodSearch (RVNS: shake only)

The local search is applied, after sampling, to a fraction of the population
(``subset_fraction``) with a shared per-generation evaluation budget
(``evaluation_budget``) -- the two knobs that control local-search intensity.

Fair accounting.  A single counter wraps the fitness function, so every
evaluation -- whether spent by the EDA or by the local search -- is counted.
Since local search spends extra evaluations, the comparison also includes a
"Baseline (matched budget)" run: the plain EDA given enough extra generations to
consume as many evaluations as the memetic pipelines.  If a memetic pipeline
beats that baseline, its advantage is genuine and not merely the effect of more
evaluations.

The script has three parts:

    1. Pipeline comparison of every optimizer on a binary NK landscape.
    2. Local-search intensity sweep (subset_fraction x evaluation_budget) for
       one optimizer, showing how the two knobs trade evaluations for quality.
    3. A non-binary (integer) NK landscape run, to confirm the optimizers work
       for discrete problems of arbitrary cardinality.

Usage
-----
    python3 local_search_eda_comparison.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.local_optimization import (
    DeterministicHillClimber,
    FirstImprovementHillClimber,
    StochasticHillClimber,
    SimulatedAnnealing,
    VariableNeighborhoodSearch,
    ReducedVariableNeighborhoodSearch,
)
from pateda.functions.discrete_binary.problems.nk_landscape import (
    create_nk_objective_function,
)
from pateda.functions.discrete_non_binary.toy_functions.integer_functions import (
    create_integer_nk_objective_function,
)


# ---------------------------------------------------------------------------
# Evaluation-counting wrapper (fair accounting across EDA + local search)
# ---------------------------------------------------------------------------

class EvalCounter:
    """Wrap a scalar fitness function and count every evaluation."""

    def __init__(self, func):
        self.func = func
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return float(self.func(x))


# ---------------------------------------------------------------------------
# Shared EDA configuration
# ---------------------------------------------------------------------------

POP_SIZE = 200
GENERATIONS = 20
TRUNCATION = 0.5
SUBSET_FRACTION = 0.5
EVAL_BUDGET = 2000           # local-search evaluations shared per generation


def run_pipeline(fitness_raw, n_vars, card, local_opt_factory, seed,
                 generations=GENERATIONS, pop_size=POP_SIZE):
    """Run one memetic EDA and return (best_fitness, total_evals, gen_found)."""
    counter = EvalCounter(fitness_raw)
    local_opt = None if local_opt_factory is None else local_opt_factory(seed)
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=pop_size),
        selection=TruncationSelection(ratio=TRUNCATION),
        replacement=ElitistReplacement(),
        local_opt=local_opt,
        stop_condition=MaxGenerations(max_gen=generations),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=counter,
        cardinality=np.full(n_vars, card),
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness_overall, counter.n, stats.generation_found


# ---------------------------------------------------------------------------
# Optimizer factories (all share subset_fraction + evaluation_budget)
# ---------------------------------------------------------------------------

def optimizer_factories(subset_fraction=SUBSET_FRACTION, budget=EVAL_BUDGET):
    """Return {name: factory(seed) -> optimizer}; all share the same knobs."""
    common = dict(subset_fraction=subset_fraction, evaluation_budget=budget)
    return {
        "DeterministicHC": lambda s: DeterministicHillClimber(seed=s, **common),
        "FirstImprovementHC": lambda s: FirstImprovementHillClimber(seed=s, **common),
        "StochasticHC(RMHC)": lambda s: StochasticHillClimber(seed=s, **common),
        "SimulatedAnnealing": lambda s: SimulatedAnnealing(seed=s, **common),
        "VNS(basic)": lambda s: VariableNeighborhoodSearch(seed=s, **common),
        "VNS(reduced)": lambda s: ReducedVariableNeighborhoodSearch(seed=s, **common),
    }


# ---------------------------------------------------------------------------
# Part 1 — pipeline comparison
# ---------------------------------------------------------------------------

def pipeline_comparison(base_seed, n_runs=5):
    n_vars, k, card = 40, 4, 2
    print("=" * 78)
    print(f"1. Memetic EDA comparison on NK landscape (n={n_vars}, k={k}, binary)")
    print("=" * 78)
    print(f"   global model=UMDA, pop={POP_SIZE}, gens={GENERATIONS}, "
          f"truncation={TRUNCATION}")
    print(f"   local search: subset_fraction={SUBSET_FRACTION}, "
          f"evaluation_budget={EVAL_BUDGET}/gen, runs={n_runs}")
    print(f"\n   {'pipeline':<24} | {'best fit':>9} | {'total evals':>11} |"
          f" {'gen*':>4}")
    print("   " + "-" * 60)

    # A fresh NK instance per seed (averaged), shared across all pipelines.
    def make_fitness(seed):
        obj = create_nk_objective_function(n_vars, k, random_seed=seed)
        return obj.nk_landscape.evaluate

    # Baseline (no local search).
    factories = {"Baseline (no LS)": None}
    factories.update(optimizer_factories())

    results = {}
    for name, factory in factories.items():
        best, evals, gens = [], [], []
        for r in range(n_runs):
            seed = base_seed + 1000 * r
            b, e, g = run_pipeline(make_fitness(seed), n_vars, card, factory, seed)
            best.append(b); evals.append(e); gens.append(g if g is not None else GENERATIONS)
        results[name] = (np.mean(best), np.mean(evals))
        print(f"   {name:<24} | {np.mean(best):>9.4f} | {np.mean(evals):>11.0f} |"
              f" {np.mean(gens):>4.1f}")

    # Matched-budget baseline: plain EDA with extra generations to consume as
    # many evaluations as the memetic pipelines (fairness control).
    ls_evals = np.mean([results[n][1] for n in results if n != "Baseline (no LS)"])
    base_evals_per_gen = results["Baseline (no LS)"][1] / GENERATIONS
    matched_gens = int(round(ls_evals / base_evals_per_gen))
    mb_best, mb_evals = [], []
    for r in range(n_runs):
        seed = base_seed + 1000 * r
        b, e, _ = run_pipeline(make_fitness(seed), n_vars, card, None, seed,
                               generations=matched_gens)
        mb_best.append(b); mb_evals.append(e)
    print("   " + "-" * 60)
    print(f"   {'Baseline (matched)':<24} | {np.mean(mb_best):>9.4f} |"
          f" {np.mean(mb_evals):>11.0f} | {matched_gens:>4d}")
    print(f"\n   'gen*' = generation the best solution was first found.")
    print(f"   The matched baseline uses ~{matched_gens} generations to spend the")
    print(f"   same evaluations the memetic pipelines spend in {GENERATIONS}.")
    print()


# ---------------------------------------------------------------------------
# Part 2 — local-search intensity sweep
# ---------------------------------------------------------------------------

def intensity_sweep(base_seed, n_runs=3):
    n_vars, k, card = 40, 4, 2
    print("=" * 78)
    print("2. Local-search intensity: subset_fraction x evaluation_budget "
          "(DeterministicHC)")
    print("=" * 78)
    print(f"   NK landscape (n={n_vars}, k={k}), UMDA global, pop={POP_SIZE}, "
          f"gens={GENERATIONS}, runs={n_runs}\n")

    fractions = [0.0, 0.25, 0.5, 1.0]
    budgets = [500, 2000, 8000]

    def make_fitness(seed):
        obj = create_nk_objective_function(n_vars, k, random_seed=seed)
        return obj.nk_landscape.evaluate

    header = "   frac \\ budget |" + "".join(f" {b:>10}" for b in budgets)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for frac in fractions:
        cells = []
        for budget in budgets:
            if frac == 0.0:
                factory = None            # no local search
            else:
                factory = (lambda s, fr=frac, bg=budget:
                           DeterministicHillClimber(
                               subset_fraction=fr, evaluation_budget=bg, seed=s))
            vals = []
            for r in range(n_runs):
                seed = base_seed + 1000 * r
                b, _, _ = run_pipeline(make_fitness(seed), n_vars, card, factory, seed)
                vals.append(b)
            cells.append(f"{np.mean(vals):>10.4f}")
        label = "no LS" if frac == 0.0 else f"{frac:.2f}"
        print(f"   {label:>12} |" + "".join(cells))
    print("\n   The dominant effect is local search vs. none (the 'no LS' row).")
    print("   Beyond that, more intensity -- a larger fraction and/or budget --")
    print("   tends to help but with diminishing returns and run-to-run noise")
    print("   (only a few seeds here); both knobs give the practitioner direct")
    print("   control over how many evaluations the local search consumes.")
    print()


# ---------------------------------------------------------------------------
# Part 3 — non-binary (integer) problem
# ---------------------------------------------------------------------------

def non_binary_demo(base_seed, n_runs=3):
    n_vars, k, card = 24, 3, 5
    print("=" * 78)
    print(f"3. Non-binary confirmation: integer NK landscape "
          f"(n={n_vars}, k={k}, cardinality={card})")
    print("=" * 78)
    print(f"   UMDA global, pop={POP_SIZE}, gens={GENERATIONS}, runs={n_runs}\n")
    print(f"   {'pipeline':<24} | {'best fit':>9} | {'total evals':>11}")
    print("   " + "-" * 50)

    def make_fitness(seed):
        obj = create_integer_nk_objective_function(n_vars, k, cardinality=card,
                                                   random_seed=seed)
        return obj.nk_landscape.evaluate

    factories = {
        "Baseline (no LS)": None,
        "DeterministicHC": lambda s: DeterministicHillClimber(
            subset_fraction=SUBSET_FRACTION, evaluation_budget=EVAL_BUDGET, seed=s),
        "SimulatedAnnealing": lambda s: SimulatedAnnealing(
            subset_fraction=SUBSET_FRACTION, evaluation_budget=EVAL_BUDGET, seed=s),
        "VNS(basic)": lambda s: VariableNeighborhoodSearch(
            subset_fraction=SUBSET_FRACTION, evaluation_budget=EVAL_BUDGET, seed=s),
    }
    for name, factory in factories.items():
        best, evals = [], []
        for r in range(n_runs):
            seed = base_seed + 1000 * r
            b, e, _ = run_pipeline(make_fitness(seed), n_vars, card, factory, seed)
            best.append(b); evals.append(e)
        print(f"   {name:<24} | {np.mean(best):>9.4f} | {np.mean(evals):>11.0f}")
    print("\n   The same optimizers work unchanged for cardinality > 2.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("#" * 78)
    print("# Local-search components for EDAs — pipeline comparison")
    print(f"# seed = {seed}")
    print("#" * 78)
    print()
    pipeline_comparison(seed)
    intensity_sweep(seed)
    non_binary_demo(seed)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)
