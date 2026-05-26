"""
Compare different selection methods on multiple benchmark functions

Demonstrates how selection pressure affects convergence on four
problems of increasing difficulty:
  1. OneMax        — unimodal, no deception
  2. Deceptive3    — 3-bit non-overlapping deceptive blocks
  3. Trap5         — 5-bit non-overlapping trap (harder deception)
  4. MH-Decep3     — alternative 3-bit deceptive landscape (two optima)
"""

import time
import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import (
    TruncationSelection,
    TournamentSelection,
    ProportionalSelection,
    RankingSelection,
    StochasticUniversalSampling,
    BoltzmannSelection,
)
from pateda.stop_conditions import MaxGenerations
from pateda.replacement import ElitistReplacement
from pateda.functions import onemax
from pateda.functions.discrete.deceptive import deceptive3
from pateda.functions.discrete.trap import trap_n
from pateda.functions.discrete.additive_decomposable import decep3_mh


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

def trap5(x: np.ndarray) -> float:
    """5-bit non-overlapping trap."""
    return trap_n(x, n_trap=5)


PROBLEMS = [
    {
        "name": "OneMax",
        "func": onemax,
        "n_vars": 30,
        "optimal": 30.0,
        "cardinality": 2,
        "max_gen": 20,
    },
    {
        "name": "Deceptive3",
        "func": deceptive3,
        "n_vars": 30,           # 10 blocks of 3
        "optimal": 30 / 3,      # = 10.0
        "cardinality": 2,
        "max_gen": 30,
    },
    {
        "name": "Trap5",
        "func": trap5,
        "n_vars": 25,           # 5 blocks of 5
        "optimal": 25.0,        # all blocks at max
        "cardinality": 2,
        "max_gen": 40,
    },
    {
        "name": "MH-Decep3",
        "func": decep3_mh,
        "n_vars": 30,           # 10 blocks of 3
        "optimal": 30.0,        # all blocks = 3 each (table max = 3)
        "cardinality": 2,
        "max_gen": 30,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_with_selection(selection_method, selection_name, problem):
    """Run UMDA with one selection method on one problem."""
    pop_size = 200
    n_vars = problem["n_vars"]
    max_gen = problem["max_gen"]
    cardinality = np.full(n_vars, int(problem["cardinality"]))

    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnUMDA(alpha=0.1),
        sampling=SampleFDA(n_samples=pop_size),
        selection=selection_method,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=problem["func"],
        cardinality=cardinality,
        components=components,
        random_seed=42,
    )

    t0 = time.time()
    statistics, _ = eda.run(verbose=False)
    elapsed = time.time() - t0

    return statistics, elapsed


SELECTION_METHODS = [
    (lambda: TruncationSelection(ratio=0.5),                             "Truncation (50%)"),
    (lambda: TournamentSelection(tournament_size=2, ratio=0.5),          "Tournament (k=2)"),
    (lambda: TournamentSelection(tournament_size=5, ratio=0.5),          "Tournament (k=5)"),
    (lambda: ProportionalSelection(ratio=0.5),                           "Proportional"),
    (lambda: RankingSelection(ratio=0.5, selection_pressure=1.5),        "Ranking (p=1.5)"),
    (lambda: StochasticUniversalSampling(ratio=0.5),                     "SUS"),
    (lambda: BoltzmannSelection(ratio=0.5, temperature=2.0),             "Boltzmann (T=2)"),
    (lambda: BoltzmannSelection(ratio=0.5, temperature=1.0),             "Boltzmann (T=1)"),
]


def run_problem(problem):
    """Run all selection methods on one problem and print results."""
    print(f"\n{'=' * 75}")
    print(f"Problem: {problem['name']}  "
          f"(n={problem['n_vars']}, optimal={problem['optimal']:.1f}, "
          f"max_gen={problem['max_gen']})")
    print(f"{'=' * 75}")
    print(f"{'Method':<28} {'Best':>8} {'%Opt':>7} {'Gen':>6} {'Time(s)':>9}")
    print("-" * 62)

    rows = []
    for make_method, name in SELECTION_METHODS:
        stats, elapsed = run_with_selection(make_method(), name, problem)
        best = stats.best_fitness_overall
        pct = best / problem["optimal"] * 100
        gen = stats.generation_found if stats.generation_found is not None else problem["max_gen"]
        rows.append((name, best, pct, gen, elapsed))
        print(f"{name:<28} {best:>8.3f} {pct:>6.1f}% {gen:>6} {elapsed:>9.3f}s")

    return rows


def main():
    print("=" * 75)
    print("Selection Method Comparison — UMDA on Four Benchmark Functions")
    print("=" * 75)

    all_results = {}
    for problem in PROBLEMS:
        all_results[problem["name"]] = run_problem(problem)

    # Cross-problem summary: best-method per problem
    print(f"\n{'=' * 75}")
    print("Summary: best method per problem (by % optimal reached)")
    print(f"{'=' * 75}")
    print(f"{'Problem':<14}", end="")
    for method_name in [name for _, name in SELECTION_METHODS]:
        print(f"  {method_name[:8]:>8}", end="")
    print()
    print("-" * (14 + 10 * len(SELECTION_METHODS)))

    for problem in PROBLEMS:
        pname = problem["name"]
        print(f"{pname:<14}", end="")
        for row in all_results[pname]:
            print(f"  {row[2]:>7.1f}%", end="")
        print()

    print()


if __name__ == "__main__":
    main()
