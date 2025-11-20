"""
Compare different selection methods

This example demonstrates how different selection methods perform
on the OneMax problem using UMDA.
"""

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
from pateda.functions import onemax


def run_with_selection(selection_method, selection_name):
    """Run UMDA with a specific selection method"""

    # Problem parameters
    pop_size = 200
    n_vars = 20
    max_gen = 15

    # Variable cardinalities (binary variables)
    cardinality = np.full(n_vars, 2)

    print(f"\n{'=' * 70}")
    print(f"Selection Method: {selection_name}")
    print(f"{'=' * 70}")

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnUMDA(alpha=0.1),
        sampling=SampleFDA(n_samples=pop_size),
        selection=selection_method,
        stop_condition=MaxGenerations(max_gen=max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=cardinality,
        components=components,
        random_seed=42,
    )

    # Run optimization
    statistics, cache = eda.run(verbose=False)

    # Print results
    print(f"Best fitness: {statistics.best_fitness_overall:.0f}/{n_vars}")
    print(f"Found at generation: {statistics.generation_found}")
    print(f"Final mean fitness: {statistics.mean_fitness[-1]:.2f}")

    return statistics


def main():
    """Compare all selection methods"""

    print("=" * 70)
    print("Comparing Selection Methods on OneMax Problem")
    print("=" * 70)

    selection_methods = [
        (TruncationSelection(ratio=0.5), "Truncation (50%)"),
        (TournamentSelection(tournament_size=2, ratio=0.5), "Tournament (size=2)"),
        (TournamentSelection(tournament_size=5, ratio=0.5), "Tournament (size=5)"),
        (ProportionalSelection(ratio=0.5), "Proportional (Roulette)"),
        (RankingSelection(ratio=0.5, selection_pressure=1.5), "Ranking (linear)"),
        (StochasticUniversalSampling(ratio=0.5), "Stochastic Universal Sampling"),
        (BoltzmannSelection(ratio=0.5, temperature=2.0), "Boltzmann (T=2.0)"),
        (BoltzmannSelection(ratio=0.5, temperature=1.0), "Boltzmann (T=1.0)"),
    ]

    results = []
    for method, name in selection_methods:
        stats = run_with_selection(method, name)
        results.append((name, stats))

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Method':<35} {'Best':<10} {'Gen':<10}")
    print("-" * 70)

    for name, stats in results:
        print(
            f"{name:<35} {stats.best_fitness_overall:<10.0f} "
            f"{stats.generation_found:<10}"
        )


if __name__ == "__main__":
    main()
