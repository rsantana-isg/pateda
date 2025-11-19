"""
Example demonstrating k-order Markov EDA (MK-EDA)

This example shows how to use the k-order Markov Chain EDA on various test problems.
The Markov model assumes sequential dependencies where each variable depends on k
previous variables.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnMarkovChain
from pateda.sampling import SampleMarkovChain
from pateda.selection import TournamentSelection
from pateda.replacement import GenerationalReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.functions.discrete import onemax, deceptive3


def markov_eda_onemax():
    """
    Example: MK-EDA on OneMax problem

    OneMax is a simple problem where fitness = sum of bits.
    Even though variables are independent, Markov model should still work.
    """
    print("=" * 70)
    print("MK-EDA on OneMax Problem")
    print("=" * 70)

    # Problem parameters
    n_vars = 20
    pop_size = 100
    cardinality = np.full(n_vars, 2)  # Binary variables

    # Create EDA with 2-order Markov model
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TournamentSelection(tournament_size=3, n_select=50),
        learning=LearnMarkovChain(k=2, alpha=0.1),
        sampling=SampleMarkovChain(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(30),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=onemax,
        components=components,
    )

    # Run EDA
    stats, cache = eda.run(verbose=True)

    print(f"\nBest solution found: {stats.best_individual_overall}")
    print(f"Best fitness: {stats.best_fitness_overall}")
    print(f"Optimal fitness (all 1s): {n_vars}")
    print(f"Success: {'Yes' if stats.best_fitness_overall >= n_vars else 'No'}")


def markov_eda_deceptive():
    """
    Example: MK-EDA on Deceptive3 problem

    Deceptive3 has non-overlapping blocks of 3 variables with deceptive structure.
    Markov model with k=2 should capture dependencies within each block.
    """
    print("\n" + "=" * 70)
    print("MK-EDA on Deceptive3 Problem")
    print("=" * 70)

    # Problem parameters (must be divisible by 3)
    n_vars = 30
    pop_size = 200
    cardinality = np.full(n_vars, 2)  # Binary variables

    # Create EDA with 3-order Markov model (to match block size)
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TournamentSelection(tournament_size=3, n_select=100),
        learning=LearnMarkovChain(k=3, alpha=0.1),
        sampling=SampleMarkovChain(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(50),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=deceptive3,
        components=components,
    )

    # Run EDA
    stats, cache = eda.run(verbose=True)

    n_blocks = n_vars // 3
    optimal_fitness = n_blocks * 3  # 3 points per block

    print(f"\nBest solution found: {stats.best_individual_overall}")
    print(f"Best fitness: {stats.best_fitness_overall}")
    print(f"Optimal fitness: {optimal_fitness}")
    print(f"Success rate: {stats.best_fitness_overall / optimal_fitness * 100:.1f}%")


def compare_markov_orders():
    """
    Compare different Markov orders on OneMax

    Shows how order k affects performance
    """
    print("\n" + "=" * 70)
    print("Comparing Different Markov Orders on OneMax")
    print("=" * 70)

    n_vars = 15
    pop_size = 100
    cardinality = np.full(n_vars, 2)
    max_gens = 25

    for k in [1, 2, 3]:
        print(f"\n--- k={k} (Markov order {k}) ---")

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TournamentSelection(tournament_size=2, n_select=50),
            learning=LearnMarkovChain(k=k, alpha=0.1),
            sampling=SampleMarkovChain(n_samples=pop_size),
            replacement=GenerationalReplacement(),
            stop_condition=MaxGenerations(max_gens),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_func=onemax,
            components=components,
        )

        stats, cache = eda.run(verbose=False)

        print(f"Best fitness: {stats.best_fitness_overall} / {n_vars}")
        # Note: generations_to_optimum not available in stats object
        print(f"Total generations: {stats.generation}")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("k-order Markov EDA (MK-EDA) Examples")
    print("*" * 70)

    # Run examples
    markov_eda_onemax()
    markov_eda_deceptive()
    compare_markov_orders()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
