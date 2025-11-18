"""
Example demonstrating k-order Markov EDA (MK-EDA)

This example shows how to use the k-order Markov Chain EDA on various test problems.
The Markov model assumes sequential dependencies where each variable depends on k
previous variables.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning.markov import LearnMarkovChain
from pateda.sampling.markov import SampleMarkovChain
from pateda.selection.tournament import TournamentSelection
from pateda.replacement.generational import GenerationalReplacement
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
    cardinality = np.full(n_vars, 2)  # Binary variables

    # Create EDA with 2-order Markov model
    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=LearnMarkovChain(k=2, alpha=0.1),
        sampling_method=SampleMarkovChain(n_samples=100),
        selection_method=TournamentSelection(tournament_size=3, n_select=50),
        replacement_method=GenerationalReplacement(),
        objective_function=onemax,
        maximize=True,  # OneMax is a maximization problem
    )

    # Run EDA
    results = eda.run(
        max_generations=30,
        population_size=100,
        verbose=True,
    )

    print(f"\nBest solution found: {results['best_individual']}")
    print(f"Best fitness: {results['best_fitness']}")
    print(f"Optimal fitness (all 1s): {n_vars}")
    print(f"Success: {'Yes' if results['best_fitness'] >= n_vars else 'No'}")


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
    cardinality = np.full(n_vars, 2)  # Binary variables

    # Create EDA with 3-order Markov model (to match block size)
    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=LearnMarkovChain(k=3, alpha=0.1),
        sampling_method=SampleMarkovChain(n_samples=200),
        selection_method=TournamentSelection(tournament_size=3, n_select=100),
        replacement_method=GenerationalReplacement(),
        objective_function=deceptive3,
        maximize=True,
    )

    # Run EDA
    results = eda.run(
        max_generations=50,
        population_size=200,
        verbose=True,
    )

    n_blocks = n_vars // 3
    optimal_fitness = n_blocks * 3  # 3 points per block

    print(f"\nBest solution found: {results['best_individual']}")
    print(f"Best fitness: {results['best_fitness']}")
    print(f"Optimal fitness: {optimal_fitness}")
    print(f"Success rate: {results['best_fitness'] / optimal_fitness * 100:.1f}%")


def compare_markov_orders():
    """
    Compare different Markov orders on OneMax

    Shows how order k affects performance
    """
    print("\n" + "=" * 70)
    print("Comparing Different Markov Orders on OneMax")
    print("=" * 70)

    n_vars = 15
    cardinality = np.full(n_vars, 2)
    max_gens = 25

    for k in [1, 2, 3]:
        print(f"\n--- k={k} (Markov order {k}) ---")

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            learning_method=LearnMarkovChain(k=k, alpha=0.1),
            sampling_method=SampleMarkovChain(n_samples=100),
            selection_method=TournamentSelection(tournament_size=2, n_select=50),
            replacement_method=GenerationalReplacement(),
            objective_function=onemax,
            maximize=True,
        )

        results = eda.run(
            max_generations=max_gens,
            population_size=100,
            verbose=False,
        )

        print(f"Best fitness: {results['best_fitness']} / {n_vars}")
        print(f"Generations to optimum: {results.get('generations_to_optimum', 'Not reached')}")


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
