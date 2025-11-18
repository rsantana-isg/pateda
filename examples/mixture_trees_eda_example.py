"""
Example demonstrating Mixture of Trees EDA (MT-EDA)

This example shows how to use the Mixture of Trees EDA on various test problems.
MT-EDA combines multiple tree-structured models with mixture weights to capture
complex dependencies.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.sampling.mixture_trees import SampleMixtureTrees
from pateda.selection.tournament import TournamentSelection
from pateda.replacement.generational import GenerationalReplacement
from pateda.functions.discrete import onemax, deceptive3


def mixture_trees_onemax():
    """
    Example: MT-EDA on OneMax problem

    OneMax is simple, but MT-EDA should still converge by learning
    multiple tree structures.
    """
    print("=" * 70)
    print("MT-EDA on OneMax Problem")
    print("=" * 70)

    # Problem parameters
    n_vars = 20
    cardinality = np.full(n_vars, 2)  # Binary variables

    # Create EDA with Mixture of Trees model
    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=LearnMixtureTrees(
            n_components=3,
            component_learning="tree",
            alpha=0.1,
            weight_learning="uniform",
            random_seed=42
        ),
        sampling_method=SampleMixtureTrees(n_samples=100),
        selection_method=TournamentSelection(tournament_size=3, n_select=50),
        replacement_method=GenerationalReplacement(),
        objective_function=onemax,
        maximize=True,
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


def mixture_trees_deceptive():
    """
    Example: MT-EDA on Deceptive3 problem

    Deceptive3 has overlapping dependencies that can benefit from
    multiple tree structures in the mixture.
    """
    print("\n" + "=" * 70)
    print("MT-EDA on Deceptive3 Problem")
    print("=" * 70)

    # Problem parameters
    n_vars = 30
    cardinality = np.full(n_vars, 2)

    # Create EDA with Mixture of Trees
    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=LearnMixtureTrees(
            n_components=5,  # More components for complex problem
            component_learning="tree",
            alpha=0.1,
            weight_learning="uniform",
            random_seed=42
        ),
        sampling_method=SampleMixtureTrees(n_samples=200),
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
    optimal_fitness = n_blocks * 3

    print(f"\nBest solution found: {results['best_individual']}")
    print(f"Best fitness: {results['best_fitness']}")
    print(f"Optimal fitness: {optimal_fitness}")
    print(f"Success rate: {results['best_fitness'] / optimal_fitness * 100:.1f}%")


def compare_mixture_sizes():
    """
    Compare different numbers of mixture components

    Shows how the number of components affects performance
    """
    print("\n" + "=" * 70)
    print("Comparing Different Numbers of Mixture Components")
    print("=" * 70)

    n_vars = 15
    cardinality = np.full(n_vars, 2)
    max_gens = 25

    for n_components in [1, 3, 5]:
        print(f"\n--- {n_components} component(s) ---")

        eda = EDA(
            n_vars=n_vars,
            cardinality=cardinality,
            learning_method=LearnMixtureTrees(
                n_components=n_components,
                component_learning="tree",
                alpha=0.1,
                weight_learning="uniform",
                random_seed=42
            ),
            sampling_method=SampleMixtureTrees(n_samples=100),
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
        print(f"Convergence rate: {results['best_fitness'] / n_vars * 100:.1f}%")


def mixture_trees_with_em_weights():
    """
    Example using EM algorithm for learning mixture weights

    Shows how to use EM instead of uniform weights
    """
    print("\n" + "=" * 70)
    print("MT-EDA with EM-learned Weights")
    print("=" * 70)

    n_vars = 20
    cardinality = np.full(n_vars, 2)

    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=LearnMixtureTrees(
            n_components=3,
            component_learning="tree",
            alpha=0.1,
            weight_learning="em",  # Use EM for weight learning
            em_iterations=5,
            random_seed=42
        ),
        sampling_method=SampleMixtureTrees(n_samples=100),
        selection_method=TournamentSelection(tournament_size=3, n_select=50),
        replacement_method=GenerationalReplacement(),
        objective_function=onemax,
        maximize=True,
    )

    results = eda.run(
        max_generations=30,
        population_size=100,
        verbose=True,
    )

    print(f"\nBest fitness: {results['best_fitness']} / {n_vars}")
    print(f"Using EM for adaptive mixture weights")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("Mixture of Trees EDA (MT-EDA) Examples")
    print("*" * 70)

    # Run examples
    mixture_trees_onemax()
    mixture_trees_deceptive()
    compare_mixture_sizes()
    mixture_trees_with_em_weights()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
