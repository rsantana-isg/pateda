"""
Default EDA for NK Random Landscape

This example demonstrates the default EDA configuration on NK random landscapes.
NK landscapes are tunably rugged fitness landscapes where N is the number of
variables and K is the epistasis level (number of interacting variables).

Based on MATEDA-2.0 DefaultEDA_NKRandom.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import create_nk_objective_function


def main():
    """Run Default EDA on NK Random Landscape"""

    # Problem parameters
    pop_size = 500
    n_vars = 50
    k = 4  # Epistasis level (each variable interacts with k neighbors)
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Create objective function with fixed random seed for reproducibility
    objective = create_nk_objective_function(n_vars=n_vars, k=k, random_seed=42)

    # Create EDA components
    learning = LearnUMDA(alpha=1.0)  # Laplace smoothing
    sampling = SampleFDA()
    selection = TruncationSelection(truncation_rate=0.5)
    replacement = ElitistReplacement(n_elite=10)
    stop_condition = MaxGenerations(max_generations=100)

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        objective_function=objective,
        learning_method=learning,
        sampling_method=sampling,
        selection_method=selection,
        replacement_method=replacement,
        stop_condition=stop_condition,
        maximize=True,
        random_seed=42,
    )

    # Run optimization
    print("Running Default EDA on NK Random Landscape...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables (N): {n_vars}")
    print(f"Epistasis level (K): {k}")
    print(f"Random seed: 42")
    print(f"Maximum generations: 100")
    print()

    statistics = eda.run()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generations run: {len(statistics['best_fitness'])}")
    print(f"Best fitness: {statistics['best_fitness'][-1]:.4f}")
    print(f"Mean fitness (final): {statistics['mean_fitness'][-1]:.4f}")
    print()
    print("Convergence curve (best fitness per generation):")
    for gen, fitness in enumerate(statistics['best_fitness'][:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(statistics['best_fitness']) > 10:
        print("  ...")
        for gen in range(len(statistics['best_fitness']) - 3, len(statistics['best_fitness'])):
            print(f"  Generation {gen}: {statistics['best_fitness'][gen]:.4f}")

    print("\nNote: NK landscapes are rugged with many local optima.")
    print("      The difficulty increases with K (epistasis level).")


if __name__ == "__main__":
    main()
