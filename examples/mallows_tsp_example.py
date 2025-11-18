"""
Example: Mallows EDA with Kendall distance for TSP

This example demonstrates how to use the Mallows model with Kendall distance
to solve the Traveling Salesman Problem (TSP).
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.selection import Truncation
from pateda.learning.mallows import LearnMallowsKendall
from pateda.sampling.mallows import SampleMallowsKendall
from pateda.replacement import Elitist
from pateda.stop_conditions import MaxGenerations
from pateda.functions.permutation import create_random_tsp


def main():
    """Run Mallows EDA on TSP"""

    # Problem setup
    n_cities = 20
    print(f"Creating random TSP instance with {n_cities} cities...")
    tsp = create_random_tsp(n_cities, seed=42)

    # EDA parameters
    pop_size = 100
    n_generations = 50
    selection_ratio = 0.5

    print(f"\nEDA Configuration:")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Selection ratio: {selection_ratio}")

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=Truncation(),
        learning=LearnMallowsKendall(),
        sampling=SampleMallowsKendall(),
        replacement=Elitist(),
        stop_condition=MaxGenerations(n_generations),
    )

    # Set learning and sampling parameters
    components.learning_params = {
        "initial_theta": 0.1,
        "upper_theta": 10.0,
        "max_iter": 100,
        "consensus_method": "borda",
    }

    components.sampling_params = {"sample_size": pop_size}
    components.selection_params = {"ratio": selection_ratio}
    components.replacement_params = {"elite_size": int(pop_size * 0.1)}

    # Create cardinality array for permutations (not actually used but required by EDA)
    cardinality = np.arange(n_cities)

    # Initialize and run EDA
    print("\nRunning Mallows EDA with Kendall distance...\n")

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_cities,
        fitness_func=tsp,
        cardinality=cardinality,
        components=components,
    )

    # Run the algorithm
    stats, cache = eda.run(verbose=True)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best fitness found: {stats.best_fitness_overall:.2f}")
    print(f"Best tour distance: {-stats.best_fitness_overall:.2f}")
    print(f"Generation found: {stats.generation_found}")
    print(f"\nBest tour:")
    print(stats.best_individual)

    # Show convergence
    print(f"\nFitness progression (first 10 generations):")
    for i in range(min(10, len(stats.best_fitness))):
        print(f"  Gen {i:2d}: Best = {stats.best_fitness[i]:8.2f}, " f"Mean = {stats.mean_fitness[i]:8.2f}")

    if len(stats.best_fitness) > 10:
        print("  ...")
        i = len(stats.best_fitness) - 1
        print(f"  Gen {i:2d}: Best = {stats.best_fitness[i]:8.2f}, " f"Mean = {stats.mean_fitness[i]:8.2f}")


if __name__ == "__main__":
    main()
