"""
Tree EDA for Goldberg's Deceptive-3 Function

This example demonstrates the Tree EDA algorithm using LearnTreeModel
on the deceptive-3 function with proportional selection and elitism.

Based on MATEDA-2.0 TreeFDA_Deceptive3.m
"""

# Add parent directory to path for running examples without installation

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.seeding import RandomInit
from pateda.selection import ProportionalSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete_binary import deceptive3


def main():
    """Run Tree EDA on Deceptive-3 function"""

    # Problem parameters
    pop_size = 500
    n_vars = 60
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Objective function
    def objective(population):
        return deceptive3(population)

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnTreeModel(alpha=0.0),
        sampling=SampleFDA(n_samples=pop_size),
        selection=ProportionalSelection(),
        replacement=ElitistReplacement(n_elite=10),
        stop_condition=MaxGenerations(max_gen=100),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=objective,
        components=components,
        random_seed=42,
    )

    # Run optimization
    print("Running Tree EDA on Deceptive-3 function...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Maximum generations: 100")
    print()

    stats, cache = eda.run(verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generations run: {len(stats.best_fitness)}")
    print(f"Best fitness: {stats.best_fitness_overall:.4f}")
    print(f"Mean fitness (final): {stats.mean_fitness[-1]:.4f}")
    print(f"Best solution: {stats.best_individual}")
    print(f"Generation found: {stats.generation_found}")
    print()
    print("Best fitness per generation:")
    for gen, fitness in enumerate(stats.best_fitness[:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(stats.best_fitness) > 10:
        print("  ...")
        for gen in range(len(stats.best_fitness) - 3, len(stats.best_fitness)):
            print(f"  Generation {gen}: {stats.best_fitness[gen]:.4f}")


if __name__ == "__main__":
    main()
