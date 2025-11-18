"""
Tree EDA for Goldberg's Deceptive-3 Function

This example demonstrates the Tree EDA algorithm using LearnTreeModel
on the deceptive-3 function with proportional selection and elitism.

Based on MATEDA-2.0 TreeFDA_Deceptive3.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import ProportionalSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import deceptive3


def main():
    """Run Tree EDA on Deceptive-3 function"""

    # Problem parameters
    pop_size = 500
    n_vars = 60
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Objective function
    def objective(population):
        return deceptive3(population)

    # Create EDA components
    learning = LearnTreeModel(alpha=0.0)
    sampling = SampleFDA()
    selection = ProportionalSelection()
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
    )

    # Run optimization
    print("Running Tree EDA on Deceptive-3 function...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
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
    print(f"Best solution: {statistics['best_solution']}")
    print()
    print("Best fitness per generation:")
    for gen, fitness in enumerate(statistics['best_fitness'][:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(statistics['best_fitness']) > 10:
        print("  ...")
        for gen in range(len(statistics['best_fitness']) - 3, len(statistics['best_fitness'])):
            print(f"  Generation {gen}: {statistics['best_fitness'][gen]:.4f}")


if __name__ == "__main__":
    main()
