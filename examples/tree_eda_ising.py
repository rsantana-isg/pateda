"""
Tree EDA for Ising Model

This example demonstrates Tree EDA on the Ising spin glass model.
The Ising model is a physics-inspired optimization problem with
pairwise interactions between binary spins.

Based on MATEDA-2.0 LearnTree_IsingModel.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import load_ising, create_ising_objective_function


def main():
    """Run Tree EDA on Ising Model"""

    # Problem parameters
    pop_size = 500
    n_vars = 64
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Load Ising instance (instance 6)
    ising_lattice, ising_interactions = load_ising(n_vars, inst=6)

    # Create objective function
    objective = create_ising_objective_function(ising_lattice, ising_interactions)

    # Create EDA components
    learning = LearnTreeModel(alpha=0.0)
    sampling = SampleFDA(n_samples=pop_size)
    selection = TruncationSelection(ratio=0.5)
    replacement = GenerationalReplacement()
    stop_condition = MaxGenerations(max_generations=150)

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
    print("Running Tree EDA on Ising Model...")
    print(f"Population size: {pop_size}")
    print(f"Number of spins: {n_vars}")
    print(f"Instance: 6")
    print(f"Maximum generations: 150")
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
    print("Best fitness per generation (first 10 and last 3):")
    for gen, fitness in enumerate(statistics['best_fitness'][:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(statistics['best_fitness']) > 10:
        print("  ...")
        for gen in range(len(statistics['best_fitness']) - 3, len(statistics['best_fitness'])):
            print(f"  Generation {gen}: {statistics['best_fitness'][gen]:.4f}")


if __name__ == "__main__":
    main()
