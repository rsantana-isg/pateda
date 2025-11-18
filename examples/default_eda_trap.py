"""
Default EDA for Trap Function

This example demonstrates the default EDA configuration on the trap function.
The trap function is decomposed into non-overlapping blocks of k variables.

Based on MATEDA-2.0 DefaultEDA_TrapFunction.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import create_trap_objective_function


def main():
    """Run Default EDA on Trap function"""

    # Problem parameters
    pop_size = 1000
    n_vars = 45
    n_trap = 5  # Trap parameter (block size)
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Objective function
    objective = create_trap_objective_function(n_trap=n_trap)

    # Create EDA components (default configuration)
    learning = LearnUMDA(alpha=1.0)  # Laplace smoothing
    sampling = SampleFDA()
    selection = TruncationSelection(truncation_rate=0.5)
    replacement = GenerationalReplacement()
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
    print("Running Default EDA on Trap Function...")
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Trap block size: {n_trap}")
    print(f"Optimal fitness: {n_vars} (all 1s)")
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
    print(f"Optimal fitness: {n_vars}")
    print(f"Gap from optimal: {n_vars - statistics['best_fitness'][-1]:.4f}")
    print(f"Best solution sum: {np.sum(statistics['best_solution'])}")
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
