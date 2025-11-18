"""
Tree EDA for HP Protein Folding

This example demonstrates Tree EDA on the HP protein folding problem.
The HP model is a simplified protein folding model where amino acids
are either hydrophobic (H=1) or polar (P=0).

Based on MATEDA-2.0 LearnTree_HPProtein.m and TreeFDA_HPProtein.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import (
    create_fibonacci_hp_sequence,
    create_hp_objective_function
)


def main():
    """Run Tree EDA on HP Protein Folding"""

    # Create HP protein sequence
    # This is a Fibonacci HP sequence used in MATEDA examples
    hp_sequence = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,  # 12 zeros (padding)
        1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0   # 12 zeros (padding)
    ], dtype=int)

    # Problem parameters
    n_vars = len(hp_sequence)
    pop_size = 1000
    # Each position can take 3 values: forward (0), left (1), right (2)
    cardinality = 3 * np.ones(n_vars, dtype=int)

    # Create objective function
    objective = create_hp_objective_function(hp_sequence)

    # Create EDA components
    learning = LearnTreeModel(alpha=0.0)
    sampling = SampleFDA()
    selection = TruncationSelection(truncation_rate=0.5)
    replacement = GenerationalReplacement()
    stop_condition = MaxGenerations(max_generations=50)

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
        maximize=True,  # Maximizing energy (minimizing negative energy)
    )

    # Run optimization
    print("Running Tree EDA on HP Protein Folding...")
    print(f"Population size: {pop_size}")
    print(f"Sequence length: {n_vars}")
    print(f"Number of H residues: {np.sum(hp_sequence)}")
    print(f"Maximum generations: 50")
    print()

    statistics = eda.run()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generations run: {len(statistics['best_fitness'])}")
    print(f"Best energy: {statistics['best_fitness'][-1]:.4f}")
    print(f"Mean energy (final): {statistics['mean_fitness'][-1]:.4f}")
    print()
    print("Best energy per generation:")
    for gen, fitness in enumerate(statistics['best_fitness'][:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(statistics['best_fitness']) > 10:
        print("  ...")
        for gen in range(len(statistics['best_fitness']) - 3, len(statistics['best_fitness'])):
            print(f"  Generation {gen}: {statistics['best_fitness'][gen]:.4f}")

    print("\nNote: To visualize the protein structure, you can use the best solution")
    print("      with a custom visualization function.")


if __name__ == "__main__":
    main()
