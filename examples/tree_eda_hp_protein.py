"""
Tree EDA for HP Protein Folding

This example demonstrates Tree EDA on the HP protein folding problem.
The HP model is a simplified protein folding model where amino acids
are either hydrophobic (H=1) or polar (P=0).

Based on MATEDA-2.0 LearnTree_HPProtein.m and TreeFDA_HPProtein.m
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.seeding import RandomInit
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
    fitness_func = create_hp_objective_function(hp_sequence)

    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnTreeModel(alpha=0.0),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(50),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        components=components,
    )

    # Run optimization
    print("Running Tree EDA on HP Protein Folding...")
    print(f"Population size: {pop_size}")
    print(f"Sequence length: {n_vars}")
    print(f"Number of H residues: {np.sum(hp_sequence)}")
    print(f"Maximum generations: 50")
    print()

    stats, cache = eda.run(verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generations run: {len(stats.best_fitness)}")
    print(f"Best energy: {stats.best_fitness_overall:.4f}")
    print(f"Mean energy (final): {stats.mean_fitness[-1]:.4f}")
    print()
    print("Best energy convergence:")
    print(f"  Initial: {stats.best_fitness[0]:.4f}")
    print(f"  Final: {stats.best_fitness_overall:.4f}")
    print(f"  Improvement: {stats.best_fitness_overall - stats.best_fitness[0]:.4f}")

    print("\nNote: To visualize the protein structure, you can use the best solution")
    print("      with a custom visualization function.")


if __name__ == "__main__":
    main()
