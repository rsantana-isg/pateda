"""
Markov Chain FDA for HP Protein Folding

This example demonstrates Markov Chain FDA on the HP protein folding problem.
A Markov chain model captures sequential dependencies where each variable
depends on k previous variables.

Based on MATEDA-2.0 MkFDA_HPProtein.m
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning import LearnMarkovChain
from pateda.sampling import SampleMarkovChain
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete import create_hp_objective_function


def main():
    """Run Markov Chain FDA on HP Protein Folding"""

    # Create HP protein sequence
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
    # Markov chain with order 2 (each variable depends on 2 previous ones)
    learning = LearnMarkovChain(order=2, alpha=1.0)
    sampling = SampleMarkovChain()
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
        maximize=True,
    )

    # Run optimization
    print("Running Markov Chain FDA on HP Protein Folding...")
    print(f"Population size: {pop_size}")
    print(f"Sequence length: {n_vars}")
    print(f"Markov chain order: 2")
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

    print("\nNote: Markov chain models capture sequential dependencies,")
    print("      which are natural for protein folding problems.")


if __name__ == "__main__":
    main()
