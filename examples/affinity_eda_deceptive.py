"""
Example: Affinity-based Factorization EDA on Deceptive Function

This example demonstrates using affinity propagation clustering to learn
factorized probabilistic models for optimization. The algorithm discovers
groups of related variables based on mutual information.

The deceptive function is a challenging benchmark where the global optimum
is surrounded by local optima that mislead optimization algorithms.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning.affinity import LearnAffinityFactorization
from pateda.sampling.fda import SampleFDA
from pateda.selection.truncation import SelectTruncation
from pateda.replacement.generational import ReplaceGenerational
from pateda.seeding.random_init import RandomInit
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.functions.discrete.deceptive import deceptive_function


def run_affinity_eda():
    """Run Affinity-based EDA on deceptive function"""

    # Problem setup
    n_vars = 30  # Number of variables (must be multiple of trap size)
    trap_size = 3  # Size of each deceptive trap
    cardinality = np.full(n_vars, 2)  # Binary variables

    # EDA parameters
    pop_size = 500
    selection_size = 250
    max_generations = 50

    # Define fitness function
    def fitness_func(population):
        return np.array([deceptive_function(ind, trap_size) for ind in population])

    # Initialize EDA components
    learning = LearnAffinityFactorization(
        max_clique_size=5,  # Maximum variables per cluster
        preference=None,  # Use median similarity as preference
        damping=0.5,  # Damping factor for message passing
        recursive=True,  # Recursively factorize large clusters
        alpha=0.1,  # Laplace smoothing
    )

    sampling = SampleFDA(n_samples=pop_size)
    selection = SelectTruncation(n_selected=selection_size)
    replacement = ReplaceGenerational()
    seeding = RandomInit(pop_size=pop_size)
    stop_condition = MaxGenerations(max_generations=max_generations)

    # Create and run EDA
    eda = EDA(
        fitness_function=fitness_func,
        n_vars=n_vars,
        cardinality=cardinality,
        learning_method=learning,
        sampling_method=sampling,
        selection_method=selection,
        replacement_method=replacement,
        seeding_method=seeding,
        stop_condition=stop_condition,
        random_seed=42,
    )

    print("Running Affinity-based EDA on Deceptive Function")
    print(f"Problem size: {n_vars} variables")
    print(f"Trap size: {trap_size}")
    print(f"Population size: {pop_size}")
    print(f"Max clique size: {learning.max_clique_size}")
    print("-" * 60)

    result = eda.run()

    # Display results
    print("\nOptimization Results:")
    print(f"Generations run: {result['generation']}")
    print(f"Best fitness: {result['best_fitness']:.4f}")
    print(f"Global optimum: {n_vars // trap_size}")  # One per trap
    print(f"Best solution: {result['best_individual']}")

    # Analyze learned structure
    model = result.get("model")
    if model:
        metadata = model.metadata
        n_cliques = metadata.get("n_cliques", 0)
        print(f"\nLearned Model Structure:")
        print(f"Number of cliques: {n_cliques}")

        # Show clique sizes
        cliques = model.structure
        clique_sizes = []
        for i in range(cliques.shape[0]):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            clique_sizes.append(n_overlap + n_new)

        print(f"Clique sizes: {sorted(clique_sizes, reverse=True)}")
        print(f"Average clique size: {np.mean(clique_sizes):.2f}")

        # Show which variables are in which cliques
        print("\nClique composition:")
        for i in range(min(5, cliques.shape[0])):  # Show first 5 cliques
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            vars_in_clique = cliques[i, 2 : 2 + n_overlap + n_new].astype(int)
            vars_in_clique = vars_in_clique[vars_in_clique >= 0]
            print(f"  Clique {i}: variables {list(vars_in_clique)}")

        if cliques.shape[0] > 5:
            print(f"  ... and {cliques.shape[0] - 5} more cliques")

    # Fitness evolution
    print("\nFitness Evolution:")
    for gen in range(0, result["generation"] + 1, max(1, result["generation"] // 10)):
        if gen < len(result["best_fitness_history"]):
            fitness = result["best_fitness_history"][gen]
            print(f"  Generation {gen:3d}: {fitness:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    run_affinity_eda()
