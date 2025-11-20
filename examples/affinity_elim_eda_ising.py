"""
Example: Affinity-based Factorization with Elimination on Ising Model

This example demonstrates the elimination strategy for affinity-based factorization.
Unlike the standard approach that recursively processes each large cluster separately,
the elimination strategy collects all variables from large clusters and processes
them together in a single recursive call.

The 2D Ising model is a physics-inspired optimization problem where spins on a
lattice interact with their neighbors.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.learning.affinity import LearnAffinityFactorizationElim
from pateda.sampling.fda import SampleFDA
from pateda.selection.truncation import SelectTruncation
from pateda.replacement.generational import ReplaceGenerational
from pateda.seeding.random_init import RandomInit
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.functions.discrete.ising import IsingModel


def run_affinity_elim_eda():
    """Run Affinity-based EDA with elimination on Ising model"""

    # Problem setup: 5x5 Ising lattice
    grid_size = 5
    n_vars = grid_size * grid_size

    # Create Ising model with ferromagnetic coupling
    ising = IsingModel(grid_size=grid_size, grid_size_y=grid_size)

    # Add ferromagnetic interactions (prefer aligned spins)
    coupling_strength = 1.0
    for i in range(grid_size):
        for j in range(grid_size):
            var_idx = i * grid_size + j

            # Right neighbor
            if j < grid_size - 1:
                neighbor_idx = i * grid_size + (j + 1)
                ising.add_coupling(var_idx, neighbor_idx, coupling_strength)

            # Bottom neighbor
            if i < grid_size - 1:
                neighbor_idx = (i + 1) * grid_size + j
                ising.add_coupling(var_idx, neighbor_idx, coupling_strength)

    cardinality = np.full(n_vars, 2)  # Binary spins: 0 or 1

    # EDA parameters
    pop_size = 400
    selection_size = 200
    max_generations = 40

    # Define fitness function (negative energy = we want to minimize energy)
    def fitness_func(population):
        return np.array([ising.evaluate(ind)[0] for ind in population])

    # Initialize EDA components with elimination strategy
    learning = LearnAffinityFactorizationElim(
        max_clique_size=4,  # Maximum variables per cluster
        preference=None,  # Use median as preference
        damping=0.9,  # Higher damping for better convergence
        max_convergence_retries=10,  # Retry if clustering doesn't converge
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

    print("Running Affinity-based EDA with Elimination on Ising Model")
    print(f"Problem: {grid_size}x{grid_size} Ising lattice ({n_vars} variables)")
    print(f"Population size: {pop_size}")
    print(f"Max clique size: {learning.max_clique_size}")
    print("-" * 60)

    result = eda.run()

    # Display results
    print("\nOptimization Results:")
    print(f"Generations run: {result['generation']}")
    print(f"Best fitness (energy): {result['best_fitness']:.4f}")

    # Visualize best solution as grid
    best_solution = result["best_individual"]
    grid = best_solution.reshape(grid_size, grid_size)

    print("\nBest solution (spin configuration):")
    print("  " + " ".join(str(i) for i in range(grid_size)))
    for i, row in enumerate(grid):
        print(f"{i} " + " ".join("↑" if s == 1 else "↓" for s in row))

    # Count aligned neighbors (measure of ferromagnetic order)
    aligned = 0
    total_neighbors = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:  # Right neighbor
                if grid[i, j] == grid[i, j + 1]:
                    aligned += 1
                total_neighbors += 1
            if i < grid_size - 1:  # Bottom neighbor
                if grid[i, j] == grid[i + 1, j]:
                    aligned += 1
                total_neighbors += 1

    alignment_ratio = aligned / total_neighbors
    print(f"\nAlignment: {aligned}/{total_neighbors} ({alignment_ratio:.1%})")

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

        # Check if spatially adjacent variables are in same cliques
        print("\nSpatial structure in learned model:")
        print("(checking if neighboring grid positions are clustered together)")

        # For first few cliques, check spatial proximity
        spatial_cliques = 0
        for i in range(min(10, cliques.shape[0])):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            vars_in_clique = cliques[i, 2 : 2 + n_overlap + n_new].astype(int)
            vars_in_clique = vars_in_clique[vars_in_clique >= 0]

            if len(vars_in_clique) >= 2:
                # Convert to grid coordinates
                coords = [(v // grid_size, v % grid_size) for v in vars_in_clique]

                # Check if any pairs are adjacent
                has_adjacent = False
                for c1 in coords:
                    for c2 in coords:
                        if c1 != c2:
                            dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                            if dist == 1:
                                has_adjacent = True
                                break
                    if has_adjacent:
                        break

                if has_adjacent:
                    spatial_cliques += 1

        print(
            f"  {spatial_cliques}/{min(10, cliques.shape[0])} "
            f"cliques contain spatially adjacent variables"
        )

    # Fitness evolution
    print("\nFitness Evolution:")
    for gen in range(0, result["generation"] + 1, max(1, result["generation"] // 10)):
        if gen < len(result["best_fitness_history"]):
            fitness = result["best_fitness_history"][gen]
            print(f"  Generation {gen:3d}: {fitness:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    run_affinity_elim_eda()
