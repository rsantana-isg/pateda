"""
Tree EDA for Ising Model

This example demonstrates Tree EDA on the Ising spin glass model.
The Ising model is a physics-inspired optimization problem with
pairwise interactions between binary spins.

Based on MATEDA-2.0 LearnTree_IsingModel.m
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations


def create_random_ising_model(grid_size):
    """
    Create a random Ising spin glass model on a 2D lattice.

    Args:
        grid_size: Size of the square lattice (grid_size x grid_size)

    Returns:
        Function that evaluates the energy for a spin configuration
    """
    n_vars = grid_size * grid_size

    # Create random interaction matrix (sparse - only neighbors interact)
    np.random.seed(42)
    interactions = {}

    for i in range(grid_size):
        for j in range(grid_size):
            var_idx = i * grid_size + j

            # Right neighbor
            if j < grid_size - 1:
                neighbor_idx = i * grid_size + (j + 1)
                # Random coupling: +1 (ferromagnetic) or -1 (antiferromagnetic)
                interactions[(var_idx, neighbor_idx)] = np.random.choice([-1.0, 1.0])

            # Bottom neighbor
            if i < grid_size - 1:
                neighbor_idx = (i + 1) * grid_size + j
                interactions[(var_idx, neighbor_idx)] = np.random.choice([-1.0, 1.0])

    def evaluate_ising(spins):
        """
        Evaluate energy of spin configuration.
        Negative energy for aligned spins with ferromagnetic coupling.
        """
        energy = 0.0
        for (i, j), coupling in interactions.items():
            # Spins are 0 or 1, convert to -1 or +1
            spin_i = 2 * spins[i] - 1
            spin_j = 2 * spins[j] - 1
            energy += coupling * spin_i * spin_j
        return -energy  # Negative because we want to maximize alignment

    return evaluate_ising


def main():
    """Run Tree EDA on Ising Model"""

    # Problem parameters
    pop_size = 500
    grid_size = 8  # 8x8 = 64 spins
    n_vars = grid_size * grid_size
    cardinality = 2 * np.ones(n_vars, dtype=int)

    # Create random Ising model
    ising_eval = create_random_ising_model(grid_size)

    # Define fitness function
    def fitness_func(population):
        # Handle both single individual and population
        if population.ndim == 1:
            return ising_eval(population)
        else:
            return np.array([ising_eval(ind) for ind in population])

    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnTreeModel(alpha=0.0),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(150),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        components=components,
        random_seed=42,
    )

    # Run optimization
    print("Running Tree EDA on Ising Model...")
    print(f"Population size: {pop_size}")
    print(f"Number of spins: {n_vars} ({grid_size}x{grid_size} lattice)")
    print(f"Coupling: Random spin glass")
    print(f"Maximum generations: 150")
    print()

    stats, cache = eda.run(verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generations run: {len(stats.best_fitness)}")
    print(f"Best fitness: {stats.best_fitness[-1]:.4f}")
    print(f"Mean fitness (final): {stats.mean_fitness[-1]:.4f}")
    print()
    print("Best fitness per generation (first 10 and last 3):")
    for gen, fitness in enumerate(stats.best_fitness[:10]):
        print(f"  Generation {gen}: {fitness:.4f}")
    if len(stats.best_fitness) > 10:
        print("  ...")
        for gen in range(len(stats.best_fitness) - 3, len(stats.best_fitness)):
            print(f"  Generation {gen}: {stats.best_fitness[gen]:.4f}")


if __name__ == "__main__":
    main()
