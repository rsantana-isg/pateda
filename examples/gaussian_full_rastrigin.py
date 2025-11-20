"""
Multivariate Gaussian EDA Example: Rastrigin Function

This example demonstrates using a full multivariate Gaussian model
to optimize the Rastrigin function, a challenging multimodal benchmark.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.continuous import rastrigin
from pateda.learning.basic_gaussian import LearnGaussianFull
from pateda.sampling.basic_gaussian import SampleGaussianFull
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit


def main():
    # Problem setup
    n_vars = 10
    lower_bounds = np.array([-5.12] * n_vars)
    upper_bounds = np.array([5.12] * n_vars)

    # Define objective function (minimize rastrigin)
    def objective(pop):
        # Handle both single individual (1D) and population (2D)
        if pop.ndim == 1:
            return -rastrigin(pop)  # Negative for maximization
        else:
            return np.array([-rastrigin(ind) for ind in pop])

    # EDA parameters
    pop_size = 200
    max_generations = 100
    var_scaling = 0.5  # Scale variance to prevent premature convergence

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),  # Select top 50%
        learning=LearnGaussianFull(),
        sampling=SampleGaussianFull(n_samples=pop_size, var_scaling=var_scaling),
        replacement=ElitistReplacement(n_elite=20),
        stop_condition=MaxGenerations(max_generations),
    )

    # Create EDA instance
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=objective,
        cardinality=np.column_stack([lower_bounds, upper_bounds]),
        components=components,
        random_seed=42,
    )

    # Run optimization
    print("Multivariate Gaussian EDA - Rastrigin Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Max generations: {max_generations}")
    print(f"Variance scaling: {var_scaling}")
    print()

    stats, cache = eda.run(verbose=True)

    # Report results
    print("\nOptimization completed!")
    print("=" * 60)
    best_solution = stats.best_individual
    best_fitness = stats.best_fitness_overall

    print(f"Best fitness (negative Rastrigin): {best_fitness:.6f}")
    print(f"Best Rastrigin value: {-best_fitness:.6f}")
    print(f"Global optimum: 0.0")
    print(f"\nBest solution:")
    print(best_solution)

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Total generations: {len(stats.best_fitness)}")
    print(f"Final mean fitness: {stats.mean_fitness[-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats.best_fitness[0]):.6f}")


if __name__ == "__main__":
    main()
