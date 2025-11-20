"""
Gaussian UMDA Example: Sphere Function

This example demonstrates using a Gaussian univariate model (Gaussian UMDA)
to optimize the sphere function, a simple continuous optimization benchmark.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.continuous import sphere
from pateda.learning.basic_gaussian import LearnGaussianUnivariate
from pateda.sampling.basic_gaussian import SampleGaussianUnivariate
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit


def main():
    # Problem setup
    n_vars = 30
    lower_bounds = np.array([-5.12] * n_vars)
    upper_bounds = np.array([5.12] * n_vars)

    # Define objective function (minimize sphere)
    def objective(pop):
        # Handle both single individual (1D) and population (2D)
        if pop.ndim == 1:
            return -sphere(pop)  # Negative for maximization
        else:
            return np.array([-sphere(ind) for ind in pop])

    # EDA parameters
    pop_size = 100
    max_generations = 50

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),  # Select top 50%
        learning=LearnGaussianUnivariate(),
        sampling=SampleGaussianUnivariate(n_samples=pop_size),
        replacement=ElitistReplacement(n_elite=10),
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
    print("Gaussian UMDA - Sphere Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Max generations: {max_generations}")
    print()

    stats, cache = eda.run(verbose=True)

    # Report results
    print("\nOptimization completed!")
    print("=" * 60)
    best_solution = stats.best_individual
    best_fitness = stats.best_fitness_overall

    print(f"Best fitness (negative sphere): {best_fitness:.6f}")
    print(f"Best sphere value: {-best_fitness:.6f}")
    print(f"Distance from optimum: {np.linalg.norm(best_solution):.6f}")
    print(f"\nFirst 5 variables of best solution:")
    print(best_solution[:5])

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Total generations: {len(stats.best_fitness)}")
    print(f"Final mean fitness: {stats.mean_fitness[-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats.best_fitness[0]):.6f}")


if __name__ == "__main__":
    main()
