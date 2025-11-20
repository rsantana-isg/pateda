"""
Mixture of Gaussians EDA Example: Rosenbrock Function

This example demonstrates using a mixture of univariate Gaussian models
to optimize the Rosenbrock function. The mixture model can better handle
multimodal distributions and diverse populations.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.continuous import rosenbrock
from pateda.learning.mixture_gaussian import LearnMixtureGaussian
from pateda.sampling.mixture_gaussian import SampleMixtureGaussian
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit


def main():
    # Problem setup
    n_vars = 10
    lower_bounds = np.array([-5.0] * n_vars)
    upper_bounds = np.array([10.0] * n_vars)

    # Define objective function (minimize rosenbrock)
    def objective(pop):
        # Handle both single individual (1D) and population (2D)
        if pop.ndim == 1:
            return -rosenbrock(pop)  # Negative for maximization
        else:
            return np.array([-rosenbrock(ind) for ind in pop])

    # EDA parameters
    pop_size = 200
    max_generations = 150
    n_clusters = 3  # Number of mixture components

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),  # Select top 50%
        learning=LearnMixtureGaussian(
            n_clusters=n_clusters,
            what_to_cluster='vars',  # Cluster based on variable values
            normalize=True,
            covariance_type='univariate'
        ),
        sampling=SampleMixtureGaussian(n_samples=pop_size),
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
    print("Mixture of Gaussians EDA - Rosenbrock Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Number of mixture components: {n_clusters}")
    print(f"Max generations: {max_generations}")
    print()

    stats, cache = eda.run(verbose=True)

    # Report results
    print("\nOptimization completed!")
    print("=" * 60)
    best_solution = stats.best_individual
    best_fitness = stats.best_fitness_overall

    print(f"Best fitness (negative Rosenbrock): {best_fitness:.6f}")
    print(f"Best Rosenbrock value: {-best_fitness:.6f}")
    print(f"Global optimum: 0.0 at (1, 1, ..., 1)")
    print(f"\nBest solution:")
    print(best_solution)
    print(f"\nDistance from optimum (1, 1, ..., 1):")
    print(f"{np.linalg.norm(best_solution - np.ones(n_vars)):.6f}")

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Total generations: {len(stats.best_fitness)}")
    print(f"Final mean fitness: {stats.mean_fitness[-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats.best_fitness[0]):.6f}")


if __name__ == "__main__":
    main()
