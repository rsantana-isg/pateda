"""
Mixture of Gaussians EDA Example: Rosenbrock Function

This example demonstrates using a mixture of univariate Gaussian models
to optimize the Rosenbrock function. The mixture model can better handle
multimodal distributions and diverse populations.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.continuous import rosenbrock
from pateda.learning.mixture_gaussian import learn_mixture_gaussian_univariate
from pateda.sampling.mixture_gaussian import sample_mixture_gaussian_univariate
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit


def main():
    # Problem setup
    n_vars = 10
    bounds = np.array([
        [-5.0] * n_vars,  # Lower bounds
        [10.0] * n_vars   # Upper bounds
    ])

    # Define objective function (minimize rosenbrock)
    def objective(pop):
        return -rosenbrock(pop)  # Negative for maximization

    # EDA parameters
    pop_size = 200
    n_selected = 100
    max_generations = 150
    n_clusters = 3  # Number of mixture components

    # Learning parameters for mixture model
    learning_params = {
        'n_clusters': n_clusters,
        'what_to_cluster': 'vars',  # Cluster based on variable values
        'normalize': True
    }

    # Create EDA instance
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        objective_func=objective,
        learning_func=lambda pop, fit, params: learn_mixture_gaussian_univariate(
            pop, fit, learning_params
        ),
        sampling_func=lambda model, n: sample_mixture_gaussian_univariate(
            model, n, bounds=bounds
        ),
        selection_func=lambda pop, fit, n: truncation_selection(
            pop, fit, n_select=n
        ),
        replacement_func=elitist_replacement,
        stop_condition=MaxGenerations(max_generations),
        bounds=bounds
    )

    # Run optimization
    print("Mixture of Gaussians EDA - Rosenbrock Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Selected individuals: {n_selected}")
    print(f"Number of mixture components: {n_clusters}")
    print(f"Max generations: {max_generations}")
    print()

    best_solutions, best_fitnesses, stats = eda.run(
        n_selected=n_selected,
        verbose=True
    )

    # Report results
    print("\nOptimization completed!")
    print("=" * 60)
    best_idx = np.argmax(best_fitnesses)
    best_solution = best_solutions[best_idx]
    best_fitness = best_fitnesses[best_idx]

    print(f"Best fitness (negative Rosenbrock): {best_fitness:.6f}")
    print(f"Best Rosenbrock value: {-best_fitness:.6f}")
    print(f"Global optimum: 0.0 at (1, 1, ..., 1)")
    print(f"\nBest solution:")
    print(best_solution)
    print(f"\nDistance from optimum (1, 1, ..., 1):")
    print(f"{np.linalg.norm(best_solution - np.ones(n_vars)):.6f}")

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Final mean fitness: {stats['mean_fitness'][-1]:.6f}")
    print(f"Final std fitness: {stats['std_fitness'][-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats['best_fitness'][0]):.6f}")


if __name__ == "__main__":
    main()
