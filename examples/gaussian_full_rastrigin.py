"""
Multivariate Gaussian EDA Example: Rastrigin Function

This example demonstrates using a full multivariate Gaussian model
to optimize the Rastrigin function, a challenging multimodal benchmark.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.functions.continuous import rastrigin
from pateda.learning.gaussian import learn_gaussian_full
from pateda.sampling.gaussian import sample_gaussian_full
from pateda.selection.truncation import truncation_selection
from pateda.replacement.elitist import elitist_replacement
from pateda.stop_conditions.max_generations import MaxGenerations


def main():
    # Problem setup
    n_vars = 10
    bounds = np.array([
        [-5.12] * n_vars,  # Lower bounds
        [5.12] * n_vars    # Upper bounds
    ])

    # Define objective function (minimize rastrigin)
    def objective(pop):
        return -rastrigin(pop)  # Negative for maximization

    # EDA parameters
    pop_size = 200
    n_selected = 100
    max_generations = 100
    var_scaling = 0.5  # Scale variance to prevent premature convergence

    # Create EDA instance
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        objective_func=objective,
        learning_func=learn_gaussian_full,
        sampling_func=lambda model, n: sample_gaussian_full(
            model, n, bounds=bounds, params={'var_scaling': var_scaling}
        ),
        selection_func=lambda pop, fit, n: truncation_selection(
            pop, fit, n_select=n
        ),
        replacement_func=elitist_replacement,
        stop_condition=MaxGenerations(max_generations),
        bounds=bounds
    )

    # Run optimization
    print("Multivariate Gaussian EDA - Rastrigin Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Selected individuals: {n_selected}")
    print(f"Max generations: {max_generations}")
    print(f"Variance scaling: {var_scaling}")
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

    print(f"Best fitness (negative Rastrigin): {best_fitness:.6f}")
    print(f"Best Rastrigin value: {-best_fitness:.6f}")
    print(f"Global optimum: 0.0")
    print(f"\nBest solution:")
    print(best_solution)

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Final mean fitness: {stats['mean_fitness'][-1]:.6f}")
    print(f"Final std fitness: {stats['std_fitness'][-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats['best_fitness'][0]):.6f}")


if __name__ == "__main__":
    main()
