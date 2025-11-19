"""
Gaussian UMDA Example: Sphere Function

This example demonstrates using a Gaussian univariate model (Gaussian UMDA)
to optimize the sphere function, a simple continuous optimization benchmark.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.continuous import sphere
from pateda.learning.basic_gaussian import learn_gaussian_univariate
from pateda.sampling.basic_gaussian import sample_gaussian_univariate
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit


def main():
    # Problem setup
    n_vars = 30
    bounds = np.array([
        [-5.12] * n_vars,  # Lower bounds
        [5.12] * n_vars    # Upper bounds
    ])

    # Define objective function (minimize sphere)
    def objective(pop):
        return -sphere(pop)  # Negative for maximization

    # EDA parameters
    pop_size = 100
    n_selected = 50
    max_generations = 50

    # Create EDA instance
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        objective_func=objective,
        learning_func=learn_gaussian_univariate,
        sampling_func=lambda model, n: sample_gaussian_univariate(
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
    print("Gaussian UMDA - Sphere Function Optimization")
    print("=" * 60)
    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Selected individuals: {n_selected}")
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

    print(f"Best fitness (negative sphere): {best_fitness:.6f}")
    print(f"Best sphere value: {-best_fitness:.6f}")
    print(f"Distance from optimum: {np.linalg.norm(best_solution):.6f}")
    print(f"\nFirst 5 variables of best solution:")
    print(best_solution[:5])

    # Statistics
    print(f"\nConvergence statistics:")
    print(f"Final mean fitness: {stats['mean_fitness'][-1]:.6f}")
    print(f"Final std fitness: {stats['std_fitness'][-1]:.6f}")
    print(f"Improvement: {(best_fitness - stats['best_fitness'][0]):.6f}")


if __name__ == "__main__":
    main()
