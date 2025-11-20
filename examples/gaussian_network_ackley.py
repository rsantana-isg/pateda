"""
Gaussian Bayesian Network EDA for Ackley Function

This script demonstrates the use of a Gaussian Bayesian Network EDA
(continuous equivalent of EBNA) on the Ackley function, a challenging
multimodal continuous optimization benchmark.

The Ackley function has many local minima but one global minimum at the origin.
A Gaussian Network EDA can learn dependencies between variables and potentially
find better paths to the global optimum.

This tests a combination not present in the MATLAB ScriptsMateda:
- Gaussian Network EDA
- Ackley function (multimodal continuous benchmark)
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.learning.basic_gaussian import LearnGaussianFull
from pateda.sampling.basic_gaussian import SampleGaussianFull


def ackley(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """
    Ackley function - a challenging multimodal benchmark

    Global minimum: f(0, 0, ..., 0) = 0

    Args:
        x: Input vector
        a, b, c: Function parameters (default values are standard)

    Returns:
        Function value (to be minimized)
    """
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1.0)


def run_gaussian_network_ackley():
    """
    Run Gaussian Network EDA on Ackley function
    """
    print("=" * 80)
    print("Gaussian Bayesian Network EDA for Ackley Function")
    print("=" * 80)
    print()

    # Problem configuration
    n_vars = 10
    pop_size = 200
    max_generations = 100

    # Search bounds
    lower_bounds = -5.0 * np.ones(n_vars)
    upper_bounds = 5.0 * np.ones(n_vars)

    print("Configuration:")
    print(f"  - Problem: Ackley function (multimodal)")
    print(f"  - Dimensions: {n_vars}")
    print(f"  - Bounds: [{lower_bounds[0]}, {upper_bounds[0]}]")
    print(f"  - Population size: {pop_size}")
    print(f"  - Max generations: {max_generations}")
    print(f"  - Global optimum: f(0,...,0) = 0")
    print()

    # We need to maximize, so negate Ackley
    def fitness_func(x):
        # Clip to bounds
        x_clipped = np.clip(x, lower_bounds, upper_bounds)
        return -ackley(x_clipped)

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),

        # Selection: top 30%
        selection=TruncationSelection(ratio=0.3),

        # Full Gaussian learning (learns full covariance matrix - captures dependencies)
        learning=LearnGaussianFull(),

        # Full Gaussian sampling
        sampling=SampleGaussianFull(
            n_samples=pop_size,
            var_scaling=0.5,  # Scale variance for exploration
        ),

        # Keep best 10 individuals
        replacement=ElitistReplacement(n_elite=10),

        stop_condition=MaxGenerations(max_generations),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.array([lower_bounds, upper_bounds]),
        components=components,
        random_seed=42,
    )

    print("Running Gaussian Network EDA...")
    print()

    # Run
    stats, cache = eda.run(verbose=True)

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    best_x = stats.best_individual
    best_x_clipped = np.clip(best_x, lower_bounds, upper_bounds)
    ackley_value = ackley(best_x_clipped)

    print(f"Best Ackley value: {ackley_value:.6f} (optimum: 0.0)")
    print(f"Best fitness: {stats.best_fitness_overall:.6f}")
    print(f"Generation found: {stats.generation_found}")
    print()
    print(f"Best solution:")
    print(f"  {best_x_clipped}")
    print()
    print(f"Distance from origin: {np.linalg.norm(best_x_clipped):.6f}")
    print()

    return stats, cache


def run_comparison_gaussian_edas():
    """
    Compare different Gaussian EDA variants on Ackley
    """
    print("=" * 80)
    print("Comparison: Gaussian EDA Variants on Ackley Function")
    print("=" * 80)
    print()

    from pateda.learning.basic_gaussian import LearnGaussianUnivariate, LearnGaussianFull
    from pateda.sampling.basic_gaussian import SampleGaussianUnivariate, SampleGaussianFull

    n_vars = 10
    pop_size = 200
    max_generations = 100
    n_runs = 5

    lower_bounds = -5.0 * np.ones(n_vars)
    upper_bounds = 5.0 * np.ones(n_vars)

    def fitness_func(x):
        x_clipped = np.clip(x, lower_bounds, upper_bounds)
        return -ackley(x_clipped)

    algorithms = [
        ("Gaussian UMDA", LearnGaussianUnivariate(), SampleGaussianUnivariate(pop_size)),
        ("Gaussian Full", LearnGaussianFull(), SampleGaussianFull(pop_size, var_scaling=0.5)),
    ]

    results = {name: [] for name, _, _ in algorithms}

    for name, learning, sampling in algorithms:
        print(f"Running {name}...")

        for run in range(n_runs):
            components = EDAComponents(
                seeding=RandomInit(),
                selection=TruncationSelection(ratio=0.3),
                learning=learning,
                sampling=sampling,
                replacement=ElitistReplacement(n_elite=10),
                stop_condition=MaxGenerations(max_generations),
            )

            eda = EDA(
                pop_size=pop_size,
                n_vars=n_vars,
                fitness_func=fitness_func,
                cardinality=np.array([lower_bounds, upper_bounds]),
                components=components,
                random_seed=42 + run,  # Different seed for each run
            )

            stats, _ = eda.run(verbose=False)

            # Convert back to Ackley value
            best_x = np.clip(stats.best_individual, lower_bounds, upper_bounds)
            ackley_value = ackley(best_x)

            results[name].append(ackley_value)
            print(f"  Run {run + 1}: Ackley = {ackley_value:.6f}")

    # Print comparison
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    print(f"{'Algorithm':<20} {'Mean Ackley':<15} {'Std':<15} {'Best':<15}")
    print("-" * 80)

    for name in results:
        values = np.array(results[name])
        print(f"{name:<20} {np.mean(values):<15.6f} {np.std(values):<15.6f} {np.min(values):<15.6f}")

    print()
    print("Note: Lower Ackley values are better (optimum = 0.0)")
    print()


if __name__ == "__main__":
    # Run single example
    stats, cache = run_gaussian_network_ackley()

    # Run comparison
    print("\n" * 2)
    run_comparison_gaussian_edas()
