"""
Examples of using MN-FDA, MN-FDAG, and MOA algorithms

Demonstrates:
1. MN-FDA with PLS sampling (using existing SampleFDA)
2. MN-FDAG with PLS sampling
3. MOA with Gibbs sampling
4. Comparison on OneMax and Trap-5 problems
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import StopCriteriaMaxGen
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import NoReplacement

# Import new learning methods
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.moa import LearnMOA

# Import sampling methods
from pateda.sampling.fda import SampleFDA  # PLS sampling
from pateda.sampling.gibbs import SampleGibbs  # Gibbs sampling


def onemax(x: np.ndarray) -> float:
    """OneMax: Maximize number of ones"""
    return float(np.sum(x))


def trap5(x: np.ndarray) -> float:
    """
    Concatenated Trap-5 function

    Deceptive function where blocks of 5 bits have:
    - All zeros: fitness = 5 (global optimum trap)
    - All ones: fitness = 6 (global optimum)
    - Otherwise: fitness = number of ones
    """
    n = len(x)
    k = 5  # Block size
    m = n // k  # Number of blocks

    total_fitness = 0.0
    for i in range(m):
        block = x[i*k:(i+1)*k]
        ones = np.sum(block)

        if ones == k:
            total_fitness += k + 1  # Global optimum
        elif ones == 0:
            total_fitness += k  # Deceptive local optimum
        else:
            total_fitness += ones

    return total_fitness


def run_mnfda_example():
    """
    Example 1: MN-FDA with PLS sampling on OneMax

    Uses chi-square test for structure learning and PLS for sampling.
    """
    print("=" * 70)
    print("Example 1: MN-FDA with PLS Sampling on OneMax")
    print("=" * 70)

    n_vars = 30
    pop_size = 100
    max_generations = 50

    # Configure components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(proportion=0.5),
        learning=LearnMNFDA(
            max_clique_size=3,
            threshold=0.05,
            return_factorized=True  # Return FactorizedModel for PLS
        ),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=NoReplacement(),
        stop_condition=StopCriteriaMaxGen(max_generations),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=np.full(n_vars, 2),
        components=components,
    )

    # Run
    stats, cache = eda.run(verbose=True)

    print(f"\nFinal Results:")
    print(f"Best fitness: {stats.best_fitness_overall:.1f}")
    print(f"Best solution: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")
    print()


def run_mnfdag_example():
    """
    Example 2: MN-FDAG with PLS sampling on Trap-5

    Uses G-test for more accurate dependency detection.
    """
    print("=" * 70)
    print("Example 2: MN-FDAG with PLS Sampling on Trap-5")
    print("=" * 70)

    n_vars = 25  # 5 blocks of 5 bits
    pop_size = 150
    max_generations = 100

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(proportion=0.3),
        learning=LearnMNFDAG(
            max_clique_size=5,  # Allow detecting 5-bit dependencies
            alpha=0.01,  # More conservative than default
            return_factorized=True
        ),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=NoReplacement(),
        stop_condition=StopCriteriaMaxGen(max_generations),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=trap5,
        cardinality=np.full(n_vars, 2),
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal Results:")
    print(f"Best fitness: {stats.best_fitness_overall:.1f} (optimum: {n_vars // 5 * 6})")
    print(f"Best solution: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")
    print()


def run_moa_example():
    """
    Example 3: MOA with Gibbs sampling on OneMax

    Uses local Markov neighborhoods and MCMC sampling.
    """
    print("=" * 70)
    print("Example 3: MOA with Gibbs Sampling on OneMax")
    print("=" * 70)

    n_vars = 30
    pop_size = 100
    max_generations = 50

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(proportion=0.5),
        learning=LearnMOA(
            k_neighbors=5,  # Max 5 neighbors per variable
            threshold_factor=1.5,  # From paper
        ),
        sampling=SampleGibbs(
            n_samples=pop_size,
            IT=4,  # From paper: iterations = IT * n * ln(n)
            temperature=1.0,
            random_order=True
        ),
        replacement=NoReplacement(),
        stop_condition=StopCriteriaMaxGen(max_generations),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=np.full(n_vars, 2),
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal Results:")
    print(f"Best fitness: {stats.best_fitness_overall:.1f}")
    print(f"Best solution: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")
    print()


def run_moa_trap5_example():
    """
    Example 4: MOA with Gibbs sampling on Trap-5

    Demonstrates MOA on a deceptive problem.
    """
    print("=" * 70)
    print("Example 4: MOA with Gibbs Sampling on Trap-5")
    print("=" * 70)

    n_vars = 25
    pop_size = 200
    max_generations = 150

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(proportion=0.3),
        learning=LearnMOA(
            k_neighbors=8,  # More neighbors for harder problem
            threshold_factor=1.5,
        ),
        sampling=SampleGibbs(
            n_samples=pop_size,
            IT=6,  # More iterations for harder problem
            temperature=1.2,  # Slightly higher for more exploration
            random_order=True,
            burnin=50  # Discard first 50 iterations
        ),
        replacement=NoReplacement(),
        stop_condition=StopCriteriaMaxGen(max_generations),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=trap5,
        cardinality=np.full(n_vars, 2),
        components=components,
    )

    stats, cache = eda.run(verbose=True)

    print(f"\nFinal Results:")
    print(f"Best fitness: {stats.best_fitness_overall:.1f} (optimum: {n_vars // 5 * 6})")
    print(f"Best solution: {stats.best_individual}")
    print(f"Found at generation: {stats.generation_found}")
    print()


def comparison_example():
    """
    Comparison of MN-FDA, MN-FDAG, and MOA on OneMax
    """
    print("=" * 70)
    print("Comparison: MN-FDA vs MN-FDAG vs MOA on OneMax(30)")
    print("=" * 70)

    n_vars = 30
    pop_size = 100
    max_generations = 40
    n_runs = 5

    algorithms = [
        ("MN-FDA + PLS", LearnMNFDA(max_clique_size=3, return_factorized=True),
         SampleFDA(pop_size)),
        ("MN-FDAG + PLS", LearnMNFDAG(max_clique_size=3, return_factorized=True),
         SampleFDA(pop_size)),
        ("MOA + Gibbs", LearnMOA(k_neighbors=5),
         SampleGibbs(pop_size, IT=4)),
    ]

    results = {name: [] for name, _, _ in algorithms}

    for name, learning, sampling in algorithms:
        print(f"\nRunning {name}...")

        for run in range(n_runs):
            components = EDAComponents(
                seeding=RandomInit(),
                selection=TruncationSelection(proportion=0.5),
                learning=learning,
                sampling=sampling,
                replacement=NoReplacement(),
                stop_condition=StopCriteriaMaxGen(max_generations),
            )

            eda = EDA(
                pop_size=pop_size,
                n_vars=n_vars,
                fitness_func=onemax,
                cardinality=np.full(n_vars, 2),
                components=components,
            )

            stats, _ = eda.run(verbose=False)
            results[name].append(stats.best_fitness_overall)
            print(f"  Run {run+1}: {stats.best_fitness_overall:.1f}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    for name in results:
        values = results[name]
        print(f"{name:<20} {np.mean(values):<10.2f} {np.std(values):<10.2f} "
              f"{np.min(values):<10.2f} {np.max(values):<10.2f}")
    print()


if __name__ == "__main__":
    # Run all examples
    run_mnfda_example()
    run_mnfdag_example()
    run_moa_example()
    run_moa_trap5_example()
    comparison_example()
