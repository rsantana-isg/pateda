"""
UMDA variant for Contiguous Block Problem

This example demonstrates a UMDA variant that incorporates three key components:

1. Seeding: Generates initial population where all solutions have exactly k ones
2. Local Optimization: Tries to arrange ones into a contiguous block
3. Repairing: After sampling, fixes solutions to have exactly k ones

Problem: Find a contiguous block of k ones in a binary vector of length n (n > k)

This is a pedagogical example that shows how these three components work together:
- Seeding ensures feasibility from the start (k ones constraint)
- Local optimization guides solutions toward contiguous arrangements
- Repairing maintains feasibility after probabilistic sampling

The optimal solution has k consecutive ones: e.g., for k=5, n=10: [0,0,1,1,1,1,1,0,0,0]
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import SeedingUnitationConstraint, RandomInit
from pateda.learning import LearnFDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations
from pateda.repairing.unitation_method import UnitationRepairing
from pateda.local_optimization.contiguous_block_opt import ContiguousBlockOptimizer
from pateda.functions.discrete.contiguous_block import create_contiguous_block_function


def print_solution(x: np.ndarray, fitness: float, k: int):
    """Pretty print a solution showing the contiguous block structure"""
    ones_positions = np.where(x == 1)[0]

    # Create visual representation
    visual = ['.' if bit == 0 else '█' for bit in x]
    visual_str = ''.join(visual)

    # Find gaps
    if len(ones_positions) > 0:
        gaps = 0
        for i in range(len(ones_positions) - 1):
            if ones_positions[i+1] - ones_positions[i] > 1:
                gaps += 1

        print(f"  [{visual_str}]  fitness={fitness:.0f}/{k}  gaps={gaps}")
    else:
        print(f"  [{visual_str}]  fitness={fitness:.0f}/{k}")


def run_umda_contiguous_block(
    n_vars: int = 20,
    k: int = 5,
    pop_size: int = 100,
    max_gen: int = 20,
    selection_ratio: float = 0.5,
    use_seeding: bool = True,
    use_local_opt: bool = True,
    use_repairing: bool = True,
    verbose: bool = True
):
    """
    Run UMDA variant on contiguous block problem.

    Args:
        n_vars: Length of binary vector
        k: Number of ones (size of block to find)
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Fraction of population to select for learning
        use_seeding: Whether to use k-ones seeding (vs random)
        use_local_opt: Whether to use local optimization
        use_repairing: Whether to use repairing
        verbose: Print detailed progress

    Returns:
        Tuple of (statistics, cache)
    """
    if k >= n_vars:
        raise ValueError(f"k ({k}) must be less than n_vars ({n_vars})")

    print("=" * 80)
    print("UMDA Variant for Contiguous Block Problem")
    print("=" * 80)
    print(f"Problem size: n={n_vars}, k={k}")
    print(f"Population size: {pop_size}")
    print(f"Maximum generations: {max_gen}")
    print(f"Selection ratio: {selection_ratio}")
    print()
    print("Components:")
    print(f"  - Seeding (k-ones constraint): {'✓' if use_seeding else '✗'}")
    print(f"  - Local optimization: {'✓' if use_local_opt else '✗'}")
    print(f"  - Repairing (k-ones constraint): {'✓' if use_repairing else '✗'}")
    print()

    # Variable cardinalities (binary)
    cardinality = np.full(n_vars, 2)

    # Create fitness function
    fitness_func = create_contiguous_block_function(k=k, with_penalty=False)

    # Configure components
    components = EDAComponents(
        # Seeding: Initialize with exactly k ones (or random)
        seeding=SeedingUnitationConstraint() if use_seeding else RandomInit(),
        seeding_params={'num_ones': k} if use_seeding else {},

        # Learning: UMDA (univariate, no structure learning)
        learning=LearnFDA(cliques=None),  # None = univariate

        # Sampling: Generate pop_size new solutions
        sampling=SampleFDA(n_samples=pop_size),

        # Selection: Select top 50%
        selection=TruncationSelection(ratio=selection_ratio),

        # Repairing: Ensure exactly k ones after sampling
        repairing=UnitationRepairing.exact_k_ones(k=k) if use_repairing else None,

        # Local optimization: Move ones together
        local_opt=ContiguousBlockOptimizer(max_iterations=10, aggressive=True) if use_local_opt else None,

        # Stop condition
        stop_condition=MaxGenerations(max_gen=max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
        random_seed=42,
    )

    # Run optimization
    statistics, cache = eda.run(verbose=verbose)

    # Print results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Best fitness found: {statistics.best_fitness_overall:.0f} (optimal: {k})")
    print(f"Generation found: {statistics.generation_found}")
    print(f"\nBest solution:")
    print_solution(statistics.best_individual, statistics.best_fitness_overall, k)

    # Show evolution of fitness
    print(f"\nFitness evolution:")
    print(f"  Generation | Best | Mean | Std")
    print(f"  {'-'*40}")
    for gen in range(len(statistics.best_fitness)):
        print(f"  {gen:10d} | {statistics.best_fitness[gen]:4.1f} | "
              f"{statistics.mean_fitness[gen]:4.2f} | {statistics.std_fitness[gen]:4.2f}")

    # Check if optimal was found
    if statistics.best_fitness_overall == k:
        print(f"\n✓ Optimal solution found! Perfect contiguous block of {k} ones.")
    else:
        print(f"\n✗ Optimal solution not found. Best: {statistics.best_fitness_overall}/{k}")

    return statistics, cache


def demonstrate_components():
    """
    Demonstrate the impact of each component by running multiple configurations.
    """
    print("\n" + "=" * 80)
    print("Demonstrating Component Impact")
    print("=" * 80)
    print("\nRunning 4 different configurations to show component impact:\n")

    n_vars = 20
    k = 6
    pop_size = 100
    max_gen = 15

    configs = [
        ("Baseline (no components)", False, False, False),
        ("With seeding only", True, False, False),
        ("With seeding + repairing", True, False, True),
        ("Full variant (all components)", True, True, True),
    ]

    results = []

    for name, use_seeding, use_local, use_repair in configs:
        print(f"\n--- {name} ---")
        stats, _ = run_umda_contiguous_block(
            n_vars=n_vars,
            k=k,
            pop_size=pop_size,
            max_gen=max_gen,
            use_seeding=use_seeding,
            use_local_opt=use_local,
            use_repairing=use_repair,
            verbose=False
        )
        results.append((name, stats.best_fitness_overall, stats.generation_found))

    # Summary table
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"{'Configuration':<40} | {'Best Fitness':<12} | {'Generation'}")
    print("-" * 80)
    for name, fitness, gen in results:
        optimal_mark = "✓" if fitness == k else " "
        print(f"{optimal_mark} {name:<38} | {fitness:4.0f}/{k:<6} | {gen}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demonstration comparing different configurations
        demonstrate_components()
    else:
        # Run single experiment with full variant
        print("Running UMDA variant with all three components")
        print("(Use 'python umda_contiguous_block.py demo' to compare configurations)\n")

        stats, cache = run_umda_contiguous_block(
            n_vars=25,
            k=8,
            pop_size=150,
            max_gen=25,
            selection_ratio=0.5,
            use_seeding=True,
            use_local_opt=True,
            use_repairing=True,
            verbose=True
        )
