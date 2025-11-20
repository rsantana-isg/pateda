"""
CUMDA (Constraint UMDA) for Posfunction

This example demonstrates CUMDA on the Posfunction, one of the test functions
from the original CUMDA paper (Santana & Ochoa).

Problem: Maximize Posfunction(x) = Σ Pos(xi), where Pos(xi) = i if xi=1, 0 otherwise

The function reaches its optimum when the last r variables are set to 1.
For example, with n=10 and r=5, the optimum is: [0,0,0,0,0,1,1,1,1,1]

Constraint: Exactly r variables must be set to 1 (unitation constraint)

CUMDA vs Standard UMDA:
- Standard UMDA would sample each variable independently
- CUMDA samples exactly r variables to be 1, using normalized probabilities
- This maintains the unitation constraint throughout the search

References:
- Santana, R., & Ochoa, A. "A Constraint Univariate Marginal Distribution Algorithm."
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import SeedingUnitationConstraint
from pateda.learning import LearnCUMDA
from pateda.sampling import SampleCUMDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


def create_posfunction(r: int):
    """
    Create Posfunction fitness function

    Args:
        r: Number of ones required (unitation value)

    Returns:
        Fitness function that maximizes sum of positions of ones
    """
    def posfunction(x: np.ndarray) -> float:
        """
        Posfunction: Sum of positions where x_i = 1

        Args:
            x: Binary vector

        Returns:
            Sum of (i+1) for all i where x[i] = 1
        """
        if x.ndim == 1:
            # Single solution
            return np.sum((np.arange(len(x)) + 1) * x)
        else:
            # Multiple solutions (vectorized)
            positions = np.arange(x.shape[1]) + 1
            return np.sum(positions * x, axis=1)

    return posfunction


def print_solution(x: np.ndarray, fitness: float, r: int, n: int):
    """Pretty print a solution"""
    ones_positions = np.where(x == 1)[0] + 1  # 1-indexed
    visual = ['.' if bit == 0 else '█' for bit in x]
    visual_str = ''.join(visual)

    # Calculate optimal fitness
    optimal_fitness = sum(range(n - r + 1, n + 1))

    print(f"  [{visual_str}]  fitness={fitness:.0f} (optimal={optimal_fitness})")
    print(f"    Ones at positions: {list(ones_positions)}")


def run_cumda_posfunction(
    n_vars: int = 30,
    r: int = 10,
    pop_size: int = 100,
    max_gen: int = 50,
    selection_ratio: float = 0.3,
    verbose: bool = True
):
    """
    Run CUMDA on Posfunction

    Args:
        n_vars: Length of binary vector
        r: Number of ones required
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Fraction to select (truncation selection parameter)
        verbose: Print detailed progress

    Returns:
        Tuple of (statistics, cache)
    """
    if r > n_vars:
        raise ValueError(f"r ({r}) must be <= n_vars ({n_vars})")

    print("=" * 80)
    print("CUMDA for Posfunction")
    print("=" * 80)
    print(f"Problem size: n={n_vars}, r={r} (exactly {r} ones)")
    print(f"Population size: {pop_size}")
    print(f"Maximum generations: {max_gen}")
    print(f"Selection ratio: {selection_ratio} (top {int(pop_size*selection_ratio)} selected)")
    print()

    # Calculate optimal fitness and solution
    optimal_fitness = sum(range(n_vars - r + 1, n_vars + 1))
    optimal_solution = np.zeros(n_vars, dtype=int)
    optimal_solution[-r:] = 1

    print(f"Optimal solution: last {r} variables set to 1")
    print(f"Optimal fitness: {optimal_fitness}")
    print()

    # Variable cardinalities (binary)
    cardinality = np.full(n_vars, 2)

    # Create fitness function
    fitness_func = create_posfunction(r)

    # Configure components
    components = EDAComponents(
        # Seeding: Initialize with exactly r ones (random positions)
        seeding=SeedingUnitationConstraint(),
        seeding_params={'num_ones': r},

        # Learning: CUMDA (learns marginal probabilities)
        learning=LearnCUMDA(alpha=0.0),  # No Laplace smoothing

        # Sampling: CUMDA sampling (samples exactly r variables to set to 1)
        sampling=SampleCUMDA(n_samples=pop_size, n_ones=r),

        # Selection: Truncation selection
        selection=TruncationSelection(ratio=selection_ratio),

        # No repairing needed: CUMDA sampling already ensures exactly r ones

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
    )

    # Run optimization
    statistics, cache = eda.run(verbose=verbose)

    # Print results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Best fitness found: {statistics.best_fitness_overall:.0f} (optimal: {optimal_fitness})")
    print(f"Generation found: {statistics.generation_found}")
    print(f"\nBest solution:")
    print_solution(statistics.best_individual, statistics.best_fitness_overall, r, n_vars)

    # Show evolution of fitness
    print(f"\nFitness evolution:")
    print(f"  Generation | Best | Mean | Std")
    print(f"  {'-'*40}")
    for gen in range(len(statistics.best_fitness)):
        print(f"  {gen:10d} | {statistics.best_fitness[gen]:6.1f} | "
              f"{statistics.mean_fitness[gen]:6.2f} | {statistics.std_fitness[gen]:5.2f}")

    # Check if optimal was found
    gap = optimal_fitness - statistics.best_fitness_overall
    if gap == 0:
        print(f"\n✓ Optimal solution found!")
    else:
        print(f"\n✗ Optimal solution not found. Gap: {gap:.0f}")

    return statistics, cache


def compare_different_r_values():
    """
    Compare CUMDA performance for different values of r
    """
    print("\n" + "=" * 80)
    print("Comparing Different r Values")
    print("=" * 80)

    n_vars = 30
    r_values = [5, 10, 15, 20, 25]
    pop_size = 100
    max_gen = 50

    results = []

    for r in r_values:
        print(f"\n--- Running CUMDA with r={r} ---")
        optimal_fitness = sum(range(n_vars - r + 1, n_vars + 1))

        stats, _ = run_cumda_posfunction(
            n_vars=n_vars,
            r=r,
            pop_size=pop_size,
            max_gen=max_gen,
            selection_ratio=0.3,
            verbose=False
        )

        gap = optimal_fitness - stats.best_fitness_overall
        results.append((r, optimal_fitness, stats.best_fitness_overall, gap, stats.generation_found))

    # Summary table
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"{'r':<5} | {'Optimal':<10} | {'Found':<10} | {'Gap':<8} | {'Generation'}")
    print("-" * 80)
    for r, opt, found, gap, gen in results:
        optimal_mark = "✓" if gap == 0 else " "
        print(f"{optimal_mark} {r:<4} | {opt:<10.0f} | {found:<10.0f} | {gap:<8.0f} | {gen}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare different r values
        compare_different_r_values()
    else:
        # Run single experiment
        print("Running CUMDA on Posfunction")
        print("(Use 'python cumda_posfunction.py compare' to test different r values)\n")

        stats, cache = run_cumda_posfunction(
            n_vars=30,
            r=10,
            pop_size=100,
            max_gen=50,
            selection_ratio=0.3,
            verbose=True
        )
