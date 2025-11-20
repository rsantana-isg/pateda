"""
CFDA (Constraint FDA) for OneMaxBlock Function

This example demonstrates CFDA on the OneMaxBlock function, one of the test
functions from the constraint FDA paper (Santana, Ochoa, & Soto).

Problem: Maximize the length of the maximum contiguous block of ones

For solutions with exactly r ones, the maximum value is r (one contiguous block).
For example, with r=5: [0,0,1,1,1,1,1,0,0] has value 5 (optimal)
                        [1,1,0,1,1,1,0,0,0] has value 3 (suboptimal)

Constraint: Exactly r variables must be set to 1

CFDA vs CUMDA:
- CUMDA uses univariate marginals (assumes independence)
- CFDA uses factorized distributions (captures dependencies)
- For OneMaxBlock, pairwise dependencies help identify contiguous patterns
- CFDA should converge faster by exploiting problem structure

References:
- Santana, R., Ochoa, A., & Soto, M. R. "Factorized Distribution Algorithms
  For Functions With Unitation Constraints."
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import SeedingUnitationConstraint
from pateda.learning import LearnCFDA, create_pairwise_chain_cliques
from pateda.sampling import SampleCFDA
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations


def create_onemaxblock_function(r: int):
    """
    Create OneMaxBlock fitness function

    Args:
        r: Number of ones required (unitation value)

    Returns:
        Fitness function that finds length of max contiguous block of ones
    """
    def onemaxblock(x: np.ndarray) -> float:
        """
        OneMaxBlock: Length of maximum contiguous block of ones

        Args:
            x: Binary vector or matrix

        Returns:
            Length of longest run of consecutive ones
        """
        if x.ndim == 1:
            # Single solution
            return _max_block_length(x)
        else:
            # Multiple solutions (vectorized)
            return np.array([_max_block_length(row) for row in x])

    def _max_block_length(x: np.ndarray) -> int:
        """Find maximum contiguous block of ones"""
        if len(x) == 0 or np.sum(x) == 0:
            return 0

        max_length = 0
        current_length = 0

        for bit in x:
            if bit == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0

        return max_length

    return onemaxblock


def print_solution(x: np.ndarray, fitness: float, r: int):
    """Pretty print a solution"""
    visual = ['.' if bit == 0 else '█' for bit in x]
    visual_str = ''.join(visual)

    # Find blocks
    blocks = []
    in_block = False
    block_start = 0

    for i, bit in enumerate(x):
        if bit == 1 and not in_block:
            block_start = i
            in_block = True
        elif bit == 0 and in_block:
            blocks.append((block_start, i - 1))
            in_block = False

    if in_block:
        blocks.append((block_start, len(x) - 1))

    print(f"  [{visual_str}]  fitness={fitness:.0f}/{r}")
    if blocks:
        block_strs = [f"[{start}:{end}]({end-start+1})" for start, end in blocks]
        print(f"    Blocks: {', '.join(block_strs)}")


def run_cfda_onemaxblock(
    n_vars: int = 30,
    r: int = 10,
    pop_size: int = 100,
    max_gen: int = 50,
    selection_ratio: float = 0.3,
    use_pairwise: bool = True,
    verbose: bool = True
):
    """
    Run CFDA on OneMaxBlock

    Args:
        n_vars: Length of binary vector
        r: Number of ones required
        pop_size: Population size
        max_gen: Maximum generations
        selection_ratio: Fraction to select
        use_pairwise: Use pairwise factorization (vs univariate)
        verbose: Print detailed progress

    Returns:
        Tuple of (statistics, cache)
    """
    if r > n_vars:
        raise ValueError(f"r ({r}) must be <= n_vars ({n_vars})")

    print("=" * 80)
    print("CFDA for OneMaxBlock")
    print("=" * 80)
    print(f"Problem size: n={n_vars}, r={r} (exactly {r} ones)")
    print(f"Population size: {pop_size}")
    print(f"Maximum generations: {max_gen}")
    print(f"Selection ratio: {selection_ratio}")
    print(f"Factorization: {'Pairwise chain' if use_pairwise else 'Univariate'}")
    print()
    print(f"Optimal fitness: {r} (one contiguous block of {r} ones)")
    print()

    # Variable cardinalities (binary)
    cardinality = np.full(n_vars, 2)

    # Create fitness function
    fitness_func = create_onemaxblock_function(r)

    # Create factorization structure
    if use_pairwise:
        # Pairwise chain: (x0,x1), (x1,x2), (x2,x3), ...
        # This captures local dependencies useful for finding contiguous blocks
        cliques = create_pairwise_chain_cliques(n_vars)
        print(f"Using pairwise chain factorization: {n_vars-1} cliques")
    else:
        # Univariate (like CUMDA)
        cliques = None
        print(f"Using univariate factorization: {n_vars} independent variables")

    print()

    # Configure components
    components = EDAComponents(
        # Seeding: Initialize with exactly r ones (random positions)
        seeding=SeedingUnitationConstraint(),
        seeding_params={'num_ones': r},

        # Learning: CFDA with specified factorization
        learning=LearnCFDA(cliques=cliques),

        # Sampling: CFDA sampling (samples from factorized distribution + constraint)
        sampling=SampleCFDA(n_samples=pop_size, n_ones=r),

        # Selection: Truncation selection
        selection=TruncationSelection(ratio=selection_ratio),

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
    print(f"Best fitness found: {statistics.best_fitness_overall:.0f} (optimal: {r})")
    print(f"Generation found: {statistics.generation_found}")
    print(f"\nBest solution:")
    print_solution(statistics.best_individual, statistics.best_fitness_overall, r)

    # Show evolution of fitness
    print(f"\nFitness evolution:")
    print(f"  Generation | Best | Mean | Std")
    print(f"  {'-'*40}")
    for gen in range(len(statistics.best_fitness)):
        print(f"  {gen:10d} | {statistics.best_fitness[gen]:4.1f} | "
              f"{statistics.mean_fitness[gen]:4.2f} | {statistics.std_fitness[gen]:4.2f}")

    # Check if optimal was found
    if statistics.best_fitness_overall == r:
        print(f"\n✓ Optimal solution found! Perfect contiguous block of {r} ones.")
    else:
        print(f"\n✗ Optimal solution not found. Gap: {r - statistics.best_fitness_overall:.0f}")

    return statistics, cache


def compare_factorizations():
    """
    Compare CFDA with different factorizations:
    - Univariate (like CUMDA)
    - Pairwise chain (captures local dependencies)
    """
    print("\n" + "=" * 80)
    print("Comparing Factorizations")
    print("=" * 80)
    print("\nComparing univariate vs pairwise factorization:\n")

    n_vars = 30
    r = 10
    pop_size = 100
    max_gen = 40
    n_runs = 5

    configs = [
        ("Univariate (CUMDA-like)", False),
        ("Pairwise chain (CFDA)", True),
    ]

    results = {}

    for name, use_pairwise in configs:
        print(f"\n--- {name} ---")
        print(f"Running {n_runs} independent runs...")

        best_fits = []
        gens_found = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ")
            stats, _ = run_cfda_onemaxblock(
                n_vars=n_vars,
                r=r,
                pop_size=pop_size,
                max_gen=max_gen,
                selection_ratio=0.3,
                use_pairwise=use_pairwise,
                verbose=False
            )
            best_fits.append(stats.best_fitness_overall)
            gens_found.append(stats.generation_found)
            print(f"fitness={stats.best_fitness_overall:.0f}, gen={stats.generation_found}")

        results[name] = {
            'mean_fitness': np.mean(best_fits),
            'std_fitness': np.std(best_fits),
            'mean_gen': np.mean(gens_found),
            'success_rate': sum(f == r for f in best_fits) / n_runs
        }

    # Summary table
    print("\n" + "=" * 80)
    print("Summary of Results (averaged over {} runs)".format(n_runs))
    print("=" * 80)
    print(f"{'Configuration':<30} | {'Mean Fitness':<15} | {'Success Rate':<12} | {'Mean Gen'}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<30} | {res['mean_fitness']:4.2f} ± {res['std_fitness']:4.2f}        | "
              f"{res['success_rate']*100:5.1f}%        | {res['mean_gen']:4.1f}")

    print()
    print("Note: Pairwise factorization should perform better by capturing")
    print("      local dependencies that favor contiguous blocks.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare different factorizations
        compare_factorizations()
    else:
        # Run single experiment with pairwise factorization
        print("Running CFDA on OneMaxBlock with pairwise factorization")
        print("(Use 'python cfda_onemaxblock.py compare' to compare factorizations)\n")

        stats, cache = run_cfda_onemaxblock(
            n_vars=30,
            r=10,
            pop_size=100,
            max_gen=40,
            selection_ratio=0.3,
            use_pairwise=True,
            verbose=True
        )
