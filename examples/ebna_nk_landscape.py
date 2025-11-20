"""
EBNA (Estimation of Bayesian Network Algorithm) for NK Landscape

This script demonstrates EBNA on the NK landscape problem, a tunable
epistatic benchmark where K controls the degree of variable interactions.

EBNA uses a Bayesian network to model dependencies between variables,
which should capture the epistatic interactions in the NK landscape.

This is a NEW combination testing:
- EBNA with K2 structure learning
- NK Landscape with varying K values
- Comparison of EBNA performance vs K parameter

The NK landscape is defined by:
- N variables
- K epistatic interactions per variable
- Random fitness contributions for each configuration

References:
- Kauffman, S. A. (1993). "The Origins of Order"
- Pelikan, M., et al. (1999). "BOA: The Bayesian Optimization Algorithm"
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.learning.bayesian_network import LearnBayesianNetwork
from pateda.sampling.bayesian_network import SampleBN


# Import NK landscape if available
try:
    from pateda.functions.discrete.nk_landscape import NKLandscape
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False
    # Simple fallback implementation
    class NKLandscape:
        def __init__(self, n, k, seed=None):
            self.n = n
            self.k = k
            if seed is not None:
                np.random.seed(seed)

            # Create random fitness tables
            self.tables = []
            for i in range(n):
                # Each variable depends on itself + k others
                table_size = 2 ** (k + 1)
                self.tables.append(np.random.random(table_size))

            # Create neighbor structure (circular)
            self.neighbors = []
            for i in range(n):
                neighbors = [(i + j + 1) % n for j in range(k)]
                self.neighbors.append(neighbors)

        def evaluate(self, x):
            fitness = 0.0
            for i in range(self.n):
                # Get configuration for this variable and its neighbors
                config_bits = [x[i]]
                for neighbor in self.neighbors[i]:
                    config_bits.append(x[neighbor])

                # Convert to index
                idx = sum(bit * (2 ** j) for j, bit in enumerate(config_bits))
                fitness += self.tables[i][int(idx)]

            return fitness / self.n  # Normalize


def run_ebna_nk_landscape(n_vars=50, k=4, seed=42):
    """
    Run EBNA on NK Landscape

    Args:
        n_vars: Number of variables (N)
        k: Number of epistatic interactions per variable (K)
        seed: Random seed for landscape generation
    """
    print("=" * 80)
    print("EBNA for NK Landscape Problem")
    print("=" * 80)
    print()

    # Create NK landscape
    nk = NKLandscape(n_vars, k, seed=seed)

    pop_size = 500
    max_generations = 100

    print("Configuration:")
    print(f"  - Problem: NK Landscape")
    print(f"  - N (variables): {n_vars}")
    print(f"  - K (epistasis): {k}")
    print(f"  - Population size: {pop_size}")
    print(f"  - Max generations: {max_generations}")
    print(f"  - Algorithm: EBNA with K2 structure learning")
    print()

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),

        # Selection: top 50%
        selection=TruncationSelection(proportion=0.5),

        # EBNA learning with K2 algorithm
        learning=LearnBayesianNetwork(
            structure_algorithm='k2',
            max_parents=min(k + 2, 5),  # Allow slightly more parents than K
            scoring_metric='bic',
        ),

        # Bayesian Network sampling
        sampling=SampleBN(n_samples=pop_size),

        # Keep best 10 individuals
        replacement=ElitistReplacement(n_elite=10),

        stop_condition=MaxGenerations(max_gen=max_generations),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=nk.evaluate,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=42,
    )

    print("Running EBNA on NK Landscape...")
    print()

    # Run
    stats, cache = eda.run(verbose=True)

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Best fitness: {stats.best_fitness_overall:.6f}")
    print(f"Generation found: {stats.generation_found}")
    print()
    print(f"Best solution: {stats.best_individual}")
    print()

    return stats, cache, nk


def run_ebna_varying_k():
    """
    Test EBNA performance as K increases (more epistasis)

    This shows how EBNA's ability to capture dependencies helps
    as the problem becomes more epistatic.
    """
    print("=" * 80)
    print("EBNA Performance vs Epistasis Level (K)")
    print("=" * 80)
    print()

    n_vars = 40
    pop_size = 500
    max_generations = 80
    n_runs = 5
    k_values = [1, 2, 4, 8]

    results = {k: [] for k in k_values}

    for k in k_values:
        print(f"\nTesting K = {k} (epistatic interactions per variable)")
        print("-" * 60)

        for run in range(n_runs):
            # Create new instance for each run
            nk = NKLandscape(n_vars, k, seed=run)

            components = EDAComponents(
                seeding=RandomInit(),
                selection=TruncationSelection(proportion=0.5),
                learning=LearnBayesianNetwork(
                    structure_algorithm='k2',
                    max_parents=min(k + 2, 5),
                    scoring_metric='bic',
                ),
                sampling=SampleBN(n_samples=pop_size),
                replacement=ElitistReplacement(n_elite=10),
                stop_condition=MaxGenerations(max_gen=max_generations),
            )

            eda = EDA(
                pop_size=pop_size,
                n_vars=n_vars,
                fitness_func=nk.evaluate,
                cardinality=np.full(n_vars, 2),
                components=components,
                random_seed=42,
            )

            stats, _ = eda.run(verbose=False)
            results[k].append(stats.best_fitness_overall)

            print(f"  Run {run + 1}: fitness = {stats.best_fitness_overall:.6f}")

    # Print comparison
    print()
    print("=" * 80)
    print("PERFORMANCE vs EPISTASIS LEVEL")
    print("=" * 80)
    print()
    print(f"{'K':<10} {'Mean Fitness':<15} {'Std':<15} {'Best':<15}")
    print("-" * 80)

    for k in k_values:
        values = np.array(results[k])
        print(f"{k:<10} {np.mean(values):<15.6f} {np.std(values):<15.6f} {np.max(values):<15.6f}")

    print()
    print("Observation: As K increases, the problem becomes harder due to more")
    print("epistatic interactions. EBNA's Bayesian network helps capture these")
    print("dependencies, but performance may degrade for very high K values.")
    print()


def run_comparison_ebna_vs_umda():
    """
    Compare EBNA vs UMDA on NK landscape

    EBNA should outperform UMDA on NK landscapes with K > 0
    because it can model dependencies.
    """
    print("=" * 80)
    print("Comparison: EBNA vs UMDA on NK Landscape (K=4)")
    print("=" * 80)
    print()

    from pateda.learning.histogram import LearnHistogram
    from pateda.sampling.histogram import SampleHistogram

    n_vars = 40
    k = 4
    pop_size = 500
    max_generations = 80
    n_runs = 5

    algorithms = [
        ("UMDA", LearnHistogram(), SampleHistogram(pop_size)),
        ("EBNA", LearnBayesianNetwork(structure_algorithm='k2', max_parents=5),
         SampleBN(pop_size)),
    ]

    results = {name: [] for name, _, _ in algorithms}

    for name, learning, sampling in algorithms:
        print(f"\nRunning {name}...")

        for run in range(n_runs):
            nk = NKLandscape(n_vars, k, seed=run)

            components = EDAComponents(
                seeding=RandomInit(),
                selection=TruncationSelection(proportion=0.5),
                learning=learning,
                sampling=sampling,
                replacement=ElitistReplacement(n_elite=10),
                stop_condition=MaxGenerations(max_gen=max_generations),
            )

            eda = EDA(
                pop_size=pop_size,
                n_vars=n_vars,
                fitness_func=nk.evaluate,
                cardinality=np.full(n_vars, 2),
                components=components,
                random_seed=42,
            )

            stats, _ = eda.run(verbose=False)
            results[name].append(stats.best_fitness_overall)

            print(f"  Run {run + 1}: fitness = {stats.best_fitness_overall:.6f}")

    # Print comparison
    print()
    print("=" * 80)
    print("EBNA vs UMDA COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Algorithm':<15} {'Mean Fitness':<15} {'Std':<15} {'Best':<15}")
    print("-" * 80)

    for name in results:
        values = np.array(results[name])
        print(f"{name:<15} {np.mean(values):<15.6f} {np.std(values):<15.6f} {np.max(values):<15.6f}")

    print()
    print("Expected: EBNA should outperform UMDA because it can model the")
    print("epistatic dependencies (K=4) present in the NK landscape.")
    print()


if __name__ == "__main__":
    # Run single example
    stats, cache, nk = run_ebna_nk_landscape(n_vars=50, k=4)

    # Run K variation experiment
    print("\n" * 2)
    run_ebna_varying_k()

    # Run comparison
    print("\n" * 2)
    run_comparison_ebna_vs_umda()
