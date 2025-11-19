"""
Example: Comparing Standard and Elimination Affinity Factorization

This example compares the two affinity-based factorization strategies:
1. LearnAffinityFactorization: Recursively processes each large cluster separately
2. LearnAffinityFactorizationElim: Collects large clusters and processes together

The trap function is used, which has multiple deceptive local optima.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning.affinity import (
    LearnAffinityFactorization,
    LearnAffinityFactorizationElim,
)
from pateda.sampling.fda import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.trap import trap_n


def run_eda_variant(learning_method, method_name):
    """Run EDA with specified learning method"""

    # Problem setup
    n_vars = 40
    trap_size = 5
    cardinality = np.full(n_vars, 2)

    # EDA parameters
    pop_size = 500
    selection_size = 250
    max_generations = 50

    def fitness_func(x):
        # Handle both single individual (1D) and population (2D)
        if x.ndim == 1:
            return trap_n(x, n_trap=trap_size)
        else:
            return np.array([trap_n(ind, n_trap=trap_size) for ind in x])

    # Initialize components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning_method,
        sampling=SampleFDA(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(max_generations),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
    )

    print(f"\n{'=' * 60}")
    print(f"Running: {method_name}")
    print(f"{'=' * 60}")

    stats, cache = eda.run(verbose=False)

    # Analyze results
    print(f"\nResults for {method_name}:")
    print(f"  Best fitness: {stats.best_fitness_overall:.4f}")
    print(f"  Target (global optimum): {n_vars}")
    print(f"  Gap: {n_vars - stats.best_fitness_overall:.4f}")

    # Get the final model from cache
    model = cache.get('models', [])[-1] if 'models' in cache and cache['models'] else None
    if model:
        metadata = model.metadata
        n_cliques = metadata.get("n_cliques", 0)

        cliques = model.structure
        clique_sizes = []
        for i in range(cliques.shape[0]):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            clique_sizes.append(n_overlap + n_new)

        print(f"\n  Model structure:")
        print(f"    Number of cliques: {n_cliques}")
        print(f"    Clique size range: {min(clique_sizes)}-{max(clique_sizes)}")
        print(f"    Average clique size: {np.mean(clique_sizes):.2f}")
        print(f"    Cliques of size {trap_size}: {clique_sizes.count(trap_size)}")

    return {'best_fitness': stats.best_fitness_overall, 'model': model, 'stats': stats, 'cache': cache}


def main():
    """Compare affinity-based factorization strategies"""

    print("Comparing Affinity-Based Factorization Strategies")
    print("Problem: Trap function (deceptive)")
    print("=" * 60)

    # Set random seed
    np.random.seed(42)

    # Strategy 1: Standard recursive factorization
    learning1 = LearnAffinityFactorization(
        max_clique_size=5,
        preference=None,
        damping=0.5,
        recursive=True,
        alpha=0.1,
    )
    result1 = run_eda_variant(learning1, "Standard Affinity Factorization")

    # Reset random seed for fair comparison
    np.random.seed(42)

    # Strategy 2: Elimination-based factorization
    learning2 = LearnAffinityFactorizationElim(
        max_clique_size=5,
        preference=None,
        damping=0.9,
        alpha=0.1,
    )
    result2 = run_eda_variant(learning2, "Elimination-based Factorization")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    print(f"\n{'Method':<40} {'Best Fitness':>15}")
    print("-" * 60)
    print(f"{'Standard Affinity Factorization':<40} {result1['best_fitness']:>15.4f}")
    print(
        f"{'Elimination-based Factorization':<40} {result2['best_fitness']:>15.4f}"
    )
    print(f"{'Global Optimum':<40} {40.0:>15.4f}")

    print("\nModel Complexity:")
    model1 = result1.get("model")
    model2 = result2.get("model")

    if model1 and model2:
        n_cliques1 = model1.metadata.get("n_cliques", 0)
        n_cliques2 = model2.metadata.get("n_cliques", 0)

        print(f"  Standard:   {n_cliques1} cliques")
        print(f"  Elimination: {n_cliques2} cliques")

    # Determine winner
    if result1["best_fitness"] > result2["best_fitness"]:
        print("\n✓ Standard Affinity Factorization performed better")
    elif result2["best_fitness"] > result1["best_fitness"]:
        print("\n✓ Elimination-based Factorization performed better")
    else:
        print("\n✓ Both methods achieved the same fitness")

    print("\nNote: Results may vary due to stochastic nature of the algorithms.")
    print(
        "The best method depends on the problem structure and parameter settings."
    )


if __name__ == "__main__":
    main()
