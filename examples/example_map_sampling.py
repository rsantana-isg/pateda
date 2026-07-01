"""
Example: MAP-based Sampling for Markov Network EDAs

This example demonstrates how to use MAP (Maximum A Posteriori) based sampling
methods with MN-FDA and MOA algorithms.

Based on Santana, R. (2013). "Message Passing Methods for Estimation of
Distribution Algorithms Based on Markov Networks"

MAP-based sampling strategies:
1. Insert-MAP (S1): Insert MAP configuration directly into population
2. Template-MAP (S2): Use MAP as template for crossover/recombination
3. Hybrid MAP (S3): Combine both strategies

The paper shows that:
- Insert-MAP generally outperforms other strategies
- Performance advantage increases with higher variable cardinality
- Exact and approximate inference (BP, decimation) show similar performance
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.moa import LearnMOA
from pateda.sampling.map_sampling import (
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleHybridMAP,
)
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.fda import SampleFDA
from pateda.selection import TruncationSelection
from pateda.seeding import RandomInit
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations


def onemax(x):
    """OneMax: maximize sum of bits"""
    return np.sum(x, axis=1)


def trap5(x):
    """Trap-5: deceptive problem with 5-bit building blocks"""
    n = x.shape[1]
    fitness = np.zeros(x.shape[0])
    for i in range(0, n, 5):
        block = x[:, i:i+5]
        ones = np.sum(block, axis=1)
        fitness += np.where(ones == 5, 5, 4 - ones)
    return fitness


def ternary_onemax(x):
    """OneMax for ternary variables (0, 1, 2)"""
    return np.sum(x, axis=1)


def run_eda(n_vars, cardinality, fitness_func, pop_size, n_generations,
            learning, sampling, random_seed=42, verbose=False):
    """Helper: build and run one EDA experiment, return (best_fitness, stats)."""
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(n_generations),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.array(cardinality),
        components=components,
        random_seed=random_seed,
    )
    stats, _ = eda.run(verbose=verbose)
    return stats


def main():
    """Run examples comparing different sampling strategies"""

    print("=" * 70)
    print("MAP-Based Sampling Examples for Markov Network EDAs")
    print("=" * 70)

    pop_size = 100

    # Example 1: MN-FDA with Insert-MAP on OneMax
    print("\n" + "-" * 70)
    print("Example 1: MN-FDA + Insert-MAP on OneMax (n=30)")
    print("-" * 70)

    n_vars = 30
    stats = run_eda(
        n_vars=n_vars,
        cardinality=[2] * n_vars,
        fitness_func=onemax,
        pop_size=pop_size,
        n_generations=50,
        learning=LearnMNFDA(max_clique_size=3, threshold=0.05, return_factorized=False),
        sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp",
                                 n_map_inserts=1, replace_worst=True),
        random_seed=42,
        verbose=True,
    )
    print(f"\nResults:")
    print(f"  Best fitness: {stats.best_fitness_overall:.1f} (optimum: {n_vars})")
    print(f"  Gen found:    {stats.generation_found}")
    print(f"  Success: {stats.best_fitness_overall == n_vars}")

    # Example 2: MN-FDA with Template-MAP on Trap-5
    print("\n" + "-" * 70)
    print("Example 2: MN-FDA + Template-MAP on Trap-5 (n=25, 5 blocks)")
    print("-" * 70)

    n_vars = 25
    optimum = 25
    stats = run_eda(
        n_vars=n_vars,
        cardinality=[2] * n_vars,
        fitness_func=trap5,
        pop_size=pop_size,
        n_generations=100,
        learning=LearnMNFDA(max_clique_size=5, threshold=0.05, return_factorized=False),
        sampling=SampleTemplateMAP(n_samples=pop_size, map_method="bp",
                                   template_prob=0.6, min_template_vars=5),
        random_seed=42,
        verbose=True,
    )
    print(f"\nResults:")
    print(f"  Best fitness: {stats.best_fitness_overall:.1f} (optimum: {optimum})")
    print(f"  Gen found:    {stats.generation_found}")
    print(f"  Success: {stats.best_fitness_overall == optimum}")

    # Example 3: MN-FDA with Hybrid MAP
    print("\n" + "-" * 70)
    print("Example 3: MN-FDA + Hybrid MAP on OneMax (n=30)")
    print("-" * 70)

    n_vars = 30
    stats = run_eda(
        n_vars=n_vars,
        cardinality=[2] * n_vars,
        fitness_func=onemax,
        pop_size=pop_size,
        n_generations=50,
        learning=LearnMNFDA(max_clique_size=3, return_factorized=False),
        sampling=SampleHybridMAP(n_samples=pop_size, map_method="bp",
                                 template_prob=0.5, n_map_inserts=1),
        random_seed=42,
        verbose=True,
    )
    print(f"\nResults:")
    print(f"  Best fitness: {stats.best_fitness_overall:.1f} (optimum: {n_vars})")
    print(f"  Success: {stats.best_fitness_overall == n_vars}")

    # Example 4: MOA with Insert-MAP
    print("\n" + "-" * 70)
    print("Example 4: MOA + Insert-MAP on OneMax (n=30)")
    print("-" * 70)

    n_vars = 30
    stats = run_eda(
        n_vars=n_vars,
        cardinality=[2] * n_vars,
        fitness_func=onemax,
        pop_size=pop_size,
        n_generations=50,
        learning=LearnMOA(k_neighbors=3, threshold_factor=1.5),
        sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp"),
        random_seed=42,
        verbose=True,
    )
    print(f"\nResults:")
    print(f"  Best fitness: {stats.best_fitness_overall:.1f} (optimum: {n_vars})")
    print(f"  Success: {stats.best_fitness_overall == n_vars}")

    # Example 5: Higher cardinality (ternary variables)
    print("\n" + "-" * 70)
    print("Example 5: MN-FDA + Insert-MAP on Ternary OneMax (n=20, k=3)")
    print("-" * 70)
    print("(Paper shows MAP methods excel with higher cardinality)")

    n_vars = 20
    optimum = n_vars * 2  # All 2's
    stats = run_eda(
        n_vars=n_vars,
        cardinality=[3] * n_vars,
        fitness_func=ternary_onemax,
        pop_size=pop_size,
        n_generations=60,
        learning=LearnMNFDA(max_clique_size=3, return_factorized=False),
        sampling=SampleInsertMAP(n_samples=pop_size, map_method="bp"),
        random_seed=42,
        verbose=True,
    )
    print(f"\nResults:")
    print(f"  Best fitness: {stats.best_fitness_overall:.1f} (optimum: {optimum})")
    print(f"  Success: {stats.best_fitness_overall == optimum}")

    # Example 6: Comparing MAP inference methods
    print("\n" + "-" * 70)
    print("Example 6: Comparing MAP Inference Methods")
    print("-" * 70)
    print("Testing: BP vs Decimation")

    n_vars = 25
    methods = {"Belief Propagation": "bp", "Decimation": "decimation"}

    print(f"\nProblem: Trap-5 (n={n_vars})")
    for method_name, method_code in methods.items():
        stats = run_eda(
            n_vars=n_vars,
            cardinality=[2] * n_vars,
            fitness_func=trap5,
            pop_size=100,
            n_generations=80,
            learning=LearnMNFDA(max_clique_size=5, return_factorized=False),
            sampling=SampleInsertMAP(n_samples=100, map_method=method_code),
            random_seed=42,
            verbose=False,
        )
        print(f"  {method_name}: Best={stats.best_fitness_overall:.1f}, "
              f"Gen found={stats.generation_found}")

    # Example 7: Comparison with baseline methods
    print("\n" + "-" * 70)
    print("Example 7: Comparing MAP-based vs Traditional Sampling")
    print("-" * 70)

    n_vars = 30
    n_runs = 3

    strategies = {
        "Insert-MAP": (
            lambda: LearnMNFDA(max_clique_size=3, return_factorized=False),
            lambda: SampleInsertMAP(n_samples=pop_size, map_method="bp"),
        ),
        "Template-MAP": (
            lambda: LearnMNFDA(max_clique_size=3, return_factorized=False),
            lambda: SampleTemplateMAP(n_samples=pop_size, map_method="bp", template_prob=0.6),
        ),
        "Hybrid-MAP": (
            lambda: LearnMNFDA(max_clique_size=3, return_factorized=False),
            lambda: SampleHybridMAP(n_samples=pop_size, map_method="bp"),
        ),
        "Gibbs": (
            lambda: LearnMNFDA(max_clique_size=3, return_factorized=False),
            lambda: SampleGibbs(n_samples=pop_size, IT=4),
        ),
        "PLS": (
            lambda: LearnMNFDA(max_clique_size=3, return_factorized=True),
            lambda: SampleFDA(n_samples=pop_size),
        ),
    }

    print(f"\nProblem: OneMax (n={n_vars}), Runs={n_runs}")
    print(f"\n{'Strategy':<15} {'Avg Fitness':<13} {'Avg Gen':<10} {'Success Rate'}")
    print("-" * 60)

    for strategy_name, (make_learner, make_sampler) in strategies.items():
        fitnesses = []
        gen_found = []
        successes = 0

        for seed in range(n_runs):
            s = run_eda(
                n_vars=n_vars,
                cardinality=[2] * n_vars,
                fitness_func=onemax,
                pop_size=pop_size,
                n_generations=50,
                learning=make_learner(),
                sampling=make_sampler(),
                random_seed=100 + seed,
                verbose=False,
            )
            fitnesses.append(s.best_fitness_overall)
            gen_found.append(s.generation_found if s.generation_found is not None else 50)
            if s.best_fitness_overall == n_vars:
                successes += 1

        print(f"{strategy_name:<15} {np.mean(fitnesses):<13.1f} "
              f"{np.mean(gen_found):<10.1f} {successes/n_runs:.0%}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nKey findings from Santana (2013):")
    print("  - Insert-MAP (S1) generally outperforms other strategies")
    print("  - Performance advantage increases with variable cardinality")
    print("  - MAP methods particularly effective on deceptive problems")
    print("  - BP and decimation MAP inference show similar performance")
    print("=" * 70)


if __name__ == "__main__":
    main()
