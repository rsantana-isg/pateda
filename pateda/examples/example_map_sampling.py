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
from pateda.core.eda import EDA
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.moa import LearnMOA
from pateda.sampling.map_sampling import (
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleHybridMAP,
)
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.fda import SampleFDA
from pateda.selection.truncation import SelectTruncation
from pateda.seeding.random import SeedRandom


# Define test problems
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


def main():
    """Run examples comparing different sampling strategies"""

    print("="*70)
    print("MAP-Based Sampling Examples for Markov Network EDAs")
    print("="*70)

    # Example 1: MN-FDA with Insert-MAP on OneMax
    print("\n" + "-"*70)
    print("Example 1: MN-FDA + Insert-MAP on OneMax (n=30)")
    print("-"*70)

    n_vars = 30
    pop_size = 100
    n_generations = 50
    cardinality = np.array([2] * n_vars)

    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_function=onemax,
        pop_size=pop_size,
        n_generations=n_generations,
        seeding=SeedRandom(),
        learning=LearnMNFDA(
            max_clique_size=3,
            threshold=0.05,
            return_factorized=False  # Return MarkovNetworkModel for MAP sampling
        ),
        sampling=SampleInsertMAP(
            n_samples=pop_size,
            map_method="bp",  # Use belief propagation for MAP inference
            n_map_inserts=1,  # Insert one MAP configuration
            replace_worst=True  # Replace worst individual with MAP
        ),
        selection=SelectTruncation(ratio=0.5),
        verbose=True,
    )

    result = eda.run()

    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']} (optimum: {n_vars})")
    print(f"  Evaluations: {result['n_evaluations']}")
    print(f"  Success: {result['best_fitness'] == n_vars}")


    # Example 2: MN-FDA with Template-MAP on Trap-5
    print("\n" + "-"*70)
    print("Example 2: MN-FDA + Template-MAP on Trap-5 (n=25, 5 blocks)")
    print("-"*70)

    n_vars = 25
    n_generations = 100

    eda = EDA(
        n_vars=n_vars,
        cardinality=np.array([2] * n_vars),
        fitness_function=trap5,
        pop_size=pop_size,
        n_generations=n_generations,
        seeding=SeedRandom(),
        learning=LearnMNFDA(
            max_clique_size=5,  # Larger cliques to capture building blocks
            threshold=0.05,
            return_factorized=False
        ),
        sampling=SampleTemplateMAP(
            n_samples=pop_size,
            map_method="bp",
            template_prob=0.6,  # 60% of variables from MAP template
            min_template_vars=5  # At least 5 variables from template
        ),
        selection=SelectTruncation(ratio=0.5),
        verbose=True,
    )

    result = eda.run()

    optimum = 25  # 5 blocks * 5 points each
    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']} (optimum: {optimum})")
    print(f"  Evaluations: {result['n_evaluations']}")
    print(f"  Success: {result['best_fitness'] == optimum}")


    # Example 3: MN-FDA with Hybrid MAP
    print("\n" + "-"*70)
    print("Example 3: MN-FDA + Hybrid MAP on OneMax (n=30)")
    print("-"*70)

    n_vars = 30

    eda = EDA(
        n_vars=n_vars,
        cardinality=np.array([2] * n_vars),
        fitness_function=onemax,
        pop_size=pop_size,
        n_generations=50,
        seeding=SeedRandom(),
        learning=LearnMNFDA(
            max_clique_size=3,
            return_factorized=False
        ),
        sampling=SampleHybridMAP(
            n_samples=pop_size,
            map_method="bp",
            template_prob=0.5,  # Balanced exploration/exploitation
            n_map_inserts=1
        ),
        selection=SelectTruncation(ratio=0.5),
        verbose=True,
    )

    result = eda.run()

    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']} (optimum: {n_vars})")
    print(f"  Success: {result['best_fitness'] == n_vars}")


    # Example 4: MOA with Insert-MAP
    print("\n" + "-"*70)
    print("Example 4: MOA + Insert-MAP on OneMax (n=30)")
    print("-"*70)

    eda = EDA(
        n_vars=n_vars,
        cardinality=np.array([2] * n_vars),
        fitness_function=onemax,
        pop_size=pop_size,
        n_generations=50,
        seeding=SeedRandom(),
        learning=LearnMOA(
            k_neighbors=3,
            threshold_factor=1.5
        ),
        sampling=SampleInsertMAP(
            n_samples=pop_size,
            map_method="bp"
        ),
        selection=SelectTruncation(ratio=0.5),
        verbose=True,
    )

    result = eda.run()

    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']} (optimum: {n_vars})")
    print(f"  Success: {result['best_fitness'] == n_vars}")


    # Example 5: Higher cardinality (ternary variables)
    print("\n" + "-"*70)
    print("Example 5: MN-FDA + Insert-MAP on Ternary OneMax (n=20, k=3)")
    print("-"*70)
    print("(Paper shows MAP methods excel with higher cardinality)")

    n_vars = 20
    cardinality = np.array([3] * n_vars)  # Ternary: {0, 1, 2}
    optimum = n_vars * 2  # All 2's

    eda = EDA(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_function=ternary_onemax,
        pop_size=pop_size,
        n_generations=60,
        seeding=SeedRandom(),
        learning=LearnMNFDA(
            max_clique_size=3,
            return_factorized=False
        ),
        sampling=SampleInsertMAP(
            n_samples=pop_size,
            map_method="bp"
        ),
        selection=SelectTruncation(ratio=0.5),
        verbose=True,
    )

    result = eda.run()

    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']} (optimum: {optimum})")
    print(f"  Success: {result['best_fitness'] == optimum}")


    # Example 6: Comparison of MAP inference methods
    print("\n" + "-"*70)
    print("Example 6: Comparing MAP Inference Methods")
    print("-"*70)
    print("Testing different MAP inference methods: BP vs Decimation")

    n_vars = 25
    methods = {
        "Belief Propagation": "bp",
        "Decimation": "decimation"
    }

    print(f"\nProblem: Trap-5 (n={n_vars})")

    for method_name, method_code in methods.items():
        eda = EDA(
            n_vars=n_vars,
            cardinality=np.array([2] * n_vars),
            fitness_function=trap5,
            pop_size=100,
            n_generations=80,
            seeding=SeedRandom(),
            learning=LearnMNFDA(max_clique_size=5, return_factorized=False),
            sampling=SampleInsertMAP(
                n_samples=100,
                map_method=method_code
            ),
            selection=SelectTruncation(ratio=0.5),
            verbose=False,
        )

        result = eda.run()

        print(f"  {method_name}: Best={result['best_fitness']}, "
              f"Evals={result['n_evaluations']}")


    # Example 7: Comparison with baseline methods
    print("\n" + "-"*70)
    print("Example 7: Comparing MAP-based vs Traditional Sampling")
    print("-"*70)

    n_vars = 30
    n_runs = 3

    strategies = {
        "Insert-MAP": (
            LearnMNFDA(max_clique_size=3, return_factorized=False),
            SampleInsertMAP(n_samples=pop_size, map_method="bp")
        ),
        "Template-MAP": (
            LearnMNFDA(max_clique_size=3, return_factorized=False),
            SampleTemplateMAP(n_samples=pop_size, map_method="bp", template_prob=0.6)
        ),
        "Hybrid-MAP": (
            LearnMNFDA(max_clique_size=3, return_factorized=False),
            SampleHybridMAP(n_samples=pop_size, map_method="bp")
        ),
        "Gibbs": (
            LearnMNFDA(max_clique_size=3, return_factorized=False),
            SampleGibbs(n_samples=pop_size, IT=4)
        ),
        "PLS": (
            LearnMNFDA(max_clique_size=3, return_factorized=True),
            SampleFDA(n_samples=pop_size)
        ),
    }

    print(f"\nProblem: OneMax (n={n_vars}), Runs={n_runs}")
    print(f"\n{'Strategy':<15} {'Avg Fitness':<12} {'Avg Evals':<12} {'Success Rate'}")
    print("-"*60)

    for strategy_name, (learner, sampler) in strategies.items():
        fitnesses = []
        evaluations = []
        successes = 0

        for _ in range(n_runs):
            eda = EDA(
                n_vars=n_vars,
                cardinality=np.array([2] * n_vars),
                fitness_function=onemax,
                pop_size=pop_size,
                n_generations=50,
                seeding=SeedRandom(),
                learning=learner,
                sampling=sampler,
                selection=SelectTruncation(ratio=0.5),
                verbose=False,
            )

            result = eda.run()
            fitnesses.append(result['best_fitness'])
            evaluations.append(result['n_evaluations'])
            if result['best_fitness'] == n_vars:
                successes += 1

        avg_fitness = np.mean(fitnesses)
        avg_evals = np.mean(evaluations)
        success_rate = successes / n_runs

        print(f"{strategy_name:<15} {avg_fitness:<12.1f} {avg_evals:<12.0f} "
              f"{success_rate:.0%}")

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nKey findings from Santana (2013):")
    print("  - Insert-MAP (S1) generally outperforms other strategies")
    print("  - Performance advantage increases with variable cardinality")
    print("  - MAP methods particularly effective on deceptive problems")
    print("  - BP and decimation MAP inference show similar performance")
    print("="*70)


if __name__ == "__main__":
    main()
