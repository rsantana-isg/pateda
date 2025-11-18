"""
Examples demonstrating discrete EDAs for combinatorial optimization.

This script shows:
1. UMDA (Univariate Marginal Distribution Algorithm)
2. BMDA (Bivariate Marginal Distribution Algorithm)
3. EBNA (Estimation of Bayesian Network Algorithm)
4. Comparison of algorithms on benchmark problems
"""

import numpy as np
from pateda.learning.umda import LearnUMDA
from pateda.learning.bmda import LearnBMDA
from pateda.learning.fda import LearnFDA
from pateda.learning.ebna import LearnEBNA
from pateda.sampling.fda import SampleFDA
from pateda.sampling.bayesian_network import SampleBayesianNetwork


# Benchmark fitness functions for binary optimization

def onemax(x):
    """
    OneMax problem: maximize number of ones.

    Parameters
    ----------
    x : np.ndarray
        Binary array of shape (n_samples, n_vars)

    Returns
    -------
    fitness : np.ndarray
        Fitness values (higher is better)
    """
    return np.sum(x, axis=1).astype(float)


def trap_3(block):
    """3-bit trap function (single block)"""
    u = np.sum(block)
    if u == 3:
        return 3.0
    else:
        return 2.0 - u


def trap_function(x, block_size=3):
    """
    Concatenated trap function.

    Deceptive problem that misleads hill-climbers.
    Requires identifying building blocks.
    """
    n_vars = x.shape[1]
    n_blocks = n_vars // block_size
    fitness = np.zeros(len(x))

    for i in range(n_blocks):
        block = x[:, i*block_size:(i+1)*block_size]
        for j in range(len(x)):
            if block_size == 3:
                fitness[j] += trap_3(block[j])
            elif block_size == 4:
                fitness[j] += trap_4(block[j])

    return fitness


def trap_4(block):
    """4-bit trap function"""
    u = np.sum(block)
    if u == 0:
        return 4.0
    elif u == 4:
        return 5.0
    else:
        return 4.0 - u


def leading_ones(x):
    """
    Leading Ones problem.

    Counts the number of consecutive ones from the start.
    This problem has strong dependencies between variables.
    """
    fitness = np.zeros(len(x))
    for i in range(len(x)):
        count = 0
        for j in range(x.shape[1]):
            if x[i, j] == 1:
                count += 1
            else:
                break
        fitness[i] = float(count)
    return fitness


def nk_landscape(x, k=2, seed=42):
    """
    NK landscape (simplified version).

    Each bit depends on k other bits, creating epistasis.
    """
    np.random.seed(seed)
    n_vars = x.shape[1]
    fitness = np.zeros(len(x))

    # Simple NK: each bit contributes based on itself and k neighbors
    for i in range(n_vars):
        neighbors = [(i + j) % n_vars for j in range(k + 1)]
        for sample_idx in range(len(x)):
            pattern = tuple(x[sample_idx, neighbors])
            # Use pattern hash as fitness contribution
            contribution = hash(pattern) % 100 / 100.0
            fitness[sample_idx] += contribution

    return fitness


def run_umda_example():
    """Example: UMDA on OneMax"""
    print("\n" + "="*60)
    print("UMDA (Univariate Marginal Distribution Algorithm)")
    print("Problem: OneMax")
    print("="*60 + "\n")

    n_vars = 30
    pop_size = 100
    n_generations = 25
    selection_size = 30

    # Initialize
    np.random.seed(42)
    population = np.random.randint(0, 2, (pop_size, n_vars))

    print(f"Problem size: {n_vars} bits")
    print(f"Population size: {pop_size}")
    print(f"Selection size: {selection_size}")
    print(f"Generations: {n_generations}\n")

    best_fitness_history = []
    mean_fitness_history = []

    for gen in range(n_generations):
        # Evaluate
        fitness = onemax(population)

        # Track progress
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

        if gen % 5 == 0 or gen == n_generations - 1:
            print(f"Generation {gen:3d}: Best = {best_fitness:.1f}/{n_vars}, Mean = {mean_fitness:.2f}")

        # Select best individuals
        idx = np.argsort(-fitness)[:selection_size]  # Negative for descending
        selected_pop = population[idx]
        selected_fit = fitness[idx]

        # Learn UMDA model
        learner = LearnUMDA(n_vars=n_vars)
        model = learner.learn(selected_pop, selected_fit)

        # Sample new population
        sampler = SampleFDA(model=model)
        population = sampler.sample(n_samples=pop_size)

    print(f"\nFinal best fitness: {best_fitness_history[-1]:.1f}/{n_vars}")
    print(f"Convergence: {best_fitness_history[0]:.1f} -> {best_fitness_history[-1]:.1f}")

    return best_fitness_history


def run_bmda_example():
    """Example: BMDA on Trap function"""
    print("\n" + "="*60)
    print("BMDA (Bivariate Marginal Distribution Algorithm)")
    print("Problem: 3-bit Trap Function")
    print("="*60 + "\n")

    n_vars = 15  # 5 blocks of 3 bits
    pop_size = 150
    n_generations = 30
    selection_size = 50

    print(f"Problem: {n_vars} bits ({n_vars//3} trap blocks)")
    print(f"Population size: {pop_size}")
    print(f"Optimal fitness: {n_vars} (all blocks solved)\n")

    # Initialize
    np.random.seed(42)
    population = np.random.randint(0, 2, (pop_size, n_vars))

    best_fitness_history = []

    for gen in range(n_generations):
        # Evaluate
        fitness = trap_function(population, block_size=3)

        best_fitness = np.max(fitness)
        best_fitness_history.append(best_fitness)

        if gen % 5 == 0 or gen == n_generations - 1:
            print(f"Generation {gen:3d}: Best = {best_fitness:.1f}/{n_vars}")

        # Select
        idx = np.argsort(-fitness)[:selection_size]

        # Learn BMDA (models pairwise dependencies)
        learner = LearnBMDA(n_vars=n_vars)
        model = learner.learn(population[idx], fitness[idx])

        # Sample
        sampler = SampleFDA(model=model)
        population = sampler.sample(n_samples=pop_size)

    print(f"\nFinal best fitness: {best_fitness_history[-1]:.1f}/{n_vars}")
    return best_fitness_history


def run_ebna_example():
    """Example: EBNA on deceptive problem"""
    print("\n" + "="*60)
    print("EBNA (Estimation of Bayesian Network Algorithm)")
    print("Problem: 4-bit Deceptive Function")
    print("="*60 + "\n")

    n_vars = 16  # 4 blocks of 4 bits
    pop_size = 200
    n_generations = 25
    selection_size = 60

    print(f"Problem: {n_vars} bits ({n_vars//4} deceptive blocks)")
    print(f"Population size: {pop_size}\n")

    # Initialize
    np.random.seed(42)
    population = np.random.randint(0, 2, (pop_size, n_vars))

    best_fitness_history = []

    for gen in range(n_generations):
        # Evaluate
        fitness = trap_function(population, block_size=4)

        best_fitness = np.max(fitness)
        best_fitness_history.append(best_fitness)

        if gen % 5 == 0 or gen == n_generations - 1:
            print(f"Generation {gen:3d}: Best = {best_fitness:.1f}")

        # Select
        idx = np.argsort(-fitness)[:selection_size]

        # Learn Bayesian Network
        learner = LearnEBNA(n_vars=n_vars)
        model = learner.learn(population[idx], fitness[idx])

        # Sample from Bayesian Network
        sampler = SampleBayesianNetwork(model=model)
        population = sampler.sample(n_samples=pop_size)

    print(f"\nFinal best fitness: {best_fitness_history[-1]:.1f}")
    return best_fitness_history


def compare_algorithms():
    """Compare UMDA, BMDA, and EBNA on different problems"""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)

    results = {}

    # Test 1: OneMax (no dependencies) - UMDA should excel
    print("\n--- Test 1: OneMax (No Dependencies) ---")
    print("Expected: UMDA performs well (simple problem)")

    n_vars = 30
    pop_size = 100

    for alg_name, learner_class in [
        ('UMDA', LearnUMDA),
        ('BMDA', LearnBMDA),
    ]:
        np.random.seed(42)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(20):
            fitness = onemax(population)
            idx = np.argsort(-fitness)[:30]

            learner = learner_class(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        final_best = np.max(onemax(population))
        results[f'OneMax_{alg_name}'] = final_best
        print(f"  {alg_name}: {final_best:.1f}/{n_vars}")

    # Test 2: Trap function (dependencies) - BMDA/EBNA should be better
    print("\n--- Test 2: Trap Function (With Dependencies) ---")
    print("Expected: BMDA/EBNA better than UMDA")

    n_vars = 15

    for alg_name, learner_class in [
        ('UMDA', LearnUMDA),
        ('BMDA', LearnBMDA),
    ]:
        np.random.seed(42)
        population = np.random.randint(0, 2, (pop_size, n_vars))

        for gen in range(25):
            fitness = trap_function(population, block_size=3)
            idx = np.argsort(-fitness)[:30]

            learner = learner_class(n_vars=n_vars)
            model = learner.learn(population[idx], fitness[idx])

            sampler = SampleFDA(model=model)
            population = sampler.sample(n_samples=pop_size)

        final_best = np.max(trap_function(population, block_size=3))
        results[f'Trap_{alg_name}'] = final_best
        print(f"  {alg_name}: {final_best:.1f}/{n_vars}")

    print("\n" + "="*60)
    print("Summary:")
    print("  - UMDA works well on problems without dependencies")
    print("  - BMDA/EBNA better on problems with pairwise/higher dependencies")
    print("="*60)

    return results


def detailed_onemax_example():
    """Detailed walkthrough of UMDA on OneMax"""
    print("\n" + "="*60)
    print("DETAILED EXAMPLE: UMDA on OneMax")
    print("="*60 + "\n")

    n_vars = 20
    pop_size = 50

    print("Problem: OneMax")
    print(f"  - Maximize the number of 1s in a binary string")
    print(f"  - String length: {n_vars}")
    print(f"  - Optimal solution: all 1s, fitness = {n_vars}\n")

    # Initialize random population
    np.random.seed(42)
    population = np.random.randint(0, 2, (pop_size, n_vars))

    print("Initial population (first 5 individuals):")
    for i in range(5):
        print(f"  {i+1}. {''.join(map(str, population[i]))} (fitness = {onemax(population[i:i+1])[0]:.0f})")

    print(f"\nRunning UMDA for 20 generations...\n")

    for gen in range(20):
        fitness = onemax(population)

        if gen % 5 == 0:
            best_idx = np.argmax(fitness)
            print(f"Generation {gen}:")
            print(f"  Best fitness: {fitness[best_idx]:.0f}/{n_vars}")
            print(f"  Mean fitness: {np.mean(fitness):.2f}")
            print(f"  Best solution: {''.join(map(str, population[best_idx]))}")

        # Select top 40%
        selection_size = 20
        idx = np.argsort(-fitness)[:selection_size]

        # Learn probabilities
        learner = LearnUMDA(n_vars=n_vars)
        model = learner.learn(population[idx], fitness[idx])

        # Show learned probabilities at generation 0
        if gen == 0 and hasattr(learner, 'probabilities'):
            print(f"\n  Learned probabilities (P(X_i=1)):")
            print(f"  {learner.probabilities}")

        # Sample new population
        sampler = SampleFDA(model=model)
        population = sampler.sample(n_samples=pop_size)

    # Final results
    final_fitness = onemax(population)
    best_idx = np.argmax(final_fitness)

    print(f"\nFinal results:")
    print(f"  Best fitness: {final_fitness[best_idx]:.0f}/{n_vars}")
    print(f"  Best solution: {''.join(map(str, population[best_idx]))}")
    print(f"  Success rate: {(final_fitness[best_idx]/n_vars)*100:.1f}%")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# DISCRETE EDA EXAMPLES")
    print("#"*60)

    # Run examples
    detailed_onemax_example()
    run_umda_example()
    run_bmda_example()

    try:
        run_ebna_example()
    except Exception as e:
        print(f"\nEBNA example skipped: {e}")

    compare_algorithms()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*60 + "\n")
