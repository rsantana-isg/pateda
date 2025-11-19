#!/usr/bin/env python3
"""
Standalone test for crossover operators

This script tests the three crossover operators on the OneMax problem.
"""

import sys
import numpy as np

# Add pateda to path
sys.path.insert(0, '/home/user/pateda')

from pateda.crossover.two_point import LearnTwoPointCrossover, SampleTwoPointCrossover
from pateda.crossover.transposition import LearnTransposition, SampleTransposition
from pateda.crossover.block import LearnBlockCrossover, SampleBlockCrossover
from pateda.mutation.bitflip import bit_flip_mutation
from pateda.functions.discrete import onemax


def test_two_point_crossover():
    """Test Two-Point Crossover on OneMax"""
    print("=" * 60)
    print("Testing Two-Point Crossover on OneMax")
    print("=" * 60)

    np.random.seed(42)
    n_vars = 30
    pop_size = 100
    n_generations = 25
    selection_size = 30

    # Initialize population
    population = np.random.randint(0, 2, (pop_size, n_vars))
    cardinality = np.ones(n_vars) * 2

    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_generations}")
    print(f"Selection size: {selection_size}\n")

    for gen in range(n_generations):
        # Evaluate
        fitness = np.apply_along_axis(onemax, 1, population)
        best_fitness = np.max(fitness)
        avg_fitness = np.mean(fitness)

        if gen % 5 == 0:
            print(f"Gen {gen:3d}: Best = {best_fitness:5.2f}, Avg = {avg_fitness:5.2f}")

        # Select best individuals
        idx = np.argsort(-fitness)[:selection_size]
        selected = population[idx]

        # Learn crossover model
        learner = LearnTwoPointCrossover(n_offspring=pop_size)
        model = learner.learn(
            generation=gen,
            n_vars=n_vars,
            cardinality=cardinality,
            population=selected,
            fitness=fitness[idx],
        )

        # Sample new population with mutation
        sampler = SampleTwoPointCrossover(
            n_samples=pop_size,
            mutation_fn=bit_flip_mutation,
            mutation_params={"mutation_prob": 1.0 / n_vars},
        )
        population = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=selected,
        )

    # Final evaluation
    final_fitness = np.apply_along_axis(onemax, 1, population)
    print(f"\nFinal: Best = {np.max(final_fitness):5.2f}, Avg = {np.mean(final_fitness):5.2f}")
    print(f"Success rate: {np.sum(final_fitness == n_vars) / pop_size * 100:.1f}% optimal\n")


def test_transposition():
    """Test Transposition on OneMax"""
    print("=" * 60)
    print("Testing Transposition on OneMax")
    print("=" * 60)

    np.random.seed(42)
    n_vars = 30
    pop_size = 100
    n_generations = 25
    selection_size = 30

    # Initialize population
    population = np.random.randint(0, 2, (pop_size, n_vars))
    cardinality = np.ones(n_vars) * 2

    print(f"Problem size: {n_vars} variables")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_generations}")
    print(f"Selection size: {selection_size}\n")

    for gen in range(n_generations):
        # Evaluate
        fitness = np.apply_along_axis(onemax, 1, population)
        best_fitness = np.max(fitness)
        avg_fitness = np.mean(fitness)

        if gen % 5 == 0:
            print(f"Gen {gen:3d}: Best = {best_fitness:5.2f}, Avg = {avg_fitness:5.2f}")

        # Select best individuals
        idx = np.argsort(-fitness)[:selection_size]
        selected = population[idx]

        # Learn transposition model
        learner = LearnTransposition(n_offspring=pop_size)
        model = learner.learn(
            generation=gen,
            n_vars=n_vars,
            cardinality=cardinality,
            population=selected,
            fitness=fitness[idx],
        )

        # Sample new population with mutation
        sampler = SampleTransposition(
            n_samples=pop_size,
            mutation_fn=bit_flip_mutation,
            mutation_params={"mutation_prob": 1.0 / n_vars},
        )
        population = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=selected,
        )

    # Final evaluation
    final_fitness = np.apply_along_axis(onemax, 1, population)
    print(f"\nFinal: Best = {np.max(final_fitness):5.2f}, Avg = {np.mean(final_fitness):5.2f}")
    print(f"Success rate: {np.sum(final_fitness == n_vars) / pop_size * 100:.1f}% optimal\n")


def test_block_crossover():
    """Test Block Crossover on OneMax"""
    print("=" * 60)
    print("Testing Block Crossover on OneMax")
    print("=" * 60)

    np.random.seed(42)
    # Create a problem with 6 blocks of 5 variables each (30 total)
    n_vars = 30
    n_classes = 6
    class_size = 5
    symmetry_index = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ])

    pop_size = 100
    n_generations = 25
    selection_size = 30

    # Initialize population
    population = np.random.randint(0, 2, (pop_size, n_vars))
    cardinality = np.ones(n_vars) * 2

    print(f"Problem size: {n_vars} variables ({n_classes} blocks of {class_size})")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_generations}")
    print(f"Selection size: {selection_size}\n")

    for gen in range(n_generations):
        # Evaluate
        fitness = np.apply_along_axis(onemax, 1, population)
        best_fitness = np.max(fitness)
        avg_fitness = np.mean(fitness)

        if gen % 5 == 0:
            print(f"Gen {gen:3d}: Best = {best_fitness:5.2f}, Avg = {avg_fitness:5.2f}")

        # Select best individuals
        idx = np.argsort(-fitness)[:selection_size]
        selected = population[idx]

        # Learn block crossover model
        learner = LearnBlockCrossover(
            n_offspring=pop_size,
            symmetry_index=symmetry_index,
        )
        model = learner.learn(
            generation=gen,
            n_vars=n_vars,
            cardinality=cardinality,
            population=selected,
            fitness=fitness[idx],
        )

        # Sample new population with mutation
        sampler = SampleBlockCrossover(
            n_samples=pop_size,
            symmetry_index=symmetry_index,
            mutation_prob=1.0 / n_vars,
        )
        population = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=selected,
        )

    # Final evaluation
    final_fitness = np.apply_along_axis(onemax, 1, population)
    print(f"\nFinal: Best = {np.max(final_fitness):5.2f}, Avg = {np.mean(final_fitness):5.2f}")
    print(f"Success rate: {np.sum(final_fitness == n_vars) / pop_size * 100:.1f}% optimal\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CROSSOVER OPERATORS TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_two_point_crossover()
        print("\n✓ Two-Point Crossover test completed successfully\n")
    except Exception as e:
        print(f"\n✗ Two-Point Crossover test failed: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_transposition()
        print("\n✓ Transposition test completed successfully\n")
    except Exception as e:
        print(f"\n✗ Transposition test failed: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_block_crossover()
        print("\n✓ Block Crossover test completed successfully\n")
    except Exception as e:
        print(f"\n✗ Block Crossover test failed: {e}\n")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
