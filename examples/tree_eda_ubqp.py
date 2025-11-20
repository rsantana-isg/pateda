"""
Tree EDA for Multi-Objective uBQP Problem

This example demonstrates solving a multi-objective unconstrained binary
quadratic programming problem using Tree-based EDA.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.functions.discrete.ubqp import generate_random_ubqp, evaluate_ubqp
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations


def pareto_dominates(obj1, obj2):
    """Check if obj1 Pareto dominates obj2 (maximization)."""
    return np.all(obj1 >= obj2) and np.any(obj1 > obj2)


def pareto_ranking(population, objectives):
    """Assign Pareto ranks to solutions."""
    n = len(population)
    ranks = np.zeros(n, dtype=int)
    dominated_count = np.zeros(n, dtype=int)
    dominating_sets = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                if pareto_dominates(objectives[i], objectives[j]):
                    dominating_sets[i].append(j)
                elif pareto_dominates(objectives[j], objectives[i]):
                    dominated_count[i] += 1

    current_front = [i for i in range(n) if dominated_count[i] == 0]
    rank = 0

    while current_front:
        for i in current_front:
            ranks[i] = rank

        next_front = []
        for i in current_front:
            for j in dominating_sets[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)

        current_front = next_front
        rank += 1

    return ranks


def multi_objective_selection(population, objectives, n_select):
    """Selection based on Pareto ranking."""
    ranks = pareto_ranking(population, objectives)
    sorted_indices = np.argsort(ranks)
    selected_pop = population[sorted_indices[:n_select]]
    fitness = -ranks
    selected_fit = fitness[sorted_indices[:n_select]]
    return selected_pop, selected_fit


def learn_tree(population, fitness, params=None):
    """Helper function to learn tree model"""
    learner = LearnTreeModel()
    n_vars = population.shape[1]
    cardinality = np.full(n_vars, 2)  # Binary variables
    model = learner.learn(
        generation=0,
        n_vars=n_vars,
        cardinality=cardinality,
        population=population,
        fitness=fitness
    )
    return model


def sample_tree_structure(model, n_samples):
    """Helper function to sample from tree model"""
    sampler = SampleFDA()
    n_vars = model.structure.shape[0]  # Number of cliques
    cardinality = np.full(n_vars, 2)  # Binary variables
    population = sampler.sample(
        n_vars=n_vars,
        model=model,
        cardinality=cardinality
    )
    # SampleFDA samples based on internal n_samples, but we need specific number
    # Resample if needed
    if len(population) != n_samples:
        # Create new sampler with correct size
        sampler = SampleFDA(n_samples=n_samples)
        population = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality
        )
    return population


def main():
    # Problem setup
    n_vars = 100
    density = 0.05
    n_objectives = 2
    seed = 42

    # Create uBQP instance
    print("Generating random multi-objective uBQP instance...")
    ubqp_instance = generate_random_ubqp(
        n_vars=n_vars,
        density=density,
        n_objectives=n_objectives,
        weight_range=(-100, 100),
        seed=seed
    )

    # Define objective function
    def objective(pop):
        results = ubqp_instance.evaluate(pop)
        return results

    # EDA parameters
    pop_size = 300
    n_selected = 150
    max_generations = 100

    # Create initial population
    initial_pop = np.random.randint(0, 2, size=(pop_size, n_vars))

    # Run optimization
    print("\nTree EDA - Multi-Objective uBQP Optimization")
    print("=" * 60)
    print(f"Number of variables: {n_vars}")
    print(f"Edge density: {density}")
    print(f"Number of objectives: {n_objectives}")
    print(f"Population size: {pop_size}")
    print(f"Selected individuals: {n_selected}")
    print(f"Max generations: {max_generations}")
    print()

    population = initial_pop.copy()
    best_per_generation = []

    for gen in range(max_generations):
        # Evaluate
        objectives = objective(population)

        # Selection using Pareto ranking
        selected_pop, selected_fit = multi_objective_selection(
            population, objectives, n_select=n_selected
        )

        # Learn tree structure
        model = learn_tree(selected_pop, selected_fit, params=None)

        # Sample new population
        new_population = sample_tree_structure(model, pop_size)

        # Replacement
        population = new_population

        # Track progress
        ranks = pareto_ranking(population, objectives)
        pareto_front_indices = ranks == 0
        pareto_objectives = objectives[pareto_front_indices]

        best_per_generation.append({
            'gen': gen,
            'pareto_size': len(pareto_objectives),
            'max_obj': np.max(pareto_objectives, axis=0),
            'mean_obj': np.mean(objectives, axis=0)
        })

        if gen % 20 == 0:
            print(f"Generation {gen}:")
            print(f"  Pareto front size: {len(pareto_objectives)}")
            print(f"  Best objectives: {np.max(pareto_objectives, axis=0)}")
            print(f"  Mean objectives: {np.mean(objectives, axis=0):.2f}")

    # Final evaluation
    final_objectives = objective(population)
    ranks = pareto_ranking(population, final_objectives)
    pareto_front_indices = ranks == 0
    pareto_solutions = population[pareto_front_indices]
    pareto_objectives = final_objectives[pareto_front_indices]

    # Report results
    print("\nOptimization completed!")
    print("=" * 60)
    print(f"Final Pareto front size: {len(pareto_solutions)}")
    print(f"\nPareto front objectives (best solutions):")

    # Sort by first objective
    sorted_indices = np.argsort(-pareto_objectives[:, 0])
    for i in sorted_indices[:10]:  # Show top 10
        obj = pareto_objectives[i]
        print(f"  Solution: Obj1 = {obj[0]:.2f}, Obj2 = {obj[1]:.2f}")

    print(f"\nBest single-objective results:")
    print(f"  Objective 1: {np.max(pareto_objectives[:, 0]):.2f}")
    print(f"  Objective 2: {np.max(pareto_objectives[:, 1]):.2f}")

    print(f"\nConvergence:")
    print(f"  Initial best: {best_per_generation[0]['max_obj']}")
    print(f"  Final best: {best_per_generation[-1]['max_obj']}")
    print(f"  Improvement: {best_per_generation[-1]['max_obj'] - best_per_generation[0]['max_obj']}")


if __name__ == "__main__":
    main()
