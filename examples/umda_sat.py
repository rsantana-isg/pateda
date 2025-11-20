"""
UMDA for Multi-Objective SAT Problem

This example demonstrates solving a multi-objective 3-SAT problem
using UMDA with Pareto ranking for selection.
"""

import numpy as np
from pateda.core.eda import EDA
from pateda.functions.discrete.sat import load_random_3sat, evaluate_sat
from pateda.learning import LearnUMDA
from pateda.sampling.fda import SampleFDA


def pareto_dominates(obj1, obj2):
    """Check if obj1 Pareto dominates obj2."""
    return np.all(obj1 >= obj2) and np.any(obj1 > obj2)


def pareto_ranking(population, objectives):
    """
    Assign Pareto ranks to solutions.

    Returns ranks where 0 is the best (non-dominated front).
    """
    n = len(population)
    ranks = np.zeros(n, dtype=int)
    dominated_count = np.zeros(n, dtype=int)
    dominating_sets = [[] for _ in range(n)]

    # Count dominations
    for i in range(n):
        for j in range(n):
            if i != j:
                if pareto_dominates(objectives[i], objectives[j]):
                    dominating_sets[i].append(j)
                elif pareto_dominates(objectives[j], objectives[i]):
                    dominated_count[i] += 1

    # Assign ranks
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
    """
    Selection based on Pareto ranking.

    Selects the best n_select individuals based on their Pareto rank.
    """
    ranks = pareto_ranking(population, objectives)
    sorted_indices = np.argsort(ranks)
    selected_pop = population[sorted_indices[:n_select]]
    # Convert ranks to fitness (lower rank = higher fitness)
    fitness = -ranks
    selected_fit = fitness[sorted_indices[:n_select]]
    return selected_pop, selected_fit


def main():
    # Problem setup
    n_vars = 50
    n_clauses = 100
    n_objectives = 2  # Multi-objective SAT
    seed = 42

    # Create SAT instance
    print("Creating random 3-SAT instance...")
    sat_instance = load_random_3sat(n_vars, n_clauses, n_objectives, seed)

    # Define objective function
    def objective(pop):
        # Returns array of shape (pop_size, n_objectives)
        results = sat_instance.evaluate(pop)
        # For multi-objective, return the objectives directly
        return results

    # EDA parameters
    pop_size = 200
    n_selected = 100
    max_generations = 50

    # Create initial population
    initial_pop = np.random.randint(0, 2, size=(pop_size, n_vars))

    # Run optimization
    print("\nUMDA - Multi-Objective 3-SAT Optimization")
    print("=" * 60)
    print(f"Number of variables: {n_vars}")
    print(f"Number of clauses per formula: {n_clauses}")
    print(f"Number of objectives: {n_objectives}")
    print(f"Population size: {pop_size}")
    print(f"Selected individuals: {n_selected}")
    print(f"Max generations: {max_generations}")
    print()

    population = initial_pop.copy()
    best_solutions = []
    best_objectives = []

    for gen in range(max_generations):
        # Evaluate
        objectives = objective(population)

        # Selection using Pareto ranking
        selected_pop, selected_fit = multi_objective_selection(
            population, objectives, n_select=n_selected
        )

        # Learn model
        learner = LearnUMDA(alpha=0.0)
        cardinality = np.full(n_vars, 2)  # Binary variables
        model = learner.learn(
            generation=gen,
            n_vars=n_vars,
            cardinality=cardinality,
            population=selected_pop,
            fitness=selected_fit
        )

        # Sample new population
        sampler = SampleFDA(n_samples=pop_size)
        new_population = sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality
        )

        # Replacement (elitist)
        population = new_population

        # Track best solutions (non-dominated front)
        ranks = pareto_ranking(population, objectives)
        pareto_front_indices = ranks == 0
        pareto_solutions = population[pareto_front_indices]
        pareto_objectives = objectives[pareto_front_indices]

        if gen % 10 == 0:
            print(f"Generation {gen}: Pareto front size = {len(pareto_solutions)}")
            print(f"  Best objectives: {np.max(pareto_objectives, axis=0)}")
            mean_objs = np.mean(objectives, axis=0)
            print(f"  Mean objectives: [{mean_objs[0]:.2f}, {mean_objs[1]:.2f}]")

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
    print(f"\nPareto front objectives:")
    for i, obj in enumerate(pareto_objectives):
        clauses_satisfied = obj
        percentages = (clauses_satisfied / n_clauses) * 100
        print(f"  Solution {i+1}: {clauses_satisfied} "
              f"({percentages[0]:.1f}%, {percentages[1]:.1f}%)")

    print(f"\nBest single-objective results:")
    print(f"  Objective 1: {np.max(pareto_objectives[:, 0])} / {n_clauses} clauses")
    print(f"  Objective 2: {np.max(pareto_objectives[:, 1])} / {n_clauses} clauses")


if __name__ == "__main__":
    main()
