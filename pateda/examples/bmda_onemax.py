"""
BMDA for OneMax problem

This example demonstrates using BMDA (Bivariate Marginal Distribution Algorithm)
to solve the OneMax problem with tournament selection.
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnBMDA
from pateda.sampling import SampleFDA
from pateda.selection import TournamentSelection
from pateda.stop_conditions import MaxGenerations
from pateda.functions import onemax


def run_bmda_onemax():
    """Run BMDA on OneMax problem"""

    # Problem parameters
    pop_size = 300
    n_vars = 30
    max_gen = 20

    # Variable cardinalities (binary variables)
    cardinality = np.full(n_vars, 2)

    print("=" * 70)
    print("BMDA for OneMax Problem")
    print("=" * 70)
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Maximum generations: {max_gen}")
    print(f"Selection: Tournament (size=3)")
    print(f"Learning: BMDA with tree structure")
    print()

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnBMDA(structure_learning="tree", alpha=0.1),
        sampling=SampleFDA(n_samples=pop_size),
        selection=TournamentSelection(tournament_size=3, ratio=0.5),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=cardinality,
        components=components,
        random_seed=42,
    )

    # Run optimization
    statistics, cache = eda.run(cache_config=[0, 0, 1, 0, 0], verbose=True)

    # Print results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Best fitness found: {statistics.best_fitness_overall:.6f}")
    print(f"Generation found: {statistics.generation_found}")
    print(f"Best individual: {statistics.best_individual}")
    print(f"\nOptimal fitness (all 1s): {n_vars}")
    print(f"Achieved: {statistics.best_fitness_overall / n_vars * 100:.1f}%")

    return statistics, cache


if __name__ == "__main__":
    stats, cache = run_bmda_onemax()
