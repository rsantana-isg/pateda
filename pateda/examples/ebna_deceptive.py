"""
EBNA for Deceptive problem

This example demonstrates using EBNA (Estimation of Bayesian Networks Algorithm)
to solve a deceptive problem with ranking selection.
"""

import numpy as np
from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnEBNA
from pateda.sampling import SampleBayesianNetwork
from pateda.selection import RankingSelection
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.deceptive import deceptive3


def run_ebna_deceptive():
    """Run EBNA on deceptive problem"""

    # Problem parameters
    pop_size = 400
    n_vars = 30  # Should be multiple of 3 for deceptive3 function
    max_gen = 30

    # Variable cardinalities (binary variables)
    cardinality = np.full(n_vars, 2)

    print("=" * 70)
    print("EBNA for Deceptive Problem")
    print("=" * 70)
    print(f"Population size: {pop_size}")
    print(f"Number of variables: {n_vars}")
    print(f"Maximum generations: {max_gen}")
    print(f"Selection: Ranking (linear, pressure=1.8)")
    print(f"Learning: EBNA with max 2 parents per variable")
    print()

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        learning=LearnEBNA(max_parents=2, score_metric="bic", alpha=0.1),
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        selection=RankingSelection(
            ratio=0.5, selection_pressure=1.8, ranking_type="linear"
        ),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )

    # Create EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=deceptive3,
        cardinality=cardinality,
        components=components,
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

    # For deceptive3 function, optimal is all 1s with fitness = n_vars/3
    optimal_fitness = n_vars / 3
    print(f"\nOptimal fitness: {optimal_fitness}")
    print(f"Achieved: {statistics.best_fitness_overall / optimal_fitness * 100:.1f}%")

    return statistics, cache


if __name__ == "__main__":
    stats, cache = run_ebna_deceptive()
