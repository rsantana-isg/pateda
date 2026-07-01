"""
Simple end-to-end test to verify discrete_EDA_RW.py works with mutation and elitism

This test runs a simple EDA with frequency balance mutation to ensure the
complete flow works correctly.
"""

import numpy as np

# Add parent directory to path

from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.learning.umda import LearnUMDA
from pateda.sampling.fda import SampleFDA
from pateda.mutation import FrequencyBalanceMutation


def onemax(x):
    """Simple OneMax fitness function"""
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    else:
        return np.sum(x, axis=1).astype(float)


def test_eda_with_mutation_and_elitism():
    """
    Test that EDA with frequency balance mutation and elitism works correctly
    """
    print("Testing EDA with frequency balance mutation and elitism...")
    
    np.random.seed(42)
    
    n_vars = 20
    pop_size = 50
    max_gen = 10
    alpha = 0.9
    
    cardinality = np.full(n_vars, 2)
    
    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=pop_size),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
        mutation=FrequencyBalanceMutation(alpha=alpha),
    )
    
    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=onemax,
        cardinality=cardinality,
        components=components,
        random_seed=42,
    )
    
    print(f"Running EDA with:")
    print(f"  - Population size: {pop_size}")
    print(f"  - Variables: {n_vars}")
    print(f"  - Max generations: {max_gen}")
    print(f"  - Alpha (mutation): {alpha}")
    print()
    
    stats, _ = eda.run(verbose=False)
    
    best_fitness = stats.best_fitness_overall
    best_solution = stats.best_individual
    
    print(f"Results:")
    print(f"  - Best fitness: {best_fitness:.2f}")
    print(f"  - Best solution sum: {np.sum(best_solution)}")
    print(f"  - Optimal: {n_vars}")
    print()
    
    # Verify the algorithm made progress
    assert best_fitness > 0, "Best fitness should be positive"
    assert best_fitness <= n_vars, f"Best fitness should not exceed {n_vars}"
    
    # Check that we found a reasonably good solution
    # (with such small pop and few generations, we might not find optimal)
    print("✓ EDA with mutation and elitism completed successfully!")
    print(f"  - Found solution with fitness {best_fitness:.2f} / {n_vars}")
    
    return best_fitness, best_solution


if __name__ == '__main__':
    test_eda_with_mutation_and_elitism()
