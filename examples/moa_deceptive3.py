"""
MOA (Markovianity Based Optimization Algorithm) for Deceptive3 Function

This script implements the MOA algorithm with Gibbs sampling to solve
Goldberg's Deceptive3 function, matching the MATLAB script:
ScriptsMateda/OptimizationScripts/MOA_Deceptive3.m

MOA uses local Markov neighborhoods to create a simpler network structure
than MN-FDA, making it efficient for problems with local dependencies.

Configuration matches MATLAB script:
- Population size: 500
- Problem size: 30 variables (10 blocks of 3 bits)
- Selection: Exponential selection (similar to Boltzmann)
- Learning: LearnMOA with k_neighbors=8, threshold_factor=1.5
- Sampling: Gibbs sampling (MOAGeneratePopulation in MATLAB)
- Replacement: Elitism (top 10 individuals)
- Temperature: Boltzmann linear schedule starting at 1.0

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- MATLAB implementation: ScriptsMateda/OptimizationScripts/MOA_Deceptive3.m
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.learning.moa import LearnMOA
from pateda.sampling.gibbs import SampleGibbs
from pateda.functions.discrete.deceptive import deceptive3


def run_moa_deceptive3():
    """
    Run MOA algorithm on Deceptive3 function

    This implementation matches the MATLAB script MOA_Deceptive3.m
    """
    print("=" * 80)
    print("MOA Algorithm for Deceptive3 Function")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Population size: 500")
    print("  - Problem size: 30 variables (10 blocks of 3 bits)")
    print("  - Algorithm: MOA with Gibbs sampling")
    print("  - Selection: Truncation (50% - similar to exponential)")
    print("  - Replacement: Elitism (10 best individuals)")
    print("  - Learning: k_neighbors=8, threshold_factor=1.5")
    print("  - Temperature: 1.0 (Boltzmann)")
    print()

    # Problem configuration
    n_vars = 30  # 10 blocks of 3 bits
    pop_size = 500
    max_generations = 100

    # Optimal fitness: 10 blocks × 1.0 = 10.0
    optimal_fitness = n_vars / 3.0

    print(f"Optimal fitness: {optimal_fitness:.1f}")
    print()

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),

        # In MATLAB: exponential selection with parameter 2
        # We use truncation selection (50%) as a simpler alternative
        # For true exponential, we would need to implement exp_selection
        selection=TruncationSelection(ratio=0.5),

        # MOA learning with parameters matching MATLAB:
        # MK_params(1:5) = {{},8,1.5,'Boltzman_linear',1.0}
        # Parameters: {}, k_neighbors=8, threshold_factor=1.5, ...
        learning=LearnMOA(
            k_neighbors=8,           # Max 8 neighbors per variable
            threshold_factor=1.5,    # Threshold for MI filtering
            prior=True,              # Laplace smoothing
        ),

        # Gibbs sampling with temperature for exploration
        # In MATLAB: MOAGeneratePopulation with 10 iterations
        sampling=SampleGibbs(
            n_samples=pop_size,
            IT=10,                   # Matches MATLAB parameter
            temperature=1.0,         # Boltzmann temperature
            random_order=True,
        ),

        # Elitism: keep best 10 individuals
        # In MATLAB: 'elitism',{10,'fitness_ordering'}
        replacement=ElitistReplacement(n_elite=10),

        # Stop condition
        stop_condition=MaxGenerations(max_gen=max_generations),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=deceptive3,
        cardinality=np.full(n_vars, 2),  # Binary variables
        components=components,
    )

    print("Running MOA...")
    print()

    # Run the algorithm
    stats, cache = eda.run(verbose=True)

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Best fitness found: {stats.best_fitness_overall:.4f}")
    print(f"Optimal fitness:    {optimal_fitness:.4f}")
    print(f"Success: {abs(stats.best_fitness_overall - optimal_fitness) < 0.01}")
    print()
    print(f"Generation found: {stats.generation_found}")
    print(f"Total generations: {stats.generation}")
    print()
    print(f"Best individual: {stats.best_individual}")
    print()

    # Decode the solution to show blocks
    print("Solution by blocks (optimal: all 111):")
    for i in range(0, n_vars, 3):
        block = stats.best_individual[i:i+3]
        block_str = ''.join(map(str, block.astype(int)))
        print(f"  Block {i//3 + 1}: {block_str}")
    print()

    return stats, cache


def run_moa_deceptive3_comparison():
    """
    Run multiple instances to compare performance
    """
    print("=" * 80)
    print("MOA on Deceptive3: Multiple Runs Comparison")
    print("=" * 80)
    print()

    n_runs = 10
    n_vars = 30
    pop_size = 500
    max_generations = 100
    optimal_fitness = n_vars / 3.0

    results = {
        'best_fitness': [],
        'generations_to_optimum': [],
        'success': [],
    }

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}...", end=' ')

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(proportion=0.5),
            learning=LearnMOA(k_neighbors=8, threshold_factor=1.5),
            sampling=SampleGibbs(n_samples=pop_size, IT=10, temperature=1.0),
            replacement=ElitistReplacement(n_elite=10),
            stop_condition=MaxGenerations(max_gen=max_generations),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=deceptive3,
            cardinality=np.full(n_vars, 2),
            components=components,
        )

        stats, _ = eda.run(verbose=False)

        # Record results
        results['best_fitness'].append(stats.best_fitness_overall)

        success = abs(stats.best_fitness_overall - optimal_fitness) < 0.01
        results['success'].append(success)

        if success:
            results['generations_to_optimum'].append(stats.generation_found)
            print(f"✓ Success at generation {stats.generation_found}")
        else:
            print(f"✗ Failed (fitness: {stats.best_fitness_overall:.4f})")

    # Print statistics
    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    best_fitness_array = np.array(results['best_fitness'])
    success_rate = np.mean(results['success']) * 100

    print(f"Success rate: {success_rate:.1f}% ({sum(results['success'])}/{n_runs})")
    print(f"Best fitness: {np.mean(best_fitness_array):.4f} ± {np.std(best_fitness_array):.4f}")
    print(f"Min fitness:  {np.min(best_fitness_array):.4f}")
    print(f"Max fitness:  {np.max(best_fitness_array):.4f}")

    if results['generations_to_optimum']:
        gen_array = np.array(results['generations_to_optimum'])
        print()
        print(f"Generations to optimum (successful runs):")
        print(f"  Mean: {np.mean(gen_array):.1f}")
        print(f"  Std:  {np.std(gen_array):.1f}")
        print(f"  Min:  {np.min(gen_array)}")
        print(f"  Max:  {np.max(gen_array)}")

    print()


if __name__ == "__main__":
    # Run single example
    stats, cache = run_moa_deceptive3()

    # Run comparison
    print("\n" * 2)
    run_moa_deceptive3_comparison()
