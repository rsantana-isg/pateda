"""
Bayesian Tree EDA with MPE Sampling for the Ising Model

This script implements a Bayesian Network Tree EDA with Most Probable Explanation
(MPE) sampling for the Ising spin glass problem, matching the MATLAB script:
ScriptsMateda/OptimizationScripts/BayesianTree_IsingModel.m

MPE sampling (also called MAP sampling) adds the most probable configuration from
the learned Bayesian network to the population during each generation, which can
significantly improve convergence.

Configuration matches MATLAB script:
- Population size: 500
- Problem size: 64 variables (8x8 Ising lattice)
- Learning: Bayesian Tree (LearnTreeModel)
- Sampling: MPE/MAP sampling (SampleMPE_BN in MATLAB)
- Stop condition: Max generations (150) OR optimal value found (86)
- Instance: 64-variable Ising model, instance 1

The Ising model is a spin glass optimization problem where we seek to find the
spin configuration that maximizes the energy (or minimizes the negative energy).

References:
- MATLAB implementation: ScriptsMateda/OptimizationScripts/BayesianTree_IsingModel.m
- Ising model: functions/ising-model/SG_64_1.txt (if available)

Note: This example requires an Ising model instance file. If not available,
we use a generated random instance for demonstration.
"""

import numpy as np
from pathlib import Path
from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import StopCondition
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.learning.tree import LearnTreeModel
from pateda.sampling.map_sampling import SampleInsertMAP


# Try to import Ising model functions
try:
    from pateda.functions.discrete.ising import load_ising, eval_ising
    ISING_AVAILABLE = True
except ImportError:
    ISING_AVAILABLE = False


class StopCriteriaMaxGenOrOptimum(StopCondition):
    """
    Stop condition: max generations OR optimal value found

    Matches MATLAB: 'maxgen_maxval',{MaxGen,MaxVal}
    """

    def __init__(self, max_generations: int, optimal_value: float, tolerance: float = 0.01):
        """
        Initialize stop criteria

        Args:
            max_generations: Maximum number of generations
            optimal_value: Optimal fitness value to reach
            tolerance: Tolerance for considering optimal value reached
        """
        self.max_generations = max_generations
        self.optimal_value = optimal_value
        self.tolerance = tolerance
        self.best_fitness_overall = -np.inf

    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params,
    ) -> bool:
        """
        Check if should stop based on max generations or optimal value

        Args:
            generation: Current generation number
            population: Current population
            fitness: Current fitness values
            **params: Additional parameters

        Returns:
            True if should stop, False otherwise
        """
        # Update best fitness seen so far
        current_best = np.max(fitness)
        if current_best > self.best_fitness_overall:
            self.best_fitness_overall = current_best

        # Stop if max generations reached
        if generation >= self.max_generations:
            return True

        # Stop if optimal value reached (within tolerance)
        if abs(self.best_fitness_overall - self.optimal_value) <= self.tolerance:
            return True

        return False

    def reset(self):
        """Reset the stop condition state"""
        self.best_fitness_overall = -np.inf


def create_ising_function(n_vars: int, instance: int = 1):
    """
    Create an Ising model fitness function

    Args:
        n_vars: Number of variables
        instance: Instance number

    Returns:
        Fitness function for the Ising model
    """
    if ISING_AVAILABLE:
        try:
            # Try to load from file
            lattice, inter = load_ising(n_vars, instance)

            def fitness_func(x):
                return eval_ising(x, lattice, inter)

            return fitness_func, lattice, inter

        except FileNotFoundError:
            print(f"Warning: Ising instance file not found. Using random instance.")

    # If not available or file not found, create a random instance
    print(f"Creating random {n_vars}-variable Ising model instance...")

    # Create a simple 2D lattice structure
    grid_size = int(np.sqrt(n_vars))
    if grid_size * grid_size != n_vars:
        raise ValueError(f"n_vars must be a perfect square, got {n_vars}")

    # Create neighbor structure (4-nearest neighbors on 2D grid)
    max_neighbors = 4
    lattice = np.zeros((n_vars, max_neighbors + 1), dtype=int)
    inter = np.random.uniform(-1, 1, (n_vars, max_neighbors))

    for i in range(n_vars):
        row = i // grid_size
        col = i % grid_size
        neighbors = []

        # Add 4 nearest neighbors (up, down, left, right)
        if row > 0:
            neighbors.append((row - 1) * grid_size + col)
        if row < grid_size - 1:
            neighbors.append((row + 1) * grid_size + col)
        if col > 0:
            neighbors.append(row * grid_size + (col - 1))
        if col < grid_size - 1:
            neighbors.append(row * grid_size + (col + 1))

        lattice[i, 0] = len(neighbors)
        for j, neighbor in enumerate(neighbors):
            lattice[i, j + 1] = neighbor + 1  # 1-indexed

    def fitness_func(x):
        r = 0.0
        for i in range(n_vars):
            if lattice[i, 0] > 0:
                for j in range(1, int(lattice[i, 0]) + 1):
                    neighbor_idx = int(lattice[i, j]) - 1
                    if i < neighbor_idx:  # Count each interaction once
                        if x[i] == x[neighbor_idx]:
                            r += inter[i, j - 1]

        return -r  # Negative because we want to maximize

    return fitness_func, lattice, inter


def run_bayesian_tree_ising_mpe():
    """
    Run Bayesian Tree EDA with MPE sampling on Ising model

    This implementation matches the MATLAB script BayesianTree_IsingModel.m
    """
    print("=" * 80)
    print("Bayesian Tree EDA with MPE Sampling for Ising Model")
    print("=" * 80)
    print()

    # Problem configuration (matches MATLAB)
    n_vars = 64  # 8x8 lattice
    pop_size = 500
    max_generations = 150
    optimal_value = 86.0  # Known optimal for this instance
    instance = 1

    print("Configuration:")
    print(f"  - Population size: {pop_size}")
    print(f"  - Problem size: {n_vars} variables (8x8 Ising lattice)")
    print(f"  - Algorithm: Bayesian Tree EDA")
    print(f"  - Sampling: MPE/MAP sampling")
    print(f"  - Stop condition: Max {max_generations} generations OR fitness >= {optimal_value}")
    print(f"  - Instance: {instance}")
    print()

    # Create Ising model fitness function
    fitness_func, lattice, inter = create_ising_function(n_vars, instance)

    print("Ising model created.")
    print()

    # Configure EDA components
    components = EDAComponents(
        seeding=RandomInit(),

        # Default selection (truncation 50%)
        selection=TruncationSelection(ratio=0.5),

        # Bayesian Tree learning
        # In MATLAB: default learning method is tree-based
        # Tree models always have max_parents=1 and use MI for edge weights
        learning=LearnTreeModel(
            alpha=0.1,  # Laplace smoothing
        ),

        # MPE/MAP sampling
        # In MATLAB: SampleMPE_BN
        # This samples using PLS and inserts the MAP (Most Probable Explanation)
        sampling=SampleInsertMAP(
            n_samples=pop_size,
            map_method='bp',  # Belief propagation for MAP inference
            n_map_inserts=1,  # Insert 1 MAP solution per generation
            k_map=1,  # Single MAP (not k-MAP)
            replace_worst=True,  # Replace worst individual
        ),

        # No replacement (generational)
        replacement=GenerationalReplacement(),

        # Stop condition: max generations OR optimal value
        stop_condition=StopCriteriaMaxGenOrOptimum(
            max_generations=max_generations,
            optimal_value=optimal_value,
            tolerance=0.1,
        ),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),  # Binary variables
        components=components,
        random_seed=42,
    )

    print("Running Bayesian Tree EDA with MPE sampling...")
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
    print(f"Optimal fitness:    {optimal_value:.4f}")
    print(f"Success: {abs(stats.best_fitness_overall - optimal_value) <= 0.1}")
    print()
    print(f"Generation found: {stats.generation_found}")
    print(f"Total generations: {stats.generation}")
    print()

    if stats.generation < max_generations:
        print(f"✓ Optimal solution found in {stats.generation_found} generations!")
    else:
        print(f"✗ Reached max generations ({max_generations})")

    print()
    print(f"Best solution: {stats.best_individual}")
    print()

    return stats, cache


def run_comparison_with_without_mpe():
    """
    Compare Bayesian Tree EDA with and without MPE sampling
    """
    print("=" * 80)
    print("Comparison: Bayesian Tree with vs without MPE Sampling")
    print("=" * 80)
    print()

    n_vars = 64
    pop_size = 500
    max_generations = 150
    optimal_value = 86.0
    n_runs = 5

    # Create Ising model
    fitness_func, _, _ = create_ising_function(n_vars, 1)

    results = {
        'With MPE': {'fitness': [], 'generations': []},
        'Without MPE (PLS)': {'fitness': [], 'generations': []},
    }

    # Test with MPE
    print("Running WITH MPE sampling...")
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=' ')

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnTreeModel(),
            sampling=SampleInsertMAP(n_samples=pop_size, map_method='bp'),
            replacement=GenerationalReplacement(),
            stop_condition=StopCriteriaMaxGenOrOptimum(max_generations, optimal_value),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=fitness_func,
            cardinality=np.full(n_vars, 2),
            components=components,
        random_seed=42,
        )

        stats, _ = eda.run(verbose=False)
        results['With MPE']['fitness'].append(stats.best_fitness_overall)
        results['With MPE']['generations'].append(stats.generation_found)
        print(f"Fitness: {stats.best_fitness_overall:.2f}, Gen: {stats.generation_found}")

    # Test without MPE (using standard PLS from FDA)
    print("\nRunning WITHOUT MPE sampling (standard PLS)...")
    from pateda.sampling.fda import SampleFDA

    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=' ')

        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnTreeModel(),
            sampling=SampleFDA(n_samples=pop_size),  # Standard PLS sampling
            replacement=GenerationalReplacement(),
            stop_condition=StopCriteriaMaxGenOrOptimum(max_generations, optimal_value),
        )

        eda = EDA(
            pop_size=pop_size,
            n_vars=n_vars,
            fitness_func=fitness_func,
            cardinality=np.full(n_vars, 2),
            components=components,
        random_seed=42,
        )

        stats, _ = eda.run(verbose=False)
        results['Without MPE (PLS)']['fitness'].append(stats.best_fitness_overall)
        results['Without MPE (PLS)']['generations'].append(stats.generation_found)
        print(f"Fitness: {stats.best_fitness_overall:.2f}, Gen: {stats.generation_found}")

    # Print comparison
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    print(f"{'Method':<25} {'Mean Fitness':<15} {'Mean Generations':<20}")
    print("-" * 80)

    for method in results:
        fitness_array = np.array(results[method]['fitness'])
        gen_array = np.array(results[method]['generations'])

        print(f"{method:<25} {np.mean(fitness_array):<15.2f} {np.mean(gen_array):<20.1f}")

    print()
    print("MPE sampling advantage:")
    mpe_fitness = np.mean(results['With MPE']['fitness'])
    pls_fitness = np.mean(results['Without MPE (PLS)']['fitness'])
    print(f"  Fitness improvement: {mpe_fitness - pls_fitness:+.2f}")

    mpe_gens = np.mean(results['With MPE']['generations'])
    pls_gens = np.mean(results['Without MPE (PLS)']['generations'])
    print(f"  Generations saved: {pls_gens - mpe_gens:.1f}")
    print()


if __name__ == "__main__":
    # Run single example
    stats, cache = run_bayesian_tree_ising_mpe()

    # Run comparison
    print("\n" * 2)
    run_comparison_with_without_mpe()
