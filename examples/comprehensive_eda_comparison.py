"""
Comprehensive EDA Algorithm Comparison

This script provides a systematic comparison of multiple EDA algorithms
on various benchmark problems. This is a NEW comprehensive test not present
in the MATLAB ScriptsMateda.

Algorithms tested:
1. UMDA (Univariate Marginal Distribution Algorithm)
2. EBNA (Estimation of Bayesian Network Algorithm)
3. Tree EDA (Tree-based FDA)
4. Affinity EDA
5. MOA (Markovianity Based Optimization Algorithm)

Problems tested:
1. OneMax (separable)
2. Deceptive3 (deceptive, 3-variable blocks)
3. Trap-5 (deceptive, 5-variable blocks)
4. NK Landscape (epistatic with varying K)

Metrics:
- Best fitness achieved
- Generations to convergence
- Success rate (reaching optimum)
- Robustness (std across runs)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement, GenerationalReplacement

# Import learning methods
from pateda.learning.histogram import LearnHistogram
from pateda.learning.bayesian_network import LearnBayesianNetwork
from pateda.learning.tree import LearnTreeModel
from pateda.learning.affinity import LearnAffinityModel
from pateda.learning.moa import LearnMOA

# Import sampling methods
from pateda.sampling.histogram import SampleHistogram
from pateda.sampling.bayesian_network import SampleBN
from pateda.sampling.fda import SampleFDA
from pateda.sampling.gibbs import SampleGibbs

# Import test functions
from pateda.functions.discrete.deceptive import deceptive3
from pateda.functions.discrete.trap import trap_k


@dataclass
class AlgorithmConfig:
    """Configuration for an EDA algorithm"""
    name: str
    learning: any
    sampling: any
    replacement: any = None


@dataclass
class BenchmarkResult:
    """Results for a single algorithm on a single problem"""
    algorithm: str
    problem: str
    best_fitness: List[float]
    generations_found: List[int]
    success_rate: float
    mean_fitness: float
    std_fitness: float
    mean_generations: float


def onemax(x: np.ndarray) -> float:
    """OneMax function - simple separable problem"""
    return float(np.sum(x))


def create_nk_landscape(n: int, k: int, seed: int):
    """Create NK landscape function"""
    from pateda.functions.discrete.nk_landscape import NKLandscape
    nk = NKLandscape(n, k, seed=seed)
    return nk.evaluate


def get_algorithms(pop_size: int) -> List[AlgorithmConfig]:
    """
    Get list of algorithms to compare

    Args:
        pop_size: Population size

    Returns:
        List of algorithm configurations
    """
    return [
        AlgorithmConfig(
            name="UMDA",
            learning=LearnHistogram(),
            sampling=SampleHistogram(pop_size),
            replacement=GenerationalReplacement(),
        ),
        AlgorithmConfig(
            name="EBNA",
            learning=LearnBayesianNetwork(
                structure_algorithm='k2',
                max_parents=3,
                scoring_metric='bic'
            ),
            sampling=SampleBN(pop_size),
            replacement=GenerationalReplacement(),
        ),
        AlgorithmConfig(
            name="Tree EDA",
            learning=LearnTreeModel(
                max_parents=1,
                scoring_method='MI'
            ),
            sampling=SampleFDA(pop_size),
            replacement=GenerationalReplacement(),
        ),
        AlgorithmConfig(
            name="Affinity EDA",
            learning=LearnAffinityModel(
                damping=0.5,
                max_iter=200
            ),
            sampling=SampleFDA(pop_size),
            replacement=GenerationalReplacement(),
        ),
        AlgorithmConfig(
            name="MOA",
            learning=LearnMOA(
                k_neighbors=5,
                threshold_factor=1.5
            ),
            sampling=SampleGibbs(
                n_samples=pop_size,
                IT=4,
                temperature=1.0
            ),
            replacement=GenerationalReplacement(),
        ),
    ]


def run_single_experiment(
    algorithm: AlgorithmConfig,
    fitness_func: callable,
    n_vars: int,
    optimal_fitness: float,
    pop_size: int = 200,
    max_generations: int = 100,
    n_runs: int = 10
) -> BenchmarkResult:
    """
    Run a single algorithm on a single problem

    Args:
        algorithm: Algorithm configuration
        fitness_func: Fitness function
        n_vars: Number of variables
        optimal_fitness: Known optimal fitness
        pop_size: Population size
        max_generations: Maximum generations
        n_runs: Number of independent runs

    Returns:
        Benchmark results
    """
    best_fitness_list = []
    generations_found_list = []
    successes = 0

    for run in range(n_runs):
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(proportion=0.5),
            learning=algorithm.learning,
            sampling=algorithm.sampling,
            replacement=algorithm.replacement if algorithm.replacement else GenerationalReplacement(),
            stop_condition=MaxGenerations(max_gen=max_generations),
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

        best_fitness_list.append(stats.best_fitness_overall)
        generations_found_list.append(stats.generation_found)

        # Check if optimal found (within tolerance)
        if abs(stats.best_fitness_overall - optimal_fitness) < 0.01:
            successes += 1

    success_rate = successes / n_runs

    return BenchmarkResult(
        algorithm=algorithm.name,
        problem="",  # Will be filled by caller
        best_fitness=best_fitness_list,
        generations_found=generations_found_list,
        success_rate=success_rate,
        mean_fitness=np.mean(best_fitness_list),
        std_fitness=np.std(best_fitness_list),
        mean_generations=np.mean(generations_found_list),
    )


def run_comprehensive_comparison():
    """
    Run comprehensive comparison of all algorithms on all problems
    """
    print("=" * 80)
    print("COMPREHENSIVE EDA ALGORITHM COMPARISON")
    print("=" * 80)
    print()

    # Problem configurations
    problems = [
        {
            'name': 'OneMax-30',
            'func': onemax,
            'n_vars': 30,
            'optimal': 30.0,
            'pop_size': 200,
            'max_gen': 50,
        },
        {
            'name': 'Deceptive3-30',
            'func': deceptive3,
            'n_vars': 30,
            'optimal': 10.0,  # 10 blocks × 1.0
            'pop_size': 300,
            'max_gen': 100,
        },
        {
            'name': 'Trap5-25',
            'func': lambda x: trap_k(x, k=5),
            'n_vars': 25,
            'optimal': 30.0,  # 5 blocks × 6.0
            'pop_size': 300,
            'max_gen': 150,
        },
    ]

    n_runs = 10
    algorithms = get_algorithms(pop_size=200)  # Will be overridden per problem

    all_results = []

    # Run experiments
    for problem in problems:
        print(f"\n{'=' * 80}")
        print(f"Problem: {problem['name']}")
        print(f"{'=' * 80}")
        print()

        # Update algorithms with correct pop_size
        algorithms = get_algorithms(problem['pop_size'])

        for algorithm in algorithms:
            print(f"  Running {algorithm.name}...", end=' ')

            result = run_single_experiment(
                algorithm=algorithm,
                fitness_func=problem['func'],
                n_vars=problem['n_vars'],
                optimal_fitness=problem['optimal'],
                pop_size=problem['pop_size'],
                max_generations=problem['max_gen'],
                n_runs=n_runs,
            )

            result.problem = problem['name']
            all_results.append(result)

            print(f"Success: {result.success_rate:.1%}, "
                  f"Mean fitness: {result.mean_fitness:.2f} ± {result.std_fitness:.2f}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print()

    for problem in problems:
        print(f"\n{problem['name']} (Optimal: {problem['optimal']:.1f})")
        print("-" * 80)
        print(f"{'Algorithm':<15} {'Success':<10} {'Mean Fit':<12} {'Std Fit':<12} {'Mean Gen':<12}")
        print("-" * 80)

        problem_results = [r for r in all_results if r.problem == problem['name']]

        for result in problem_results:
            print(f"{result.algorithm:<15} {result.success_rate:<10.1%} "
                  f"{result.mean_fitness:<12.3f} {result.std_fitness:<12.3f} "
                  f"{result.mean_generations:<12.1f}")

    # Create visualizations
    create_comparison_plots(all_results, problems)

    return all_results


def create_comparison_plots(results: List[BenchmarkResult], problems: List[Dict]):
    """
    Create visualization plots for comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Success Rate Comparison
    ax1 = axes[0, 0]
    for problem in problems:
        problem_results = [r for r in results if r.problem == problem['name']]
        algorithms = [r.algorithm for r in problem_results]
        success_rates = [r.success_rate * 100 for r in problem_results]

        x = np.arange(len(algorithms))
        ax1.bar(x + problems.index(problem) * 0.25, success_rates,
               width=0.25, label=problem['name'])

    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate Comparison')
    ax1.set_xticks(x + 0.25)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Mean Fitness Comparison
    ax2 = axes[0, 1]
    for problem in problems:
        problem_results = [r for r in results if r.problem == problem['name']]
        algorithms = [r.algorithm for r in problem_results]
        mean_fitness = [r.mean_fitness for r in problem_results]

        x = np.arange(len(algorithms))
        ax2.bar(x + problems.index(problem) * 0.25, mean_fitness,
               width=0.25, label=problem['name'])

    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Mean Best Fitness')
    ax2.set_title('Mean Fitness Comparison')
    ax2.set_xticks(x + 0.25)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Robustness (Std Dev)
    ax3 = axes[1, 0]
    for problem in problems:
        problem_results = [r for r in results if r.problem == problem['name']]
        algorithms = [r.algorithm for r in problem_results]
        std_fitness = [r.std_fitness for r in problem_results]

        x = np.arange(len(algorithms))
        ax3.bar(x + problems.index(problem) * 0.25, std_fitness,
               width=0.25, label=problem['name'])

    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Fitness Std Dev')
    ax3.set_title('Robustness (Lower is Better)')
    ax3.set_xticks(x + 0.25)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Efficiency (Mean Generations)
    ax4 = axes[1, 1]
    for problem in problems:
        problem_results = [r for r in results if r.problem == problem['name']]
        algorithms = [r.algorithm for r in problem_results]
        mean_gen = [r.mean_generations for r in problem_results]

        x = np.arange(len(algorithms))
        ax4.bar(x + problems.index(problem) * 0.25, mean_gen,
               width=0.25, label=problem['name'])

    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Mean Generations to Best')
    ax4.set_title('Efficiency (Lower is Better)')
    ax4.set_xticks(x + 0.25)
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_comparison_results.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: eda_comparison_results.png")


if __name__ == "__main__":
    results = run_comprehensive_comparison()

    print()
    print("=" * 80)
    print("OBSERVATIONS")
    print("=" * 80)
    print()
    print("Expected trends:")
    print("- UMDA: Good on separable problems (OneMax), struggles with dependencies")
    print("- EBNA: Better on epistatic problems (Trap, NK) due to BN dependencies")
    print("- Tree EDA: Good balance, efficient learning")
    print("- Affinity EDA: Automatic structure discovery, variable performance")
    print("- MOA: Good on problems with local dependencies, MCMC sampling overhead")
    print()
    print("Use these results to select appropriate algorithms for your problem!")
    print()
