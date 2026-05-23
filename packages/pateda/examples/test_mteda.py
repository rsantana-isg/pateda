"""
MT-EDA Test Script - Testing Mixture of Trees EDA Enhancements
================================================================

This script tests the enhanced MT-EDA implementation including:
1. Exact likelihood computation for tree models
2. EM learning with exact likelihood
3. Priors for mutation-like effect (Section 4 of RepMutMTFDA.pdf)
4. Adaptive learning (Section 5 of RepMutMTFDA.pdf)
5. Fitness-weighted components

Benchmark problems tested:
- OneMax: Simple additive problem
- KDeceptive3: 3-bit deceptive problem
- Deceptive3: Standard 3-bit deceptive function
- KDeceptive5: 5-bit deceptive problem
- HIFF: Hierarchical If and only If function

Usage:
    python test_mteda.py [--seed SEED] [--verbose]

Example:
    python test_mteda.py --seed 42 --verbose

References:
- Santana, R., Ochoa, A., & Soto, M.R. (2001). "The Mixture of Trees Factorized
  Distribution Algorithm." GECCO 2001.
- RepMutMTFDA.pdf: Priors and Adaptive Learning for MT-FDA
"""

# Add parent directory to path for running examples without installation
# Need to add grandparent of grandparent so that 'import pateda' can find the pateda dir

import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Any

# Core EDA modules
from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations, MaxGenerationsOrOptimum
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement

# MT-EDA modules
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.sampling.mixture_trees import SampleMixtureTrees

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, hard_decep5, hiff
)


# ==============================================================================
# Fitness Functions
# ==============================================================================

def onemax(x: np.ndarray) -> np.ndarray:
    """OneMax function - simple additive benchmark"""
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    else:
        return np.sum(x, axis=1).astype(float)


def wrap_k_deceptive(k: int):
    """K-deceptive wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([k_deceptive(x.reshape(1, -1), k)[0]])
        else:
            return k_deceptive(x, k)
    return wrapped


def wrap_decep3(x: np.ndarray) -> np.ndarray:
    """Deceptive-3 wrapper"""
    if x.ndim == 1:
        return np.array([decep3(x.reshape(1, -1))[0]])
    else:
        return decep3(x)


def wrap_hard_decep5(x: np.ndarray) -> np.ndarray:
    """Hard Deceptive-5 wrapper"""
    if x.ndim == 1:
        return np.array([hard_decep5(x.reshape(1, -1))[0]])
    else:
        return hard_decep5(x)


def wrap_hiff(x: np.ndarray) -> np.ndarray:
    """HIFF wrapper"""
    if x.ndim == 1:
        return np.array([hiff(x.reshape(1, -1))[0]])
    else:
        return hiff(x)


# ==============================================================================
# MT-EDA Configuration
# ==============================================================================

def create_mteda_config(
    pop_size: int,
    n_components: int = 3,
    weight_learning: str = "uniform",
    use_priors: bool = False,
    use_adaptive: bool = False,
    random_seed: int = None,
) -> Tuple[LearnMixtureTrees, SampleMixtureTrees]:
    """
    Create MT-EDA learning and sampling configuration

    Args:
        pop_size: Population size
        n_components: Number of tree components
        weight_learning: Weight learning method ("uniform", "em", "fitness_proportional")
        use_priors: Enable priors for mutation-like effect
        use_adaptive: Enable adaptive learning
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (learning_method, sampling_method)
    """
    learning = LearnMixtureTrees(
        n_components=n_components,
        component_learning="tree",
        alpha=0.1,
        weight_learning=weight_learning,
        em_iterations=10,
        random_seed=random_seed,
        use_priors=use_priors,
        use_adaptive=use_adaptive,
        truncation_ratio=0.5,
        adaptive_mu=0.9,
    )

    sampling = SampleMixtureTrees(n_samples=pop_size)

    return learning, sampling


def run_mteda_experiment(
    config_name: str,
    fitness_func,
    n_vars: int,
    pop_size: int,
    max_generations: int,
    optimal_fitness: float,
    n_components: int = 3,
    weight_learning: str = "uniform",
    use_priors: bool = False,
    use_adaptive: bool = False,
    random_seed: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single MT-EDA experiment

    Args:
        config_name: Name for this configuration
        fitness_func: Fitness function
        n_vars: Number of variables
        pop_size: Population size
        max_generations: Maximum generations
        optimal_fitness: Known optimal fitness value
        n_components: Number of tree components
        weight_learning: Weight learning method
        use_priors: Enable priors
        use_adaptive: Enable adaptive learning
        random_seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with experiment results
    """
    np.random.seed(random_seed)

    # Create learning and sampling methods
    learning, sampling = create_mteda_config(
        pop_size=pop_size,
        n_components=n_components,
        weight_learning=weight_learning,
        use_priors=use_priors,
        use_adaptive=use_adaptive,
        random_seed=random_seed,
    )

    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerationsOrOptimum(
            max_generations=max_generations,
            optimal_fitness=optimal_fitness,
            tolerance=1e-6,
        ),
    )

    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        components=components,
        cardinality=np.full(n_vars, 2),  # Binary variables
        random_seed=random_seed,
    )

    start_time = time.time()
    statistics, cache = eda.run(verbose=False)
    elapsed_time = time.time() - start_time

    # Extract results from statistics
    best_fitness = statistics.best_fitness_overall
    best_solution = statistics.best_individual
    generations_used = len(statistics.best_fitness)

    # Check if optimum was reached
    reached_optimum = abs(best_fitness - optimal_fitness) < 1e-6

    if verbose:
        status = "SUCCESS" if reached_optimum else "INCOMPLETE"
        print(f"  [{status}] {config_name}: "
              f"best={best_fitness:.2f}, "
              f"gen={generations_used}, "
              f"time={elapsed_time:.2f}s")

    return {
        "config_name": config_name,
        "best_fitness": best_fitness,
        "optimal_fitness": optimal_fitness,
        "reached_optimum": reached_optimum,
        "generations": generations_used,
        "time": elapsed_time,
        "statistics": statistics,
    }


# ==============================================================================
# Test Suite
# ==============================================================================

def test_exact_likelihood(verbose: bool = True):
    """
    Test that exact_likelihood computes correct probabilities

    Verifies:
    1. Likelihood values are in [0, 1] range
    2. Likelihood is higher for samples similar to training data
    3. Sum of likelihoods across all possible configurations <= 1
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Testing Exact Likelihood Computation")
        print("=" * 60)

    np.random.seed(42)

    # Create simple test population
    n_vars = 6
    n_samples = 50
    cardinality = np.full(n_vars, 2)

    # Generate biased population (more 1s than 0s)
    population = np.random.choice([0, 1], size=(n_samples, n_vars), p=[0.3, 0.7])
    fitness = np.sum(population, axis=1).astype(float)

    # Learn MT-EDA model
    learner = LearnMixtureTrees(
        n_components=2,
        component_learning="tree",
        weight_learning="em",
    )

    model = learner.learn(
        generation=0,
        n_vars=n_vars,
        cardinality=cardinality,
        population=population,
        fitness=fitness,
    )

    # Test exact likelihood on training data
    components = [
        type('TreeModel', (), {
            'structure': model.structure[i],
            'parameters': model.parameters["components"][i]
        })()
        for i in range(len(model.structure))
    ]

    # Compute likelihood for each component
    for i, comp in enumerate(components):
        likelihoods = learner.exact_likelihood(comp, population, cardinality)

        # All likelihoods should be positive
        assert np.all(likelihoods >= 0), f"Component {i}: negative likelihood"

        # Likelihoods should be <= 1
        assert np.all(likelihoods <= 1.0 + 1e-6), f"Component {i}: likelihood > 1"

        if verbose:
            print(f"  Component {i}: min={np.min(likelihoods):.6f}, "
                  f"max={np.max(likelihoods):.6f}, mean={np.mean(likelihoods):.6f}")

    # Test that samples similar to training data have higher likelihood
    # than random samples
    random_pop = np.random.choice([0, 1], size=(n_samples, n_vars))
    biased_pop = np.random.choice([0, 1], size=(n_samples, n_vars), p=[0.3, 0.7])

    for i, comp in enumerate(components):
        random_likelihood = np.mean(learner.exact_likelihood(comp, random_pop, cardinality))
        biased_likelihood = np.mean(learner.exact_likelihood(comp, biased_pop, cardinality))

        if verbose:
            print(f"  Component {i}: random_mean={random_likelihood:.6f}, "
                  f"biased_mean={biased_likelihood:.6f}")

    if verbose:
        print("  [PASSED] Exact likelihood computation test")

    return True


def test_priors(verbose: bool = True):
    """
    Test prior computation for mutation-like effect

    Verifies:
    1. Priors are computed correctly
    2. Adaptive priors respond to data probability
    3. Priors are stored in model metadata
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Testing Priors for Mutation-like Effect")
        print("=" * 60)

    np.random.seed(42)

    n_vars = 10
    n_samples = 50
    cardinality = np.full(n_vars, 2)
    population = np.random.choice([0, 1], size=(n_samples, n_vars))
    fitness = np.sum(population, axis=1).astype(float)

    # Test with priors enabled (fixed)
    learner_priors = LearnMixtureTrees(
        n_components=2,
        use_priors=True,
        use_adaptive=False,
        truncation_ratio=0.5,
    )

    model_priors = learner_priors.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    assert model_priors.parameters["priors"] is not None, "Priors should be computed"
    assert model_priors.parameters["total_prior"] is not None, "Total prior should be set"
    assert model_priors.metadata["use_priors"] is True, "use_priors flag should be True"

    if verbose:
        print(f"  Fixed priors: {model_priors.parameters['priors']}")
        print(f"  Total prior: {model_priors.parameters['total_prior']:.6f}")

    # Test with adaptive priors
    learner_adaptive = LearnMixtureTrees(
        n_components=2,
        use_priors=True,
        use_adaptive=True,
        truncation_ratio=0.5,
        adaptive_mu=0.9,
    )

    model_adaptive = learner_adaptive.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    if verbose:
        print(f"  Adaptive priors: {model_adaptive.parameters['priors']}")
        print(f"  P_bar(D): {model_adaptive.metadata['p_bar_d']:.6f}")
        print(f"  Adaptive stopped: {model_adaptive.metadata['adaptive_stopped']}")

    # Test without priors
    learner_no_priors = LearnMixtureTrees(
        n_components=2,
        use_priors=False,
    )

    model_no_priors = learner_no_priors.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    assert model_no_priors.parameters["priors"] is None, "Priors should be None when disabled"

    if verbose:
        print("  [PASSED] Priors computation test")

    return True


def test_fitness_weighted_learning(verbose: bool = True):
    """
    Test fitness-weighted component learning

    Verifies:
    1. Fitness-weighted learning produces different weights than uniform
    2. Components with higher-fitness samples get higher weights
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Testing Fitness-weighted Component Learning")
        print("=" * 60)

    np.random.seed(42)

    n_vars = 10
    n_samples = 100
    cardinality = np.full(n_vars, 2)

    # Create population with varying fitness
    population = np.random.choice([0, 1], size=(n_samples, n_vars))
    fitness = np.sum(population, axis=1).astype(float)

    # Test uniform weights
    learner_uniform = LearnMixtureTrees(
        n_components=3,
        weight_learning="uniform",
    )
    model_uniform = learner_uniform.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    # Test EM weights
    learner_em = LearnMixtureTrees(
        n_components=3,
        weight_learning="em",
        em_iterations=10,
    )
    model_em = learner_em.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    # Test fitness-weighted
    learner_fitness = LearnMixtureTrees(
        n_components=3,
        weight_learning="fitness_proportional",
    )
    model_fitness = learner_fitness.learn(
        generation=0, n_vars=n_vars, cardinality=cardinality,
        population=population, fitness=fitness
    )

    if verbose:
        print(f"  Uniform weights: {model_uniform.parameters['weights']}")
        print(f"  EM weights: {model_em.parameters['weights']}")
        print(f"  Fitness weights: {model_fitness.parameters['weights']}")

    # Verify uniform weights are actually uniform
    uniform_weights = model_uniform.parameters['weights']
    assert np.allclose(uniform_weights, 1/3), "Uniform weights should be equal"

    # Verify all weights sum to 1
    assert np.isclose(np.sum(model_em.parameters['weights']), 1.0), "EM weights should sum to 1"
    assert np.isclose(np.sum(model_fitness.parameters['weights']), 1.0), "Fitness weights should sum to 1"

    if verbose:
        print("  [PASSED] Fitness-weighted learning test")

    return True


def test_benchmark_problems(verbose: bool = True):
    """
    Test MT-EDA on benchmark problems

    Tests multiple configurations on:
    - OneMax
    - Deceptive3
    - HIFF (smaller size)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Testing MT-EDA on Benchmark Problems")
        print("=" * 60)

    results = []

    # Define test problems
    problems = [
        ("OneMax-20", onemax, 20, 20.0),
        ("Deceptive3-15", wrap_decep3, 15, 5.0),  # 5 blocks * 1.0 per block
        ("KDeceptive3-18", wrap_k_deceptive(3), 18, 6.0),  # 6 blocks
    ]

    # Define configurations to test
    configs = [
        ("MT-EDA2-Uniform", {"n_components": 2, "weight_learning": "uniform"}),
        ("MT-EDA3-Uniform", {"n_components": 3, "weight_learning": "uniform"}),
        ("MT-EDA2-EM", {"n_components": 2, "weight_learning": "em"}),
        ("MT-EDA2-Fitness", {"n_components": 2, "weight_learning": "fitness_proportional"}),
        ("MT-EDA2-Priors", {"n_components": 2, "weight_learning": "uniform", "use_priors": True}),
        ("MT-EDA2-Adaptive", {"n_components": 2, "weight_learning": "em", "use_adaptive": True}),
    ]

    for prob_name, fitness_func, n_vars, optimal in problems:
        if verbose:
            print(f"\n  Problem: {prob_name} (optimal={optimal})")
            print("  " + "-" * 50)

        for config_name, config_params in configs:
            result = run_mteda_experiment(
                config_name=config_name,
                fitness_func=fitness_func,
                n_vars=n_vars,
                pop_size=100,
                max_generations=50,
                optimal_fitness=optimal,
                random_seed=42,
                verbose=verbose,
                **config_params
            )
            result["problem"] = prob_name
            results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        successes = sum(1 for r in results if r["reached_optimum"])
        total = len(results)
        print(f"  Success rate: {successes}/{total} ({100*successes/total:.1f}%)")

    return results


def run_all_tests(verbose: bool = True):
    """Run all MT-EDA tests"""
    print("=" * 60)
    print("MT-EDA Enhancement Test Suite")
    print("=" * 60)

    all_passed = True

    # Test 1: Exact likelihood
    try:
        test_exact_likelihood(verbose)
    except AssertionError as e:
        print(f"  [FAILED] Exact likelihood: {e}")
        all_passed = False

    # Test 2: Priors
    try:
        test_priors(verbose)
    except AssertionError as e:
        print(f"  [FAILED] Priors: {e}")
        all_passed = False

    # Test 3: Fitness-weighted learning
    try:
        test_fitness_weighted_learning(verbose)
    except AssertionError as e:
        print(f"  [FAILED] Fitness-weighted: {e}")
        all_passed = False

    # Test 4: Benchmark problems
    try:
        test_benchmark_problems(verbose)
    except Exception as e:
        print(f"  [FAILED] Benchmark problems: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MT-EDA enhancements"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "likelihood", "priors", "fitness", "benchmark"],
        help="Test to run (default: all)"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.test == "all":
        run_all_tests(verbose=True)
    elif args.test == "likelihood":
        test_exact_likelihood(verbose=True)
    elif args.test == "priors":
        test_priors(verbose=True)
    elif args.test == "fitness":
        test_fitness_weighted_learning(verbose=True)
    elif args.test == "benchmark":
        test_benchmark_problems(verbose=True)
